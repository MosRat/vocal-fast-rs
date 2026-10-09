//! Fixed RMVPE graph and Applio/WebUI rolling pitch history. The pending history
//! is committed only after Synth succeeds; failed inference cannot advance F0.
use crate::{CpuConfig, CpuError, CpuF0Config, CpuGraph};
use vocal_graph::{
    Precision, PrecisionConfig,
    graph::{DType, GraphBuilder, Op},
    models,
};
use vocal_model::ModelPackage;

pub(crate) const HISTORY_FRAMES: usize = 1024;

pub(crate) fn build_graph(
    package: &ModelPackage,
    config: &CpuConfig,
) -> Result<GraphBuilder, CpuError> {
    let samples = config.f0.samples(config.window_samples);
    let g = GraphBuilder::default();
    let wave = g.input("waveform", &[1, samples], DType::F32);
    let threshold = g.input("threshold", &[1], DType::F32);
    let basis = g.parameter(package, "rmvpe.mel_basis", Precision::Fp32)?;
    let mel = g.node(
        Op::StftMel,
        &[&wave, &basis],
        vec![1, 128, samples / 160 + 1],
        DType::F32,
        vec![],
        [0.; 4],
        vec![],
        "rmvpe.stft_mel".into(),
    );
    let f0 = models::rmvpe_from_mel(
        &g,
        package,
        &mel,
        &PrecisionConfig::fp32(),
        &threshold,
        false,
    )?;
    g.mark("pitchf", &f0);
    Ok(g)
}

#[derive(Clone)]
struct PitchHistory([f32; HISTORY_FRAMES]);
impl Default for PitchHistory {
    fn default() -> Self {
        Self([0.; HISTORY_FRAMES])
    }
}
impl PitchHistory {
    fn next(&self, raw: &[f32], advance: usize) -> Result<Self, CpuError> {
        if raw.len() <= 4
            || advance == 0
            || advance > raw.len() - 4
            || raw.len() - 4 > HISTORY_FRAMES
            || raw.iter().any(|value| !value.is_finite() || *value < 0.)
        {
            return Err(CpuError::Runtime(
                "invalid rolling RMVPE output/advance".into(),
            ));
        }
        let mut next = self.clone();
        next.0.copy_within(advance.., 0);
        let interior = &raw[3..raw.len() - 1];
        next.0[HISTORY_FRAMES - interior.len()..].copy_from_slice(interior);
        Ok(next)
    }
}

pub(crate) struct CpuPitchStage {
    pub graph: CpuGraph,
    policy: CpuF0Config,
    raw: Vec<f32>,
    history: PitchHistory,
    pending: Option<PitchHistory>,
}
impl CpuPitchStage {
    pub fn new(graph: CpuGraph, policy: CpuF0Config) -> Self {
        Self {
            graph,
            policy,
            raw: Vec::new(),
            history: PitchHistory::default(),
            pending: None,
        }
    }
    pub fn reset(&mut self) {
        self.history.0.fill(0.);
        self.pending = None;
    }
    pub fn run(&mut self, wave: &[f32], aligned: &mut Vec<f32>) -> Result<(), CpuError> {
        let threshold = 0.03f32.to_le_bytes();
        match self.policy {
            CpuF0Config::FullWindow => self.graph.run_f32_into(
                &[
                    ("waveform", crate::runtime::bytes(wave)),
                    ("threshold", &threshold),
                ],
                "pitchf",
                aligned,
            ),
            CpuF0Config::Rolling {
                advance_samples,
                rmvpe_samples,
            } => {
                self.graph.run_f32_into(
                    &[
                        (
                            "waveform",
                            crate::runtime::bytes(&wave[wave.len() - rmvpe_samples..]),
                        ),
                        ("threshold", &threshold),
                    ],
                    "pitchf",
                    &mut self.raw,
                )?;
                if self.raw.len() != rmvpe_samples / 160 + 1 {
                    return Err(CpuError::Runtime(
                        "unexpected rolling RMVPE frame count".into(),
                    ));
                }
                let next = self.history.next(&self.raw, advance_samples / 160)?;
                aligned.clear();
                aligned.extend_from_slice(&next.0[HISTORY_FRAMES - wave.len() / 160..]);
                self.pending = Some(next);
                Ok(())
            }
        }
    }
    pub fn commit(&mut self) {
        if let Some(next) = self.pending.take() {
            self.history = next;
        }
    }
    pub fn raw<'a>(&'a self, aligned: &'a [f32]) -> &'a [f32] {
        match self.policy {
            CpuF0Config::FullWindow => aligned,
            CpuF0Config::Rolling { .. } => &self.raw,
        }
    }
    pub fn history(&self) -> Option<&[f32]> {
        match self.policy {
            CpuF0Config::FullWindow => None,
            CpuF0Config::Rolling { .. } => Some(&self.history.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_advances_one_block_and_overwrites_overlapping_interior() {
        let initial = PitchHistory::default();
        let raw = (0..32).map(|i| i as f32).collect::<Vec<_>>();
        let first = initial.next(&raw, 25).unwrap();
        assert!(initial.0.iter().all(|value| *value == 0.));
        assert_eq!(&first.0[996..], &raw[3..31]);
        assert!(first.0[..996].iter().all(|value| *value == 0.));
        let raw2 = raw.iter().map(|value| value + 100.).collect::<Vec<_>>();
        let second = first.next(&raw2, 25).unwrap();
        assert_eq!(&second.0[971..996], &raw[3..28]);
        assert_eq!(&second.0[996..], &raw2[3..31]);
        // A prepared result leaves committed history untouched until accepted.
        assert_eq!(&first.0[996..], &raw[3..31]);
    }

    #[test]
    fn invalid_frames_do_not_produce_history() {
        let history = PitchHistory::default();
        assert!(history.next(&[1.; 32], 29).is_err());
        assert!(history.next(&[1.; 32], 0).is_err());
        assert!(history.next(&[f32::NAN; 32], 25).is_err());
        assert!(history.next(&[-1.; 32], 25).is_err());
    }

    #[test]
    fn stage_reset_and_discarded_inference_preserve_history_contract() {
        let g = GraphBuilder::default();
        let wave = g.input("waveform", &[1, 4960], DType::F32);
        let threshold = g.input("threshold", &[1], DType::F32);
        let zero = g.mul(&threshold, &g.scalar(0., Precision::Fp32));
        let raw = g.add(&g.slice(&wave, 1, 0, 32, 1), &zero);
        g.mark("pitchf", &raw);
        let config = CpuConfig {
            threads: 1,
            ..CpuConfig::default()
        };
        let graph = CpuGraph::compile(g, &config).unwrap();
        let mut stage = CpuPitchStage::new(graph, CpuF0Config::rvc_realtime());
        let mut aligned = Vec::new();
        stage.run(&vec![120.; 44960], &mut aligned).unwrap();
        let first = aligned.clone();
        assert_eq!(aligned.len(), 281);
        assert_eq!(&aligned[253..], &[120.; 28]);
        assert!(stage.history().unwrap().iter().all(|value| *value == 0.));
        stage.commit();
        // Simulate a failed downstream Synth: prepare, but do not commit.
        stage.run(&vec![240.; 44960], &mut aligned).unwrap();
        stage.run(&vec![360.; 44960], &mut aligned).unwrap();
        assert_eq!(&aligned[228..253], &[120.; 25]);
        assert_eq!(&aligned[253..], &[360.; 28]);
        stage.commit();
        stage.reset();
        assert!(stage.history().unwrap().iter().all(|value| *value == 0.));
        stage.run(&vec![120.; 44960], &mut aligned).unwrap();
        assert_eq!(aligned, first);
    }
}
