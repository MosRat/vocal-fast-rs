//! Full-context file inference with bounded stage lifetimes.
//!
//! Stages exchange host tensors for offline conversion. Releasing each compiled
//! graph before constructing the next bounds GPU memory without shortening the
//! HuBERT/F0 context. Retrieval alone is batched: queries are independent.
use crate::{
    DirectMlConfig, DirectMlControls, DirectMlDevice, DirectMlError, DirectMlSources, F0Config,
    GpuTimings, GraphOutput, Precision,
    graph::{DType, GraphBuilder, Op},
    models,
    pipeline::{fill_normal, frontend},
};
use std::{sync::Arc, time::Instant};
use vocal_model::{ModelPackage, RetrievalIndex};

pub struct DirectMlOfflineGraph {
    config: DirectMlConfig,
    synth: Arc<ModelPackage>,
    hubert: Arc<ModelPackage>,
    rmvpe: Arc<ModelPackage>,
    index: Option<RetrievalIndex>,
    device: DirectMlDevice,
    pub frames: usize,
    pub hubert_frames: usize,
    pub latent_channels: usize,
    pub feature_width: usize,
    pub sample_rate: u32,
    seed: u64,
}
pub struct DirectMlOfflineOutput {
    pub audio: Vec<f32>,
    pub preparation_ms: f64,
    pub inference_ms: f64,
    pub debug: GraphOutput,
}
fn bytes(values: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}
fn stage(error: DirectMlError, name: &str) -> DirectMlError {
    DirectMlError::Runtime(format!("offline {name}: {error}"))
}
impl DirectMlOfflineGraph {
    pub fn config(&self) -> &DirectMlConfig {
        &self.config
    }
    pub fn from_sources(sources: &DirectMlSources) -> Result<Self, DirectMlError> {
        let config = sources.config.clone();
        config.validate()?;
        if config.append_hubert_frame
            || config.skip_head_frames != 0
            || config.f0 != F0Config::FullWindow
        {
            return Err(DirectMlError::Config(
                "offline graphs require full context and no tail crop".into(),
            ));
        }
        let synth = vocal_model::load_voice_cache_shared(&sources.synth)?;
        let hubert = vocal_model::load_cache_shared(&sources.hubert)?;
        let rmvpe = vocal_model::load_cache_shared(&sources.rmvpe)?;
        let c = synth.validate_rvc_synth()?;
        let version = vocal_model::RvcVersion::from_feature_width(c.feature_width)?;
        hubert.validate_hubert_for_rvc(version)?;
        let index = if sources.enable_retrieval {
            RetrievalIndex::from_package(&synth)?
        } else {
            None
        };
        if sources.enable_retrieval && index.is_none() {
            return Err(DirectMlError::Config(
                "retrieval enabled without an index".into(),
            ));
        }
        if index
            .as_ref()
            .is_some_and(|index| index.dimensions() != version.feature_width())
        {
            return Err(DirectMlError::Config(
                "retrieval index dimensions disagree with RVC model version".into(),
            ));
        }
        let hubert_frames = (config.window_samples - 400) / 320 + 1;
        let device = DirectMlDevice::from_config(&config)?;
        Ok(Self {
            frames: (config.window_samples / 160).min(2 * hubert_frames),
            hubert_frames,
            latent_channels: c.latent_channels as usize,
            feature_width: c.feature_width as usize,
            sample_rate: c.sample_rate,
            config,
            synth,
            hubert,
            rmvpe,
            index,
            device,
            seed: 0x1234_5678_9abc_def0,
        })
    }
    pub fn process(
        &mut self,
        wave: &[f32],
        controls: &DirectMlControls,
    ) -> Result<DirectMlOfflineOutput, DirectMlError> {
        controls
            .auto_pitch
            .validate()
            .map_err(|e| DirectMlError::Config(e.to_string()))?;
        let c = self.synth.validate_rvc_synth()?;
        let upsample = c.upsample_rates.iter().product::<u64>() as usize;
        if wave.len() != self.config.window_samples
            || wave.iter().any(|v| !v.is_finite())
            || controls.valid_frames != self.frames
            || controls.speaker_id >= c.speakers as usize
            || !controls.pitch_semitones.is_finite()
            || !(-96. ..=96.).contains(&controls.pitch_semitones)
            || !controls.index_ratio.is_finite()
            || !(0. ..=1.).contains(&controls.index_ratio)
            || (controls.index_ratio > 0. && self.index.is_none())
            || !controls.protect.is_finite()
            || !(0. ..=0.5).contains(&controls.protect)
        {
            return Err(DirectMlError::Config(
                "invalid offline waveform or controls".into(),
            ));
        }
        let started = Instant::now();
        let mut preparation_ms = 0.;
        let mut gpu_timings = GpuTimings::default();
        let threshold = 0.03f32.to_le_bytes();
        let build = Instant::now();
        let mut graph = self
            .device
            .compile_hubert_for_rvc(
                &self.hubert,
                wave.len(),
                &self.config.precision,
                vocal_model::RvcVersion::from_feature_width(c.feature_width)?,
            )
            .map_err(|e| stage(e, "HuBERT build"))?;
        preparation_ms += build.elapsed().as_secs_f64() * 1000.;
        let (raw, timings) = graph.run_f32("features", &[("waveform", bytes(wave))])?;
        gpu_timings += timings.gpu_timings;
        drop(graph);

        let build = Instant::now();
        let mut graph = self
            .device
            .compile_rmvpe(&self.rmvpe, wave.len(), &self.config.precision, false)
            .map_err(|e| stage(e, "RMVPE build"))?;
        preparation_ms += build.elapsed().as_secs_f64() * 1000.;
        let (f0, timings) = graph.run_f32(
            "pitchf",
            &[("waveform", bytes(wave)), ("threshold", &threshold)],
        )?;
        gpu_timings += timings.gpu_timings;
        drop(graph);

        let mut features = raw.clone();
        if let Some(index) = &self.index {
            if controls.index_ratio > 0. {
                let batch = self.hubert_frames.min(128);
                let build = Instant::now();
                let mut graph = self
                    .device
                    .compile_retrieval_with_search(index, batch, self.config.retrieval_search)
                    .map_err(|e| stage(e, "retrieval build"))?;
                preparation_ms += build.elapsed().as_secs_f64() * 1000.;
                let ratio = controls.index_ratio.to_le_bytes();
                let mut query = vec![0.; batch * c.feature_width as usize];
                for rows in features.chunks_mut(query.len()) {
                    query.fill(0.);
                    query[..rows.len()].copy_from_slice(rows);
                    let (result, timings) = graph.run_f32(
                        "features",
                        &[("features", bytes(&query)), ("index_ratio", &ratio)],
                    )?;
                    rows.copy_from_slice(&result[..rows.len()]);
                    gpu_timings += timings.gpu_timings;
                }
            }
        }
        let mut rnd = vec![0.; self.frames * self.latent_channels];
        let mut noise = vec![0.; self.frames * upsample];
        fill_normal(&mut rnd, &mut self.seed);
        fill_normal(&mut noise, &mut self.seed);
        let build = Instant::now();
        let g = GraphBuilder::default();
        let h = g.input(
            "hubert",
            &[1, self.hubert_frames, c.feature_width as usize],
            DType::F32,
        );
        let raw_h = g.input("hubert_raw", &h.shape, DType::F32);
        let f0_input = g.input("f0", &[1, f0.len()], DType::F32);
        let shift = g.input("pitch_shift", &[1], DType::F32);
        let protect = g.input("protect", &[1], DType::F32);
        let length = g.input("p_len", &[1], DType::I32);
        let sid = g.input("sid", &[1], DType::I32);
        let random = g.input("rnd", &[1, self.latent_channels, self.frames], DType::F32);
        let nsf_noise = g.input("nsf_noise", &[1, 1, noise.len()], DType::F32);
        let (mut feats, pitch, pitchf) = frontend(&g, &h, &f0_input, self.frames, &shift);
        if self.index.is_some() && c.uses_f0 {
            let (original, _, _) = frontend(&g, &raw_h, &f0_input, self.frames, &shift);
            let voiced = g.binary(Op::Greater, &pitchf, &g.scalar(0., Precision::Fp32));
            let enabled = g.binary(Op::Less, &protect, &g.scalar(0.5, Precision::Fp32));
            let amount = g.reshape(
                &g.select(
                    &enabled,
                    &g.select(&voiced, &g.scalar(1., Precision::Fp32), &protect),
                    &g.scalar(1., Precision::Fp32),
                ),
                &[1, self.frames, 1],
            );
            feats = g.add(
                &g.mul(&feats, &amount),
                &g.mul(&original, &g.sub(&g.scalar(1., Precision::Fp32), &amount)),
            );
        }
        if self.config.debug_outputs {
            g.mark("feats", &feats);
            g.mark("pitch", &pitch);
            g.mark("pitchf", &pitchf);
        }
        // Stage diagnostics deliberately avoid exporting decoder activations.
        let audio = models::synth(
            &g,
            &self.synth,
            &feats,
            &pitch,
            &pitchf,
            &length,
            &sid,
            &random,
            &nsf_noise,
            0,
            &self.config.precision,
            false,
        )?;
        g.mark("audio", &audio);
        let mut graph = self
            .device
            .compile(&g)
            .map_err(|e| stage(e, "Synth build"))?;
        preparation_ms += build.elapsed().as_secs_f64() * 1000.;
        let mut matcher = vocal_types::PitchMatcher::default();
        let effective_pitch =
            matcher.resolve(&f0, controls.auto_pitch, controls.pitch_semitones, false);
        let shift = 2f32.powf(effective_pitch / 12.).to_le_bytes();
        let protect = controls.protect.to_le_bytes();
        let length = (self.frames as i32).to_le_bytes();
        let sid = (controls.speaker_id as i32).to_le_bytes();
        let bindings = [
            ("hubert", bytes(&features)),
            ("hubert_raw", bytes(&raw)),
            ("f0", bytes(&f0)),
            ("pitch_shift", &shift),
            ("protect", &protect),
            ("p_len", &length),
            ("sid", &sid),
            ("rnd", bytes(&rnd)),
            ("nsf_noise", bytes(&noise)),
        ];
        let inputs = bindings
            .into_iter()
            .filter(|(n, _)| graph.input_names().any(|name| name == *n))
            .collect::<Vec<_>>();
        let (audio, mut debug) = if self.config.debug_outputs {
            let result = graph.run(&inputs)?;
            (result.f32("audio")?, result)
        } else {
            graph.run_f32("audio", &inputs)?
        };
        gpu_timings += debug.gpu_timings;
        if self.config.debug_outputs {
            debug.push_f32(
                "auto_pitch_offset",
                &[effective_pitch - controls.pitch_semitones],
            );
            for (name, values) in [
                ("hubert_raw", &raw),
                ("hubert_features", &features),
                ("rmvpe_pitchf", &f0),
                ("synth_random", &rnd),
                ("synth_noise", &noise),
            ] {
                debug.push_f32(name, values);
            }
        }
        if audio.iter().any(|v| !v.is_finite()) {
            return Err(DirectMlError::Runtime(
                "non-finite offline decoder output".into(),
            ));
        }
        let inference_ms = (started.elapsed().as_secs_f64() * 1000. - preparation_ms).max(0.);
        debug.gpu_ms = gpu_timings.dispatch_ms;
        debug.gpu_timings = gpu_timings;
        Ok(DirectMlOfflineOutput {
            audio,
            preparation_ms,
            inference_ms,
            debug,
        })
    }
}
