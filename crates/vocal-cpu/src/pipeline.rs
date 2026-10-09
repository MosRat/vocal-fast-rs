use crate::{
    CpuConfig, CpuError, CpuGraph, CpuNodeTiming, CpuQuantization, CpuQuantizationPreset,
    CpuQuantizedOperator, pitch::CpuPitchStage, runtime::bytes,
};
use std::{path::PathBuf, time::Instant};
use vocal_graph::{
    PrecisionConfig,
    graph::{DType, GraphBuilder},
    models,
};
use vocal_types::{ConversionConfig, PitchMatcher};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CpuSources {
    pub synth: PathBuf,
    pub hubert: PathBuf,
    pub rmvpe: PathBuf,
}
pub struct CpuPipeline {
    pub config: CpuConfig,
    pub sample_rate: u32,
    pub frames: usize,
    pub hubert_frames: usize,
    pub latent_channels: usize,
    pub feature_width: usize,
    pub uses_f0: bool,
    pub upsample: usize,
    speakers: usize,
    hubert: CpuGraph,
    rmvpe: CpuPitchStage,
    synth: CpuGraph,
    index: Option<vocal_model::RetrievalIndex>,
    random: Vec<f32>,
    noise: Vec<f32>,
    seed: u64,
    matcher: PitchMatcher,
    buffers: PipelineBuffers,
}
#[derive(Default)]
struct PipelineBuffers {
    raw: Vec<f32>,
    features: Vec<f32>,
    f0: Vec<f32>,
    pitchf: Vec<f32>,
    feats: Vec<f32>,
    pitch_bytes: Vec<u8>,
}
pub struct CpuPipelineOutput {
    pub audio: Vec<f32>,
    pub feature_ms: f64,
    pub pitch_ms: f64,
    pub retrieval_ms: f64,
    pub synth_ms: f64,
    pub total_ms: f64,
    pub debug: std::collections::BTreeMap<String, Vec<u8>>,
    pub timings: Vec<CpuNodeTiming>,
}
impl CpuPipeline {
    pub fn memory_usage(&self) -> std::collections::BTreeMap<&'static str, crate::CpuMemory> {
        std::collections::BTreeMap::from([
            ("hubert", self.hubert.memory_usage()),
            ("rmvpe", self.rmvpe.graph.memory_usage()),
            ("synth", self.synth.memory_usage()),
        ])
    }
    pub fn has_retrieval_index(&self) -> bool {
        self.index.is_some()
    }
    pub fn from_sources(sources: &CpuSources, config: CpuConfig) -> Result<Self, CpuError> {
        Self::from_sources_impl(sources, config, None, false)
    }
    pub fn from_sources_calibration(
        sources: &CpuSources,
        config: CpuConfig,
        recipe: &CpuQuantization,
    ) -> Result<Self, CpuError> {
        Self::from_sources_impl(sources, config, Some(recipe), true)
    }
    pub fn from_sources_quantized(
        sources: &CpuSources,
        config: CpuConfig,
        recipe: &CpuQuantization,
    ) -> Result<Self, CpuError> {
        Self::from_sources_impl(sources, config, Some(recipe), false)
    }
    /// Compile an opt-in precision preset with caller-supplied calibration.
    /// Reject partial graph coverage; do not calibrate from inference inputs.
    pub fn from_sources_preset(
        sources: &CpuSources,
        config: CpuConfig,
        preset: CpuQuantizationPreset,
        calibration: std::collections::BTreeMap<String, Vec<f32>>,
    ) -> Result<Self, CpuError> {
        let version = vocal_model::inspect_model_artifact(&sources.synth)?
            .manifest
            .rvc_version()?;
        let recipe = preset.calibrated_recipe_for_rvc(calibration, version)?;
        let pipeline = Self::from_sources_quantized(sources, config, &recipe)?;
        preset.validate_coverage_for_rvc(&pipeline.quantized_operators(), version)?;
        Ok(pipeline)
    }
    pub fn calibration(&self) -> Result<std::collections::BTreeMap<String, Vec<f32>>, CpuError> {
        let mut result = self.hubert.calibration()?;
        result.extend(self.rmvpe.graph.calibration()?);
        result.extend(self.synth.calibration()?);
        Ok(result)
    }
    pub fn quantized_operators(&self) -> Vec<CpuQuantizedOperator> {
        [
            self.hubert.quantized_operators(),
            self.rmvpe.graph.quantized_operators(),
            self.synth.quantized_operators(),
        ]
        .concat()
    }
    fn from_sources_impl(
        sources: &CpuSources,
        config: CpuConfig,
        recipe: Option<&CpuQuantization>,
        calibration_only: bool,
    ) -> Result<Self, CpuError> {
        config.validate()?;
        let compile = |g| match recipe {
            Some(recipe) if calibration_only => CpuGraph::compile_calibration(g, &config, recipe),
            Some(recipe) => CpuGraph::compile_quantized(g, &config, recipe),
            None => CpuGraph::compile(g, &config),
        };
        let precision = PrecisionConfig::fp32();
        // The Synth metadata selects the acoustic layer before compilation.
        // Drop this temporary reference so stage packing keeps bounded lifetimes.
        let version = vocal_model::inspect_model_artifact(&sources.synth)?
            .manifest
            .rvc_version()?;
        let feature_width = version.feature_width();
        // Finish packing and release each source package before loading the
        // next stage. Raw packages, graph constants and packed weights are
        // otherwise simultaneously resident during the entire compilation.
        let (hubert, hubert_frames) = {
            let package = vocal_model::load_cache_shared(&sources.hubert)?;
            let g = GraphBuilder::default();
            let wave = g.input("waveform", &[1, config.window_samples], DType::F32);
            let features = models::hubert_for_rvc(&g, &package, &wave, &precision, version)?;
            let frames = features.shape[1];
            g.mark("features", &features);
            (compile(g)?, frames)
        };
        let rmvpe = {
            let package = vocal_model::load_cache_shared(&sources.rmvpe)?;
            let g = crate::pitch::build_graph(&package, &config)?;
            CpuPitchStage::new(compile(g)?, config.f0.clone())
        };
        let synth = vocal_model::load_voice_cache_shared(&sources.synth)?;
        let c = synth.validate_rvc_synth()?;
        let frames = (config.window_samples / 160)
            .min(2 * (hubert_frames + usize::from(config.append_hubert_frame)));
        if config.skip_head_frames >= frames {
            return Err(CpuError::Config(
                "Synth crop exceeds actual features".into(),
            ));
        }
        let g = GraphBuilder::default();
        let feats = g.input("feats", &[1, frames, feature_width], DType::F32);
        let pitch = g.input("pitch", &[1, frames], DType::I32);
        let pitchf = g.input("pitchf", &[1, frames], DType::F32);
        let length = g.input("p_len", &[1], DType::I32);
        let sid = g.input("sid", &[1], DType::I32);
        let rnd = g.input("rnd", &[1, c.latent_channels as usize, frames], DType::F32);
        let upsample = c.upsample_rates.iter().product::<u64>() as usize;
        let noise = g.input("nsf_noise", &[1, 1, frames * upsample], DType::F32);
        let audio = models::synth(
            &g,
            &synth,
            &feats,
            &pitch,
            &pitchf,
            &length,
            &sid,
            &rnd,
            &noise,
            config.skip_head_frames,
            &precision,
            false,
        )?;
        g.mark("audio", &audio);
        let graph = compile(g)?;
        let index = vocal_model::RetrievalIndex::from_package(&synth)?;
        if index
            .as_ref()
            .is_some_and(|index| index.dimensions() != feature_width)
        {
            return Err(CpuError::Config(
                "retrieval index dimensions disagree with RVC model version".into(),
            ));
        }
        Ok(Self {
            config,
            sample_rate: c.sample_rate,
            frames,
            hubert_frames,
            latent_channels: c.latent_channels as usize,
            feature_width,
            uses_f0: c.uses_f0,
            upsample,
            speakers: c.speakers as usize,
            hubert,
            rmvpe,
            synth: graph,
            index,
            random: vec![0.; frames * c.latent_channels as usize],
            noise: vec![0.; frames * upsample],
            seed: 0x1234_5678_9abc_def0,
            matcher: PitchMatcher::default(),
            buffers: PipelineBuffers {
                features: vec![0.; hubert_frames * feature_width],
                pitchf: vec![0.; frames],
                feats: vec![0.; frames * feature_width],
                pitch_bytes: vec![0; frames * 4],
                ..PipelineBuffers::default()
            },
        })
    }
    pub fn reset_stream(&mut self) {
        self.matcher.reset();
        self.rmvpe.reset();
    }
    pub fn process(
        &mut self,
        wave: &[f32],
        conversion: ConversionConfig,
        speaker: usize,
        debug: bool,
    ) -> Result<CpuPipelineOutput, CpuError> {
        let started = Instant::now();
        let mut random = std::mem::take(&mut self.random);
        let mut noise = std::mem::take(&mut self.noise);
        fill_normal(&mut random, &mut self.seed);
        fill_normal(&mut noise, &mut self.seed);
        let result = self.process_exact(wave, conversion, speaker, &random, &noise, debug);
        self.random = random;
        self.noise = noise;
        result.map(|mut output| {
            output.total_ms = started.elapsed().as_secs_f64() * 1000.;
            output
        })
    }
    pub fn process_exact(
        &mut self,
        wave: &[f32],
        conversion: ConversionConfig,
        speaker: usize,
        random: &[f32],
        noise: &[f32],
        debug: bool,
    ) -> Result<CpuPipelineOutput, CpuError> {
        conversion
            .validate()
            .map_err(|e| CpuError::Config(e.to_string()))?;
        if wave.len() != self.config.window_samples
            || speaker >= self.speakers
            || random.len() != self.frames * self.latent_channels
            || noise.len() != self.frames * self.upsample
            || wave
                .iter()
                .chain(random)
                .chain(noise)
                .any(|x| !x.is_finite())
            || (conversion.index_ratio > 0. && self.index.is_none())
        {
            return Err(CpuError::Config(
                "invalid waveform, random tensors, speaker or unavailable retrieval".into(),
            ));
        }
        let start = Instant::now();
        let stage = Instant::now();
        self.hubert.run_f32_into(
            &[("waveform", bytes(wave))],
            "features",
            &mut self.buffers.raw,
        )?;
        let feature_ms = stage.elapsed().as_secs_f64() * 1000.;
        let stage = Instant::now();
        self.rmvpe.run(wave, &mut self.buffers.f0)?;
        let pitch_ms = stage.elapsed().as_secs_f64() * 1000.;
        let stage = Instant::now();
        let PipelineBuffers {
            raw,
            features,
            f0,
            pitchf,
            feats,
            pitch_bytes,
        } = &mut self.buffers;
        features.copy_from_slice(raw);
        if let Some(index) = &self.index {
            if conversion.index_ratio > 0. {
                for row in features.chunks_exact_mut(self.feature_width) {
                    crate::retrieval::blend(index, row, conversion.index_ratio, self.config.avx2)?;
                }
            }
        }
        let retrieval_ms = stage.elapsed().as_secs_f64() * 1000.;
        let stage = Instant::now();
        let semitones = self.matcher.resolve(
            &f0,
            conversion.auto_pitch,
            conversion.pitch_semitones,
            self.config.append_hubert_frame,
        );
        let shift = 2f32.powf(semitones / 12.);
        for (dst, src) in pitchf.iter_mut().zip(&f0[..self.frames]) {
            *dst = src * shift;
        }
        coarse_pitch_into(pitchf, pitch_bytes);
        for frame in 0..self.frames {
            let src = (frame / 2).min(self.hubert_frames - 1) * self.feature_width;
            let amount = if self.uses_f0 && pitchf[frame] == 0. && conversion.protect < 0.5 {
                conversion.protect
            } else {
                1.
            };
            for dim in 0..self.feature_width {
                feats[frame * self.feature_width + dim] =
                    features[src + dim] * amount + raw[src + dim] * (1. - amount);
            }
        }
        let p_len = (self.frames as i32).to_le_bytes();
        let sid = (speaker as i32).to_le_bytes();
        let bindings = [
            ("feats", bytes(&feats)),
            ("pitch", pitch_bytes.as_slice()),
            ("pitchf", bytes(&pitchf)),
            ("p_len", &p_len),
            ("sid", &sid),
            ("rnd", bytes(random)),
            ("nsf_noise", bytes(noise)),
        ];
        let bindings = bindings
            .into_iter()
            .filter(|(name, _)| self.synth.input_names().any(|n| n == *name))
            .collect::<Vec<_>>();
        let mut audio = Vec::new();
        self.synth.run_f32_into(&bindings, "audio", &mut audio)?;
        if audio.iter().any(|x| !x.is_finite()) {
            return Err(CpuError::Runtime("non-finite Synth audio".into()));
        }
        let synth_ms = stage.elapsed().as_secs_f64() * 1000.;
        self.rmvpe.commit();
        let mut trace = std::collections::BTreeMap::new();
        if debug {
            for (name, data) in [
                ("hubert_raw", raw.as_slice()),
                ("hubert_features", features.as_slice()),
                ("rmvpe_pitchf", self.rmvpe.raw(f0)),
                ("feats", feats.as_slice()),
                ("pitchf", pitchf.as_slice()),
                ("synth_random", random),
                ("synth_noise", noise),
                ("audio", &audio),
            ] {
                trace.insert(name.into(), bytes(data).to_vec());
            }
            trace.insert("pitch".into(), pitch_bytes.clone());
            if let Some(history) = self.rmvpe.history() {
                trace.insert("f0_cache".into(), bytes(history).to_vec());
            }
        }
        let timings = [
            ("hubert", self.hubert.timings()),
            ("rmvpe", self.rmvpe.graph.timings()),
            ("synth", self.synth.timings()),
        ]
        .into_iter()
        .flat_map(|(stage, timings)| {
            timings.into_iter().map(move |mut timing| {
                timing.stage = stage.into();
                timing
            })
        })
        .collect();
        Ok(CpuPipelineOutput {
            audio,
            feature_ms,
            pitch_ms,
            retrieval_ms,
            synth_ms,
            total_ms: start.elapsed().as_secs_f64() * 1000.,
            debug: trace,
            timings,
        })
    }
}
fn coarse_pitch_into(f0: &[f32], destination: &mut [u8]) {
    let min = 1127. * (50f32 / 700.).ln_1p();
    let max = 1127. * (1100f32 / 700.).ln_1p();
    for (hz, bytes) in f0.iter().zip(destination.chunks_exact_mut(4)) {
        let pitch = ((1127. * (1. + hz / 700.).ln() - min) * (254. / (max - min)) + 1.)
            .clamp(1., 255.)
            .round_ties_even() as i32;
        bytes.copy_from_slice(&pitch.to_le_bytes());
    }
}
fn fill_normal(values: &mut [f32], seed: &mut u64) {
    for pair in values.chunks_mut(2) {
        let mut uniform = || {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            ((*seed >> 40) as f32 + 0.5) / 16777216.
        };
        let radius = (-2. * uniform().ln()).sqrt();
        let angle = std::f32::consts::TAU * uniform();
        pair[0] = radius * angle.cos();
        if pair.len() == 2 {
            pair[1] = radius * angle.sin();
        }
    }
}
