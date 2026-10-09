use crate::{
    DirectMlConfig, DirectMlDevice, DirectMlError, DirectMlGraph, GraphOutput, Precision,
    graph::{DType, GraphBuilder, Op, Tensor},
    models,
};
use std::time::Instant;
use vocal_model::{ModelPackage, RetrievalIndex};

pub use crate::sources::DirectMlSources;

/// Values that change without recompiling the fixed graph.
#[derive(Clone, Debug, serde::Serialize)]
pub struct DirectMlControls {
    pub speaker_id: usize,
    pub pitch_semitones: f32,
    pub auto_pitch: vocal_types::AutoPitchConfig,
    pub index_ratio: f32,
    pub protect: f32,
    pub valid_frames: usize,
}
impl DirectMlControls {
    pub fn new(valid_frames: usize) -> Self {
        Self {
            speaker_id: 0,
            pitch_semitones: 0.,
            // Low-level diagnostics retain explicit controls. Engine defaults
            // are supplied by the shared ConversionConfig, which enables it.
            auto_pitch: vocal_types::AutoPitchConfig::disabled(),
            index_ratio: 0.,
            protect: 0.33,
            valid_frames,
        }
    }
}
pub struct DirectMlPipeline {
    graph: DirectMlGraph,
    config: DirectMlConfig,
    pub sample_rate: u32,
    pub frames: usize,
    pub hubert_frames: usize,
    pub speakers: usize,
    pub latent_channels: usize,
    pub feature_width: usize,
    pub upsample: usize,
    pub build_ms: f64,
    has_retrieval: bool,
    seed: u64,
    random_workspace: Vec<f32>,
    noise_workspace: Vec<f32>,
    last_auto_pitch: Option<vocal_types::AutoPitchConfig>,
    input_order: Vec<usize>,
}
pub struct DirectMlPipelineOutput {
    pub audio: Vec<f32>,
    pub gpu_ms: f64,
    pub total_ms: f64,
    pub debug: GraphOutput,
}
/// Graph constants are self-contained. Source packages and parsed indices may
/// be released before the driver compiles and initializes its GPU resources.
struct PreparedPipeline {
    builder: GraphBuilder,
    config: DirectMlConfig,
    sample_rate: u32,
    frames: usize,
    hubert_frames: usize,
    speakers: usize,
    latent_channels: usize,
    feature_width: usize,
    upsample: usize,
    has_retrieval: bool,
    started: Instant,
}
impl PreparedPipeline {
    fn compile(self) -> Result<DirectMlPipeline, DirectMlError> {
        tracing::info!(
            nodes = self.builder.nodes.borrow().len(),
            window = self.config.window_samples,
            execution = ?self.config.execution,
            precision = ?self.config.precision,
            "compiling native DirectML RVC graph"
        );
        let device = DirectMlDevice::from_config(&self.config)?;
        let graph = device.compile(&self.builder)?;
        let names = [
            "waveform",
            "threshold",
            "pitch_shift",
            "auto_pitch_enabled",
            "auto_pitch_target",
            "index_ratio",
            "protect",
            "p_len",
            "sid",
            "rnd",
            "nsf_noise",
        ];
        let input_order = graph
            .input_names()
            .map(|name| {
                names
                    .iter()
                    .position(|n| *n == name)
                    .ok_or_else(|| DirectMlError::Graph(format!("unknown pipeline input {name}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(DirectMlPipeline {
            graph,
            config: self.config,
            sample_rate: self.sample_rate,
            frames: self.frames,
            hubert_frames: self.hubert_frames,
            speakers: self.speakers,
            latent_channels: self.latent_channels,
            feature_width: self.feature_width,
            upsample: self.upsample,
            build_ms: self.started.elapsed().as_secs_f64() * 1000.,
            has_retrieval: self.has_retrieval,
            seed: 0x1234_5678_9abc_def0,
            random_workspace: vec![0.; self.latent_channels * self.frames],
            noise_workspace: vec![0.; self.frames * self.upsample],
            last_auto_pitch: None,
            input_order,
        })
    }
}
pub(crate) fn frontend(
    g: &GraphBuilder,
    h: &Tensor,
    pitchf: &Tensor,
    frames: usize,
    shift: &Tensor,
) -> (Tensor, Tensor, Tensor) {
    let indices = (0..frames)
        .map(|i| (i / 2).min(h.shape[1] - 1) as i32)
        .collect::<Vec<_>>();
    let feats = g.gather(h, &g.integers(&indices, &[frames]), 1);
    let pitchf = g.mul(&g.slice(pitchf, 1, 0, frames, 1), shift);
    let min = 1127. * (50f32 / 700.).ln_1p();
    let max = 1127. * (1100f32 / 700.).ln_1p();
    let mel = g.scale(
        &g.unary(
            Op::Log,
            &g.add(&g.scale(&pitchf, 1. / 700.), &g.scalar(1., Precision::Fp32)),
        ),
        1127.,
    );
    let coarse = g.add(
        &g.scale(
            &g.sub(&mel, &g.scalar(min, Precision::Fp32)),
            254. / (max - min),
        ),
        &g.scalar(1., Precision::Fp32),
    );
    let coarse = g.binary(
        Op::Min,
        &g.binary(Op::Max, &coarse, &g.scalar(1., Precision::Fp32)),
        &g.scalar(255., Precision::Fp32),
    );
    (
        feats,
        g.cast(&g.unary(Op::Round, &coarse), DType::I32),
        pitchf,
    )
}
impl DirectMlPipeline {
    pub fn from_sources(sources: &DirectMlSources) -> Result<Self, DirectMlError> {
        let synth = vocal_model::load_voice_cache_shared(&sources.synth)?;
        let hubert = vocal_model::load_cache_shared(&sources.hubert)?;
        let rmvpe = vocal_model::load_cache_shared(&sources.rmvpe)?;
        let index = if sources.enable_retrieval {
            RetrievalIndex::from_package(&synth)?
        } else {
            None
        };
        if sources.enable_retrieval && index.is_none() {
            return Err(DirectMlError::Config(
                "retrieval enabled without an index in VFRM".into(),
            ));
        }
        let prepared = Self::prepare(
            &synth,
            &hubert,
            &rmvpe,
            index.as_ref(),
            sources.config.clone(),
        )?;
        drop((synth, hubert, rmvpe, index));
        // Loading canonical sources and converting graph constants can free
        // hundreds of MiB on an otherwise idle control thread. Return those
        // pages before the driver allocates its compiled operator resources.
        // This construction boundary is never part of inference replay.
        #[cfg(windows)]
        unsafe {
            libmimalloc_sys::mi_collect(true);
        }
        prepared.compile()
    }
    pub fn build(
        synth: &ModelPackage,
        hubert: &ModelPackage,
        rmvpe: &ModelPackage,
        index: Option<&RetrievalIndex>,
        config: DirectMlConfig,
    ) -> Result<Self, DirectMlError> {
        Self::prepare(synth, hubert, rmvpe, index, config)?.compile()
    }
    fn prepare(
        synth: &ModelPackage,
        hubert: &ModelPackage,
        rmvpe: &ModelPackage,
        index: Option<&RetrievalIndex>,
        config: DirectMlConfig,
    ) -> Result<PreparedPipeline, DirectMlError> {
        config.validate()?;
        let started = Instant::now();
        let sc = synth.validate_rvc_synth()?;
        let version = vocal_model::RvcVersion::from_feature_width(sc.feature_width)?;
        hubert.validate_hubert_for_rvc(version)?;
        if index.is_some_and(|index| index.dimensions() != version.feature_width()) {
            return Err(DirectMlError::Config(
                "retrieval index dimensions disagree with RVC model version".into(),
            ));
        }
        let stage_mask = (1u32 << sc.upsample_rates.len()) - 1;
        if (config.precision.synth_fp32_stage_mask | config.precision.synth_noise_fp16_stage_mask)
            & !stage_mask
            != 0
        {
            return Err(DirectMlError::Config(
                "Synth precision mask exceeds model stages".into(),
            ));
        }
        let hubert_frames = (config.window_samples - 400) / 320 + 1;
        let frames = if config.append_hubert_frame {
            config.window_samples / 160
        } else {
            (config.window_samples / 160).min(hubert_frames * 2)
        };
        let upsample = sc.upsample_rates.iter().product::<u64>() as usize;
        let g = GraphBuilder::default();
        let wave = g.input("waveform", &[1, config.window_samples], DType::F32);
        let threshold = g.input("threshold", &[1], DType::F32);
        let shift = g.input("pitch_shift", &[1], DType::F32);
        let rate = g.input("index_ratio", &[1], DType::F32);
        let protect = g.input("protect", &[1], DType::F32);
        let length = g.input("p_len", &[1], DType::I32);
        let sid = g.input("sid", &[1], DType::I32);
        let rnd = g.input("rnd", &[1, sc.latent_channels as usize, frames], DType::F32);
        let noise = g.input("nsf_noise", &[1, 1, frames * upsample], DType::F32);
        if config.debug_outputs {
            g.mark("synth_random", &rnd);
            g.mark("synth_noise", &noise);
        }
        let h = models::hubert_for_rvc(&g, hubert, &wave, &config.precision, version)?;
        let hubert_frames = h.shape[1];
        if config.debug_outputs {
            g.mark("hubert_raw", &h);
        }
        let original = h.clone();
        let h = if let Some(index) = index {
            models::retrieval(&g, &h, index, &rate, config.retrieval_search)?
        } else {
            h
        };
        if config.debug_outputs {
            g.mark("hubert_features", &h);
        }
        let pitchf = crate::f0::build(&g, rmvpe, &wave, &threshold, &config)?;
        let shift = crate::pitch_matching::build(&g, &pitchf, &shift, config.debug_outputs);
        let (mut feats, pitch, pitchf) = frontend(&g, &h, &pitchf, frames, &shift);
        if index.is_some() && sc.uses_f0 {
            let (raw, _, _) = frontend(
                &g,
                &original,
                &pitchf,
                frames,
                &g.scalar(1., Precision::Fp32),
            );
            let voiced = g.binary(Op::Greater, &pitchf, &g.scalar(0., Precision::Fp32));
            let enabled = g.binary(Op::Less, &protect, &g.scalar(0.5, Precision::Fp32));
            let amount = g.select(
                &enabled,
                &g.select(&voiced, &g.scalar(1., Precision::Fp32), &protect),
                &g.scalar(1., Precision::Fp32),
            );
            let amount = g.reshape(&amount, &[1, frames, 1]);
            feats = g.add(
                &g.mul(&feats, &amount),
                &g.mul(&raw, &g.sub(&g.scalar(1., Precision::Fp32), &amount)),
            );
        }
        if config.debug_outputs {
            g.mark("feats", &feats);
            g.mark("pitchf", &pitchf);
            g.mark("pitch", &pitch);
        }
        let audio = models::synth(
            &g,
            synth,
            &feats,
            &pitch,
            &pitchf,
            &length,
            &sid,
            &rnd,
            &noise,
            config.skip_head_frames,
            &config.precision,
            config.debug_outputs,
        )?;
        g.mark("audio", &audio);
        Ok(PreparedPipeline {
            builder: g,
            config,
            sample_rate: sc.sample_rate,
            frames,
            hubert_frames,
            speakers: sc.speakers as usize,
            latent_channels: sc.latent_channels as usize,
            feature_width: sc.feature_width as usize,
            upsample,
            started,
            has_retrieval: index.is_some(),
        })
    }
    pub fn has_retrieval_index(&self) -> bool {
        self.has_retrieval
    }
    /// Shape/precision changes require a newly compiled graph. Expose its
    /// actual policy read-only so callers cannot silently change replay rules.
    pub fn config(&self) -> &DirectMlConfig {
        &self.config
    }
    pub fn memory(&self) -> crate::GraphMemory {
        self.graph.memory()
    }
    pub fn reset(&mut self) -> Result<(), DirectMlError> {
        self.graph.reset_states()?;
        self.seed = 0x1234_5678_9abc_def0;
        Ok(())
    }
    pub fn process(
        &mut self,
        waveform: &[f32],
        controls: &DirectMlControls,
    ) -> Result<DirectMlPipelineOutput, DirectMlError> {
        let started = Instant::now();
        let mut random = std::mem::take(&mut self.random_workspace);
        let mut noise = std::mem::take(&mut self.noise_workspace);
        fill_normal(&mut random, &mut self.seed);
        fill_normal(&mut noise, &mut self.seed);
        let result = self.process_exact(waveform, controls, &random, &noise);
        self.random_workspace = random;
        self.noise_workspace = noise;
        let mut result = result?;
        result.total_ms = started.elapsed().as_secs_f64() * 1000.;
        Ok(result)
    }
    pub fn process_exact(
        &mut self,
        waveform: &[f32],
        controls: &DirectMlControls,
        random: &[f32],
        noise: &[f32],
    ) -> Result<DirectMlPipelineOutput, DirectMlError> {
        let started = Instant::now();
        let DirectMlControls {
            speaker_id: speaker,
            pitch_semitones: semitones,
            auto_pitch,
            index_ratio,
            protect,
            valid_frames,
        } = *controls;
        auto_pitch
            .validate()
            .map_err(|e| DirectMlError::Config(e.to_string()))?;
        if waveform.len() != self.config.window_samples
            || waveform.iter().any(|v| !v.is_finite())
            || speaker >= self.speakers
            || valid_frames == 0
            || valid_frames > self.frames
            || !semitones.is_finite()
            || !(-96. ..=96.).contains(&semitones)
            || !index_ratio.is_finite()
            || !(0. ..=1.).contains(&index_ratio)
            || (index_ratio > 0. && !self.has_retrieval)
            || !protect.is_finite()
            || !(0. ..=1.).contains(&protect)
            || random.len() != self.frames * self.latent_channels
            || noise.len() != self.frames * self.upsample
            || random.iter().chain(noise).any(|v| !v.is_finite())
        {
            return Err(DirectMlError::Config(
                "invalid waveform, speaker, pitch, retrieval or random binding".into(),
            ));
        }
        let threshold = 0.03f32.to_le_bytes();
        if self.last_auto_pitch.is_some_and(|old| old != auto_pitch) {
            self.graph.reset_states()?;
        }
        self.last_auto_pitch = Some(auto_pitch);
        let auto_enabled = f32::from(u8::from(auto_pitch.enabled)).to_le_bytes();
        let auto_target = auto_pitch.target_hz.to_le_bytes();
        let shift = 2f32.powf(semitones / 12.).to_le_bytes();
        let ratio = index_ratio.to_le_bytes();
        let len = (valid_frames as i32).to_le_bytes();
        let sid = (speaker as i32).to_le_bytes();
        let protect = protect.to_le_bytes();
        fn bytes(slice: &[f32]) -> &[u8] {
            unsafe {
                std::slice::from_raw_parts(
                    slice.as_ptr() as *const u8,
                    std::mem::size_of_val(slice),
                )
            }
        }
        let candidates = [
            ("waveform", bytes(waveform)),
            ("threshold", &threshold),
            ("pitch_shift", &shift),
            ("auto_pitch_enabled", &auto_enabled),
            ("auto_pitch_target", &auto_target),
            ("index_ratio", &ratio),
            ("protect", &protect),
            ("p_len", &len),
            ("sid", &sid),
            ("rnd", bytes(random)),
            ("nsf_noise", bytes(noise)),
        ];
        let mut ordered = [("", &[][..]); 11];
        for (slot, index) in ordered.iter_mut().zip(&self.input_order) {
            *slot = candidates[*index];
        }
        let inputs = &ordered[..self.input_order.len()];
        let (audio, result) = if self.config.debug_outputs {
            let result = self.graph.run(inputs)?;
            (result.f32("audio")?, result)
        } else {
            self.graph.run_f32("audio", inputs)?
        };
        if audio.iter().any(|v| !v.is_finite()) {
            return Err(DirectMlError::Runtime(
                "non-finite DirectML audio output".into(),
            ));
        }
        Ok(DirectMlPipelineOutput {
            audio,
            gpu_ms: result.gpu_ms,
            total_ms: started.elapsed().as_secs_f64() * 1000.,
            debug: result,
        })
    }
}

pub(crate) fn fill_normal(values: &mut [f32], seed: &mut u64) {
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
