//! Standalone graph compilation for parity tests and stage profiling.
use vocal_model::ModelPackage;

use crate::{
    DirectMlDevice, DirectMlError, DirectMlGraph, PrecisionConfig,
    graph::{DType, GraphBuilder},
    models,
};

impl DirectMlDevice {
    pub fn compile_retrieval(
        &self,
        index: &vocal_model::RetrievalIndex,
        frames: usize,
    ) -> Result<DirectMlGraph, DirectMlError> {
        self.compile_retrieval_with_search(index, frames, crate::RetrievalSearch::Auto)
    }
    pub fn compile_retrieval_with_search(
        &self,
        index: &vocal_model::RetrievalIndex,
        frames: usize,
        search: crate::RetrievalSearch,
    ) -> Result<DirectMlGraph, DirectMlError> {
        if frames == 0 || frames > 5000 {
            return Err(DirectMlError::Config(
                "invalid retrieval frame count".into(),
            ));
        }
        let g = GraphBuilder::default();
        let features = g.input("features", &[1, frames, index.dimensions()], DType::F32);
        let ratio = g.input("index_ratio", &[1], DType::F32);
        g.mark(
            "features",
            &models::retrieval(&g, &features, index, &ratio, search)?,
        );
        self.compile(&g)
    }
    pub fn compile_hubert(
        &self,
        package: &ModelPackage,
        samples: usize,
        precision: &PrecisionConfig,
    ) -> Result<DirectMlGraph, DirectMlError> {
        self.compile_hubert_for_rvc(package, samples, precision, vocal_model::RvcVersion::V2)
    }

    pub fn compile_hubert_for_rvc(
        &self,
        package: &ModelPackage,
        samples: usize,
        precision: &PrecisionConfig,
        version: vocal_model::RvcVersion,
    ) -> Result<DirectMlGraph, DirectMlError> {
        validate_window(samples)?;
        let g = GraphBuilder::default();
        let waveform = g.input("waveform", &[1, samples], DType::F32);
        g.mark(
            "features",
            &models::hubert_for_rvc(&g, package, &waveform, precision, version)?,
        );
        self.compile(&g)
    }

    pub fn compile_rmvpe(
        &self,
        package: &ModelPackage,
        samples: usize,
        precision: &PrecisionConfig,
        debug: bool,
    ) -> Result<DirectMlGraph, DirectMlError> {
        validate_window(samples)?;
        let g = GraphBuilder::default();
        let waveform = g.input("waveform", &[1, samples], DType::F32);
        let threshold = g.input("threshold", &[1], DType::F32);
        g.mark(
            "pitchf",
            &models::rmvpe(&g, package, &waveform, precision, &threshold, debug)?,
        );
        self.compile(&g)
    }

    pub fn compile_synth(
        &self,
        package: &ModelPackage,
        frames: usize,
        skip: usize,
        precision: &PrecisionConfig,
        debug: bool,
    ) -> Result<DirectMlGraph, DirectMlError> {
        let c = package.validate_rvc_synth()?;
        if frames == 0 || frames > 5000 || skip >= frames {
            return Err(DirectMlError::Config(
                "invalid Synth frame count or crop".into(),
            ));
        }
        let g = GraphBuilder::default();
        let feats = g.input("feats", &[1, frames, c.feature_width as usize], DType::F32);
        let pitch = g.input("pitch", &[1, frames], DType::I32);
        let pitchf = g.input("pitchf", &[1, frames], DType::F32);
        let length = g.input("p_len", &[1], DType::I32);
        let sid = g.input("sid", &[1], DType::I32);
        let rnd = g.input("rnd", &[1, c.latent_channels as usize, frames], DType::F32);
        let up = c.upsample_rates.iter().product::<u64>() as usize;
        let noise = g.input("nsf_noise", &[1, 1, frames * up], DType::F32);
        g.mark(
            "audio",
            &models::synth(
                &g, package, &feats, &pitch, &pitchf, &length, &sid, &rnd, &noise, skip, precision,
                debug,
            )?,
        );
        self.compile(&g)
    }
}

fn validate_window(samples: usize) -> Result<(), DirectMlError> {
    crate::DirectMlConfig::offline(samples).validate()
}
