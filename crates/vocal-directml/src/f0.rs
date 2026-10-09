//! RMVPE window selection and reference-compatible GPU history update.
use crate::{
    DirectMlConfig, DirectMlError, F0Config,
    graph::{DType, GraphBuilder, Tensor},
    models,
};
use vocal_model::ModelPackage;

pub(crate) const HISTORY_FRAMES: usize = 1024;

pub(crate) fn build(
    g: &GraphBuilder,
    package: &ModelPackage,
    waveform: &Tensor,
    threshold: &Tensor,
    config: &DirectMlConfig,
) -> Result<Tensor, DirectMlError> {
    let window = match config.f0 {
        F0Config::FullWindow => waveform.clone(),
        F0Config::Rolling { rmvpe_samples, .. } => g.slice(
            waveform,
            1,
            config.window_samples - rmvpe_samples,
            rmvpe_samples,
            1,
        ),
    };
    let raw = models::rmvpe(
        g,
        package,
        &window,
        &config.precision,
        threshold,
        config.debug_outputs,
    )?;
    if config.debug_outputs {
        g.mark("rmvpe_pitchf", &raw);
    }
    match config.f0 {
        F0Config::FullWindow => Ok(raw),
        F0Config::Rolling {
            advance_samples, ..
        } => {
            let history = g.input("f0_history", &[1, HISTORY_FRAMES], DType::F32);
            let interior = raw.shape[1] - 4;
            let retained = g.slice(
                &history,
                1,
                advance_samples / 160,
                HISTORY_FRAMES - interior,
                1,
            );
            // Applio/WebUI shift left one block, overwrite the tail with [3:-1].
            let next = g.concat(&[retained, g.slice(&raw, 1, 3, interior, 1)], 1);
            g.feedback(&history, &next);
            if config.debug_outputs {
                g.mark("f0_cache", &next);
            }
            let frames = config.window_samples / 160;
            Ok(g.slice(&next, 1, HISTORY_FRAMES - frames, frames, 1))
        }
    }
}
