//! Experiment policy and diagnostics, kept outside the timed streaming loop.
use serde::Deserialize;
use std::{
    collections::BTreeSet,
    error::Error,
    path::{Path, PathBuf},
};
use vocal_cpu::{CpuConfig, CpuPipeline, CpuQuantization, CpuSources};
use vocal_types::ConversionConfig;

#[derive(Deserialize, serde::Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Experiment {
    pub baseline: Option<CpuConfig>,
    pub candidate: Option<CpuConfig>,
    /// Fully calibrated recipe. Model/corpus fingerprints are recorded by the
    /// orchestrating script before launching this example.
    pub recipe_path: Option<PathBuf>,
    pub trace: bool,
    pub profile: bool,
}
impl Default for Experiment {
    fn default() -> Self {
        Self {
            baseline: None,
            candidate: None,
            recipe_path: None,
            trace: true,
            profile: false,
        }
    }
}
impl Experiment {
    pub fn recipe(&self) -> Result<Option<CpuQuantization>, Box<dyn Error>> {
        let recipe: Option<CpuQuantization> = self
            .recipe_path
            .as_ref()
            .map(|path| -> Result<_, Box<dyn Error>> {
                Ok(serde_json::from_slice(&std::fs::read(path)?)?)
            })
            .transpose()?;
        if let Some(recipe) = &recipe {
            recipe.validate()?;
            if recipe.rules.is_empty()
                || recipe
                    .rules
                    .iter()
                    .any(|r| !recipe.calibration.contains_key(&r.prefix))
            {
                return Err("streaming experiments require nonempty exact-name rules with complete calibration".into());
            }
        }
        Ok(recipe)
    }
    pub fn validate(
        &self,
        baseline: &CpuConfig,
        candidate: &CpuConfig,
    ) -> Result<(), Box<dyn Error>> {
        for config in [baseline, candidate] {
            config.validate()?;
            if config.profile
                || config.window_samples != 44960
                || config.skip_head_frames != 250
                || !config.append_hubert_frame
            {
                return Err("streaming experiment requires 44960/4000 geometry and disabled node clocks in configuration".into());
            }
            if let vocal_cpu::CpuF0Config::Rolling {
                advance_samples, ..
            } = config.f0
            {
                if advance_samples != super::ADVANCE {
                    return Err("F0 history advance must match the streaming block".into());
                }
            }
        }
        if baseline.threads != candidate.threads
            || baseline.spin_wait_us != candidate.spin_wait_us
            || baseline.pin_to_performance_cores != candidate.pin_to_performance_cores
        {
            return Err("paired graphs must share a thread policy".into());
        }
        Ok(())
    }
}
pub fn pipeline(
    sources: &CpuSources,
    config: CpuConfig,
    recipe: Option<&CpuQuantization>,
) -> Result<CpuPipeline, Box<dyn Error>> {
    Ok(match recipe {
        Some(recipe) => CpuPipeline::from_sources_quantized(sources, config, recipe)?,
        None => CpuPipeline::from_sources(sources, config)?,
    })
}
pub fn validate_coverage(
    pipeline: &CpuPipeline,
    recipe: Option<&CpuQuantization>,
) -> Result<(), Box<dyn Error>> {
    let expected = recipe
        .into_iter()
        .flat_map(|r| r.rules.iter().map(|r| r.prefix.as_str()))
        .collect::<BTreeSet<_>>();
    let operators = pipeline.quantized_operators();
    let actual = operators
        .iter()
        .map(|r| r.name.as_str())
        .collect::<BTreeSet<_>>();
    if actual != expected || actual.len() != operators.len() {
        return Err("quantized operator coverage differs from exact-name recipe".into());
    }
    Ok(())
}
pub fn profile(
    sources: &CpuSources,
    config: &CpuConfig,
    recipe: Option<&CpuQuantization>,
    audio: &[f32],
    controls: ConversionConfig,
    directory: &Path,
) -> Result<(), Box<dyn Error>> {
    let mut config = config.clone();
    config.profile = true;
    let mut pipeline = pipeline(sources, config.clone(), recipe)?;
    validate_coverage(&pipeline, recipe)?;
    let mut window = vec![0.; config.window_samples];
    let mut rows = Vec::new();
    for (i, chunk) in audio.chunks(super::ADVANCE).enumerate() {
        super::window_push(&mut window, chunk);
        let output = pipeline.process(&window, controls, 0, false)?;
        if [20, 40, 60, 80, 100].contains(&i) {
            rows.push(serde_json::json!({"chunk": i, "timings": output.timings}));
        }
    }
    std::fs::write(
        directory.join("profile.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "config": config, "quantized": pipeline.quantized_operators(), "rows": rows,
            "note": "diagnostic per-node clock reads; excluded from latency benchmarks"
        }))?,
    )?;
    println!(
        "{}",
        serde_json::json!({"profile": directory.join("profile.json")})
    );
    Ok(())
}
