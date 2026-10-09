//! cargo run -p vocal-model --example bundle-rvc -- SYNTH HUBERT RMVPE OUTPUT
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths: Vec<_> = std::env::args_os().skip(1).collect();
    if paths.len() != 4 {
        return Err("usage: bundle-rvc SYNTH HUBERT RMVPE OUTPUT.vfrm".into());
    }
    vocal_model::bundle_rvc_cache(&paths[0], &paths[1], &paths[2], &paths[3])?;
    println!(
        "{}",
        serde_json::to_string_pretty(&vocal_model::inspect_model_artifact(&paths[3])?)?
    );
    Ok(())
}
