//! Convert a checkpoint + index with the restricted Rust loader, then bundle acoustic weights.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths: Vec<_> = std::env::args_os().skip(1).collect();
    if paths.len() != 4 {
        return Err("usage: prepare-rvc MODEL_DIRECTORY HUBERT RMVPE OUTPUT.vfrm".into());
    }
    let root = std::path::Path::new(&paths[0]);
    let manifest = vocal_model::load_manifest(root.join("manifest.json"))?;
    let package = vocal_model::load_rvc_directory(root, manifest)?;
    package.validate_rvc_synth()?;
    let synth = root.join("synth.vfrm");
    vocal_model::write_cache(&synth, &package)?;
    drop(package);
    vocal_model::bundle_rvc_cache(&synth, &paths[1], &paths[2], &paths[3])?;
    println!(
        "{}",
        serde_json::to_string_pretty(&vocal_model::inspect_model_artifact(&paths[3])?)?
    );
    Ok(())
}
