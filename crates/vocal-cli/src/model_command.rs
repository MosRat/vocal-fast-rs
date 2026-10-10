//! Offline model import commands. No inference backend or Python is required.
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use serde_json::Value;
use vocal_model::{
    RetrievalIndex, Tensor, bundle_rvc_cache, inspect_model_artifact, load_cache, load_manifest,
    load_rvc_onnx, load_rvc_pth, load_rvc_pth_auto, write_cache,
};

use crate::cli_error::CliResult;

static TEMP_ID: AtomicU64 = AtomicU64::new(0);

pub fn run(command: &str, args: &[String]) -> CliResult<()> {
    match (command, args) {
        ("convert", [input, output, options @ ..]) => convert(input, output, options),
        ("bundle", [synth, hubert, rmvpe, output]) => {
            bundle_rvc_cache(synth, hubert, rmvpe, output)?;
            describe(output)
        }
        ("describe", [path]) => describe(path),
        ("verify", [path]) => {
            let package = load_cache(path)?;
            RetrievalIndex::from_package(&package)?;
            describe(path)
        }
        _ => Err(format!("invalid arguments for model {command}; see help").into()),
    }
}

fn describe(path: impl AsRef<Path>) -> CliResult<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(&inspect_model_artifact(path)?)?
    );
    Ok(())
}

fn options(args: &[String]) -> CliResult<BTreeMap<&str, &str>> {
    let mut result = BTreeMap::new();
    let mut pairs = args.chunks_exact(2);
    for pair in &mut pairs {
        let key = pair[0].as_str();
        if !matches!(
            key,
            "--manifest"
                | "--index"
                | "--name"
                | "--id"
                | "--source"
                | "--provenance"
                | "--library-info"
        ) {
            return Err(format!("unknown conversion option {key}").into());
        }
        if pair[1].is_empty() || result.insert(key, pair[1].as_str()).is_some() {
            return Err(format!("empty or duplicate option {key}").into());
        }
    }
    if !pairs.remainder().is_empty() {
        return Err("conversion option requires a value".into());
    }
    Ok(result)
}

fn json_object(path: &str) -> CliResult<Value> {
    if fs::metadata(path)?.len() > 1024 * 1024 {
        return Err("import metadata exceeds 1 MiB".into());
    }
    let value: Value = serde_json::from_slice(&fs::read(path)?)?;
    if !value.is_object() {
        return Err("import metadata must be a JSON object".into());
    }
    Ok(value)
}

fn convert(input: &str, output: &str, args: &[String]) -> CliResult<()> {
    let options = options(args)?;
    let input = Path::new(input);
    let output = Path::new(output);
    if output.exists() {
        return Err("conversion output already exists".into());
    }
    if !matches!(
        output.extension().and_then(|s| s.to_str()),
        Some("vfr" | "vfrm")
    ) {
        return Err("conversion output must end in .vfr or .vfrm".into());
    }
    let manifest = options.get("--manifest").map(load_manifest).transpose()?;
    let default_name = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("imported-model");
    let name = options.get("--name").copied().unwrap_or(default_name);
    let source = options.get("--source").copied().unwrap_or("local-import");
    let mut package = match (
        input
            .extension()
            .and_then(|s| s.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        manifest,
    ) {
        (Some("pth"), Some(manifest)) => load_rvc_pth(input, manifest)?,
        (Some("pth"), None) => load_rvc_pth_auto(input, name, source)?,
        (Some("onnx"), Some(manifest)) => load_rvc_onnx(input, manifest)?,
        (Some("onnx"), None) => {
            return Err("ONNX conversion requires --manifest; sample rate is not guessed".into());
        }
        _ => return Err("expected an RVC .pth or .onnx input".into()),
    };
    if let Some(name) = options.get("--name") {
        if name.trim().is_empty() || name.len() > 512 {
            return Err("model name must contain 1–512 UTF-8 bytes".into());
        }
        package.manifest.display_name = (*name).into();
    }
    if let Some(id) = options.get("--id") {
        if id.is_empty()
            || id.len() > 128
            || !id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
        {
            return Err("model id must contain 1–128 ASCII letters, digits, -, _ or .".into());
        }
        package.manifest.id = (*id).into();
    }
    if let Some(source) = options.get("--source") {
        package.manifest.source = (*source).into();
    }
    if let Some(path) = options.get("--index") {
        // Reject malformed or mismatched FAISS before writing any output.
        if fs::metadata(path)?.len() > 2 * 1024 * 1024 * 1024 {
            return Err("retrieval index exceeds 2 GiB".into());
        }
        package
            .tensors
            .push(Tensor::byte_tensor("aux.retrieval.faiss", fs::read(path)?));
        RetrievalIndex::from_package(&package)?;
        package.manifest.index_file = Some(
            Path::new(path)
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into(),
        );
    } else {
        package.manifest.index_file = None;
    }
    for (option, key) in [
        ("--provenance", "provenance"),
        ("--library-info", "libraryInfo"),
    ] {
        if let Some(path) = options.get(option) {
            package.metadata[key] = json_object(path)?;
        }
    }
    package.validate_rvc_synth()?;
    publish_cache(output, |temporary| {
        write_cache(temporary, &package)?;
        inspect_model_artifact(temporary)?;
        Ok(())
    })?;
    describe(output)
}

/// Publish a fully validated file without overwriting a concurrently created output.
fn publish_cache(output: &Path, write: impl FnOnce(&Path) -> CliResult<()>) -> CliResult<()> {
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let temporary = Temporary(parent.join(format!(
        ".vfrm-import-{}-{}.tmp.vfrm",
        std::process::id(),
        TEMP_ID.fetch_add(1, Ordering::Relaxed)
    )));
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary.0)?;
    write(&temporary.0)?;
    fs::OpenOptions::new()
        .write(true)
        .open(&temporary.0)?
        .sync_all()?;
    // Same-directory hard link is an atomic create-if-absent on NTFS and Unix.
    fs::hard_link(&temporary.0, output)?;
    Ok(())
}

struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reject_ambiguous_options() {
        for args in [
            vec!["--name"],
            vec!["--force", "true"],
            vec!["--name", "a", "--name", "b"],
        ] {
            assert!(options(&args.into_iter().map(String::from).collect::<Vec<_>>()).is_err());
        }
    }
    #[test]
    fn failed_import_cleans_temporary_and_preserves_existing_output() {
        let directory = std::env::temp_dir().join(format!("vfrm-cli-test-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let output = directory.join("model.vfrm");
        fs::write(&output, b"original").unwrap();
        assert!(
            publish_cache(&output, |p| {
                fs::write(p, b"new")?;
                Ok(())
            })
            .is_err()
        );
        assert_eq!(fs::read(&output).unwrap(), b"original");
        fs::remove_file(&output).unwrap();
        assert!(
            publish_cache(&output, |p| {
                fs::write(p, b"partial")?;
                Err("failed validation".into())
            })
            .is_err()
        );
        assert!(!output.exists());
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 0);
        publish_cache(&output, |p| {
            assert_eq!(p.extension().and_then(|s| s.to_str()), Some("vfrm"));
            fs::write(p, b"complete")?;
            Ok(())
        })
        .unwrap();
        assert_eq!(fs::read(&output).unwrap(), b"complete");
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
        fs::remove_file(&output).unwrap();
        fs::remove_dir(&directory).unwrap();
    }
}
