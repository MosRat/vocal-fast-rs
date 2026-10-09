use super::*;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
};
use vocal_model::{
    RetrievalIndex, Tensor, bundle_rvc_cache, inspect_model_artifact, load_rvc_pth_auto,
    materialize_acoustic_sources, write_cache,
};
type Progress<'a> =
    &'a mut dyn FnMut(ImportPhase, String, Option<String>, u64, Option<u64>) -> LibraryResult<()>;
const MAX_FILE: u64 = 4 * 1024 * 1024 * 1024;

struct Staging(PathBuf);
impl Drop for Staging {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub(crate) fn execute(
    request: &ModelImportRequest,
    cancel: &AtomicBool,
    progress: Progress<'_>,
) -> LibraryResult<(PathBuf, bool, Vec<String>)> {
    let root = request
        .library_directory
        .clone()
        .unwrap_or_else(default_library_directory);
    fs::create_dir_all(&root)?;
    let root = fs::canonicalize(root)?;
    let unique = format!(
        "import-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );
    let stage = root.join(format!(".{unique}"));
    fs::create_dir(&stage)?;
    let _cleanup = Staging(stage.clone());
    let (mut model, mut index, source) = match &request.source {
        ImportSource::Local {
            model_path,
            index_path,
        } => (
            fs::canonicalize(model_path)?,
            index_path.as_ref().map(fs::canonicalize).transpose()?,
            "Local import".to_string(),
        ),
        ImportSource::Remote {
            repository,
            model_file,
            index_file,
        } => {
            progress(
                ImportPhase::Listing,
                "Resolving repository revision".into(),
                None,
                0,
                None,
            )?;
            let client = client()?;
            let catalog = provider::list_with_client(&client, repository.clone())?;
            let find = |path: &str| {
                catalog
                    .files
                    .iter()
                    .find(|f| f.path == path)
                    .ok_or_else(|| {
                        LibraryError::new("notFound", format!("Repository file not found: {path}"))
                    })
            };
            let m = find(model_file)?;
            if m.kind == "index" {
                return Err(LibraryError::new(
                    "invalidModel",
                    "Select a VFRM, PTH or ZIP model file",
                ));
            }
            let cache = root.join(".downloads");
            let model = download::download(
                &client,
                &catalog.repository,
                m,
                &cache,
                cancel,
                &mut |n, total| {
                    let _ = progress(
                        ImportPhase::Downloading,
                        "Downloading model weights".into(),
                        Some(m.path.clone()),
                        n,
                        total,
                    );
                },
            )?;
            check_cancel(cancel)?;
            let index = if let Some(p) = index_file {
                let f = find(p)?;
                if f.kind != "index" {
                    return Err(LibraryError::new(
                        "invalidIndex",
                        "Select an .index file for retrieval",
                    ));
                }
                Some(download::download(
                    &client,
                    &catalog.repository,
                    f,
                    &cache,
                    cancel,
                    &mut |n, total| {
                        let _ = progress(
                            ImportPhase::Downloading,
                            "Downloading retrieval index".into(),
                            Some(f.path.clone()),
                            n,
                            total,
                        );
                    },
                )?)
            } else {
                None
            };
            (
                model,
                index,
                format!(
                    "{:?}: {}@{}",
                    catalog.repository.provider,
                    catalog.repository.repository,
                    catalog.repository.revision.as_deref().unwrap_or_default()
                ),
            )
        }
    };
    check_cancel(cancel)?;
    if extension(&model) == "zip" {
        progress(
            ImportPhase::Verifying,
            "Inspecting model archive".into(),
            None,
            0,
            None,
        )?;
        let extracted = extract(&model, &stage, cancel)?;
        model = extracted.0;
        if index.is_none() {
            index = extracted.1;
        }
    }
    let model_size = fs::metadata(&model)?.len();
    if model_size == 0 || model_size > MAX_FILE {
        return Err(LibraryError::new(
            "size",
            "Model file exceeds the 4 GiB import limit",
        ));
    }
    if fs2::available_space(&root)? < model_size.saturating_mul(2) + 256 * 1024 * 1024 {
        return Err(LibraryError::new(
            "disk",
            "Not enough disk space to convert and publish this model",
        ));
    }
    let synth = stage.join("voice.vfrm");
    let output = stage.join("model.vfrm");
    match extension(&model).as_str() {
        "pth" => {
            progress(
                ImportPhase::Converting,
                "Converting RVC checkpoint to canonical VFRM".into(),
                Some(model.display().to_string()),
                0,
                None,
            )?;
            let mut package = load_rvc_pth_auto(&model, &request.display_name, &source)?;
            package.manifest.id = unique.clone();
            if let Some(path) = &index {
                let data = read_bounded(path, MAX_FILE, cancel)?;
                let parsed = RetrievalIndex::from_faiss(&data)?;
                if parsed.dimensions() != package.manifest.feature_dims() {
                    return Err(LibraryError::new(
                        "invalidIndex",
                        "Retrieval index feature width does not match the voice model",
                    ));
                }
                drop(parsed);
                package.manifest.index_file = path.file_name().map(|s| s.to_string_lossy().into());
                package
                    .tensors
                    .push(Tensor::byte_tensor("aux.retrieval.faiss", data));
            }
            check_cancel(cancel)?;
            write_cache(&synth, &package)?;
            drop(package);
        }
        "vfr" | "vfrm" => {
            progress(
                ImportPhase::Verifying,
                "Validating VFRM components".into(),
                Some(model.display().to_string()),
                0,
                None,
            )?;
            let inspection = inspect_model_artifact(&model)?;
            if !inspection.components.synth.present {
                return Err(LibraryError::new(
                    "invalidModel",
                    "Selected VFRM contains acoustic weights only, not a voice model",
                ));
            }
            if index.is_some() {
                return Err(LibraryError::new(
                    "invalidIndex",
                    "VFRM imports preserve their embedded index; add an index while importing a PTH checkpoint",
                ));
            }
            copy_cancel(&model, &synth, cancel)?;
            let metadata = vocal_model::read_model_metadata(&synth)?;
            let mut fields = metadata.fields;
            fields.display_name = request.display_name.clone();
            vocal_model::edit_model_metadata(&synth, &metadata.revision, fields)?;
        }
        _ => {
            return Err(LibraryError::new(
                "format",
                "Import accepts RVC .pth, .vfr/.vfrm or a ZIP containing one voice model",
            ));
        }
    }
    check_cancel(cancel)?;
    let inspection = inspect_model_artifact(&synth)?;
    let mut warnings = Vec::new();
    if inspection.components.automatic_tensor_rt {
        fs::rename(&synth, &output)?;
    } else {
        progress(
            ImportPhase::Bundling,
            "Resolving shared HuBERT and RMVPE weights".into(),
            None,
            0,
            None,
        )?;
        let sources = acoustic_sources(request, &root, cancel)?;
        if let Some((hubert, rmvpe)) = sources {
            let estimate = fs::metadata(&synth)?.len()
                + fs::metadata(&hubert)?.len()
                + fs::metadata(&rmvpe)?.len()
                + 64 * 1024 * 1024;
            if fs2::available_space(&root)? < estimate {
                return Err(LibraryError::new(
                    "disk",
                    "Not enough disk space to bundle acoustic weights",
                ));
            }
            check_cancel(cancel)?;
            bundle_rvc_cache(&synth, &hubert, &rmvpe, &output)?;
        } else {
            warnings.push("Synth-only model: import shared HuBERT and RMVPE VFRM weights, or choose an existing complete model, to enable CPU, DirectML and automatic TensorRT preparation.".into());
            fs::rename(&synth, &output)?;
        }
    }
    check_cancel(cancel)?;
    let inspection = inspect_model_artifact(&output)?;
    if !inspection.components.retrieval.present {
        warnings.push("No retrieval index is included. Set index ratio to 0, or import the matching index with a PTH checkpoint.".into());
    }
    // Store source checksums next to the model for provenance and later audits.
    progress(
        ImportPhase::Verifying,
        "Recording model provenance".into(),
        None,
        0,
        None,
    )?;
    let hash = hash_file(&model, cancel)?;
    fs::write(stage.join("provenance.json"),serde_json::to_vec_pretty(&serde_json::json!({"source":&request.source,"modelSha256":hash,"displayName":request.display_name,"complete":inspection.components.automatic_tensor_rt})).map_err(|e|LibraryError::new("metadata",e.to_string()))?)?;
    check_cancel(cancel)?;
    progress(
        ImportPhase::Publishing,
        "Adding verified model to library".into(),
        None,
        0,
        None,
    )?;
    // Same filesystem directory rename publishes model and provenance together.
    if synth.exists() {
        fs::remove_file(&synth)?;
    }
    let archive = stage.join("archive");
    if archive.exists() {
        fs::remove_dir_all(archive)?;
    }
    let published = root.join(&unique);
    fs::rename(&stage, &published)?;
    Ok((
        published.join("model.vfrm"),
        inspection.components.automatic_tensor_rt,
        warnings,
    ))
}

fn acoustic_sources(
    request: &ModelImportRequest,
    root: &Path,
    cancel: &AtomicBool,
) -> LibraryResult<Option<(PathBuf, PathBuf)>> {
    if request.hubert_source.is_some() || request.rmvpe_source.is_some() {
        let h = request.hubert_source.clone().ok_or_else(|| {
            LibraryError::new(
                "acoustic",
                "Select both HuBERT and RMVPE canonical VFRM files",
            )
        })?;
        let r = request.rmvpe_source.clone().ok_or_else(|| {
            LibraryError::new(
                "acoustic",
                "Select both HuBERT and RMVPE canonical VFRM files",
            )
        })?;
        if !inspect_model_artifact(&h)?.components.hubert.present
            || !inspect_model_artifact(&r)?.components.rmvpe.present
        {
            return Err(LibraryError::new(
                "acoustic",
                "Acoustic source kinds do not match",
            ));
        }
        return Ok(Some((h, r)));
    }
    let selected = request.acoustic_bundle.clone();
    let bundle = if let Some(path) = selected {
        Some(path)
    } else {
        find_bundle(root, cancel)?
    };
    if let Some(bundle) = bundle {
        let [h, r] = materialize_acoustic_sources(&bundle, root.join(".acoustic-cache"))?;
        return Ok(Some((h, r)));
    }
    Ok(None)
}
fn find_bundle(root: &Path, cancel: &AtomicBool) -> LibraryResult<Option<PathBuf>> {
    let mut pending = vec![(root.to_path_buf(), 0)];
    let mut seen = 0;
    while let Some((dir, depth)) = pending.pop() {
        for e in fs::read_dir(dir)? {
            check_cancel(cancel)?;
            let e = e?;
            seen += 1;
            if seen > 2000 {
                return Ok(None);
            }
            let p = e.path();
            let t = e.file_type()?;
            if t.is_dir() && depth < 4 && !e.file_name().to_string_lossy().starts_with('.') {
                pending.push((p, depth + 1));
            } else if t.is_file()
                && matches!(extension(&p).as_str(), "vfr" | "vfrm")
                && inspect_model_artifact(&p).is_ok_and(|i| i.components.automatic_tensor_rt)
            {
                return Ok(Some(p));
            }
        }
    }
    Ok(None)
}
fn extension(path: &Path) -> String {
    path.extension()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
}
fn read_bounded(path: &Path, max: u64, cancel: &AtomicBool) -> LibraryResult<Vec<u8>> {
    if fs::metadata(path)?.len() > max {
        return Err(LibraryError::new("size", "File exceeds import limits"));
    }
    let mut file = File::open(path)?;
    let mut data = Vec::new();
    let mut bytes = [0; 128 * 1024];
    loop {
        check_cancel(cancel)?;
        let n = file.read(&mut bytes)?;
        if n == 0 {
            break;
        }
        if data.len() as u64 + n as u64 > max {
            return Err(LibraryError::new("size", "File grew beyond import limits"));
        }
        data.extend_from_slice(&bytes[..n]);
    }
    Ok(data)
}
fn copy_cancel(input: &Path, output: &Path, cancel: &AtomicBool) -> LibraryResult<()> {
    let mut from = File::open(input)?;
    let mut to = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)?;
    let mut bytes = [0; 128 * 1024];
    loop {
        check_cancel(cancel)?;
        let n = from.read(&mut bytes)?;
        if n == 0 {
            break;
        }
        to.write_all(&bytes[..n])?;
    }
    to.sync_all()?;
    Ok(())
}
fn hash_file(path: &Path, cancel: &AtomicBool) -> LibraryResult<String> {
    let mut input = File::open(path)?;
    let mut hash = Sha256::new();
    let mut b = [0; 128 * 1024];
    loop {
        check_cancel(cancel)?;
        let n = input.read(&mut b)?;
        if n == 0 {
            break;
        }
        hash.update(&b[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn extract(
    path: &Path,
    stage: &Path,
    cancel: &AtomicBool,
) -> LibraryResult<(PathBuf, Option<PathBuf>)> {
    let mut zip = zip::ZipArchive::new(File::open(path)?)
        .map_err(|e| LibraryError::new("archive", e.to_string()))?;
    if zip.len() > 10_000 {
        return Err(LibraryError::new(
            "archive",
            "Archive contains too many entries",
        ));
    }
    let dest = stage.join("archive");
    fs::create_dir(&dest)?;
    let mut models = Vec::new();
    let mut indices = Vec::new();
    let mut total = 0_u64;
    let mut names = std::collections::HashSet::new();
    for i in 0..zip.len() {
        check_cancel(cancel)?;
        let mut entry = zip
            .by_index(i)
            .map_err(|e| LibraryError::new("archive", e.to_string()))?;
        let name = entry.name().to_owned();
        validate_relative(name.trim_end_matches('/'))?;
        if entry.is_symlink() {
            return Err(LibraryError::new(
                "archive",
                "Archive symlinks are not supported",
            ));
        }
        if entry.is_dir() {
            continue;
        }
        total = total
            .checked_add(entry.size())
            .ok_or_else(|| LibraryError::new("size", "Archive size overflow"))?;
        if total > MAX_FILE
            || entry.size() > MAX_FILE
            || (entry.size() > 16 * 1024 * 1024
                && entry.size() / entry.compressed_size().max(1) > 1000)
        {
            return Err(LibraryError::new(
                "archive",
                "Archive expanded size exceeds limits",
            ));
        }
        let ext = extension(Path::new(&name));
        if !matches!(ext.as_str(), "pth" | "vfr" | "vfrm" | "index") {
            continue;
        }
        if !names.insert(name.to_lowercase()) {
            return Err(LibraryError::new(
                "archive",
                "Archive has duplicate file names",
            ));
        }
        let output = dest.join(&name);
        fs::create_dir_all(output.parent().unwrap())?;
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&output)?;
        let mut b = [0; 128 * 1024];
        let mut nbytes = 0;
        loop {
            check_cancel(cancel)?;
            let n = entry.read(&mut b)?;
            if n == 0 {
                break;
            }
            nbytes += n as u64;
            if nbytes > MAX_FILE {
                return Err(LibraryError::new("size", "Archive entry exceeds limits"));
            }
            file.write_all(&b[..n])?;
        }
        file.sync_all()?;
        if ext == "index" {
            indices.push(output);
        } else {
            models.push(output);
        }
    }
    if models.len() != 1 || indices.len() > 1 {
        return Err(LibraryError::new(
            "archive",
            "Archive must contain one voice checkpoint and at most one index. Extract it locally and import the matching files separately.",
        ));
    }
    Ok((models.remove(0), indices.pop()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn archive_traversal_is_rejected_without_writing_outside_staging() {
        let root = std::env::temp_dir().join(format!(
            "vocal-zip-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        let _cleanup = Staging(root.clone());
        let archive = root.join("model.zip");
        let mut writer = zip::ZipWriter::new(File::create(&archive).unwrap());
        writer
            .start_file("../escape.pth", zip::write::SimpleFileOptions::default())
            .unwrap();
        writer.write_all(b"weights").unwrap();
        writer.finish().unwrap();
        assert_eq!(
            extract(&archive, &root, &AtomicBool::new(false))
                .unwrap_err()
                .code,
            "path"
        );
        assert!(!root.parent().unwrap().join("escape.pth").exists());
    }
}
