//! Reproducible listing/import smoke tool shared with the GUI implementation.
use vocal_runtime::model_library::*;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("list") {
        let provider = match args[2].as_str() {
            "ms" => ModelProvider::ModelScope,
            "mirror" => ModelProvider::HfMirror,
            _ => ModelProvider::HuggingFace,
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&list_remote_models(RemoteRepository {
                provider,
                repository: args[3].clone(),
                revision: None
            })?)?
        );
        return Ok(());
    }
    let request: ModelImportRequest = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let service = ModelLibrary::new(|s| println!("{}", serde_json::to_string(&s).unwrap()));
    service.start(request)?;
    while service.status().phase.is_active() {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let status = service.status();
    if let Some(error) = status.error {
        return Err(error.into());
    }
    let path = status.output_path.ok_or("no published model")?;
    let before = vocal_model::read_model_metadata(&path)?;
    let mut fields = before.fields.clone();
    fields.display_name.push_str(" · renamed");
    let context = vocal_runtime::RuntimeContext::new(vocal_runtime::RuntimeConfig::default())?;
    context.load_model(path.display().to_string())?;
    let first = context.snapshot()?;
    let view = context.edit_model_metadata(path.display().to_string(), before.revision, fields)?;
    let after = context.snapshot()?;
    assert_eq!(
        first.engine.as_ref().unwrap().processed_chunks,
        after.engine.as_ref().unwrap().processed_chunks
    );
    assert_eq!(
        after.model.as_ref().unwrap().display_name,
        view.fields.display_name
    );
    assert_eq!(first.config_revision, after.config_revision);
    context.start_processing()?;
    assert!(
        context
            .edit_model_metadata(path.display().to_string(), view.revision, view.fields)
            .is_err()
    );
    println!("metadata synchronization and active-session guard passed");
    Ok(())
}
