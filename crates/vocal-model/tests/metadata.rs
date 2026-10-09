use std::{fs, path::PathBuf};
use vocal_model::*;
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!(
            "vocal-metadata-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&dir).unwrap();
        let package = ModelPackage {
            manifest: RvcModelManifest {
                id: "voice".into(),
                display_name: "Original".into(),
                architecture: "rvc-v2".into(),
                version: "v2".into(),
                sample_rate: 40000,
                model_file: "voice.pth".into(),
                index_file: Some("voice.index".into()),
                source: "fixture".into(),
                notes: None,
            },
            metadata: serde_json::json!({"future":{"execution":"preserved"}}),
            tensors: vec![
                Tensor::byte_tensor("rvc.pth", (0..255).collect()),
                Tensor::byte_tensor("aux.retrieval.faiss", vec![9; 111]),
            ],
        };
        write_cache(dir.join("voice.vfrm"), &package).unwrap();
        Self(dir)
    }
    fn path(&self) -> PathBuf {
        self.0.join("voice.vfrm")
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn display_edits_preserve_tensors_and_computational_metadata() {
    let fixture = Fixture::new();
    let before = load_cache(fixture.path()).unwrap();
    let view = read_model_metadata(fixture.path()).unwrap();
    let mut edit = view.fields;
    edit.display_name = "中文模型 / Edited".repeat(9);
    edit.notes = Some("Longer metadata moves the tensor data section".repeat(50));
    edit.author = "Author".into();
    edit.tags = vec!["Chinese".into()];
    let result = edit_model_metadata(fixture.path(), &view.revision, edit).unwrap();
    assert_ne!(view.revision, result.revision);
    let after = load_cache(fixture.path()).unwrap();
    assert_eq!(before.metadata["future"], after.metadata["future"]);
    assert_eq!(before.manifest.sample_rate, after.manifest.sample_rate);
    assert_eq!(before.manifest.id, after.manifest.id);
    for (a, b) in before.tensors.iter().zip(&after.tensors) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.data, b.data);
        assert_eq!(a.shape, b.shape);
    }
    assert_eq!(result.fields.author, "Author");
    assert!(result.inspection.components.retrieval.present);
    // Shrink metadata again; relocation works in both directions.
    let mut edit = result.fields;
    edit.display_name = "Short".into();
    edit.notes = None;
    edit_model_metadata(fixture.path(), &result.revision, edit).unwrap();
    assert_eq!(
        load_cache(fixture.path()).unwrap().tensors[0].data,
        before.tensors[0].data
    );
}
#[test]
fn stale_and_invalid_edits_leave_original_intact() {
    let f = Fixture::new();
    let view = read_model_metadata(f.path()).unwrap();
    let original = fs::read(f.path()).unwrap();
    assert!(edit_model_metadata(f.path(), "stale", view.fields.clone()).is_err());
    let mut blank = view.fields.clone();
    blank.display_name = "  ".into();
    assert!(edit_model_metadata(f.path(), &view.revision, blank).is_err());
    assert_eq!(fs::read(f.path()).unwrap(), original);
    let mut edit = view.fields.clone();
    edit.display_name = "Updated".into();
    edit_model_metadata(f.path(), &view.revision, edit).unwrap();
    let updated = fs::read(f.path()).unwrap();
    assert!(edit_model_metadata(f.path(), &view.revision, view.fields).is_err());
    assert_eq!(fs::read(f.path()).unwrap(), updated);
    assert_eq!(fs::read_dir(&f.0).unwrap().count(), 1);
}
