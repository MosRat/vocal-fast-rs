use super::{LibraryError, LibraryResult, client, validate_relative};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashSet, io::Read};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ModelProvider {
    HuggingFace,
    HfMirror,
    ModelScope,
}
impl ModelProvider {
    fn endpoint(self) -> &'static str {
        match self {
            Self::HuggingFace => "https://huggingface.co",
            Self::HfMirror => "https://hf-mirror.com",
            Self::ModelScope => "https://modelscope.cn",
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RemoteRepository {
    pub provider: ModelProvider,
    pub repository: String,
    pub revision: Option<String>,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RemoteFile {
    pub path: String,
    pub size_bytes: Option<u64>,
    pub sha256: Option<String>,
    pub kind: String,
    pub revision: Option<String>,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RemoteModelCatalog {
    pub repository: RemoteRepository,
    pub files: Vec<RemoteFile>,
    pub total_files: usize,
}

fn repo_id(input: &str, provider: ModelProvider) -> LibraryResult<String> {
    let id = if input.starts_with("https://") {
        let url = reqwest::Url::parse(input)
            .map_err(|_| LibraryError::new("repository", "Invalid repository URL"))?;
        let expected = reqwest::Url::parse(provider.endpoint()).unwrap();
        if url.host_str() != expected.host_str()
            || !url.username().is_empty()
            || url.password().is_some()
        {
            return Err(LibraryError::new(
                "repository",
                "Repository URL does not match the selected provider",
            ));
        }
        let parts: Vec<_> = url.path().trim_matches('/').split('/').collect();
        let offset = usize::from(
            matches!(provider, ModelProvider::ModelScope) && parts.first() == Some(&"models"),
        );
        if parts.len() < offset + 2 {
            return Err(LibraryError::new(
                "repository",
                "Use an owner/model repository",
            ));
        }
        format!("{}/{}", parts[offset], parts[offset + 1])
    } else {
        input.trim().trim_matches('/').into()
    };
    if id.split('/').count() != 2
        || id.len() > 256
        || id.split('/').any(|s| {
            s.is_empty()
                || s == "."
                || s == ".."
                || !s
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
        })
    {
        return Err(LibraryError::new(
            "repository",
            "Use an owner/model repository ID",
        ));
    }
    Ok(id)
}
fn url(provider: ModelProvider, segments: &[&str]) -> reqwest::Url {
    let mut u = reqwest::Url::parse(provider.endpoint()).unwrap();
    u.path_segments_mut().unwrap().extend(segments);
    u
}
fn json(client: &reqwest::blocking::Client, url: reqwest::Url) -> LibraryResult<Value> {
    let response = client.get(url).send()?;
    let status = response.status();
    if !status.is_success() {
        if response.url().host_str() == Some("hf-mirror.com") {
            return Err(LibraryError::new(
                "http",
                format!(
                    "HF Mirror returned HTTP {status}. Switch Source to Hugging Face, or retry later."
                ),
            ));
        }
        return Err(LibraryError::new(
            if status.as_u16() == 404 {
                "notFound"
            } else if matches!(status.as_u16(), 401 | 403) {
                "accessDenied"
            } else {
                "http"
            },
            format!(
                "Repository request returned HTTP {status}. Public repositories are supported."
            ),
        ));
    }
    let mut bytes = Vec::new();
    response.take(8 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > 8 * 1024 * 1024 {
        return Err(LibraryError::new(
            "repository",
            "Repository response exceeds 8 MiB",
        ));
    }
    let v: Value = serde_json::from_slice(&bytes)
        .map_err(|e| LibraryError::new("repository", e.to_string()))?;
    if v.get("Code")
        .and_then(Value::as_u64)
        .is_some_and(|c| c != 200)
    {
        return Err(LibraryError::new(
            "repository",
            v["Message"].as_str().unwrap_or("ModelScope request failed"),
        ));
    }
    Ok(v)
}
fn file(path: &str, size: Option<u64>, hash: Option<&str>) -> LibraryResult<Option<RemoteFile>> {
    let ext = path
        .rsplit('.')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let kind = match ext.as_str() {
        "pth" => "checkpoint",
        "vfr" | "vfrm" => "vfrm",
        "index" => "index",
        "zip" => "archive",
        _ => return Ok(None),
    };
    validate_relative(path)?;
    Ok(Some(RemoteFile {
        path: path.into(),
        size_bytes: size,
        sha256: hash
            .filter(|h| h.len() == 64 && h.chars().all(|c| c.is_ascii_hexdigit()))
            .map(str::to_ascii_lowercase),
        kind: kind.into(),
        revision: None,
    }))
}
pub fn list_remote_models(repository: RemoteRepository) -> LibraryResult<RemoteModelCatalog> {
    list_with_client(&client()?, repository)
}
pub(crate) fn list_with_client(
    client: &reqwest::blocking::Client,
    mut repository: RemoteRepository,
) -> LibraryResult<RemoteModelCatalog> {
    repository.repository = repo_id(&repository.repository, repository.provider)?;
    let mut revision = repository
        .revision
        .clone()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| {
            if matches!(repository.provider, ModelProvider::ModelScope) {
                "master"
            } else {
                "main"
            }
            .into()
        });
    if revision.len() > 256 || revision.chars().any(|c| c.is_control()) {
        return Err(LibraryError::new("repository", "Invalid revision"));
    }
    let parts: Vec<_> = repository.repository.split('/').collect();
    let mut files = Vec::new();
    let mut total = 0;
    if matches!(repository.provider, ModelProvider::ModelScope) {
        // The official legacy API truncates recursive listings. Enumerate each
        // directory independently and fail visibly on a suspiciously large one.
        let mut pending = vec![String::new()];
        let mut visited = HashSet::new();
        while let Some(root) = pending.pop() {
            if !visited.insert(root.clone()) {
                continue;
            }
            if visited.len() > 512 {
                return Err(LibraryError::new(
                    "repository",
                    "Repository has too many directories",
                ));
            }
            let mut u = url(
                repository.provider,
                &["api", "v1", "models", parts[0], parts[1], "repo", "files"],
            );
            u.query_pairs_mut()
                .append_pair("Revision", &revision)
                .append_pair("Recursive", "False")
                .append_pair("Root", &root);
            let v = json(client, u)?;
            let entries = v
                .pointer("/Data/Files")
                .or_else(|| v.get("Data"))
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    LibraryError::new("repository", "ModelScope returned no file list")
                })?;
            if entries.len() >= 1000 {
                return Err(LibraryError::new(
                    "repository",
                    "ModelScope directory listing may be truncated; choose a smaller repository",
                ));
            }
            for e in entries {
                let path = e["Path"]
                    .as_str()
                    .or_else(|| e["Name"].as_str())
                    .unwrap_or_default();
                validate_relative(path)?;
                if matches!(e["Type"].as_str(), Some("tree" | "directory" | "dir")) {
                    pending.push(path.into());
                    continue;
                }
                total += 1;
                if total > 10_000 {
                    return Err(LibraryError::new(
                        "repository",
                        "Repository has too many files",
                    ));
                }
                if let Some(mut f) = file(path, e["Size"].as_u64(), e["Sha256"].as_str())? {
                    f.revision = e["Revision"]
                        .as_str()
                        .filter(|s| s.len() == 40 && s.chars().all(|c| c.is_ascii_hexdigit()))
                        .map(str::to_owned);
                    files.push(f);
                }
            }
        }
    } else {
        let mut u = url(
            repository.provider,
            &["api", "models", parts[0], parts[1], "revision", &revision],
        );
        u.query_pairs_mut().append_pair("blobs", "true");
        let v = json(client, u)?;
        revision = v["sha"]
            .as_str()
            .filter(|s| s.len() == 40 && s.chars().all(|c| c.is_ascii_hexdigit()))
            .ok_or_else(|| {
                LibraryError::new("repository", "Hub response has no immutable revision")
            })?
            .into();
        let siblings = v["siblings"]
            .as_array()
            .ok_or_else(|| LibraryError::new("repository", "Hub response has no files"))?;
        total = siblings.len();
        if total > 10_000 {
            return Err(LibraryError::new(
                "repository",
                "Repository has too many files",
            ));
        }
        for e in siblings {
            if let Some(f) = file(
                e["rfilename"].as_str().unwrap_or_default(),
                e["size"]
                    .as_u64()
                    .or_else(|| e.pointer("/lfs/size").and_then(Value::as_u64)),
                e.pointer("/lfs/sha256").and_then(Value::as_str),
            )? {
                files.push(f);
            }
        }
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));
    repository.revision = Some(revision);
    Ok(RemoteModelCatalog {
        repository,
        files,
        total_files: total,
    })
}
pub(crate) fn download_url(
    repository: &RemoteRepository,
    path: &str,
) -> LibraryResult<reqwest::Url> {
    validate_relative(path)?;
    let id = repo_id(&repository.repository, repository.provider)?;
    let p: Vec<_> = id.split('/').collect();
    let revision = repository.revision.as_deref().unwrap_or("main");
    if matches!(repository.provider, ModelProvider::ModelScope) {
        let mut u = url(
            repository.provider,
            &["api", "v1", "models", p[0], p[1], "repo"],
        );
        u.query_pairs_mut()
            .append_pair("Revision", revision)
            .append_pair("FilePath", path);
        Ok(u)
    } else {
        let mut segments = vec![p[0], p[1], "resolve", revision];
        segments.extend(path.split('/'));
        Ok(url(repository.provider, &segments))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn repository_and_paths_are_bounded() {
        assert_eq!(
            repo_id(
                "https://hf-mirror.com/user/model/tree/main",
                ModelProvider::HfMirror
            )
            .unwrap(),
            "user/model"
        );
        assert!(repo_id("https://evil.example/user/model", ModelProvider::HfMirror).is_err());
        for p in [
            "../a.pth",
            "C:/a.pth",
            "dir/CON.pth",
            "dir/a.pth:ads",
            "a\\b.pth",
        ] {
            assert!(validate_relative(p).is_err());
        }
    }
    #[test]
    fn chinese_download_paths_are_encoded() {
        let r = RemoteRepository {
            provider: ModelProvider::HfMirror,
            repository: "owner/model".into(),
            revision: Some("abc".into()),
        };
        assert!(
            download_url(&r, "weights/男声.pth")
                .unwrap()
                .as_str()
                .contains("%E7%94%B7")
        );
    }
}
