use super::{
    LibraryError, LibraryResult, RemoteFile, RemoteRepository, check_cancel, provider::download_url,
};
use fs2::FileExt;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
    time::Instant,
};
const MAX_DOWNLOAD: u64 = 4 * 1024 * 1024 * 1024;

pub(crate) fn download(
    client: &reqwest::blocking::Client,
    repository: &RemoteRepository,
    entry: &RemoteFile,
    cache: &Path,
    cancel: &AtomicBool,
    progress: &mut impl FnMut(u64, Option<u64>),
) -> LibraryResult<PathBuf> {
    let mut resolved = repository.clone();
    if let Some(revision) = &entry.revision {
        resolved.revision = Some(revision.clone());
    }
    let immutable = resolved
        .revision
        .as_ref()
        .is_some_and(|s| s.len() == 40 && s.chars().all(|c| c.is_ascii_hexdigit()));
    if !immutable && entry.sha256.is_none() {
        return Err(LibraryError::new(
            "revision",
            "Repository did not supply an immutable file revision or SHA-256; safe cache/resume is unavailable",
        ));
    }
    download_at(
        client,
        &resolved,
        entry,
        cache,
        cancel,
        progress,
        download_url(&resolved, &entry.path)?,
    )
}

fn download_at(
    client: &reqwest::blocking::Client,
    repository: &RemoteRepository,
    entry: &RemoteFile,
    cache: &Path,
    cancel: &AtomicBool,
    progress: &mut impl FnMut(u64, Option<u64>),
    url: reqwest::Url,
) -> LibraryResult<PathBuf> {
    if entry.size_bytes.is_some_and(|s| s == 0 || s > MAX_DOWNLOAD) {
        return Err(LibraryError::new(
            "size",
            "Model file must be between 1 byte and 4 GiB",
        ));
    }
    fs::create_dir_all(cache)?;
    let identity = serde_json::to_vec(&(repository, &entry.path, entry.size_bytes, &entry.sha256))
        .map_err(|e| LibraryError::new("repository", e.to_string()))?;
    let key = format!("{:x}", Sha256::digest(identity));
    let ext = entry.path.rsplit('.').next().unwrap_or("bin");
    let output = cache.join(format!("{key}.{ext}"));
    let part = cache.join(format!("{key}.part"));
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(cache.join(format!("{key}.lock")))?;
    lock.try_lock_exclusive().map_err(|_| {
        LibraryError::new(
            "busy",
            "This file is being downloaded by another application instance",
        )
    })?;
    if output.exists() {
        if verify(&output, entry, cancel).is_ok() {
            progress(fs::metadata(&output)?.len(), entry.size_bytes);
            return Ok(output);
        }
        check_cancel(cancel)?;
        fs::remove_file(&output)?;
    }
    let mut last_error = None;
    for _ in 0..4 {
        check_cancel(cancel)?;
        let offset = fs::metadata(&part).map_or(0, |m| m.len());
        if entry.size_bytes == Some(offset) && offset != 0 {
            break;
        }
        if offset > MAX_DOWNLOAD {
            fs::remove_file(&part)?;
            continue;
        }
        let free = fs2::available_space(cache)?;
        if free
            < entry
                .size_bytes
                .unwrap_or(256 * 1024 * 1024)
                .saturating_sub(offset)
                + 64 * 1024 * 1024
        {
            return Err(LibraryError::new(
                "disk",
                "Not enough free disk space for this download",
            ));
        }
        let mut request = client.get(url.clone());
        if offset > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
        }
        let mut response = match request.send() {
            Ok(r) => r,
            Err(e) => {
                last_error = Some(LibraryError::from(e));
                continue;
            }
        };
        let status = response.status();
        if !status.is_success() {
            return Err(LibraryError::new(
                if matches!(status.as_u16(), 401 | 403) {
                    "accessDenied"
                } else {
                    "http"
                },
                format!("Download returned HTTP {status}"),
            ));
        }
        let resumed = status.as_u16() == 206;
        if resumed {
            let range = response
                .headers()
                .get(reqwest::header::CONTENT_RANGE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or_default();
            if !range.starts_with(&format!("bytes {offset}-")) {
                return Err(LibraryError::new(
                    "download",
                    "Server returned an inconsistent resume range",
                ));
            }
        }
        let start = if resumed { offset } else { 0 };
        let total = entry
            .size_bytes
            .or_else(|| response.content_length().and_then(|n| n.checked_add(start)));
        if total.is_some_and(|n| n > MAX_DOWNLOAD) {
            return Err(LibraryError::new("size", "Remote file exceeds 4 GiB"));
        }
        let mut output_file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(resumed)
            .truncate(!resumed)
            .open(&part)?;
        let mut downloaded = start;
        let mut chunk = [0; 128 * 1024];
        let mut last_progress = Instant::now();
        let mut failed = false;
        progress(downloaded, total);
        loop {
            check_cancel(cancel)?;
            let n = match response.read(&mut chunk) {
                Ok(n) => n,
                Err(e) => {
                    last_error = Some(LibraryError::new("network", e.to_string()));
                    failed = true;
                    break;
                }
            };
            if n == 0 {
                break;
            }
            downloaded += n as u64;
            if downloaded > MAX_DOWNLOAD || total.is_some_and(|t| downloaded > t) {
                return Err(LibraryError::new(
                    "size",
                    "Download exceeded its declared size",
                ));
            }
            output_file.write_all(&chunk[..n])?;
            if last_progress.elapsed().as_millis() >= 200 {
                progress(downloaded, total);
                last_progress = Instant::now();
            }
        }
        output_file.sync_all()?;
        progress(downloaded, total);
        if !failed {
            last_error = None;
            break;
        }
    }
    if let Some(error) = last_error {
        return Err(error);
    }
    verify(&part, entry, cancel)?;
    fs::rename(&part, &output)?;
    Ok(output)
}

fn verify(path: &Path, entry: &RemoteFile, cancel: &AtomicBool) -> LibraryResult<()> {
    let size = fs::metadata(path)?.len();
    if size == 0 || size > MAX_DOWNLOAD || entry.size_bytes.is_some_and(|n| n != size) {
        return Err(LibraryError::new(
            "size",
            "Downloaded size does not match repository metadata",
        ));
    }
    let mut input = File::open(path)?;
    let mut hash = Sha256::new();
    let mut bytes = [0; 128 * 1024];
    loop {
        check_cancel(cancel)?;
        let n = input.read(&mut bytes)?;
        if n == 0 {
            break;
        }
        hash.update(&bytes[..n]);
    }
    let actual = format!("{:x}", hash.finalize());
    if entry
        .sha256
        .as_ref()
        .is_some_and(|expected| expected != &actual)
    {
        fs::remove_file(path)?;
        return Err(LibraryError::new(
            "checksum",
            "Downloaded SHA-256 does not match the repository",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{BufRead, BufReader},
        net::TcpListener,
        sync::{Arc, Mutex},
    };
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let p = std::env::temp_dir().join(format!(
                "vocal-download-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            fs::create_dir(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn server(
        responses: Vec<&'static str>,
    ) -> (
        reqwest::Url,
        Arc<Mutex<Vec<String>>>,
        std::thread::JoinHandle<()>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = reqwest::Url::parse(&format!(
            "http://{}/weights",
            listener.local_addr().unwrap()
        ))
        .unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let observed = requests.clone();
        let thread = std::thread::spawn(move || {
            for response in responses {
                let (mut socket, _) = listener.accept().unwrap();
                socket
                    .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                    .unwrap();
                let mut reader = BufReader::new(socket.try_clone().unwrap());
                let mut headers = String::new();
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" || line.is_empty() {
                        break;
                    }
                    headers.push_str(&line);
                }
                observed.lock().unwrap().push(headers);
                socket.write_all(response.as_bytes()).unwrap();
            }
        });
        (url, requests, thread)
    }
    fn entry() -> RemoteFile {
        RemoteFile {
            path: "voice.pth".into(),
            size_bytes: Some(6),
            sha256: Some(format!("{:x}", Sha256::digest(b"abcdef"))),
            kind: "checkpoint".into(),
            revision: None,
        }
    }
    fn repo() -> RemoteRepository {
        RemoteRepository {
            provider: super::super::ModelProvider::HuggingFace,
            repository: "owner/model".into(),
            revision: Some("immutable".into()),
        }
    }
    fn test_client() -> reqwest::blocking::Client {
        reqwest::blocking::Client::builder()
            .no_proxy()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap()
    }
    #[test]
    fn interrupted_transfer_resumes_valid_range_and_reuses_verified_cache() {
        let f = Fixture::new();
        let (url, requests, t) = server(vec![
            "HTTP/1.1 200 OK\r\nContent-Length: 6\r\nConnection: close\r\n\r\nabc",
            "HTTP/1.1 206 Partial Content\r\nContent-Length: 3\r\nContent-Range: bytes 3-5/6\r\nConnection: close\r\n\r\ndef",
        ]);
        let cancel = AtomicBool::new(false);
        let path = download_at(
            &test_client(),
            &repo(),
            &entry(),
            &f.0,
            &cancel,
            &mut |_, _| {},
            url.clone(),
        )
        .unwrap();
        t.join().unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"abcdef");
        assert!(
            requests.lock().unwrap()[1]
                .to_lowercase()
                .contains("range: bytes=3-")
        );
        assert_eq!(
            download_at(
                &test_client(),
                &repo(),
                &entry(),
                &f.0,
                &cancel,
                &mut |_, _| {},
                url
            )
            .unwrap(),
            path
        );
    }
    #[test]
    fn checksum_failure_removes_partial_and_cancellation_never_publishes() {
        let f = Fixture::new();
        let (url, _, t) = server(vec![
            "HTTP/1.1 200 OK\r\nContent-Length: 6\r\nConnection: close\r\n\r\nxxxxxx",
        ]);
        let cancel = AtomicBool::new(false);
        assert_eq!(
            download_at(
                &test_client(),
                &repo(),
                &entry(),
                &f.0,
                &cancel,
                &mut |_, _| {},
                url
            )
            .unwrap_err()
            .code,
            "checksum"
        );
        t.join().unwrap();
        assert!(
            !fs::read_dir(&f.0).unwrap().any(|e| e
                .unwrap()
                .path()
                .extension()
                .is_some_and(|e| e == "part"))
        );
        cancel.store(true, std::sync::atomic::Ordering::Release);
        assert_eq!(
            download_at(
                &test_client(),
                &repo(),
                &entry(),
                &f.0,
                &cancel,
                &mut |_, _| {},
                reqwest::Url::parse("http://127.0.0.1:1/").unwrap()
            )
            .unwrap_err()
            .code,
            "cancelled"
        );
    }
    #[test]
    fn unsafe_resume_response_does_not_join_different_content() {
        let f = Fixture::new();
        let identity =
            serde_json::to_vec(&(repo(), &entry().path, entry().size_bytes, &entry().sha256))
                .unwrap();
        let key = format!("{:x}", Sha256::digest(identity));
        let part = f.0.join(format!("{key}.part"));
        fs::write(&part, b"abc").unwrap();
        let (url, _, t) = server(vec![
            "HTTP/1.1 206 Partial Content\r\nContent-Length: 3\r\nContent-Range: bytes 0-2/6\r\nConnection: close\r\n\r\nxyz",
        ]);
        assert_eq!(
            download_at(
                &test_client(),
                &repo(),
                &entry(),
                &f.0,
                &AtomicBool::new(false),
                &mut |_, _| {},
                url
            )
            .unwrap_err()
            .code,
            "download"
        );
        t.join().unwrap();
        assert_eq!(fs::read(part).unwrap(), b"abc");
    }
}
