use std::fs;
use std::path::{Path, PathBuf};

use burn_dataset::Dataset;

use crate::Error;

/// Canonical TinyShakespeare corpus, as used by Karpathy's char-rnn / nano-gpt.
pub const TINY_SHAKESPEARE_URL: &str =
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

const CACHE_FILE_NAME: &str = "tinyshakespeare.txt";

#[derive(Debug, Clone)]
pub struct TinyShakespeare {
    text: String,
    line_offsets: Vec<(usize, usize)>,
}

impl TinyShakespeare {
    /// Build the dataset from an in-memory corpus.
    pub fn from_text(text: String) -> Self {
        let line_offsets = build_line_offsets(&text);
        Self { text, line_offsets }
    }

    /// Load the dataset from a local text file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        let text = fs::read_to_string(path)?;
        Ok(Self::from_text(text))
    }

    /// Fetch the TinyShakespeare corpus, caching it under `cache_dir`.
    /// Re-uses the cached file on subsequent calls.
    pub fn download(cache_dir: impl AsRef<Path>) -> Result<Self, Error> {
        Self::download_from(TINY_SHAKESPEARE_URL, cache_dir)
    }

    /// Like [`Self::download`] but allows overriding the source URL — useful
    /// for tests and mirrors.
    pub fn download_from(url: &str, cache_dir: impl AsRef<Path>) -> Result<Self, Error> {
        let cache_dir = cache_dir.as_ref();
        let path: PathBuf = cache_dir.join(CACHE_FILE_NAME);
        if !path.exists() {
            fs::create_dir_all(cache_dir)?;
            tracing::info!(url, target = %path.display(), "downloading TinyShakespeare");
            let body = ureq::get(url).call()?.body_mut().read_to_string()?;
            fs::write(&path, body)?;
        } else {
            tracing::debug!(target = %path.display(), "using cached TinyShakespeare");
        }
        Self::from_file(path)
    }

    /// Raw corpus text, useful for downstream tokenizer training / windowing.
    pub fn text(&self) -> &str {
        &self.text
    }
}

impl Dataset<String> for TinyShakespeare {
    fn get(&self, index: usize) -> Option<String> {
        let (start, end) = *self.line_offsets.get(index)?;
        Some(self.text[start..end].to_string())
    }

    fn len(&self) -> usize {
        self.line_offsets.len()
    }
}

fn build_line_offsets(text: &str) -> Vec<(usize, usize)> {
    let mut offsets = Vec::new();
    let mut start = 0;
    for (i, byte) in text.bytes().enumerate() {
        if byte == b'\n' {
            offsets.push((start, i));
            start = i + 1;
        }
    }
    if start < text.len() {
        offsets.push((start, text.len()));
    }
    offsets
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::io::Write;

    use super::*;

    #[test]
    fn from_text_yields_lines() {
        let ds = TinyShakespeare::from_text("first\nsecond\nthird".to_string());
        assert_eq!(ds.len(), 3);
        assert_eq!(ds.get(0).as_deref(), Some("first"));
        assert_eq!(ds.get(1).as_deref(), Some("second"));
        assert_eq!(ds.get(2).as_deref(), Some("third"));
        assert_eq!(ds.get(3), None);
    }

    #[test]
    fn from_text_drops_trailing_empty_line() {
        let ds = TinyShakespeare::from_text("a\nb\n".to_string());
        assert_eq!(ds.len(), 2);
        assert_eq!(ds.get(0).as_deref(), Some("a"));
        assert_eq!(ds.get(1).as_deref(), Some("b"));
    }

    #[test]
    fn from_text_preserves_full_corpus() {
        let raw = "alpha\nbeta\ngamma";
        let ds = TinyShakespeare::from_text(raw.to_string());
        assert_eq!(ds.text(), raw);
    }

    #[test]
    fn from_file_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corpus.txt");
        let mut f = fs::File::create(&path).unwrap();
        f.write_all(b"hello\nworld").unwrap();
        drop(f);

        let ds = TinyShakespeare::from_file(&path).unwrap();
        assert_eq!(ds.len(), 2);
        assert_eq!(ds.text(), "hello\nworld");
    }
}
