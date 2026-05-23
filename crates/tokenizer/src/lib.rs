//! Tokenizers for nano-gpt.
//!
//! Exposes the [`Tokenizer`] trait — `encode` / `decode` / `vocab_size` — and
//! a [`CharTokenizer`] implementation that maps each unique `char` of a
//! training corpus to a single token id. BPE is intended to live alongside
//! it as a second implementation, selectable via config.

mod char_tokenizer;
mod error;

pub use char_tokenizer::CharTokenizer;
pub use error::Error;

/// A reversible mapping between text and integer token ids.
///
/// `encode` may fail when the input contains symbols outside the
/// tokenizer's vocabulary; `decode` may fail when a token id is out of
/// range.
pub trait Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, Error>;
    fn decode(&self, tokens: &[u32]) -> Result<String, Error>;
    fn vocab_size(&self) -> usize;
}
