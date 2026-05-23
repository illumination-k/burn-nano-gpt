use std::collections::BTreeSet;

use crate::{Error, Tokenizer};

/// Character-level tokenizer: every unique `char` in the training corpus
/// becomes one token. IDs are assigned by sorted code-point order so the
/// vocabulary is reproducible across runs and machines.
#[derive(Debug, Clone)]
pub struct CharTokenizer {
    // Sorted; index is the token id.
    id_to_char: Vec<char>,
}

impl CharTokenizer {
    /// Build a tokenizer from a corpus.
    pub fn from_text(text: &str) -> Self {
        let chars: BTreeSet<char> = text.chars().collect();
        Self {
            id_to_char: chars.into_iter().collect(),
        }
    }

    fn lookup(&self, c: char) -> Option<u32> {
        // `id_to_char` is sorted, so binary search is the natural lookup.
        self.id_to_char.binary_search(&c).ok().map(|i| i as u32)
    }
}

impl Tokenizer for CharTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        text.chars()
            .map(|c| self.lookup(c).ok_or(Error::UnknownChar(c)))
            .collect()
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, Error> {
        tokens
            .iter()
            .map(|&id| {
                self.id_to_char
                    .get(id as usize)
                    .copied()
                    .ok_or(Error::UnknownToken(id))
            })
            .collect()
    }

    fn vocab_size(&self) -> usize {
        self.id_to_char.len()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn vocab_size_matches_unique_chars() {
        let tok = CharTokenizer::from_text("hello");
        // unique: h, e, l, o
        assert_eq!(tok.vocab_size(), 4);
    }

    #[test]
    fn empty_corpus_yields_empty_vocab() {
        let tok = CharTokenizer::from_text("");
        assert_eq!(tok.vocab_size(), 0);
        assert_eq!(tok.encode("").unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn encode_decode_round_trip() {
        let tok = CharTokenizer::from_text("hello world");
        let ids = tok.encode("hello").unwrap();
        let text = tok.decode(&ids).unwrap();
        assert_eq!(text, "hello");
    }

    #[test]
    fn encode_unknown_char_errors() {
        let tok = CharTokenizer::from_text("abc");
        let err = tok.encode("ad").unwrap_err();
        assert!(matches!(err, Error::UnknownChar('d')));
    }

    #[test]
    fn decode_unknown_token_errors() {
        let tok = CharTokenizer::from_text("abc");
        let err = tok.decode(&[42]).unwrap_err();
        assert!(matches!(err, Error::UnknownToken(42)));
    }

    #[test]
    fn vocab_is_deterministic_regardless_of_input_order() {
        let a = CharTokenizer::from_text("cba");
        let b = CharTokenizer::from_text("abc");
        assert_eq!(a.encode("abc").unwrap(), b.encode("abc").unwrap());
    }

    #[test]
    fn handles_non_ascii_chars() {
        let tok = CharTokenizer::from_text("あいう");
        let ids = tok.encode("いあう").unwrap();
        assert_eq!(tok.decode(&ids).unwrap(), "いあう");
    }
}
