//! End-to-end check: build a tokenizer from a `TinyShakespeare` corpus and
//! confirm that every dataset item round-trips through encode/decode.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use burn_dataset::Dataset;
use dataset::TinyShakespeare;
use tokenizer::{CharTokenizer, Tokenizer};

#[test]
fn tokenizer_round_trips_every_line_in_dataset() {
    let raw =
        "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.";
    let ds = TinyShakespeare::from_text(raw.to_string());
    let tok = CharTokenizer::from_text(ds.text());

    assert!(tok.vocab_size() > 0);

    for i in 0..ds.len() {
        let line = ds.get(i).unwrap();
        let ids = tok.encode(&line).unwrap();
        let back = tok.decode(&ids).unwrap();
        assert_eq!(back, line, "round-trip mismatch on line {i}");
    }
}

#[test]
fn tokenizer_vocab_covers_full_corpus() {
    let raw = "abc\ndef\nghi";
    let ds = TinyShakespeare::from_text(raw.to_string());
    let tok = CharTokenizer::from_text(ds.text());

    // Encoding the entire corpus must succeed — no UnknownChar — and decode
    // back to the original bytes-for-bytes.
    let ids = tok.encode(ds.text()).unwrap();
    assert_eq!(tok.decode(&ids).unwrap(), raw);
}
