//! Datasets for nano-gpt training.
//!
//! Currently supports the TinyShakespeare corpus used by the nano-gpt reference
//! implementation. Each item exposed via [`burn_dataset::Dataset`] is one
//! newline-delimited line of the source text; the full corpus is also
//! accessible via [`TinyShakespeare::text`] for downstream tokenization /
//! windowing.

mod error;
mod tiny_shakespeare;

pub use error::Error;
pub use tiny_shakespeare::{TINY_SHAKESPEARE_URL, TinyShakespeare};
