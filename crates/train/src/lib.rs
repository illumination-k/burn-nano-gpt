//! Training utilities for nano-GPT.
//!
//! The crate exposes three small primitives that the training binary
//! composes together:
//!
//! - [`sample_batch`] — pick random `[batch, block_size]` windows from a
//!   token stream, paired with their shifted-by-one targets.
//! - [`language_model_loss`] — flatten `[batch, seq, vocab]` logits and
//!   `[batch, seq]` targets into a single cross-entropy call.
//! - [`train_step`] — one forward / backward / optimizer step.

mod batch;
mod loss;
mod step;

pub use batch::sample_batch;
pub use loss::language_model_loss;
pub use step::train_step;
