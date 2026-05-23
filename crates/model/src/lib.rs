//! nano-GPT model in burn.
//!
//! The model is a decoder-only Transformer language model whose sole purpose
//! is to be readable — every block is built from primitive `burn::nn`
//! modules rather than the pre-assembled `nn::transformer::*`, so the code
//! traces 1:1 with the GPT papers and Karpathy's nano-gpt reference.
//!
//! See [`NanoGptConfig`] for the hyperparameters and [`NanoGpt::forward`]
//! for the inference path.

mod attention;
mod block;
mod config;
mod model;

pub use attention::{MultiHeadSelfAttention, MultiHeadSelfAttentionConfig};
pub use block::{Block, BlockConfig};
pub use config::NanoGptConfig;
pub use model::NanoGpt;
