use burn::config::Config;
use burn::tensor::backend::Backend;

use crate::block::BlockConfig;
use crate::model::NanoGpt;

/// Hyperparameters for [`NanoGpt`].
///
/// All magic numbers live here so the training script can inject them and
/// the model code stays free of literal constants.
#[derive(Config, Debug)]
pub struct NanoGptConfig {
    /// Number of distinct token ids the model accepts.
    pub vocab_size: usize,
    /// Maximum context length (in tokens) the positional embedding covers.
    pub block_size: usize,
    /// Number of stacked Transformer blocks.
    pub n_layer: usize,
    /// Number of attention heads per block. `d_model` must be divisible by `n_head`.
    pub n_head: usize,
    /// Width of token / position embeddings and of every block's residual stream.
    pub d_model: usize,
    /// Inner width of the position-wise feed-forward network. Typically `4 * d_model`.
    pub d_ff: usize,
    /// Dropout probability applied inside attention and FFN.
    #[config(default = 0.0)]
    pub dropout: f64,
}

impl NanoGptConfig {
    /// Build a randomly-initialised model on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> NanoGpt<B> {
        assert!(
            self.d_model.is_multiple_of(self.n_head),
            "d_model ({}) must be divisible by n_head ({})",
            self.d_model,
            self.n_head,
        );

        NanoGpt::new(self, device)
    }

    pub(crate) fn block_config(&self) -> BlockConfig {
        BlockConfig::new(self.d_model, self.n_head, self.d_ff).with_dropout(self.dropout)
    }
}
