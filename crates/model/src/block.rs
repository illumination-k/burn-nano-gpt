use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use crate::attention::{MultiHeadSelfAttention, MultiHeadSelfAttentionConfig};

/// Configuration for [`Block`].
#[derive(Config, Debug)]
pub struct BlockConfig {
    pub d_model: usize,
    pub n_head: usize,
    /// Inner width of the feed-forward network.
    pub d_ff: usize,
    #[config(default = 0.0)]
    pub dropout: f64,
}

/// A single Transformer decoder block: pre-norm + causal self-attention,
/// then pre-norm + position-wise feed-forward, each wrapped in a residual.
#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    ln_attn: LayerNorm<B>,
    attn: MultiHeadSelfAttention<B>,
    ln_mlp: LayerNorm<B>,
    mlp: FeedForward<B>,
}

impl BlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Block<B> {
        let attn_cfg =
            MultiHeadSelfAttentionConfig::new(self.d_model, self.n_head).with_dropout(self.dropout);
        let mlp_cfg = FeedForwardConfig::new(self.d_model, self.d_ff).with_dropout(self.dropout);

        Block {
            ln_attn: LayerNormConfig::new(self.d_model).init(device),
            attn: attn_cfg.init(device),
            ln_mlp: LayerNormConfig::new(self.d_model).init(device),
            mlp: mlp_cfg.init(device),
        }
    }
}

impl<B: Backend> Block<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.ln_attn.forward(x));
        x.clone() + self.mlp.forward(self.ln_mlp.forward(x))
    }
}

#[derive(Config, Debug)]
struct FeedForwardConfig {
    d_model: usize,
    d_ff: usize,
    #[config(default = 0.0)]
    dropout: f64,
}

/// Two-layer position-wise feed-forward network with GELU activation.
#[derive(Module, Debug)]
struct FeedForward<B: Backend> {
    fc1: Linear<B>,
    gelu: Gelu,
    fc2: Linear<B>,
    dropout: Dropout,
}

impl FeedForwardConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        FeedForward {
            fc1: LinearConfig::new(self.d_model, self.d_ff).init(device),
            gelu: Gelu::new(),
            fc2: LinearConfig::new(self.d_ff, self.d_model).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> FeedForward<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);
        let x = self.gelu.forward(x);
        let x = self.fc2.forward(x);
        self.dropout.forward(x)
    }
}
