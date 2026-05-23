use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor};

/// Configuration for [`MultiHeadSelfAttention`].
#[derive(Config, Debug)]
pub struct MultiHeadSelfAttentionConfig {
    /// Residual-stream width. Must be divisible by `n_head`.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_head: usize,
    #[config(default = 0.0)]
    pub dropout: f64,
}

/// Causal multi-head self-attention.
///
/// Projects the input into per-head queries, keys, and values, applies
/// scaled dot-product attention with a triangular mask so each position
/// only attends to itself and earlier positions, and projects the
/// concatenated heads back to the residual-stream width.
#[derive(Module, Debug)]
pub struct MultiHeadSelfAttention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    /// Projects the concatenated per-head outputs back to `d_model`.
    proj: Linear<B>,
    dropout: Dropout,
    n_head: usize,
    head_dim: usize,
}

impl MultiHeadSelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadSelfAttention<B> {
        assert!(
            self.d_model.is_multiple_of(self.n_head),
            "d_model ({}) must be divisible by n_head ({})",
            self.d_model,
            self.n_head,
        );
        let head_dim = self.d_model / self.n_head;
        let linear = || LinearConfig::new(self.d_model, self.d_model).init(device);

        MultiHeadSelfAttention {
            query: linear(),
            key: linear(),
            value: linear(),
            proj: linear(),
            dropout: DropoutConfig::new(self.dropout).init(),
            n_head: self.n_head,
            head_dim,
        }
    }
}

impl<B: Backend> MultiHeadSelfAttention<B> {
    /// `x: [batch, seq, d_model] -> [batch, seq, d_model]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, d_model] = x.dims();
        let device = x.device();

        let q = split_heads(
            self.query.forward(x.clone()),
            batch,
            seq,
            self.n_head,
            self.head_dim,
        );
        let k = split_heads(
            self.key.forward(x.clone()),
            batch,
            seq,
            self.n_head,
            self.head_dim,
        );
        let v = split_heads(
            self.value.forward(x),
            batch,
            seq,
            self.n_head,
            self.head_dim,
        );

        // Scaled dot-product: [B, nh, T, hd] @ [B, nh, hd, T] -> [B, nh, T, T].
        let scale = (self.head_dim as f32).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)).div_scalar(scale);

        // Causal mask: forbid attending to future positions.
        let mask = causal_mask::<B>(seq, &device);
        let scores = scores.mask_fill(mask, f32::NEG_INFINITY);

        let weights = softmax(scores, 3);
        let weights = self.dropout.forward(weights);

        // [B, nh, T, T] @ [B, nh, T, hd] -> [B, nh, T, hd] -> [B, T, C].
        let out = weights.matmul(v);
        let out = merge_heads(out, batch, seq, d_model);

        self.proj.forward(out)
    }
}

/// `[B, T, C] -> [B, n_head, T, head_dim]`.
fn split_heads<B: Backend>(
    x: Tensor<B, 3>,
    batch: usize,
    seq: usize,
    n_head: usize,
    head_dim: usize,
) -> Tensor<B, 4> {
    x.reshape([batch, seq, n_head, head_dim]).swap_dims(1, 2)
}

/// `[B, n_head, T, head_dim] -> [B, T, d_model]`.
fn merge_heads<B: Backend>(
    x: Tensor<B, 4>,
    batch: usize,
    seq: usize,
    d_model: usize,
) -> Tensor<B, 3> {
    x.swap_dims(1, 2).reshape([batch, seq, d_model])
}

/// `[1, 1, seq, seq]` boolean mask that is `true` on positions a query is
/// forbidden from attending to (the strict upper triangle = future tokens).
/// Broadcasts naturally over the batch and head dims of the score tensor.
fn causal_mask<B: Backend>(seq: usize, device: &B::Device) -> Tensor<B, 4, Bool> {
    let mask: Tensor<B, 2> = Tensor::ones([seq, seq], device).triu(1);
    mask.greater_elem(0.5).unsqueeze::<4>()
}
