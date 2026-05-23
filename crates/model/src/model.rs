use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::block::Block;
use crate::config::NanoGptConfig;

/// nano-GPT: token + position embeddings, a stack of [`Block`]s, a final
/// LayerNorm, and a linear projection to vocab logits.
#[derive(Module, Debug)]
pub struct NanoGpt<B: Backend> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    blocks: Vec<Block<B>>,
    ln_final: LayerNorm<B>,
    /// Untied LM head — kept separate from `token_embedding` for clarity;
    /// weight tying can be added later as an optimisation.
    head: Linear<B>,
    block_size: usize,
}

impl<B: Backend> NanoGpt<B> {
    pub(crate) fn new(config: &NanoGptConfig, device: &B::Device) -> Self {
        let blocks = (0..config.n_layer)
            .map(|_| config.block_config().init(device))
            .collect();

        Self {
            token_embedding: EmbeddingConfig::new(config.vocab_size, config.d_model).init(device),
            position_embedding: EmbeddingConfig::new(config.block_size, config.d_model)
                .init(device),
            blocks,
            ln_final: LayerNormConfig::new(config.d_model).init(device),
            head: LinearConfig::new(config.d_model, config.vocab_size).init(device),
            block_size: config.block_size,
        }
    }

    /// Run a forward pass.
    ///
    /// - `tokens`: `[batch, seq]` token ids. `seq` must be `<= block_size`.
    /// - returns logits of shape `[batch, seq, vocab_size]`.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq] = tokens.dims();
        assert!(
            seq <= self.block_size,
            "sequence length {} exceeds block_size {}",
            seq,
            self.block_size,
        );
        let device = tokens.device();

        let tok = self.token_embedding.forward(tokens);

        // Position ids 0..seq, broadcast across the batch by reshape+expand.
        let positions = Tensor::<B, 1, Int>::arange(0..seq as i64, &device)
            .reshape([1, seq])
            .expand([batch, seq]);
        let pos = self.position_embedding.forward(positions);

        let mut x = tok + pos;
        for block in &self.blocks {
            x = block.forward(x);
        }
        let x = self.ln_final.forward(x);
        self.head.forward(x)
    }

    /// Greedy autoregressive decode: starting from `prompt`
    /// (`[batch, seq]`), repeatedly take the argmax over the final
    /// position's logits and append it. Returns the full
    /// `[batch, seq + n_new_tokens]` token tensor.
    ///
    /// When the running context exceeds `block_size`, the oldest tokens
    /// are dropped — the positional embedding only covers a `block_size`
    /// window.
    pub fn generate(&self, prompt: Tensor<B, 2, Int>, n_new_tokens: usize) -> Tensor<B, 2, Int> {
        let mut context = prompt;
        for _ in 0..n_new_tokens {
            let [batch, seq] = context.dims();
            let cropped = if seq > self.block_size {
                context
                    .clone()
                    .slice([0..batch, seq - self.block_size..seq])
            } else {
                context.clone()
            };
            let logits = self.forward(cropped);
            let [_, t, _] = logits.dims();

            // Last position's logits: [batch, vocab]. argmax along vocab → [batch, 1].
            let last = logits.slice([0..batch, t - 1..t]).squeeze_dim::<2>(1);
            let next = last.argmax(1);

            context = Tensor::cat(vec![context, next], 1);
        }
        context
    }
}
