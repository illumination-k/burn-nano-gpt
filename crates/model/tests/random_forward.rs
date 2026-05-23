//! Sanity check: a randomly-initialised `NanoGpt` produces logits of the
//! expected shape and finite values for a single forward pass.
//!
//! Uses the `NdArray` backend so the test is CPU-only and deterministic —
//! the `wgpu` backend is the production target but isn't a good fit for
//! unit tests (requires a GPU adapter at runtime).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::tensor::{Int, Tensor};

use model::NanoGptConfig;

type Backend = NdArray<f32>;

#[test]
fn random_init_forward_pass_yields_finite_logits() {
    let device = NdArrayDevice::default();

    let vocab_size = 32;
    let block_size = 16;
    let n_layer = 2;
    let n_head = 2;
    let d_model = 16;
    let d_ff = 32;
    let config = NanoGptConfig::new(vocab_size, block_size, n_layer, n_head, d_model, d_ff);
    let model = config.init::<Backend>(&device);

    let batch = 1;
    let seq = 8;
    let tokens = Tensor::<Backend, 1, Int>::arange(0..seq as i64, &device).reshape([batch, seq]);

    let logits = model.forward(tokens);

    assert_eq!(logits.dims(), [batch, seq, vocab_size]);

    let values: Vec<f32> = logits.to_data().to_vec().unwrap();
    assert!(
        values.iter().all(|v| v.is_finite()),
        "logits contained non-finite values",
    );
}
