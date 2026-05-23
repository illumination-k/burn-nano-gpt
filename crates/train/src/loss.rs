use burn::nn::loss::CrossEntropyLossConfig;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

/// Next-token cross-entropy loss for an autoregressive language model.
///
/// The model emits logits of shape `[batch, seq, vocab]` and the batcher
/// supplies targets of shape `[batch, seq]`. `CrossEntropyLoss` operates
/// on `[N, vocab]` / `[N]`, so we flatten the batch and sequence axes
/// together and let it average over every token position.
pub fn language_model_loss<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 1> {
    let [batch, seq, vocab] = logits.dims();
    let device = logits.device();

    let logits = logits.reshape([batch * seq, vocab]);
    let targets = targets.reshape([batch * seq]);

    CrossEntropyLossConfig::new()
        .init(&device)
        .forward(logits, targets)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::tensor::{ElementConversion, TensorData};

    use super::*;

    type B = NdArray<f32>;

    #[test]
    fn perfect_prediction_yields_near_zero_loss() {
        // One batch, two timesteps, two classes. Logits put nearly all mass on
        // the correct class, so cross-entropy should be ~0.
        let device = NdArrayDevice::default();
        let logits = Tensor::<B, 3>::from_data(
            TensorData::new(vec![10.0_f32, -10.0, -10.0, 10.0], [1, 2, 2]),
            &device,
        );
        let targets =
            Tensor::<B, 2, Int>::from_data(TensorData::new(vec![0_i32, 1], [1, 2]), &device);

        let loss: f32 = language_model_loss(logits, targets).into_scalar().elem();
        assert!(loss < 1e-3, "expected near-zero loss, got {loss}");
    }
}
