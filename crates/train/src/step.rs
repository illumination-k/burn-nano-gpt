use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Int, Tensor};
use model::NanoGpt;

use crate::loss::language_model_loss;

/// Run a single training step: forward → loss → backward → optimizer step.
///
/// Returns the updated model and the scalar loss value (extracted before
/// `backward` consumes the loss tensor).
pub fn train_step<B, O>(
    model: NanoGpt<B>,
    optimizer: &mut O,
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    lr: f64,
) -> (NanoGpt<B>, f32)
where
    B: AutodiffBackend,
    O: Optimizer<NanoGpt<B>, B>,
{
    let logits = model.forward(inputs);
    let loss = language_model_loss(logits, targets);

    let loss_value: f32 = loss.clone().into_scalar().elem();

    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    let model = optimizer.step(lr, model, grads);

    (model, loss_value)
}
