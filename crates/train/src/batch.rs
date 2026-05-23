use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use rand::Rng;

/// Sample a training mini-batch from a pre-tokenized corpus.
///
/// Each row of `inputs` is a contiguous window of `block_size` tokens
/// drawn from a uniformly random start position. The matching row of
/// `targets` is the same window shifted by one token — exactly the
/// next-token prediction setup of nano-gpt.
///
/// Both tensors have shape `[batch_size, block_size]`.
///
/// # Panics
///
/// Panics if `tokens.len() <= block_size`, since no shifted-by-one window
/// would fit.
pub fn sample_batch<B: Backend, R: Rng>(
    tokens: &[u32],
    batch_size: usize,
    block_size: usize,
    rng: &mut R,
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
    assert!(
        tokens.len() > block_size,
        "corpus must contain more than block_size ({block_size}) tokens, got {}",
        tokens.len(),
    );

    // Inclusive upper bound for the start index — the last token of the
    // target window is at `start + block_size`, which must be in range.
    let max_start = tokens.len() - block_size - 1;

    let total = batch_size * block_size;
    let mut input_ids: Vec<i32> = Vec::with_capacity(total);
    let mut target_ids: Vec<i32> = Vec::with_capacity(total);

    for _ in 0..batch_size {
        let start = rng.random_range(0..=max_start);
        for j in 0..block_size {
            input_ids.push(tokens[start + j] as i32);
            target_ids.push(tokens[start + j + 1] as i32);
        }
    }

    let shape = [batch_size, block_size];
    let inputs = Tensor::<B, 2, Int>::from_data(TensorData::new(input_ids, shape), device);
    let targets = Tensor::<B, 2, Int>::from_data(TensorData::new(target_ids, shape), device);
    (inputs, targets)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;

    type B = NdArray<f32>;

    #[test]
    fn batch_shapes_match_request() {
        let tokens: Vec<u32> = (0..64).collect();
        let mut rng = StdRng::seed_from_u64(0);
        let (x, y) = sample_batch::<B, _>(&tokens, 4, 8, &mut rng, &NdArrayDevice::default());
        assert_eq!(x.dims(), [4, 8]);
        assert_eq!(y.dims(), [4, 8]);
    }

    #[test]
    fn targets_are_inputs_shifted_by_one() {
        // Deterministic corpus + seeded rng so we can verify the shift.
        let tokens: Vec<u32> = (0..32).collect();
        let mut rng = StdRng::seed_from_u64(42);
        let (x, y) = sample_batch::<B, _>(&tokens, 2, 4, &mut rng, &NdArrayDevice::default());

        let x_data: Vec<i64> = x.to_data().convert::<i64>().to_vec().unwrap();
        let y_data: Vec<i64> = y.to_data().convert::<i64>().to_vec().unwrap();
        for (xi, yi) in x_data.iter().zip(y_data.iter()) {
            assert_eq!(*yi, *xi + 1);
        }
    }

    #[test]
    #[should_panic(expected = "corpus must contain more than block_size")]
    fn rejects_short_corpus() {
        let tokens: Vec<u32> = vec![0, 1, 2, 3];
        let mut rng = StdRng::seed_from_u64(0);
        let _ = sample_batch::<B, _>(&tokens, 1, 4, &mut rng, &NdArrayDevice::default());
    }
}
