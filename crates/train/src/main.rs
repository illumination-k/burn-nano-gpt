//! End-to-end training demo: download TinyShakespeare, build a small
//! nano-GPT, and train it for a configurable number of optimizer steps.
//!
//! The defaults are deliberately small so the loop fits in a few minutes
//! on CPU — the point is to verify that loss actually goes down, not to
//! produce a competitive model.
//!
//! Knobs (all optional, via env var):
//! - `BURN_NANO_GPT_ITERS` — number of optimizer steps (default 1000)
//! - `BURN_NANO_GPT_LOG_EVERY` — log every N steps (default 100)
//! - `BURN_NANO_GPT_SEED` — RNG seed for batch sampling (default 0)
//! - `BURN_NANO_GPT_CACHE` — directory for the cached corpus (default `./.cache`)
//! - `BURN_NANO_GPT_PROMPT` — generation prompt (default `"Hello"`)
//! - `BURN_NANO_GPT_GENERATE` — number of tokens to generate (default 200)
//! - `BURN_NANO_GPT_LOSS_CSV` — if set, write per-step `step,loss` rows here

#![allow(clippy::print_stdout)]

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::optim::AdamWConfig;
use burn::tensor::{Int, Tensor, TensorData};
use dataset::TinyShakespeare;
use model::{NanoGpt, NanoGptConfig};
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokenizer::{CharTokenizer, Tokenizer};
use train::{sample_batch, train_step};

type Backend = Autodiff<NdArray<f32>>;
type InferenceBackend = <Backend as burn::tensor::backend::AutodiffBackend>::InnerBackend;

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    if let Err(err) = run() {
        tracing::error!(error = %err, "training failed");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let device = NdArrayDevice::default();

    let iters = env_usize("BURN_NANO_GPT_ITERS", 1000);
    let log_every = env_usize("BURN_NANO_GPT_LOG_EVERY", 100).max(1);
    let seed = env_u64("BURN_NANO_GPT_SEED", 0);
    let prompt_text = std::env::var("BURN_NANO_GPT_PROMPT").unwrap_or_else(|_| "Hello".to_string());
    let n_generate = env_usize("BURN_NANO_GPT_GENERATE", 200);
    let loss_csv = std::env::var_os("BURN_NANO_GPT_LOSS_CSV").map(PathBuf::from);

    // 1. Corpus + tokenizer.
    let cache_dir: PathBuf = std::env::var_os("BURN_NANO_GPT_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".cache"));
    let dataset = TinyShakespeare::download(&cache_dir)?;
    let tokenizer = CharTokenizer::from_text(dataset.text());
    let tokens = tokenizer.encode(dataset.text())?;
    tracing::info!(
        vocab_size = tokenizer.vocab_size(),
        n_tokens = tokens.len(),
        "corpus tokenized",
    );

    // 2. Model — kept small so each step is a few hundred ms on CPU.
    let block_size = 64;
    let batch_size = 16;
    let config = NanoGptConfig::new(
        tokenizer.vocab_size(),
        block_size,
        /* n_layer */ 2,
        /* n_head */ 2,
        /* d_model */ 64,
        /* d_ff */ 256,
    );
    let mut model = config.init::<Backend>(&device);

    // 3. Sample from the random-initialised model so the README can show a
    //    before/after comparison.
    let untrained_sample = greedy_generate(
        &model.valid(),
        &tokenizer,
        &prompt_text,
        n_generate,
        &device,
    )?;
    println!("\n--- before training (random init) ---");
    println!("prompt:    {prompt_text:?}");
    println!("generated: {untrained_sample:?}");

    // 4. Optimizer.
    let mut optimizer = AdamWConfig::new().init();
    let lr = 3e-4;

    // 5. Train loop. Each iteration samples a fresh random batch; nano-gpt
    //    treats "epoch" as a fuzzy concept and reports per-iteration loss.
    let mut rng = StdRng::seed_from_u64(seed);
    let start = Instant::now();

    let mut csv_writer = loss_csv
        .as_ref()
        .map(|path| -> Result<_, Box<dyn std::error::Error>> {
            let file = File::create(path)?;
            let mut w = BufWriter::new(file);
            writeln!(w, "step,loss")?;
            Ok(w)
        })
        .transpose()?;

    println!(
        "\ntraining: iters={iters}, batch={batch_size}, block={block_size}, lr={lr}, vocab={}",
        tokenizer.vocab_size(),
    );

    for step in 1..=iters {
        let (inputs, targets) =
            sample_batch::<Backend, _>(&tokens, batch_size, block_size, &mut rng, &device);
        let (next_model, loss) = train_step(model, &mut optimizer, inputs, targets, lr);
        model = next_model;

        if let Some(w) = csv_writer.as_mut() {
            writeln!(w, "{step},{loss}")?;
        }

        if step == 1 || step % log_every == 0 || step == iters {
            let elapsed = start.elapsed().as_secs_f32();
            tracing::info!(step, iters, loss, elapsed_s = elapsed, "train step");
            println!("step {step:>5}/{iters}  loss={loss:.4}  elapsed={elapsed:6.1}s");
        }
    }

    if let Some(mut w) = csv_writer {
        w.flush()?;
    }

    // 6. Say hello — greedy decode from the prompt on the inference
    //    (non-autodiff) view of the trained weights.
    let trained_sample = greedy_generate(
        &model.valid(),
        &tokenizer,
        &prompt_text,
        n_generate,
        &device,
    )?;
    println!("\n--- after training ---");
    println!("prompt:    {prompt_text:?}");
    println!("generated: {trained_sample:?}");

    Ok(())
}

fn greedy_generate(
    model: &NanoGpt<InferenceBackend>,
    tokenizer: &CharTokenizer,
    prompt: &str,
    n_new_tokens: usize,
    device: &NdArrayDevice,
) -> Result<String, Box<dyn std::error::Error>> {
    let prompt_ids: Vec<i32> = tokenizer
        .encode(prompt)?
        .into_iter()
        .map(|id| id as i32)
        .collect();
    let prompt_len = prompt_ids.len();
    let prompt_tensor = Tensor::<InferenceBackend, 2, Int>::from_data(
        TensorData::new(prompt_ids, [1, prompt_len]),
        device,
    );

    let full = model.generate(prompt_tensor, n_new_tokens);
    let ids: Vec<i64> = full.into_data().convert::<i64>().to_vec()?;
    let ids: Vec<u32> = ids.into_iter().map(|i| i as u32).collect();
    Ok(tokenizer.decode(&ids)?)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
