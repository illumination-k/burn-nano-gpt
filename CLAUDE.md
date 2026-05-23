# AGENTS Guideline

This repository is pre-alpha and under active development. The API is not stable and may change without a major version bump, so backwards compatibility is not guaranteed at this stage.
So developers of this repository DO NOT need to worry about breaking changes or maintaining backwards compatibility. We prefer to iterate quickly and make breaking changes as needed, rather than trying to maintain backwards compatibility.

## Policy

Follow the YANGI, SOLID, DRY, and KISS principles in all code and documentation. Prioritize simplicity, readability, and maintainability over cleverness or optimization. Avoid premature optimization and over-engineering. Strive for clear and concise code that is easy to understand and modify.

## Development Process

Run `mise install` first to install the toolchain and project tools.

At the end of a session, run `mise run ci` and make sure it passes. Use the narrower tasks while iterating:

```bash
mise run fmt      # Format
mise run lint     # Lint and policy checks
mise run test     # Tests
mise run ci       # Full required verification
```

## Commands

Run `mise install` first to install all tools.

```bash
mise run ci    # Run all ci:* tasks
mise run fmt   # Run all fmt:* tasks
mise run lint  # Run all lint:* tasks
mise run test  # Run all test:* tasks
```

## Tools

All tools are managed by mise. Run `mise install` to install them.

| Tool           | Purpose                                 |
| -------------- | --------------------------------------- |
| uv             | Python package manager                  |
| dprint         | Code formatter                          |
| prek           | Pre-commit hook runner                  |
| shfmt          | Shell script formatter                  |
| actionlint     | GitHub Actions linter                   |
| zizmor         | GitHub Actions security linter          |
| shellcheck     | Shell script linter                     |
| ghalint        | GitHub Actions linter                   |
| pinact         | Pin GitHub Actions versions to SHAs     |
| rust           | Rust toolchain                          |
| cargo-binstall |                                         |
| cargo-nextest  | Fast Rust test runner                   |
| cargo-deny     | Dependency license and advisory checker |
| cargo-audit    | Security advisory checker for Rust      |
| cargo-mutants  |                                         |
| cargo-llvm-cov |                                         |

# CLAUDE.md

## Purpose

Rust製の深層学習フレームワーク [burn](https://burn.dev/) を使って nano-gpt
（小さなTransformer言語モデル）を学習する、学習・動作確認用リポジトリ。
burnのAPIとTransformer学習の仕組みを手を動かして理解することが目的で、
プロダクションでの大規模学習は目的としない。

## Architecture

- `src/model/` — Transformerブロック・マルチヘッドAttention・nano-gptモデル定義
- `src/data/` — TinyShakespeareデータセットのダウンロード・バッチ生成
- `src/tokenizer/` — `Tokenizer` trait と実装（char-level / BPE 切り替え可能）
- `src/train/` — 学習ループ、optimizer、loss、checkpoint保存
- `src/sample.rs` — 学習済みモデルからのテキスト生成

### Backend

`wgpu` backend を中心に据える（Apple Silicon / cross-platformで動かしやすいため）。
将来 backend を増やしたくなったら feature flag で `ndarray` / `candle` を切り替えられるようにする。

### Tokenizer

`Tokenizer` trait で `encode` / `decode` / `vocab_size` を抽象化し、以下を実装する：

- **char-level** — TinyShakespeareの全文字を集めて構築するシンプルな実装（nano-gpt原典に近い）
- **BPE** — GPT-2互換BPE（`tiktoken-rs` か `tokenizers` crateを利用）

学習時は config で切り替え、checkpoint と一緒に tokenizer の種類・語彙を保存する。

## Coding guidelines

- 読みやすさ重視。nano-gptのリファレンス実装として、コードを読んで仕組みを学べることを優先する
- マジックナンバーは `Config` 構造体に集約し、学習スクリプトから注入する
- 過度な抽象化は避け、まずは1つのモデル・1つのデータセットで動くシンプルな構成にする
