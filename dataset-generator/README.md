# Schema‑Conditioned DSL → JSON AST (Gemma 3 270M student)

This repository is a **complete, teacher-in-the-loop pipeline** to build a small model that can:

- Read a **JSON Schema** describing a DSL
- Read a natural language **query** (EN/FR)
- Output **one JSON object** (an AST) that **validates against the schema** and matches the query intent, or a refusal with a reason + suggestion when the schema cannot satisfy the request

The teacher is a local **Ollama** model (set via `--teacher-model`, config, or `LLM_MODEL`). The student is **Gemma 3 270M** fine-tuned with **Unsloth LoRA**. After SFT, the repo runs a **teacher‑aligned preference tuning step (DPO)** using teacher verdicts.

## What you run (high level)

There are four stages:

1) **Dataset generation (teacher)** → JSONL artifacts under `outputs/d_0*_*.jsonl`  
2) **SFT training (student)** → LoRA checkpoints + `training_metrics.jsonl`  
3) **Teacher evaluation** → per-sample + summary evaluation JSON files  
4) **Teacher-aligned DPO** → a second LoRA checkpoint aligned to teacher preferences

If you only run one command, use `runAll.sh` (see Quickstart).

---

## Requirements

- **Python 3.12+** and **uv** (this repo uses `uv sync` + `uv run`)
- **CUDA GPU** strongly recommended for training (`runTraining.sh` sets CUDA env vars used in this environment)
- **Ollama** installed locally and a teacher model pulled (default: `gpt-oss:120b`)

### Teacher model configuration

The teacher model id comes from (in practice):
- CLI: `--teacher-model`
- Config: `dataset_generation.teacher_model` (defaults to `gpt-oss:120b`)
- Environment: `LLM_MODEL` (used as a fallback/default for scripts that accept `--teacher-model`)

---

## Quickstart (recommended)

```bash
# 0) Install deps
uv sync

# 1) Run everything: dataset -> SFT -> eval -> DPO alignment
./runAll.sh --with-training --with-evals --with-alignment
```

Notes:
- `--with-alignment` requires the **Ollama teacher** to be available (alignment is built from teacher verdicts).
- `--teacher-model` sets the teacher used for domain specs, dataset generation, evals, and alignment (required for those stages).
- Defaults come from `config/defaults.json` unless overridden with CLI flags.
- `runAll.sh` auto-resumes SFT and DPO training from the latest `checkpoint-*` it finds in their output dirs (remove checkpoints to force a fresh run).

---

## Configuration (`config/defaults.json`)

The repository is driven by `config/defaults.json`:

- `dataset_generation`: sizes/ratios and output file paths for schema/data creation
- `training`: SFT hyperparameters and the student run directory
- `alignment`: DPO hyperparameters and file paths for preference pairs

You can either:
- Edit `config/defaults.json`, or
- Pass overrides on the command line to the scripts/runners.

---

## Pipeline outputs (what files to look at)

### Dataset generation artifacts

All are JSONL:

- `outputs/d_01_domain_specs.jsonl`: teacher-generated domain field/value specs
- `outputs/d_02_final_schemas.jsonl`: final JSON Schemas (one per domain/schema_id)
- `outputs/d_03_dataset.jsonl`: (schema, query, ast_json, is_valid, …)
- `outputs/d_04_training_corpus.jsonl`: training corpus with `prompt` + `completion`

Prettified copies are written to `outputs/pretty/*.json` for inspection.

### SFT training artifacts

Directory: `outputs/student_runs/gemma3-270m/` (default)

- `checkpoint-*`: periodic LoRA checkpoints
- `checkpoint-final/`: final LoRA adapter + tokenizer (when training completes)
- `training_metrics.jsonl`: streaming training/eval metrics (JSON per line)
- `run_config.json`: resolved training config for reproducibility

### Teacher evaluation artifacts

Directory: `outputs/student_runs/eval/` (default)

- `evaluation_results.jsonl`: per-sample eval records (student output + schema checks + teacher signals)
- `evaluation_summary.json`: aggregate rates (parsed/schema valid/teacher accept/etc)

### DPO alignment artifacts

Directory: `outputs/student_runs/gemma3-270m-dpo/` (default)

- `checkpoint-final/`: aligned LoRA adapter + tokenizer
- Pairs file: `outputs/student_runs/alignment/dpo_pairs.jsonl` (default)

---

## How evaluation works (important)

There are **two different “evaluation” concepts** in this repo:

1) **Trainer `eval_loss` during SFT** (teacher-free)  
   - Runs every `eval_steps` on a held-out split of `outputs/d_04_training_corpus.jsonl`.  
   - Measures **cross-entropy against the stored target completions** (AST JSON).  
   - Useful for early overfitting signals, but it does *not* tell you if the AST matches the query semantically.

2) **Teacher evaluation (`scripts/evaluate_student.py`)** (teacher-in-the-loop)  
   - Actually runs generation, parses JSON, validates schema, and asks the teacher to judge semantics.  
   - Produces decision‑grade metrics like `teacher_accept_rate` and `semantic_pass_rate`.

Alignment (DPO) uses the **teacher evaluation output**.

---

## Reading metrics (with examples)

### 1) SFT training metrics (`training_metrics.jsonl`)

File: `outputs/student_runs/<run>/training_metrics.jsonl`

Two common log shapes:

**Train step log**
```json
{"step": 60, "epoch": 0.25, "loss": 0.159, "grad_norm": 1.399, "learning_rate": 4.999885e-05}
```

- `step`: optimizer step (after gradient accumulation)
- `epoch`: fractional epoch progress (e.g. `0.25` = 25% through epoch 0→1)
- `loss`: average training cross-entropy on supervised tokens over the last logging window
- `grad_norm`: global gradient norm (sanity signal: spikes often mean a bad batch or instability)
- `learning_rate`: current LR from the scheduler

**Eval step log**
```json
{"step": 50, "epoch": 0.21, "eval_loss": 0.17636, "eval_runtime": 68.847, "eval_samples_per_second": 2.905, "eval_steps_per_second": 0.726}
```

- `eval_loss`: cross-entropy on the held-out set (teacher-free; targets are the stored completions)
- `eval_runtime`: seconds spent evaluating
- `eval_samples_per_second` / `eval_steps_per_second`: throughput metrics for that eval pass

How to interpret:
- If `loss` keeps going down but `eval_loss` starts going up → likely overfitting.
- If both are `NaN` or `eval_loss` becomes `NaN` → your label masking is broken (this repo now sanity-checks and filters empty-supervision eval batches).

### 2) Teacher evaluation metrics (`evaluation_summary.json`)

File: `outputs/student_runs/eval/evaluation_summary.json`

Key fields:
- `parsed_rate`: % outputs that were valid JSON objects
- `schema_valid_rate`: % outputs that pass JSON Schema validation
- `exact_match_rate`: % outputs exactly match the stored target AST (canonical JSON string)
- `teacher_accept_rate`: % outputs the teacher judged as semantically correct
- `semantic_pass_rate`: % outputs that are both schema-valid **and** teacher-accepted (the best “did it work?” number)

Teacher calls are retried on parse errors according to `evaluation.teacher_retries` and `evaluation.teacher_retry_wait` in `config/defaults.json` (`-1` = retry indefinitely to keep progressing up to `--max-samples`).

### 3) DPO alignment metrics

During DPO training (`scripts/train_alignment_dpo.py`), TRL logs step metrics like:
```json
{"loss": 0.0, "grad_norm": 0.00011142194125568494, "learning_rate": 4.425252525252526e-06, "rewards/chosen": 12.19339656829834, "rewards/rejected": -2.8378841876983643, "rewards/accuracies": 1.0, "rewards/margins": 15.031280517578125, "logps/chosen": -153.79283142089844, "logps/rejected": -45.93531036376953, "logits/chosen": -2.3316988945007324, "logits/rejected": -2.4716598987579346, "epoch": 206}
```

- `loss`: DPO objective (log-sigmoid of the reward gap); trends toward 0 when the model already prefers the teacher pick by a wide margin.
- `grad_norm` / `learning_rate`: sanity checks for stability and schedule.
- `rewards/chosen` and `rewards/rejected`: DPO “rewards” = beta × (policy log p − reference log p) for each completion; higher `chosen` and lower (often negative) `rejected` is good.
- `rewards/accuracies`: fraction of pairs where `rewards/chosen` > `rewards/rejected` (want → 1).
- `rewards/margins`: average reward gap (`chosen − rejected`); larger positive margins mean clearer preferences and usually faster convergence.
- `logps/*`: total log-likelihood of the chosen/rejected completions under the current policy (without the reference adapter); should get less negative as the policy fits the data.
- `logits/*`: mean raw logits gathered for the chosen/rejected tokens; mainly a numerical sanity signal (watch for inf/NaN).
- `epoch`: fractional pass through the pairs dataset.

If you see `loss` saturating at ~0 with `rewards/accuracies` near 1.0, the model is confidently preferring the teacher picks; if `loss` stalls high or `rewards/margins` stay near 0, the model is not separating chosen vs rejected. The *real* way to judge alignment, though, is to:

1) Re-run teacher evaluation on the aligned adapter
2) Compare `teacher_accept_rate` and `semantic_pass_rate` against the SFT adapter

---

## Runners (recommended interface)

All runners:
- run `uv sync`
- then call the underlying Python scripts

### `runDatasetGeneration.sh`

Generates `outputs/d_01` → `outputs/d_05` (and `outputs/pretty/*`).

```bash
./runDatasetGeneration.sh --config config/defaults.json --teacher-model gpt-oss:120b
```

Flags:
- `--config`: path to config JSON
- `--teacher-model`: Ollama model id to use for domain specs and dataset generation
Resumes automatically if a dataset checkpoint file from a prior run is present.

### Fast dev config (tiny end-to-end runs)

Use `config/defaults.dev.json` for a ~30-line dataset and small training loop. Examples:

- All stages: `./runAll.sh --config config/defaults.dev.json --with-training --with-evals --with-alignment`
- Dataset only: `./runDatasetGeneration.sh --config config/defaults.dev.json --teacher-model gpt-oss:20b`
- Training: `./runTraining.sh --config config/defaults.dev.json`
- Evals: `./runEvals.sh --config config/defaults.dev.json --teacher-model gpt-oss:20b`
- Alignment: `./runAlignment.sh --config config/defaults.dev.json --teacher-model gpt-oss:20b`

### `runTraining.sh`

Runs SFT training (`scripts/train_student_unsloth.py`).

```bash
./runTraining.sh --config config/defaults.json
```

Any extra args are forwarded to `scripts/train_student_unsloth.py`.

### `runEvals.sh`

Runs evaluation (`scripts/evaluate_student.py`) and then builds a Markdown+plots report (`scripts/report_training.py`).

```bash
./runEvals.sh --adapter outputs/student_runs/gemma3-270m/checkpoint-final --teacher-model gpt-oss:120b
```

If an aligned checkpoint exists at `alignment.output_dir/checkpoint-final` (defaults to `outputs/student_runs/gemma3-270m-dpo/checkpoint-final`), `runEvals.sh` will also run a second evaluation for it and emit an `alignment_report.md` alongside the standard student report in the reporting directory (defaults to `outputs/reports/gemma-3-270m-it`).

Resumes evaluation automatically when a checkpoint JSON sits next to the evaluation results path.

### `runAlignment.sh` (teacher-required)

Runs the full alignment stage:

1) `scripts/evaluate_student.py` (**must use the Ollama teacher**)  
2) `scripts/build_alignment_pairs.py` (build chosen/rejected pairs from teacher outputs)  
3) `scripts/train_alignment_dpo.py` (DPO preference fine-tune)

```bash
./runAlignment.sh \
  --config config/defaults.json \
  --adapter outputs/student_runs/gemma3-270m/checkpoint-final \
  --teacher-model gpt-oss:120b
```

### `runAll.sh`

Orchestrates the pipeline:

```bash
./runAll.sh --config config/defaults.json --with-training --with-evals --with-alignment
```

Flags:
- `--teacher-model`: override the teacher for domain specs, dataset generation, evals, and alignment (required for those stages)
- `--with-training`: run SFT training
- `--with-evals`: run evaluation + report
- `--with-alignment`: run teacher eval + DPO alignment

---

## Script reference (what each one does + key flags)

Tip: every script supports `--help` for the exhaustive flag list. The sections below explain the flags you will actually tune most often.

### Dataset generation scripts

These are called by `runDatasetGeneration.sh` in order:

#### `scripts/generate_domain_specs.py`

Purpose: ask the teacher to produce domain-specific fields + enum values + example queries.

Key flags:
- `--config`: uses `dataset_generation.*` defaults
- `--teacher-model`: Ollama model id (teacher)
- `--prompts`: domain prompt file (`data/domain_prompts.jsonl`)
- `--examples-per-schema`: how many example queries to ask for per schema
- `--offline-fallback`: skip Ollama and generate deterministic stubs (testing only)

#### `scripts/build_schemas.py`

Purpose: inject generated fields/enums/operators into `data/base_schema_template.json` to create final JSON Schemas.

Key flags:
- `--domain-specs`: input from `generate_domain_specs.py`
- `--operator-catalog`: operator pool (`data/operator_catalog.json`)
- `--min-operators` / `--max-operators`: sampled per schema to vary operator vocab
- Operator catalog entries now include `synonyms`; each retained operator is assigned one alias per schema to force readers to rely on the descriptions.

#### `scripts/generate_dataset.py`

Purpose: ask the teacher to produce validated `(query, AST)` pairs for each schema (using its example queries), validate them against JSON Schema, and write them as SFT targets.

Key flags:
- `--teacher-model`: required; Ollama model id to generate ASTs
- `--samples-per-schema`: how many validated rows to keep per schema
- `--teacher-retries` / `--teacher-retry-wait`: retry policy when parsing the teacher output fails
- `--seed`: controls deterministic shuffling of example queries
- Checkpointing: `--checkpoint-path` + `--checkpoint-every` for periodic saves; `--resume` continues from an on-disk checkpoint (enabled by default via config and the runner).

#### `scripts/build_training_corpus.py`

Purpose: convert dataset rows into the actual SFT corpus (`prompt` + `completion`) used by the student trainer.

Key flags:
- `--include-invalid`: include invalid rows in the corpus (usually keep `true` early, then try a clean-only phase)
- `--max-samples`: cap output size for quick iteration

### `scripts/train_student_unsloth.py` (SFT training)

Purpose: fine-tune `unsloth/gemma-3-270m-it` with LoRA on the training corpus.

Inputs/outputs:
- Input: `--training-corpus` (defaults to `training.training_corpus`)
- Output: `--output-dir` (defaults to `training.output_dir`)

Key flags (grouped):
- Data split:
  - `--val-split`: fraction held out for `eval_loss` (0 disables eval)
  - `--eval-max-samples`: cap eval set size for faster eval
- Sequence budget:
  - `--max-seq-length`: token budget per example (higher = more context, much more memory)
- Training schedule:
  - `--max-steps` or `--num-epochs`: stop condition
  - `--logging-steps`, `--eval-steps`, `--save-steps`, `--save-total-limit`
  - `--early-stop-min-delta`, `--early-stop-patience`: stop when `eval_loss` fails to improve by at least `min_delta` for `patience` consecutive evals
- Optimization:
  - `--learning-rate`, `--weight-decay`, `--warmup-steps`
  - `--batch-size`, `--grad-accum` (effective batch = batch_size × grad_accum)
- Precision/memory:
  - `--load-in-4bit`, `--bf16`, `--use-gradient-checkpointing`
  - `--eval-accumulation-steps`: accumulate eval batches to fit long contexts (e.g., 32K tokens)
- LoRA:
  - `--lora-r`, `--lora-alpha`, `--lora-dropout`

### `scripts/evaluate_student.py` (teacher evaluation)

Purpose: generate student outputs, parse JSON, validate schema, and (optionally) compare to teacher outputs.

Important: for alignment, you must run with a teacher model (`--teacher-model` or `LLM_MODEL` env).

Key flags:
- `--adapter`: LoRA checkpoint dir to evaluate
- `--dataset`: dataset JSONL (defaults to `outputs/d_03_dataset.jsonl`)
- `--teacher-model`: Ollama model id (required for DPO alignment inputs)
- `--eval-results`: where to write per-sample JSONL (defaults to `alignment.eval_results`; `--out-dir` still works)
- Generation: `--max-new-tokens`, `--temperature`, `--top-p` (defaults from `evaluation.*` in config)
- Sampling: `--max-samples` (shuffles with `--sample-seed` first for diversity), `--include-invalid`
- Teacher robustness: `--teacher-retries` (`-1` = infinite), `--teacher-retry-wait`
- Checkpointing: `--checkpoint-path` (defaults next to `--eval-results`), `--checkpoint-every`, `--resume` for long evals; the runner auto-resumes when it finds a checkpoint file.

### `scripts/build_alignment_pairs.py` (preference dataset builder)

Purpose: convert teacher eval outputs into DPO-ready pairs:

- prompt: the exact prompt used during eval (`prompt_text`)
- chosen: teacher canonical AST (`teacher_canonical`)
- rejected: student canonical AST (`student_canonical`)

Key flags:
- `--config`: uses `alignment.eval_results` + `alignment.pairs_out` defaults
- `--only-rejected`: by default keeps only teacher-rejected examples (higher signal)
- `--max-pairs`: cap dataset size for quick experiments

### `scripts/train_alignment_dpo.py` (DPO alignment)

Purpose: teacher-aligned preference tuning starting from the SFT LoRA adapter.

Key flags:
- `--pairs`: pairs JSONL (`alignment.pairs_out`)
- `--adapter`: starting checkpoint (`alignment.adapter`)
- `--output-dir`: where to save aligned adapter (`alignment.output_dir`)
- Training: `--max-steps`, `--batch-size`, `--grad-accum`, `--learning-rate`, `--warmup-steps`, `--logging-steps`
- Checkpointing: `--save-steps`, `--save-total-limit`, `--resume-from-checkpoint` (point at a saved checkpoint dir to keep going)
- DPO: `--beta`
- Precision/memory: `--load-in-4bit`, `--bf16`, `--max-seq-length`

### `scripts/report_training.py` (plots + Markdown report)

Purpose: turn `training_metrics.jsonl` + evaluation files into plots and a Markdown report.

Key flags:
- `--training-metrics`: usually `outputs/student_runs/<run>/training_metrics.jsonl`
- `--eval-summary` / `--eval-results`: from `evaluate_student.py`
- `--out-dir`: where to write `outputs/reports/*`

---

## Defaults explained (`config/defaults.json`)

Everything in this repo is wired so that:

- `config/defaults.json` provides defaults
- scripts accept `--config` to load those defaults
- CLI flags override config values when you need to experiment

Naming convention:
- config keys use `snake_case` (e.g. `max_seq_length`)
- CLI flags usually use `kebab-case` (e.g. `--max-seq-length`)

Below is a **learners-first explanation** of what each default does and what tradeoffs it implies.

### `dataset_generation` (teacher-driven data)

These settings control how much data you generate, how diverse it is, and where the JSONL files go.

- `prompts_path` (`data/domain_prompts.jsonl`): seed “domain prompts” given to the teacher to invent fields/enums for each domain.
- `examples_per_schema` (`60`): how many example NL queries to ask for *inside* each domain spec (used by `generate_domain_specs.py`). Higher = richer spec context, but more teacher tokens/cost.
- `min_operators` / `max_operators` (`4` / `7`): number of allowed operators sampled into each schema. This intentionally forces the student to pay attention to the operator vocabulary per schema.
- `seed` (`11`): RNG seed to make sampling/mutations reproducible.
- `include_invalid` (`true`): whether to keep invalid/negative rows when exporting the SFT corpus (via `build_training_corpus.py`). Keeping negatives can teach “don’t do this”, but too many can hurt pure generation quality; a common pattern is: start with negatives, then do a clean-only fine-tune.
- Output paths:
  - `schema_specs_out` (`outputs/d_01_domain_specs.jsonl`): domain specs (fields/enums/examples) generated by the teacher.
  - `final_schemas_out` (`outputs/d_02_final_schemas.jsonl`): fully built JSON Schemas (one per domain/schema_id).
  - `dataset_out` (`outputs/d_03_dataset.jsonl`): main dataset (schema_json, query, ast_json, is_valid, …).
  - `training_corpus_out` (`outputs/d_04_training_corpus.jsonl`): prompt/completion pairs for SFT.
- Teacher-driven sample generation (in `generate_dataset.py`):
  - `teacher_model` (required): Ollama id to call for AST generation.
  - `samples_per_schema` (`60`): how many teacher-validated rows to keep per schema (dataset size knob).
  - `teacher_retries` (`3`) / `teacher_retry_wait` (`1.0`): retry policy when parsing the teacher output fails.

### `training` (SFT / LoRA on Gemma 3 270M)

These settings control supervised fine-tuning where the student learns to reproduce the target AST completions.

**Inputs/outputs**
- `training_corpus` (`outputs/d_04_training_corpus.jsonl`): the SFT dataset (text prompts + completions).
- `output_dir` (`outputs/student_runs/gemma3-270m`): where checkpoints, metrics, and `checkpoint-final/` are written.
- `base_model` (`unsloth/gemma-3-270m-it`): the Hugging Face model id for the student base weights.
- `seed` (`3407`): controls dataset splitting and various randomized components (helps reproducibility).
- `max_samples` (`null`): optional cap on how many training samples to load (useful for debugging/fast experiments).

**Sequence length (the biggest compute knob)**
- `max_seq_length` (`32768`): token budget per training example after tokenization. This caps how much schema+query+answer can fit. Higher is dramatically slower/more memory (attention scales ~O(L²)). If you get OOM or eval throughput becomes unusable, this is the first knob to reduce.

**Batching**
- `batch_size` (`4`): per-device batch size.
- `grad_accum` (`4`): gradient accumulation steps. Effective batch size is `batch_size × grad_accum` (here 16). Increase `grad_accum` when you can’t raise `batch_size` due to memory.

**Optimization**
- `learning_rate` (`5e-5`): LoRA learning rate. If training is unstable (loss spikes), lower it; if learning is too slow, raise slightly.
- `weight_decay` (`0.01`): regularization on trained weights (LoRA params).
- `warmup_steps` (`50`): linear warmup steps to avoid early instability.

**How long you train**
- `max_steps` (`3000`): hard cap on training optimizer steps (takes precedence over epochs).
- `num_epochs` (`20.0`): used only if `max_steps` is unset.

**Validation split + speed**
- `val_split` (`0.05`): fraction held out for `eval_loss` during training.
- `eval_max_samples` (`256`): cap eval set size (keeps eval fast; great when using long contexts).
- `eval_steps` (`50`): run eval every N training steps.
- `eval_accumulation_steps` (`2`): reduces eval memory by accumulating outputs across micro-batches. Helpful for long contexts (32K) when eval OOMs.

**Logging/checkpointing**
- `logging_steps` (`10`): how often to log training metrics.
- `save_steps` (`200`): checkpoint frequency.
- `save_total_limit` (`3`): number of checkpoints to keep.
- Dataset + eval scripts also checkpoint mid-run: use `--checkpoint-path`/`--checkpoint-every` and `--resume` (runners auto-detect existing checkpoints).

**Stability**
- `max_grad_norm` (`1.0`): gradient clipping. This caps rare excursions without usually hurting quality.
- `early_stop_min_delta` (`0.002`) + `early_stop_patience` (`2`): stops training if `eval_loss` fails to improve by at least `min_delta` for `patience` consecutive evals (saves GPU once you plateau).

**Precision/memory**
- `load_in_4bit` (`true`): loads base weights quantized for memory savings.
- `bf16` (`true`): use bfloat16 on GPU when available.
- `use_gradient_checkpointing` (`false`): trades compute for memory by recomputing activations.

**LoRA**
- `lora_r` (`64`): LoRA rank. Higher = more capacity, more memory/compute.
- `lora_alpha` (`128`): LoRA scaling.
- `lora_dropout` (`0.05`): regularization for LoRA adapters.

### `alignment` (teacher‑aligned DPO)

These settings control the post-SFT preference tuning step.

**What “teacher preference” means here**
- The teacher is used *before* DPO: we run `evaluate_student.py --teacher-model ...` to get `teacher_canonical` and `teacher_verdict`.
- We then build a pairs dataset where, for the same prompt:
  - `chosen` = teacher AST
  - `rejected` = student AST
- DPO training then optimizes the student to prefer `chosen` over `rejected`.

**Inputs/outputs**
- `eval_results` (`outputs/student_runs/eval/evaluation_results.jsonl`): must contain teacher fields (so eval must be run with `--teacher-model`).
- `eval_summary` (`outputs/student_runs/eval/evaluation_summary.json`): aggregate metrics written alongside `eval_results`.
- `pairs_out` (`outputs/student_runs/alignment/dpo_pairs.jsonl`): the preference dataset produced by `build_alignment_pairs.py`.
- `adapter` (`outputs/student_runs/gemma3-270m/checkpoint-final`): starting point for DPO (your SFT LoRA).
- `output_dir` (`outputs/student_runs/gemma3-270m-dpo`): where the aligned adapter is saved.

**Model + sequence**
- `base_model` (`unsloth/gemma-3-270m-it`): base model used to load for alignment.
- `max_seq_length` (`32768`): DPO token budget. Same tradeoffs as SFT: extremely expensive at 32K; reduce this first if alignment is slow/OOM.

**Training**
- `batch_size` (`2`) and `grad_accum` (`8`): effective batch is 16.
- `learning_rate` (`5e-06`): usually smaller than SFT because DPO is a “fine adjustment” stage.
- `warmup_steps` (`50`)
- `max_steps` (`500`)
- `logging_steps` (`10`)
- `save_steps` (`100`): checkpoint frequency.
- `save_total_limit` (`3`): number of checkpoints to keep (older ones pruned).
- Resume a run with `--resume-from-checkpoint <path>` pointing to a saved DPO checkpoint under `alignment.output_dir`.

**DPO-specific**
- `beta` (`0.1`): strength of the preference objective (higher = more aggressively “follow teacher over rejected”).

**Precision/memory**
- `load_in_4bit` (`true`), `bf16` (`true`)

---

## Troubleshooting

### `eval_loss: NaN`

This almost always means the eval batches have **zero supervised tokens** (all labels masked to `-100`), typically due to:
- response markers not matching the chat template
- samples so long that the response is truncated away

This repo includes:
- an eval-batch sanity check before training
- filtering to drop samples where the response would be truncated before the token budget

If you still see NaNs:
- lower `max_seq_length` to something realistic for your GPU/model
- inspect the longest examples in `outputs/d_04_training_corpus.jsonl`

### “Gemma3ForCausalLM does not accept `num_items_in_batch`”

This is an Unsloth/TRL integration warning about an extra collator field. The trainer in this repo strips it before forwarding to the model.

---

## Global objective (in one sentence)

Generate teacher‑supervised, schema‑conditioned training data and train a small local model that turns natural language requests into schema‑valid JSON AST programs, then align it to teacher semantics via preference tuning.
