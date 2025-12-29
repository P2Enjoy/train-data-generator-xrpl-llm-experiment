# Schema‑Conditioned DSL → JSON AST (Gemma 3 270M student)

This repository is a **complete, teacher-in-the-loop pipeline** to build a small model that can:

- Read a **JSON Schema** describing a DSL
- Read a natural language **query** (EN/FR)
- Output **one JSON object** (an AST) that **validates against the schema** and matches the query intent

The teacher is a local **Ollama** model (configured via `.llmrc` or `LLM_MODEL`). The student is **Gemma 3 270M** fine-tuned with **Unsloth LoRA**. After SFT, the repo runs a **teacher‑aligned preference tuning step (DPO)** using teacher verdicts.

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

The teacher model id is resolved in this order:

1) `LLM_MODEL` environment variable  
2) `.llmrc` file at repo root (one line: model id)  
3) fallback: `gpt-oss:120b`

Example:

```bash
echo "gpt-oss:120b" > .llmrc
# or:
export LLM_MODEL="gpt-oss:120b"
```

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
- If you want to override the teacher model for dataset generation, use `./runAll.sh --model <ollama_id> ...`.
- Defaults come from `config/defaults.json` unless overridden with CLI flags.

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
- `outputs/d_03_schema_queries.jsonl`: example queries per schema
- `outputs/d_04_dataset.jsonl`: (schema, query, ast_json, is_valid, …)
- `outputs/d_05_training_corpus.jsonl`: training corpus with `prompt` + `completion`

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
   - Runs every `eval_steps` on a held-out split of `outputs/d_05_training_corpus.jsonl`.  
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

### 3) DPO alignment metrics

During DPO training (`scripts/train_alignment_dpo.py`), TRL prints step logs to stdout (loss/reward signals vary by TRL version/config). The *real* way to evaluate DPO improvements is to:

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
./runDatasetGeneration.sh --config config/defaults.json --model gpt-oss:120b
```

Flags:
- `--config`: path to config JSON
- `--model`: Ollama model id to use for schema/query generation

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
- `--model`: teacher model override for dataset generation steps
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
- `--model`: Ollama model id (teacher)
- `--prompts`: domain prompt file (`data/domain_prompts.jsonl`)
- `--examples-per-schema`: how many example queries to ask for per schema
- `--offline-fallback`: skip Ollama and generate deterministic stubs (testing only)

#### `scripts/build_schemas.py`

Purpose: inject generated fields/enums/operators into `data/base_schema_template.json` to create final JSON Schemas.

Key flags:
- `--domain-specs`: input from `generate_domain_specs.py`
- `--operator-catalog`: operator pool (`data/operator_catalog.json`)
- `--min-operators` / `--max-operators`: sampled per schema to vary operator vocab

#### `scripts/generate_example_queries.py`

Purpose: ask the teacher for more NL queries per schema to diversify the dataset.

Key flags:
- `--schemas`: input from `build_schemas.py`
- `--per-schema`: how many queries per schema
- `--model`: Ollama model id (teacher)
- `--offline-fallback`: testing only

#### `scripts/generate_dataset.py`

Purpose: build `(schema, query) → AST` training rows by calling the teacher and validating against JSON Schema; then synthesize invalid negatives.

Key flags:
- `--positives-per-schema`: how many validated teacher rows to keep per schema
- `--negative-ratio`: how many invalid negatives per positive
- `--seed`: controls deterministic sampling/mutations

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

Important: for alignment, you must run with a teacher model (`--teacher-model` or `.llmrc`/`LLM_MODEL`).

Key flags:
- `--adapter`: LoRA checkpoint dir to evaluate
- `--dataset`: dataset JSONL (defaults to `outputs/d_04_dataset.jsonl`)
- `--teacher-model`: Ollama model id (required for DPO alignment inputs)
- `--out-dir`: where to write `evaluation_results.jsonl` + `evaluation_summary.json`
- Generation: `--max-new-tokens`, `--temperature`, `--top-p`
- Sampling: `--max-samples`, `--include-invalid`

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
- DPO: `--beta`
- Precision/memory: `--load-in-4bit`, `--bf16`, `--max-seq-length`

### `scripts/report_training.py` (plots + Markdown report)

Purpose: turn `training_metrics.jsonl` + evaluation files into plots and a Markdown report.

Key flags:
- `--training-metrics`: usually `outputs/student_runs/<run>/training_metrics.jsonl`
- `--eval-summary` / `--eval-results`: from `evaluate_student.py`
- `--out-dir`: where to write `outputs/reports/*`

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
- inspect the longest examples in `outputs/d_05_training_corpus.jsonl`

### “Gemma3ForCausalLM does not accept `num_items_in_batch`”

This is an Unsloth/TRL integration warning about an extra collator field. The trainer in this repo strips it before forwarding to the model.

---

## Global objective (in one sentence)

Generate teacher‑supervised, schema‑conditioned training data and train a small local model that turns natural language requests into schema‑valid JSON AST programs, then align it to teacher semantics via preference tuning.
