# Project Manifest – Schema-Conditioned DSL → AST LLM (SMOL 1.7B)

## 1. Strategic Objective

Build a small, on-device friendly LLM (SMOL 1.7B class) that:

* Ingests a **JSON Schema** that defines a DSL (fields, operators, values, structure, semantics via descriptions).
* Ingests a **natural language query** in English or French only (EN/FR).
* Outputs a **JSON AST** that is a valid instance of the DSL (and of the given JSON Schema), suitable for deterministic execution by downstream engines.

Target usage:

* Local / mobile inference for interactive query → DSL translation.
* Pluggable parser engine for multiple domains (events, work, cooking, meetings, etc.) with the **same structural template** but different field/value vocabularies.

## 2. Target Model – SMOL 1.7B

* Base model: a small, open HF model in the ~1.5–2B range ("SMOL 1.7B" class).
* Properties required:

  * Instruction-tuned / chat-ready baseline.
  * Context window ≥ 8K tokens (ideal 16K) to handle large JSON Schemas.
  * Good performance on structured / code-like tasks.
* Fine-tuning strategy:

  * Parameter-efficient (QLoRA / LoRA) using **Unsloth**.
  * Quantization for deployment: 4-bit (GGUF or equivalent) for mobile/edge inference.

## 3. High-Level Architecture – 3 Pipelines

### 3.1 Pipeline A – Schema & Domain Generator (Meta-Schema LLM)

**Goal:** Generate diverse DSL schemas for different domains while keeping the **core JSON Schema structure fixed**.

* Inputs:

  * A base JSON Schema template (e.g. FunnelDefinition) that defines:

    * Structural layout (object → steps → conditions, timeframe, etc.).
    * Node types and constraints (required fields, enums, oneOf, additionalProperties=false, etc.).
  * Domain prompts (e.g. "events", "work tasks", "cooking recipes", "meeting planner").

* Process:

  * A “meta-prompt” LLM generates **domain specializations**:

    * A list of `field` identifiers (e.g. `c.firstname`, `task.priority`, `recipe.type`).
    * For each field where relevant, a set of allowed enum `value`s.
    * Optionally, short descriptions for fields and values.
  * Output format: **JSONL**, one document per domain schema:

    ```json
    {
      "schema_id": "events_funnel_v1",
      "domain": "events",
      "fields": [
        { "const": "ec.team", "description": "High-level event team" },
        { "const": "c.firstname", "description": "Contact first name" }
      ],
      "enums": {
        "ec.team": ["All", "Launch", "Learn", "Build", "Adopt", "Engage"],
        "ec.persona": ["Developers", "Corporates", "Community", "Internal"]
      }
    }
    ```

* Post-processing:

  * A deterministic Python builder injects fields/enums into the **fixed base template** to produce the final JSON Schema for that domain.
  * The structure is never edited directly by the LLM, only filled through controlled slots.

### 3.2 Pipeline B – Dataset Generator (Teacher LLM + Validator)

**Goal:** Produce a large, high-quality dataset of (schema, query, AST) triples, plus synthetically corrupted negatives.

#### 3.2.1 Positive Examples (Teacher LLM)

* For each `schema_id`:

  * Generate multiple **natural language queries** per schema:

    * LLM-generated based on schema descriptions.
    * Optionally, curated manual prompts for priority domains.
  * For each (schema, query) pair:

    * Call a strong teacher LLM (e.g. GPT-4.x-class) with:

      * Developer message: instructions to output **only JSON AST**, strictly schema-compliant, with internal reasoning but no explanation.
      * Assistant message: “I MUST ADHERE TO THIS SCHEMA” followed by the full JSON Schema.
      * User message: the query.
    * Validate the returned JSON instance using a JSON Schema validator.
    * If valid → store as **positive sample**.

* Positive sample format (JSONL):

  ```json
  {
    "schema_id": "events_funnel_v1",
    "schema_json": "{...full JSON Schema as string...}",
    "query": "All events from Paris and concerning the company employees",
    "ast_json": "{...AST JSON...}",
    "is_valid": true,
    "validation_error": null
  }
  ```

#### 3.2.2 Negative Examples (Synthetic Mutations + Natural Errors)

* Start from valid ASTs (positives) and generate **controlled corruptions**:

  * Structural:

    * Remove required fields (e.g. drop `timeframe` or `conditions`).
    * Add unexpected properties under `additionalProperties`: false.
  * Type / enum:

    * Change `value` to a non-enum string when enum is enforced.
    * Change field type (string → number, etc.).
  * DSL semantics:

    * Use a mismatched field for a concept (e.g. mapping “employees” to a persona unrelated to employees).
    * Break business rules (e.g. `timeframe.end` < `timeframe.start`).

* Optionally keep some **raw teacher mistakes** as negatives.

* Negative sample format:

  ```json
  {
    "schema_id": "events_funnel_v1",
    "schema_json": "{...}",
    "query": "All events from Paris and concerning the company employees",
    "ast_json": "{...corrupted AST...}",
    "is_valid": false,
    "validation_error": "Additional properties are not allowed ('foo' was unexpected)",
    "error_type": "extra_property"
  }
  ```

* Dataset design:

  * Balanced valid/invalid ratio (target ~70% valid, 30% invalid for pretraining the student; can add a final clean phase with only valid data).
  * Domain diversity: events, work, cooking, meetings, etc., all sharing the same structural schema template.

### 3.3 Pipeline C – Student Training (Unsloth + SMOL 1.7B)

**Goal:** Distill teacher behaviour into SMOL 1.7B so it can:

* Read a full JSON Schema (DSL definition) + natural language query.
* Emit a **schema-compliant AST JSON**.

#### 3.3.1 Training Sample Format

* **Input text (prompt)**

  ```text
  You are a JSON AST generator.
  You must output a single JSON object that satisfies the following JSON Schema
  and represents the program matching the user request.

  [SCHEMA]
  {...full JSON Schema...}
  [/SCHEMA]

  [QUERY]
  All events from Paris and concerning the company employees
  [/QUERY]

  [OUTPUT]
  ```

* **Output text (target)**

  ```text
  {
    "prompt": "All events from Paris and concerning the company employees",
    "steps": [...],
    "timeframe": {...}
  }
  ```

* Loss masking:

  * Use causal LM setup; mask the loss on the prompt part, apply loss only on the AST tokens.

#### 3.3.2 Training Configuration

* Framework: Unsloth + PyTorch / HF Transformers.
* Fine-tuning method: QLoRA / LoRA on top of SMOL 1.7B.
* Hyperparameters (baseline):

  * LR: ~1–2e-4 for adapters.
  * Batch size (effective) tuned for GPU constraints using gradient accumulation.
  * Training steps / epochs: tuned based on validation loss plateau.
* Checkpoints:

  * Save intermediate adapters often; run schema-based evaluation between phases.

#### 3.3.3 Evaluation & Metrics

* Standard metrics:

  * Exact match of AST JSON (strict string equality after canonicalization).
  * Structural match (ignoring whitespace and field ordering) using JSON diff.
* Schema-based metrics:

  * JSON Schema validation pass rate on generated ASTs.
  * Distribution of validation error types.
* Semantic metrics:

  * NL → AST alignment sanity checks for curated test cases.
  * Teacher adjudication of schema-valid student outputs: the teacher inspects the candidate AST plus schema and query and returns `is_good`/`reason`; an answer only passes if it is schema-valid **and** the teacher confirms it satisfies the query.
  * Teacher is enabled by default using the local ollama model configured in `.llmrc` (or `LLM_MODEL`); evaluation will fail fast if no teacher is configured.

## CLI reference (all flags, defaults in `config/defaults.json` unless noted)

### Dataset generation
- `scripts/generate_domain_specs.py`: `--config` defaults JSON; `--prompts` domain prompts JSONL; `--out` specs JSONL; `--model` ollama id (from `.llmrc`/`LLM_MODEL` via `default_model()`); `--seed` for deterministic fallbacks; `--examples-per-schema` query count to request from LLM; `--offline-fallback` to skip ollama and emit stub specs.
- `scripts/build_schemas.py`: `--config`; `--base-template` base JSON Schema; `--domain-specs` specs JSONL; `--out` final schemas JSONL; `--operator-catalog` pool to sample operators; `--min-operators` / `--max-operators` counts; `--seed` for operator sampling.
- `scripts/generate_example_queries.py`: `--config`; `--schemas` final schemas JSONL; `--out` schema+queries JSONL; `--per-schema` number of queries per schema; `--model` ollama id; `--offline-fallback` to use deterministic stubs; `--seed` for stubs; `--max-retries` when parsing LLM output.
- `scripts/generate_dataset.py`: `--config`; `--schemas` final schemas JSONL; `--out` dataset JSONL; `--positives-per-schema` validated samples per schema; `--negative-ratio` negatives per positive; `--seed`.
- `scripts/build_training_corpus.py`: `--config`; `--dataset` dataset JSONL; `--out` prompt/target JSONL; `--include-invalid` to keep negatives; `--max-samples` optional cap.

### Training and evaluation
- `scripts/train_student_unsloth.py`: `--config`; `--training-corpus`; `--output-dir`; `--base-model`; `--max-seq-length`; `--batch-size`; `--grad-accum`; `--learning-rate`; `--weight-decay`; `--warmup-steps`; `--num-epochs`; `--max-steps`; `--logging-steps`; `--eval-steps`; `--save-steps`; `--save-total-limit`; `--val-split`; `--max-samples`; `--seed`; `--load-in-4bit`; `--bf16`; `--use-gradient-checkpointing`; `--resume-from-checkpoint`; `--lora-r`; `--lora-alpha`; `--lora-dropout`; `--eval-max-samples`.
- `scripts/evaluate_student.py`: `--dataset`; `--adapter`; `--base-model`; `--max-seq-length`; `--max-new-tokens`; `--max-samples`; `--include-invalid`; `--temperature`; `--top-p`; `--load-in-4bit`; `--out-dir`; `--teacher-model` (required, defaults to `.llmrc`/`LLM_MODEL` via `default_model()`); `--log-every`.
- `scripts/report_training.py`: `--training-metrics`; `--eval-summary`; `--eval-results`; `--run-config`; `--out-dir`; `--report-name`.

### Teacher sanity checks
- `scripts/test_teacher.py`: `--training-corpus`; `--model` ollama id (default `.llmrc`/`LLM_MODEL`); `--max-samples`.
- `scripts/test_inference_teacher.py`: `--schema-id`; `--prompt`; `--schema-source` schemas JSONL; `--model` ollama id (default `.llmrc`/`LLM_MODEL`); `--current-date` ISO date for prompt context.

### Utilities
- `scripts/pretty_jsonl.py`: `--input` JSONL; `--out` prettified JSON; `--limit` cap records.

### Convenience runners
- `runDatasetGeneration.sh`: `--model` ollama id override for dataset steps; `--config` defaults JSON path.
- `runTraining.sh`: accepts `-h/--help`, otherwise forwards all args to `scripts/train_student_unsloth.py` after `uv sync`.
- `runEvals.sh`: accepts `--out-dir` (eval output root), `--adapter` (LoRA checkpoint dir), `-h/--help`; forwards other args to `scripts/evaluate_student.py`, then builds a report.
- `runAll.sh`: `--model` (passes to dataset generation), `--config`, `--with-training`, `--with-evals`; runs dataset generation always, optional training/evals.

## 4. Canonicalization & Normalization

To stabilize training for a small model:

* **JSON canonicalization:**

  * Deterministic property ordering in all ASTs.
  * Consistent indentation and whitespace.
  * No comments, no trailing commas.

* **Prompt canonicalization:**

  * Same marker tokens `[SCHEMA]`, `[QUERY]`, `[OUTPUT]` for all examples.
  * Same intro instructions across the entire dataset.

* **Schema preprocessing:**

  * Retain: titles, descriptions, enums, constraints relevant to semantic mapping.
  * Optionally strip redundant comments to reduce context size while preserving meaning.

## 5. Inference Contract (Runtime API)

The runtime API for the distilled model should be stable and constrained to English and French (EN/FR).

* **Input:**

  ```json
  {
    "schema_id": "events_funnel_v1",
    "schema_json": "{...}",
    "query": "All events from Paris and concerning the company employees"
  }
  ```

* **Internal prompt building:**

  * Embed into the canonical text prompt with `[SCHEMA]`, `[QUERY]`, `[OUTPUT]` markers.

* **Output (expected):**

  ```json
  {
    "prompt": "All events from Paris and concerning the company employees",
    "steps": [...],
    "timeframe": {...}
  }
  ```

* **Post-processing:**

  * Parse model output as JSON.
  * Validate against the schema.
  * If invalid:

    * Optionally run a rule-based fixer / second-pass prompt.
    * Or return a structured error to the caller.

## 6. Tech Stack & Tooling

* **Data & orchestration:**

  * Python for pipelines A/B/C.
  * JSONL for datasets and schema descriptions.
  * `jsonschema` (or equivalent) for validation.

* **Modeling & training:**

  * Hugging Face Transformers ecosystem.
  * Unsloth for efficient QLoRA / LoRA fine-tuning.
  * Weights & Biases (or similar) for experiment tracking (optional but recommended).

* **Deployment:**

  * Quantized weights (4-bit) in a mobile/edge-friendly format.
  * Simple HTTP/gRPC microservice or local binding for mobile apps.

## 7. Roadmap & Milestones

1. **MVP DSL & Schema Template**

   * Finalize base JSON Schema template (FunnelDefinition-style).
   * Implement deterministic schema builder from domain field/enum specs.

2. **Synthetic Data v1**

   * Generate first set of domain specializations (events + 1–2 others).
   * Build data generator for positives and synthetic negatives.
   * Obtain ~10k–50k training examples.

3. **First Distillation Run (SMOL 1.7B)**

   * Fine-tune with Unsloth using the v1 dataset.
   * Evaluate on held-out schemas and curated test queries.

4. **Iteration & Hardening**

   * Add more domains and harder NL queries.
   * Improve negative sampling and semantic test suites.
   * Refine prompt / JSON canonicalization.

5. **Packaging & Deployment**

   * Export quantized model.
   * Wrap with an inference API that hides prompt-building internals.
   * Integrate into target mobile / local environments.

## 8. Open Questions / Design Decisions

 * The model strictly emits JSON ASTs only; no explanation or debug mode is required.
 * Which domains are strategic enough to justify manual curation of gold test sets?
 * Backward compatibility of ASTs is **not** a requirement; schema and model versions may introduce breaking changes.

This manifest serves as the reference backbone for subsequent design docs, implementation tasks, and experiment tracking for the schema-conditioned AST LLM built on SMOL 1.7B.

## Local Pipeline Scripts

Multi-step scripts now call the local Ollama model `gpt-oss:120b` to generate compliant JSON Schemas with rich descriptions and to surface example natural-language queries per schema. Each step reads JSONL and writes JSONL so you can inspect outputs or swap files manually.

0) Fast bootstrap via `runAll.sh` (ensures `uv sync` runs and each step executes if its output is missing):
```bash
./runAll.sh
```

1) Domain spec synthesis (prompts → fields/enums/descriptions + example queries, via Ollama):
```bash
python scripts/generate_domain_specs.py \
  --prompts data/domain_prompts.jsonl \
  --out outputs/d_01_domain_specs.jsonl \
  --model gpt-oss:120b
```

2) Schema build (inject specs into the base template similar to the provided FunnelDefinition example, with `allOf` enum guards):
```bash
python scripts/build_schemas.py \
  --domain-specs outputs/d_01_domain_specs.jsonl \
  --base-template data/base_schema_template.json \
  --operator-catalog data/operator_catalog.json \
  --out outputs/d_02_final_schemas.jsonl
```
`build_schemas.py` also samples a per-schema operator set (defaults to 4–7) from `data/operator_catalog.json` so each schema exposes a slightly different operator vocabulary. Set `--min-operators` / `--max-operators` / `--seed` to control that spread.

3) Example queries for each generated schema (LLM-generated, EN/FR mix):
```bash
python scripts/generate_example_queries.py \
  --schemas outputs/d_02_final_schemas.jsonl \
  --out outputs/d_03_schema_queries.jsonl \
  --per-schema 6 \
  --model gpt-oss:120b
```

4) Dataset creation (positive + synthetic negative ASTs with JSON Schema validation):
```bash
python scripts/generate_dataset.py \
  --schemas outputs/d_02_final_schemas.jsonl \
  --out outputs/d_04_dataset.jsonl \
  --positives-per-schema 8 \
  --negative-ratio 0.4
```
Operators in each AST are sampled from the schema-specific operator list so the student model must pay attention to the allowed operator vocabulary per schema; negatives include unsupported operator mutations.

5) Training text export (canonical prompt/target pairs):
```bash
python scripts/build_training_corpus.py \
  --dataset outputs/d_04_dataset.jsonl \
  --out outputs/d_05_training_corpus.jsonl
```

Notes:
- `runAll.sh` wraps the entire pipeline, calls `uv sync`, and uses `uv run python …` so you never have to spell the `uv` invocations yourself. It skips steps whose outputs already exist and always continues with downstream stages.
- `runAll.sh` also retries a step automatically if its output logs contain `[warn]`, so intermittent Ollama glitches will trigger a second attempt before failing.
- `reset.sh` removes `outputs/` so you can start fresh before running `./runAll.sh` (it recreates everything).
- The default Ollama model is stored in `.llmrc` (set to `gpt-oss:120b`). `runAll.sh` reads this file, but you can pass `--model <name>` or set `LLM_MODEL` in the environment when you need a different model.
- `data/base_schema_template.json` mirrors the FunnelDefinition shape (prompt, steps with named conditions using `operator` and `value`, timeframe with start/end dates). `build_schemas.py` injects field `oneOf` entries and per-field `allOf` enum constraints produced by the LLM.
- `generate_domain_specs.py` and `generate_example_queries.py` call Ollama by default; pass `--offline-fallback` to use deterministic stubs if the model is unavailable. Ensure `ollama run gpt-oss:120b` works locally before running the pipeline.
- Default inputs live under `data/`; numbered outputs go to `outputs/d_0*_*.jsonl`. Validation is performed during dataset creation to label valid vs. invalid rows.

### Pretty JSON outputs

After generating the JSONL artifacts, `runAll.sh` also runs `scripts/pretty_jsonl.py` to emit a human-readable copy of each file under `outputs/pretty/…`. These files are JSON arrays with 2-space indentation so you can quickly inspect full schema/dataset dumps without manual tooling.

## Teacher smoke test

Use `scripts/test_teacher.py` to replay prompts from `outputs/d_05_training_corpus.jsonl` against `gpt-oss:120b`. It shows the LLM’s JSON output, the stored completion, and a diff if they diverge:

```bash
uv run python scripts/test_teacher.py --training-corpus outputs/d_05_training_corpus.jsonl --model gpt-oss:120b --max-samples 2
```

Inspect the printed diffs to verify the teacher model can still produce schema-compliant ASTs before progressing to student fine-tuning.

## Interactive teacher inference

`test_inference_teacher.py` lets you try a new prompt for any schema. It loads `outputs/d_02_final_schemas.jsonl`, builds the canonical `[SCHEMA]`/`[QUERY]` prompt, calls `gpt-oss:120b`, and prints both the raw model reply and the parsed JSON AST.

```bash
uv run python scripts/test_inference_teacher.py \
  --schema-id events_funnel_v1 \
  --prompt "All events attended by Guillaume last year" \
  --model gpt-oss:120b
```

Use this script to sanity-check a single NL request before adding it to your dataset or training loops.

## Student training – Gemma 3 270M (Unsloth)

Fine-tune a small student on the generated corpus using Unsloth LoRA adapters, then evaluate and report:

- Install deps (Unsloth, TRL, Transformers): `UV_CACHE_DIR=.uv-cache uv sync`
- Train: `./runTraining.sh --training-corpus outputs/d_05_training_corpus.jsonl --output-dir outputs/student_runs/gemma3-270m`
- Evaluate the saved adapter on valid samples (optionally compare to the Ollama teacher):  
  `./runEvals.sh --adapter outputs/student_runs/gemma3-270m/checkpoint-final --teacher-model gpt-oss:120b`
- Build plots + Markdown summary (auto-run by `runEvals.sh`): `./runEvals.sh`

## Runner scripts

- `runDatasetGeneration.sh [--model MODEL]` → steps 1–5 (domain specs → training corpus) with prettified outputs.  
- `runTraining.sh [train_student_unsloth.py args…]` → LoRA fine-tune Gemma-3-270M student.  
- `runEvals.sh [evaluate_student.py args…]` → evaluate student (and optional teacher comparison) then generate report.  
- `runAll.sh [--model MODEL] [--with-training] [--with-evals]` → wrapper that calls the above in order.  

To avoid permission issues, the runners set `UV_CACHE_DIR=${UV_CACHE_DIR:-.uv-cache}`; override it if you prefer a different cache location.
