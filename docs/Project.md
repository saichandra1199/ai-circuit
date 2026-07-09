# Project Architecture — Agentic ML Engineer on H&M Fashion Data

## 1. Overview

5-class H&M fashion image classification used as a benchmark to demonstrate an **autonomous AI ML engineering loop**. The classifier is the substrate; the agent is the artifact.

**Central hypothesis:** an AI agent given a training framework, a dataset, and an evaluation metric can autonomously discover better hyperparameter configurations through iterative experimentation — the same way a human ML engineer would.

The agent never modifies `train.py`. It only modifies configuration, reads metrics, and decides what to try next.

> The project is judged on AI-driven experimentation, reflection, decision making, and optimization — not on building the most sophisticated vision model.

---

## 2. Two Workflows

### Workflow 1 — Human Data Prep + Agent Trains (`run_human_agent.py`)

```
Human:   raw_data/ ──[preprocessing]──► data/full/ + data/sample/
                                               │
                                   training_config.yaml (baseline)
                                               │
                         ┌─────────────────────▼──────────────────────┐
                         │            AGENTIC LOOP                     │
                         │                                             │
                         │  init_iter ──► train.py ──► evaluate        │
                         │      ▲                          │           │
                         │      │                     (continue?)      │
                         │      │                          │           │
                         │   improve ◄── notes ◄───────────┘           │
                         └─────────────────────────────────────────────┘
```

Human handles all data decisions: class selection, stratified splitting, class weights, directory structure. Agent's job is purely optimization — receive a working pipeline, improve macro F1 through config changes.

**Intentionally reserved for the agent** (baseline has these OFF):
- MixUp, CutMix, RandAugment, AutoAugment, RandomErasing
- Label Smoothing, Focal Loss
- Higher resolution, different backbone, different scheduler/optimizer

### Workflow 2 — End-to-End Autonomous Agent (`run_full_agent.py`)

```
Agent:   raw_data/articles.csv + raw_data/images/
              │
         [hm_data_prep_agent.py]
         LLM decides: which classes, how many, how to split, class weights
              │
         data/auto/  ←  agent-created structure
              │
         auto_training_config.yaml
              │
         Same agentic loop as Workflow 1
         train → evaluate → notes → improve → repeat
```

Agent receives only raw CSV + images. No human provides class selection, splits, or metadata. The `hm_data_prep_agent.py` handles everything data-related; then the identical training loop runs.

---

## 3. Benchmark

All runs are compared using:

| Metric | Description |
|--------|-------------|
| Macro F1 / Accuracy / Precision / Recall | Model quality |
| Training time | Compute efficiency |
| Number of experiments | How quickly agent converges |
| Agent decisions | What changes were made and why |
| Improvement over baseline | Delta from first run |
| Human intervention | How much setup was required |

The benchmark focuses on the **engineering workflow** rather than only final accuracy.

---

## 4. Dataset

**Source:** H&M Personalized Fashion (Kaggle)  
**Label column:** `product_group_name` from `articles.csv`  
**Total images:** ~92,286 (after filtering to 5 classes)  
**Split:** stratified 80/10/10 train/val/test (no leakage — each image in exactly one split)

### Why `product_group_name`?

Other candidate columns were rejected:
- `index_group_name` (5 classes) — too easy, no room for optimization
- `index_name` (10 classes) — some labels correspond to size ranges, not visually distinguishable
- `garment_group_name` (21 classes) — fine-grained, high visual overlap (e.g. "Jersey Basic" vs "Jersey Fancy")
- `product_type_name` (131 classes) — too many, extreme imbalance

`product_group_name` gives 5 visually distinct classes that converge well with transfer learning.

### Classes and Imbalance (~8:1 ratio)

| Class | Train (full) | Train (sample) | Weight |
|-------|-------------|----------------|--------|
| Garment Upper body | 34,144 | 500 | 0.43 |
| Garment Lower body | 15,816 | 500 | 0.93 |
| Garment Full body | 10,620 | 500 | 1.38 |
| Accessories | 8,804 | 500 | 1.67 |
| Shoes | 4,125 | 500 | 3.56 |

**Imbalance strategy:** inverse-frequency class weights applied to both `WeightedCrossEntropy` and `WeightedRandomSampler`. No augmented copies generated on disk — the agent controls augmentation as an optimization lever.

### Data Directories

```
data/
├── sample/         # 500 train / 63 val / 63 test per class (agent default)
├── full/           # full scale (symlink → processed_data/)
├── class_weights.json
└── class_mapping.json
```

`data/sample/` created by `utils/create_sample_data.py` with seed 42 — reproducible across runs.

### Image ID Fix

`articles.csv` stores article IDs without leading zeros (e.g. `108775015`), but image filenames use zero-padded 10-digit IDs (`0108775015.jpg`). Always apply `str.zfill(10)` before matching.

---

## 5. Training Pipeline (`train.py`)

Config-driven, fixed code. Agent modifies only the YAML — never the script.

**Model:** `timm.create_model(backbone, pretrained=True, num_classes=5, drop_rate=dropout)`  
Any timm backbone supported. Warm-starting via `model.checkpoint`.

**Transforms (applied online during training, not during preprocessing):**

| Split | Transforms |
|-------|-----------|
| Train | Resize(256) → RandomResizedCrop(224) → RandomHorizontalFlip → ColorJitter → ToTensor → Normalize(ImageNet) |
| Val/Test | Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet) |

Advanced augmentations (MixUp, CutMix, RandAugment, RandomErasing) start OFF — reserved for agent.

**Loss:** `weighted_ce` (CrossEntropy + class weights + label smoothing) or `focal` (configurable gamma).

**Optimizer:** AdamW / Adam / SGD from config.

**Scheduler:** CosineAnnealingLR / StepLR / OneCycleLR / ReduceLROnPlateau.

**Mixed precision:** `torch.amp.autocast` — auto-disabled on CPU.

**Subsampling:** `max_samples_per_class` caps dataset at runtime without touching disk.

### Output per Run

Each run writes to `experiments/run_N/`:

```
run_N/
├── config.yaml       # exact config snapshot for this run
├── best_model.pth    # best checkpoint (by val macro F1)
├── metrics.json      # full history + test metrics + confusion matrix
├── notes.md          # LLM-written analysis (agent runs only)
└── tensorboard/      # TensorBoard event files
```

`metrics.json` schema:
```json
{
  "experiment_name": "agent_run_3",
  "backbone": "convnext_base",
  "epochs_trained": 15,
  "best_val_macro_f1": 0.743,
  "test": {
    "macro_f1": 0.738,
    "accuracy": 0.751,
    "per_class": { "Shoes": { "f1-score": 0.81, "precision": ..., "recall": ... }, ... },
    "confusion_matrix": [[...], ...]
  },
  "history": [{ "epoch": 1, "train_loss": ..., "val_macro_f1": ... }, ...]
}
```

---

## 6. Agentic Loop (`agents/hm_training_agent.py`)

Built with **LangGraph StateGraph**. LLM: `gpt-4o-mini` via `utils/llm_api.py`.

### Graph

```
init_iter ──► run_train ──► evaluate ──┬──(done)──► END
                                        │
                                   (continue)
                                        │
                                  generate_notes
                                        │
                                     improve
                                        │
                                   init_iter (loop)
```

### State (`HMTrainingState`)

| Field | Type | Purpose |
|-------|------|---------|
| `run_num` | int | current iteration counter |
| `current_config` | dict | config for the next run |
| `last_diff` | dict | changes applied to reach current_config |
| `last_metrics` | dict | metrics.json from last run |
| `notes_history` | list | rolling last-3 notes (LLM memory) |
| `experiment_log` | list | all runs: F1, accuracy, diff, checkpoint |
| `best_macro_f1` | float | best test macro F1 seen |
| `best_checkpoint_path` | str | path to best_model.pth from best run |
| `plateau_count` | int | consecutive runs without +0.005 F1 gain |
| `max_iterations` / `target_f1` | int/float | stop conditions |

### Node Details

**`init_iter`** — increments `run_num`, writes `current_config` to `experiments/run_N/config.yaml`, sets `paths.output_dir`.

**`run_train`** — `subprocess.run(["python", "train.py", "--config", ...])`. On non-zero exit: sets `done=True` with error.

**`evaluate`** — reads `metrics.json`, detects plateau (`macro_f1 < best + 0.005`), updates `experiment_log.json`. Routes to END if target reached or max iterations hit.

**`generate_notes`** — LLM call with `NOTES_PROMPT`. Writes `notes.md`. Maintains rolling 3-note history for next improve call.

**`improve`** — LLM call with `IMPROVE_PROMPT`. Returns JSON diff. `_apply_diff()` deep-merges via dot-notation. On plateau ≥ 2: prompt pushes bolder changes (backbone swap, MixUp/CutMix, different loss).

---

## 7. Config-Diff Design

The agent returns a JSON diff of config changes:

```json
{ "optimizer.lr": 0.00005, "augmentations.mixup": true, "augmentations.mixup_alpha": 0.4 }
```

Applied via dot-notation traversal to `current_config`. Each run's exact config is snapshotted in `config.yaml` — any run is fully reproducible.

**Advantages:**
- **Auditable** — every change logged in `experiment_log.json`, the diff IS the hypothesis
- **Safe** — LLM cannot introduce code bugs, only hyperparameter changes
- **Reversible** — revert any run by loading its `config.yaml`

### Warm-Start Checkpointing

When backbone stays the same, agent can include `model.checkpoint`:
```json
{ "model.checkpoint": "experiments/run_2/best_model.pth", "optimizer.lr": 0.00005 }
```

`train.py` loads the checkpoint before training. If architecture changed (incompatible state dict), falls back to ImageNet pretrained with a warning.

### Plateau Detection

`plateau_count` increments when improvement < 0.005. When ≥ 2: prompt switches to bold-change mode — try different backbone, enable MixUp/CutMix, change loss function.

---

## 8. LLM Prompts (`agents/prompts.py`)

### `IMPROVE_PROMPT`
**Input:** full config YAML + run metrics + per-class F1 + last 3 notes + plateau warning or checkpoint hint + enumerated config levers with valid ranges.  
**Output:** JSON object with 2–5 config changes. LLM is grounded in available levers — cannot invent keys that `train.py` doesn't read.

### `NOTES_PROMPT`
**Input:** run metrics vs target, per-class F1, config diff applied this run.  
**Output:** 3-section markdown:
- `## Changes Made` — what changed and why
- `## Results Analysis` — what worked, what failed, confusion patterns
- `## Further Improvements` — 2–3 specific next steps

Notes serve dual purpose: human-readable audit trail + LLM memory for the improve step.

---

## 9. Data Prep Agent (`agents/hm_data_prep_agent.py`) — Workflow 2 only

Invoked by `run_full_agent.py` before the training loop.

1. Loads `articles.csv`, counts images per class
2. LLM selects 3–6 visually distinct classes with ≥500 images each
3. Filters and stratified-splits the selected classes (80/10/10)
4. Copies images to `data/auto/{train,val,test}/{class}/`
5. Writes `class_mapping.json` and `class_weights.json`
6. Returns paths dict → patched into training config

---

## 10. Files Reference

| File | Purpose |
|------|---------|
| `train.py` | Config-driven training pipeline. Fixed — agent never modifies. |
| `training_config.yaml` | Baseline config. Agent reads as starting point. |
| `run_human_agent.py` | Workflow 1 entry point. |
| `run_full_agent.py` | Workflow 2 entry point (autonomous data prep + training). |
| `agents/hm_training_agent.py` | LangGraph agent — full loop logic. |
| `agents/hm_data_prep_agent.py` | LLM-driven data prep for Workflow 2. |
| `agents/prompts.py` | `IMPROVE_PROMPT` and `NOTES_PROMPT` templates. |
| `utils/llm_api.py` | OpenAI `gpt-4o-mini` chat wrapper. Reads `OPENAI_API_KEY` from `.env`. |
| `utils/create_sample_data.py` | Creates `data/sample/` from `processed_data/`. |
| `data/` | Agent-facing data root: `sample/`, `full/`, metadata JSON. |
| `experiments/` | All agent run outputs + `experiment_log.json`. |

---

## 11. Design Decisions

**Why config-diff and not script rewriting?**  
Config changes cover 95% of the optimization space for image classification. Rewriting `train.py` risks code bugs that obscure the F1 signal. Config-diff keeps the loop clean and auditable.

**Why `gpt-4o-mini`?**  
The improve step is pattern-matching on metrics and returning a small JSON — not novel reasoning. Fast and cheap for 5–10 sequential iterations.

**Why 500/class for sample data instead of runtime subsampling?**  
Prebuilt `data/sample/` means `train.py` does no filtering and results are reproducible across runs (same 500 images each time, fixed seed 42).

**Why rolling 3-note memory?**  
Enough context to avoid repeating failed approaches, short enough to stay within token budget. Fixed 3-section note structure makes LLM extraction reliable.

**Why not modify images during preprocessing?**  
Training pipeline owns all augmentations. This gives the agent full control over augmentation as an optimization lever — it cannot experiment with what's already baked into the images.
