# Project Architecture — Agentic ML Engineer

## 1. Overview

Image classification used as a benchmark to demonstrate an **autonomous AI ML engineering loop**. The classifier is the substrate; the agent is the artifact.

**Central hypothesis:** an AI agent given a training framework, a dataset, and an evaluation metric can autonomously discover better hyperparameter configurations through iterative experimentation — the same way a human ML engineer would.

The agent never modifies `train.py`. It only modifies configuration, reads metrics, and decides what to try next.

> The project is judged on AI-driven experimentation, reflection, decision making, and optimization — not on building the most sophisticated vision model.

---

## 2. Two Workflows

### Workflow 1 — Human Data Prep + Agent Trains (`run_human_agent.py`)

```
Human:   raw data ──[preprocessing]──► data/ (train/val/test structure)
                                             │
                              training_config.yaml (baseline)
                                             │
                    ┌────────────────────────▼──────────────────────────┐
                    │                  AGENTIC LOOP                      │
                    │                                                    │
                    │  init_iter ──► train.py ──► evaluate               │
                    │      ▲                          │                  │
                    │      │                     (continue?)             │
                    │      │                          │                  │
                    │   improve ◄── notes ◄───────────┘                  │
                    └────────────────────────────────────────────────────┘
```

Human handles all data decisions: class selection, stratified splitting, class weights, directory structure. Agent's job is purely optimization — receive a working pipeline, improve macro F1 through config changes.

Edit `training_config.yaml` → **WORKFLOW 1** section (`paths.*`).

### Workflow 2 — End-to-End Autonomous Agent (`run_full_agent.py`)

```
Agent:   raw_data/articles.csv + raw_data/images/
              │
         [data_prep_agent.py]
         LLM decides: which classes, how many, how to split, class weights
         Writes: data_prep_notes.md (dataset analysis + quality report)
              │
         data/auto/  ←  agent-created structure
              │
         Same agentic loop as Workflow 1
         train → evaluate → notes → improve → repeat
```

Agent receives only raw CSV + images. `data_prep_agent.py` handles everything data-related; then the identical training loop runs.

Edit `training_config.yaml` → **WORKFLOW 2** section (`data_prep.*`). Optional: set `data_prep.instructions` to guide class selection.

---

## 3. Single Config File

Both workflows share `training_config.yaml`, divided into three sections:

```yaml
# ══ SHARED ══
agent:          # max_iterations, target_f1, llm_model
model:          # backbone, image_size, dropout, checkpoint
training:       # epochs, batch_size, early_stopping, mixed_precision
optimizer:      # type, lr, weight_decay
scheduler:      # type, min_lr, step_size, gamma
loss:           # type, label_smoothing, focal_gamma
sampler:        # use_weighted
augmentations:  # mixup, cutmix, randaugment, color_jitter, etc.

# ══ WORKFLOW 1 ══
paths:
  data_dir, class_mapping, class_weights, experiment_dir

# ══ WORKFLOW 2 ══
data_prep:
  raw_data_dir, max_train_per_class, instructions
```

---

## 4. Dataset

Works with any image classification dataset that has:
- A CSV with a label column and an ID column
- An `images/` directory with files named `{id}.jpg`

Default columns: `label_col="product_group_name"`, `id_col="article_id"`. Override in `prepare_data()` for other schemas.

**Split:** stratified 80/10/10 train/val/test (no leakage).  
**Imbalance:** inverse-frequency class weights applied to `WeightedCrossEntropy` and `WeightedRandomSampler`.  
**Capping:** `max_train_per_class` caps training set; val/test are capped at `max_train_per_class // 8` to preserve the 80/10/10 ratio.

### Data Directories

```
data/
├── sample/             # prebuilt sample dataset (500 train / ~63 val / ~63 test per class)
├── class_weights.json
└── class_mapping.json

data/auto/              # Workflow 2 output (agent-created)
├── train/val/test/<class>/
├── class_mapping_auto.json
├── class_weights_auto.json
└── data_prep_notes.md  # LLM-written dataset analysis (quality, imbalance, decisions)
```

---

## 5. Evaluation Metric — Macro F1

### What is F1?

F1 score is the harmonic mean of **precision** and **recall** for a single class:

```
Precision = TP / (TP + FP)   ← of all predictions for this class, how many were right?
Recall    = TP / (TP + FN)   ← of all actual items of this class, how many did we find?

F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

F1 ranges from 0 (worst) to 1 (perfect). It penalizes both false positives and false negatives equally — a model can't game it by only predicting the easy class or by predicting everything as one class.

### Why "Macro"?

With multiple classes, you can aggregate F1 in different ways:

| Aggregation | How | Effect |
|---|---|---|
| **Micro F1** | Pool all TPs/FPs/FNs across classes, then compute one F1 | Dominated by the largest class — big class gets more votes |
| **Macro F1** | Compute F1 per class, then average (unweighted) | Every class counts equally regardless of size |
| **Weighted F1** | Compute F1 per class, then average weighted by class size | Similar to micro — large classes still dominate |

**Macro F1 is the right metric for imbalanced classification.**

Example: 3 classes — Shoes (5000 imgs), Accessories (500 imgs), Underwear (100 imgs).

- A model that perfectly classifies Shoes but ignores Underwear gets ~80% accuracy.
- Macro F1 gives equal weight to Underwear's F1 — if the model never predicts Underwear, Underwear F1 = 0, which drags macro F1 down severely.
- This forces the agent to optimize for every class, not just the majority.

### How the Agent Uses It

- **Target:** `agent.target_f1` in config — agent stops when test macro F1 exceeds this.
- **Best checkpoint:** `train.py` saves `best_model.pth` based on *validation* macro F1.
- **Per-class breakdown:** `metrics.json` includes per-class F1 so the agent can identify which classes underperform and target specific fixes (e.g., more augmentation, class weight adjustment).
- **Plateau detection:** if macro F1 improves by < 0.005 for 2 consecutive runs, the agent switches to bold-change mode.

---

## 6. Training Pipeline (`train.py`)

Config-driven, fixed code. Agent modifies only the YAML — never the script.

**Model:** `timm.create_model(backbone, pretrained=True, num_classes=N, drop_rate=dropout)`  
Any timm backbone is supported. Warm-starting via `model.checkpoint`.

**Transforms (applied online, not baked into images):**

| Split | Transforms |
|---|---|
| Train | Resize → RandomResizedCrop → RandomHorizontalFlip → ColorJitter/RandAugment/AutoAugment → ToTensor → Normalize |
| Val/Test | Resize → CenterCrop → ToTensor → Normalize |

Advanced augmentations (MixUp, CutMix, RandAugment, RandomErasing) start OFF — reserved for agent to enable.

**Loss:** `weighted_ce` (CrossEntropy + class weights + label smoothing) or `focal`.  
**Optimizer:** AdamW / Adam / SGD.  
**Scheduler:** CosineAnnealingLR / StepLR / OneCycleLR / ReduceLROnPlateau.  
**Mixed precision:** `torch.amp.autocast` — auto-disabled on CPU.

**Live progress:** tqdm batch-level bar with running loss/acc; each epoch prints loss, val F1, LR, epoch time, and ETA. New best val F1 marked with ★.

### Output per Run

Each run writes to `experiments/<session_dir>/run_N/`:

```
run_N/
├── config.yaml       # exact config snapshot — any run is fully reproducible
├── best_model.pth    # best checkpoint (by val macro F1)
├── metrics.json      # full history + test metrics + confusion matrix
├── notes.md          # LLM-written analysis (changes made, results, next steps)
└── tensorboard/
```

`metrics.json` schema:
```json
{
  "experiment_name": "agent_run_3",
  "backbone": "convnext_base.fb_in22k_ft_in1k",
  "epochs_trained": 15,
  "best_val_macro_f1": 0.743,
  "test": {
    "macro_f1": 0.738,
    "accuracy": 0.751,
    "per_class": { "Shoes": { "f1-score": 0.81, ... }, ... },
    "confusion_matrix": [[...], ...]
  },
  "history": [{ "epoch": 1, "train_loss": ..., "val_macro_f1": ... }, ...]
}
```

---

## 7. Agentic Loop (`agents/training_agent.py`)

Built with **LangGraph StateGraph**. LLM configurable via `agent.llm_model` in config.

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

### State (`TrainingState`)

| Field | Type | Purpose |
|---|---|---|
| `session_dir` | str | timestamped dir for this agent session (`experiments/YYYYMMDD_HHMMSS`) |
| `workflow` | str | `"human+agent"` or `"full_agent"` — recorded in master log |
| `llm_model` | str | LLM used for all agent calls |
| `run_num` | int | current iteration counter |
| `current_config` | dict | config for the next run |
| `last_diff` | dict | changes applied to reach current_config |
| `last_metrics` | dict | metrics.json from last run |
| `notes_history` | list | rolling last-3 notes (LLM context window) |
| `experiment_log` | list | all runs in this session |
| `best_macro_f1` | float | best test macro F1 seen |
| `best_checkpoint_path` | str | path to best_model.pth from best run |
| `plateau_count` | int | consecutive runs without +0.005 F1 gain |
| `max_iterations` / `target_f1` | int/float | stop conditions |

### Node Details

**`init_iter`** — increments `run_num`, creates `<session_dir>/run_N/`, writes config snapshot.

**`run_train`** — `subprocess.Popen(["python", "-u", "train.py", ...])` with stdout streamed live. Stderr (tqdm) goes directly to terminal. On non-zero exit: sets `done=True` with error tail.

**`evaluate`** — reads `metrics.json`, detects plateau, appends to session log + master log (`experiments/master_log.json`). Routes to END if target reached or max iterations hit.

**`generate_notes`** — LLM call with `NOTES_PROMPT`. Writes `notes.md`. Maintains rolling 3-note history.

**`improve`** — LLM call with `IMPROVE_PROMPT`. Returns JSON diff. `_apply_diff()` merges via dot-notation. On plateau ≥ 2: prompt pushes bolder changes (backbone swap, MixUp/CutMix, different loss).

---

## 8. Experiment Logs

### Session log — `<session_dir>/experiment_log.json`

All runs within one agent session. Run sequentially by the agent.

### Master log — `experiments/master_log.json`

Accumulates every run across all sessions ever. Append-only. Use this to compare results across experiments and find the best checkpoint.

```json
{
  "session_dir": "experiments/20240709_143022",
  "timestamp": "2024-07-09 14:30:22",
  "workflow": "human+agent",
  "run": 1,
  "backbone": "convnext_base.fb_in22k_ft_in1k",
  "macro_f1": 0.7234,
  "accuracy": 0.7891,
  "val_macro_f1": 0.7456,
  "epochs_trained": 15,
  "target_f1": 0.75,
  "gap_to_target": -0.0266,
  "is_best_in_session": true,
  "diff": { "optimizer.lr": 0.0001, "augmentations.mixup": true },
  "checkpoint": "experiments/20240709_143022/run_1/best_model.pth",
  "data": {
    "data_dir": "data/sample",
    "classes": ["Accessories", "Garment Lower body", "Garment Upper body", "Shoes"],
    "split_counts": { "train": {...}, "val": {...}, "test": {...} },
    "class_weights": { "Accessories": 1.24, "Shoes": 0.91 }
  }
}
```

---

## 9. Config-Diff Design

The agent returns a JSON diff of config changes:

```json
{ "optimizer.lr": 0.00005, "augmentations.mixup": true, "augmentations.mixup_alpha": 0.4 }
```

Applied via dot-notation traversal to `current_config`. Each run's exact config is snapshotted in `config.yaml` — any run is fully reproducible.

**Advantages:**
- **Auditable** — every change logged; the diff IS the hypothesis
- **Safe** — LLM cannot introduce code bugs, only hyperparameter changes
- **Reversible** — rerun any experiment by loading its `config.yaml`

### Warm-Start Checkpointing

When backbone stays the same, agent can warm-start:
```json
{ "model.checkpoint": "experiments/.../run_2/best_model.pth", "optimizer.lr": 0.00005 }
```

`train.py` loads the checkpoint before training. If architecture changed (incompatible state dict), falls back to ImageNet pretrained with a warning.

### Plateau Detection

`plateau_count` increments when improvement < 0.005. When ≥ 2: prompt switches to bold-change mode — different backbone, MixUp/CutMix, different loss function.

---

## 10. LLM Prompts (`agents/prompts.py`)

### `IMPROVE_PROMPT`

**Input:** full config YAML + run metrics + per-class F1 + last 3 notes + plateau warning or checkpoint hint + enumerated config levers with valid ranges.  
**Output:** JSON with 2–5 config changes. LLM is grounded in available levers — cannot invent keys `train.py` doesn't read.

### `NOTES_PROMPT`

**Input:** run metrics vs target, per-class F1, config diff applied this run.  
**Output:** 3-section markdown:
- `## Changes Made` — what changed and why
- `## Results Analysis` — what worked, what failed, confusion patterns
- `## Further Improvements` — 2–3 specific next steps

Notes serve dual purpose: human-readable audit trail + LLM context for the improve step.

---

## 11. Data Prep Agent (`agents/data_prep_agent.py`) — Workflow 2 only

Invoked by `run_full_agent.py` before the training loop.

1. Loads CSV, counts all classes and total items
2. LLM selects 3–6 visually distinct classes with ≥500 images — respects `data_prep.instructions` if set
3. Filters to selected classes, tracks missing images
4. Stratified 80/10/10 split; caps val/test at `max_train_per_class // 8` to preserve ratio
5. Copies images to `data/auto/{train,val,test}/{class}/`
6. Computes inverse-frequency class weights
7. LLM writes `data_prep_notes.md` — dataset overview, data quality flags (imbalance, missing images), preparation decisions
8. Returns paths dict → patched into training config

---

## 12. Files Reference

| File | Purpose |
|---|---|
| `train.py` | Config-driven training pipeline. Fixed — agent never modifies. |
| `training_config.yaml` | Single config for both workflows. Three labelled sections. |
| `run_human_agent.py` | Workflow 1 entry point. |
| `run_full_agent.py` | Workflow 2 entry point (autonomous data prep + training). |
| `agents/training_agent.py` | LangGraph agent — full loop logic. |
| `agents/data_prep_agent.py` | LLM-driven data prep for Workflow 2. |
| `agents/prompts.py` | `IMPROVE_PROMPT` and `NOTES_PROMPT` templates. |
| `utils/llm_api.py` | OpenAI chat wrapper. Model configurable via `agent.llm_model`. |
| `utils/create_sample_data.py` | Creates `data/sample/` from raw data. |
| `data/` | Agent-facing data root. |
| `experiments/master_log.json` | Cross-session master record — all runs ever. |
| `experiments/<session>/` | One timestamped dir per agent run session. |

---

## 13. Design Decisions

**Why config-diff and not script rewriting?**  
Config changes cover 95% of the optimization space for image classification. Rewriting `train.py` risks code bugs that obscure the F1 signal. Config-diff keeps the loop clean and auditable.

**Why configurable LLM model?**  
`gpt-4o-mini` is fast and cheap for the improve step (pattern-matching on metrics, returning small JSON). `gpt-4o` is better for the notes and data prep analysis. User can tune cost vs quality via `agent.llm_model`.

**Why timestamped session directories?**  
Multiple agent runs would overwrite `run_1/`, `run_2/`, etc. Timestamped dirs (`20240709_143022/`) make every session collision-free and self-contained. `master_log.json` provides the cross-session view.

**Why rolling 3-note memory?**  
Enough context to avoid repeating failed approaches, short enough to stay within token budget. Fixed 3-section note structure makes LLM extraction reliable.

**Why not modify images during preprocessing?**  
Training pipeline owns all augmentations. This gives the agent full control over augmentation as an optimization lever — it cannot experiment with what's already baked into the images.

**Why stream subprocess stdout instead of capturing?**  
`capture_output=True` hides all training output until the run finishes. Streaming via `Popen` with `-u` (unbuffered) shows the tqdm batch bar and epoch lines in real time — user sees progress and ETA throughout training.
