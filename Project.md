# Project.md — AI Circuit: Autonomous ML Engineer
## Complete Technical Reference & Jury Q&A

---

## Table of Contents

1. [Project Overview & Hypothesis](#1-project-overview--hypothesis)
2. [System Architecture](#2-system-architecture)
3. [Two Workflows](#3-two-workflows)
4. [Agentic Loop — LangGraph StateGraph](#4-agentic-loop--langgraph-stategraph)
5. [LLM Integration & Prompt Design](#5-llm-integration--prompt-design)
6. [Training Pipeline — train.py](#6-training-pipeline--trainpy)
7. [Data Preparation Pipeline](#7-data-preparation-pipeline)
8. [Full Agent: Data Prep Agent](#8-full-agent-data-prep-agent)
9. [Evaluation — evaluate.py](#9-evaluation--evaluatepy)
10. [Experiment Tracking & Logs](#10-experiment-tracking--logs)
11. [Dashboard — Streamlit](#11-dashboard--streamlit)
12. [Configuration System](#12-configuration-system)
13. [Code File Reference](#13-code-file-reference)
14. [Design Decisions & Tradeoffs](#14-design-decisions--tradeoffs)
15. [Dataset Details — H&M Fashion](#15-dataset-details--hm-fashion)
16. [Metric: Macro F1 Explained](#16-metric-macro-f1-explained)
17. [Supported Model Backbones](#17-supported-model-backbones)
18. [Security & Safety Constraints](#18-security--safety-constraints)
19. [Jury Q&A](#19-jury-qa)

---

## 1. Project Overview & Hypothesis

**AI Circuit** uses image classification as a benchmark to demonstrate an **autonomous AI ML engineering loop**. The classifier is the substrate; the LLM-powered agent is the artifact.

**Central hypothesis:** an AI agent given a training framework, a dataset, and an evaluation metric can autonomously discover better hyperparameter configurations through iterative experimentation — the same way a human ML engineer would.

### What the agent does

- Reads metric outputs (macro F1, per-class F1, confusion matrix, loss curve)
- Reasons about what configurations could improve performance
- Proposes targeted changes (optimizer, augmentation, architecture)
- Applies them and trains a new run
- Reflects on what worked and why
- Repeats until a target is reached or budget is exhausted

### What the agent does NOT do

- Write or modify Python code
- Access the internet or training data directly
- Hallucinate config keys — grounded in an enumerated list of valid keys with valid ranges

> The project is judged on **AI-driven experimentation, reflection, decision making** — not on achieving the highest possible vision model accuracy.

---

## 2. System Architecture

```
+-------------------------------------------------------------------------+
|                          AI CIRCUIT SYSTEM                              |
|                                                                         |
|  +----------------------------------------------------------------------+
|  |                     FULL AGENT WORKFLOW                              |
|  |  raw_data/ (CSV + images)                                            |
|  |       |                                                              |
|  |  [data_prep_agent.py]                                                |
|  |   LLM selects classes -> analyzes stats -> LLM picks pipeline        |
|  |   -> run_pipeline() -> data/auto/ (train/val/test splits)            |
|  +-------+--------------------------------------------------------------+
|          |
|  +-------v--------------------------------------------------------------+
|  |               SHARED: AGENTIC TRAINING LOOP                          |
|  |                (agents/training_agent.py — LangGraph)                |
|  |                                                                      |
|  |   init_iter --> run_train --> evaluate --+(done?)--> END             |
|  |       ^                                  |                           |
|  |       |                             (continue)                       |
|  |       |         improve <-- generate_notes <----+                    |
|  |       +-----------(LLM JSON diff)                                    |
|  +-------+--------------------------------------------------------------+
|          |
|  +-------v--------------------------------------------------------------+
|  |                  TRAINING ENGINE (train.py)                          |
|  |  timm backbone -> WeightedCrossEntropy/Focal -> AdamW/SGD            |
|  |  CosineScheduler -> MixUp/CutMix/RandAugment -> TensorBoard          |
|  |  -> best_model.pth + metrics.json + tensorboard/                     |
|  +----------------------------------------------------------------------+
|                                                                         |
|  +---------------------------+  +-------------------------------------+ |
|  |   EXPERIMENT TRACKING     |  |         DASHBOARD                   | |
|  |   master_log.json         |  |  streamlit run reports/dashboard.py | |
|  |   experiment_log.json     |  |  F1 trend, per-class bars, notes    | |
|  |   run_N/metrics.json      |  |  cross-session comparison           | |
|  |   run_N/notes.md          |  |                                     | |
|  +---------------------------+  +-------------------------------------+ |
+-------------------------------------------------------------------------+
```

### Component Responsibilities

| Component | File | Role |
|---|---|---|
| **Agentic loop** | `agents/training_agent.py` | LangGraph StateGraph; orchestrates all nodes |
| **Data prep agent** | `agents/data_prep_agent.py` | Full Agent only: LLM-driven class selection + dataset pipeline |
| **Prompts** | `agents/prompts.py` | All LLM prompt strings: IMPROVE_PROMPT, NOTES_PROMPT, CLASS_DECISION_PROMPT |
| **Training engine** | `train.py` | Config-driven PyTorch training; never modified by agent |
| **Evaluator** | `evaluate.py` | Loads checkpoint, runs inference, returns metrics dict |
| **LLM wrapper** | `utils/llm_api.py` | Thin OpenAI API wrapper, temperature=0.3 |
| **Data prep (manual)** | `utils/data_prep.py` | 11-step dataset pipeline for Human+Agent workflow |
| **Data prep (pipeline)** | `utils/data_prep_pipeline.py` | Configurable step-based pipeline for Full Agent |
| **Dashboard** | `reports/dashboard.py` | Streamlit app for experiment visualization |
| **Session compare** | `utils/compare_experiments.py` | CLI diff between two session directories |

---

## 3. Two Workflows

### Human+Agent — Human Data Prep + Agent Trains (`run_human_agent.py`)

```
Human role:
  - Run utils/data_prep.py to prepare dataset (11-step pipeline)
  - Set paths.data_dir in training_config.yaml
  - Optionally set agent.initial_checkpoint for warm-start

Agent role:
  - Train -> evaluate -> write notes -> propose config changes
  - Repeat max_iterations times or until target_f1 is reached
```

Human handles all data decisions: class selection, stratified splitting, class weights. Agent's job is purely optimization — receive a working pipeline, improve macro F1 through config changes.

**Entry point (`run_human_agent.py`):**
```python
cfg = yaml.safe_load(open(args.config))
run(
    config_path=args.config,
    max_iterations=cfg["agent"]["max_iterations"],
    target_f1=cfg["agent"]["target_f1"],
    workflow="human+agent",
)
```

### Full Agent — End-to-End Autonomous (`run_full_agent.py`)

```
Human role:
  - Set data_prep.raw_data_dir in config
  - Optionally set data_prep.force_classes or instructions

Agent role:
  Step 1: data_prep_agent.prepare_data()
    - LLM selects classes from articles.csv label distribution
    - Analyzes data stats (counts, missing images, imbalance ratio)
    - LLM picks pipeline steps (validate, dedup, resize, etc.)
    - run_pipeline() copies + transforms images to data/auto/
  Step 2: Agentic training loop (same as Human+Agent)
    - Agent can re-trigger data prep if it decides more data is needed
      (changes data_prep.max_train_per_class or force_classes)
```

**Entry point (`run_full_agent.py`):**
```python
data_paths = prepare_data(raw_data_dir=..., force_classes=..., ...)
cfg["paths"].update(data_paths)   # patch config in-memory (YAML never overwritten)
run_training(config_path=cfg, workflow="full_agent", data_prep_config=dp_cfg)
```

---

## 4. Agentic Loop — LangGraph StateGraph

### Graph Definition

```python
graph = StateGraph(TrainingState)
graph.add_node("init_iter",       init_iter)
graph.add_node("run_train",       run_train)
graph.add_node("evaluate",        evaluate)
graph.add_node("generate_notes",  generate_notes)
graph.add_node("improve",         improve)

graph.set_entry_point("init_iter")
graph.add_edge("init_iter",      "run_train")
graph.add_edge("run_train",      "evaluate")
graph.add_conditional_edges(
    "evaluate",
    lambda s: END if s["done"] else "generate_notes"
)
graph.add_edge("generate_notes", "improve")
graph.add_edge("improve",        "init_iter")
```

### State — `TrainingState` TypedDict

| Field | Type | Description |
|---|---|---|
| `session_dir` | str | Timestamped session path: `experiments/YYYYMMDD_HHMMSS_<name>` |
| `workflow` | str | `"human+agent"` or `"full_agent"` — recorded in every log entry |
| `llm_model` | str | `"gpt-4o-mini"` / `"gpt-4o"` — configurable via `agent.llm_model` |
| `run_num` | int | Current run counter (incremented by `init_iter`) |
| `base_config` | dict | Original config — never mutated |
| `current_config` | dict | Working config for the next run |
| `last_diff` | dict | The JSON diff that produced `current_config` |
| `last_metrics` | dict | `metrics.json` from last completed run |
| `notes_history` | list | Rolling last-3 LLM notes (context for `improve` node) |
| `experiment_log` | list | All run summaries in this session |
| `best_macro_f1` | float | Best test macro F1 seen in this session |
| `best_checkpoint_path` | str | Path to `best_model.pth` from best run |
| `plateau_count` | int | Consecutive runs without ≥ 0.005 F1 improvement |
| `max_iterations` | int | Run budget (hard stop) |
| `target_f1` | float | Stop early when macro F1 ≥ this value |
| `done` | bool | Set by `evaluate` when stopping condition is met |
| `error` | str | Set if `train.py` exited non-zero |
| `data_prep_config` | dict | Full Agent only — current data prep settings |
| `data_prep_output_dir` | str | Where `prepare_data` writes files |
| `needs_data_prep` | bool | Set by `improve` when data config changed |

### Node Details

#### `init_iter`
- Re-runs `prepare_data()` if `needs_data_prep=True` (Full Agent can request more data mid-session)
- Increments `run_num`, creates `<session_dir>/run_N/`
- Writes the full current config to `run_N/config.yaml` — every run is independently reproducible
- Prints diff from previous run for full transparency

#### `run_train`
- `subprocess.Popen([sys.executable, "-u", "train.py", "--config", "run_N/config.yaml"])`
- `-u` = unbuffered: tqdm batch bar + epoch lines stream to terminal in real time
- `stderr=None` = tqdm progress goes directly to terminal (not captured)
- Captures stdout for error tail; on non-zero exit, sets `done=True` with last 40 lines

#### `evaluate`
- Reads `run_N/metrics.json` (written by `train.py`)
- Computes plateau: F1 improved by < 0.005 → increment `plateau_count`
- Appends entry to `<session_dir>/experiment_log.json`
- Appends entry to `experiments/master_log.json` (global append-only log)
- Sets `done=True` if `macro_f1 >= target_f1` or `run_num >= max_iterations`

#### `generate_notes`
- Calls LLM with `NOTES_PROMPT` template
- Output: 3-section markdown — Changes Made, Results Analysis, Next Steps
- Written to `run_N/notes.md`
- Maintains rolling `notes_history` (last 3 notes) — fed to `improve` as context

#### `improve`
- Calls LLM with `IMPROVE_PROMPT` template
- LLM receives: full config, metrics, per-class F1, notes history, all valid config keys with ranges
- LLM returns: JSON dict of 2–5 dot-notation config changes
- `_apply_diff(current_config, diff)` applies changes via deep-copy + dot-notation traversal
- On plateau ≥ 2: prompt includes "bold change" directive (different backbone, augmentation strategies)
- Sets `needs_data_prep=True` if LLM changed `data_prep.*` keys (Full Agent only)

---

## 5. LLM Integration & Prompt Design

### LLM API (`utils/llm_api.py`)

```python
def chat(prompt: str, model: str = "gpt-4o-mini") -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,   # low: deterministic + grounded; some exploration
    )
    return resp.choices[0].message.content.strip()
```

Temperature 0.3: low enough to avoid hallucinating keys, high enough to explore different strategies across runs.

### `IMPROVE_PROMPT` — Hyperparameter Optimization

**Inputs injected into template:**
- Number of classes, class names, target F1, gap to target
- Full current YAML config
- Run metrics: macro F1, accuracy, val loss, epochs trained
- Per-class F1 table
- Last 3 experiment notes (rolling context window)
- **Enumerated list of all valid config keys** with valid ranges and constraints
- Optional plateau warning + bold-change directive
- Optional warm-start checkpoint hint

**Output requirement:** ONLY valid JSON, no prose:
```json
{"optimizer.lr": 0.00005, "augmentations.mixup": true, "augmentations.mixup_alpha": 0.4}
```

**Grounding strategy:** The prompt explicitly enumerates every key the agent can change with valid ranges and conditional dependencies (e.g., `scheduler.warmup_epochs` REQUIRED for transformer backbones). This prevents hallucination of non-existent config keys.

**Plateau mode (when `plateau_count >= 2`):**
```
PLATEAU DETECTED — F1 has not improved in the last 2 runs.
Try bolder changes: different backbone, MixUp or CutMix, different loss function,
or increase max_train_per_class.
```

### `NOTES_PROMPT` — Experiment Analysis

**Output:** 3-section markdown:
- `## Changes Made` — each config change + reasoning behind it
- `## Results Analysis` — what worked, what failed, which classes underperformed
- `## Next Steps` — 2–3 specific, actionable suggestions for the next run

Notes serve two purposes:
1. **Human-readable audit trail** — every decision is documented and saved to `notes.md`
2. **LLM context for `improve`** — rolling 3-note window prevents repeating failed approaches

### `CLASS_DECISION_PROMPT` — Class Selection (Full Agent only)

Receives label distribution from `articles.csv`. LLM selects 3–6 visually distinct classes with:
- Sufficient images (≥500 per class after filtering)
- High visual distinctiveness (makes the task learnable)
- Reasonable balance (avoids severe class imbalance)

Returns: `{"classes": [...], "rationale": "..."}`

### `PIPELINE_CONFIG_PROMPT` — Data Pipeline Config (Full Agent only)

Receives data stats (counts, missing%, imbalance ratio). LLM decides which preprocessing steps to enable and with what parameters. Returns JSON matching `DEFAULT_PIPELINE_CONFIG` schema.

---

## 6. Training Pipeline — `train.py`

Config-driven, fixed code. The agent only changes the YAML — **never `train.py`**.

### Model Construction

```python
import timm
model = timm.create_model(
    backbone,           # e.g., "efficientnet_b2", "convnext_tiny"
    pretrained=True,    # ImageNet weights
    num_classes=N,      # number of target classes
    drop_rate=dropout,  # configurable dropout rate
)
```

If `model.checkpoint` is set, loads state dict before training (warm-start). Falls back to ImageNet pretrained with a warning if architecture changed (incompatible state dict).

### Transform Pipeline

Applied **online** (not baked into stored images) — gives the agent full control over augmentation as an optimization lever.

| Split | Transform Chain |
|---|---|
| **Train** | Resize → RandomResizedCrop(sz) → RandomHorizontalFlip → [ColorJitter OR RandAugment OR AutoAugment] → ToTensor → Normalize → [RandomErasing] |
| **Val/Test** | Resize → CenterCrop(sz) → ToTensor → Normalize |

- **Normalization:** ImageNet mean/std `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` when `pretrained=True`; falls back to dataset-computed stats when `pretrained=False`
- **Image size scaling:** changing `model.image_size` automatically scales all crop/resize operations
- **Augmentation exclusivity:** `autoaugment` and `randaugment` are mutually exclusive (prompt enforces this)

### MixUp and CutMix (Batch-Level)

```python
def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    return lam*x + (1-lam)*x[idx], y, y[idx], lam

def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    # cut random rectangular region; paste from another image in the batch
    ...
```

Mixed loss: `lam * loss(pred, y_a) + (1-lam) * loss(pred, y_b)`

### Loss Functions

| Config | Implementation | When to use |
|---|---|---|
| `weighted_ce` | `nn.CrossEntropyLoss(weight=class_weights, label_smoothing=ls)` | Default; good for moderately imbalanced data |
| `focal` | `FocalLoss(gamma=focal_gamma)` | Down-weights easy examples; useful when some classes are systematically harder |

### Optimizer & Scheduler Options

| Config | Implementation |
|---|---|
| `optimizer.type: adamw` | `torch.optim.AdamW(lr, weight_decay)` |
| `optimizer.type: sgd` | `torch.optim.SGD(lr, momentum, weight_decay)` |
| `scheduler.type: cosine` | `CosineAnnealingLR(T_max=epochs, eta_min=min_lr)` + optional linear warmup |
| `scheduler.type: onecycle` | `OneCycleLR(max_lr=lr, total_steps=...)` |
| `scheduler.type: plateau` | `ReduceLROnPlateau(factor=gamma, patience=step_size)` |
| `scheduler.type: step` | `StepLR(step_size, gamma)` |

`warmup_epochs` applies a linear LR ramp from `min_lr` to `lr` over the first N epochs — **required for transformer backbones** (Swin, ViT) to avoid early instability.

### Weighted Random Sampler

When `sampler.use_weighted=true`: each image's sampling probability = `class_weight[class]`. Combined with `WeightedCrossEntropy`, this doubly emphasizes minority classes.

### Mixed Precision

`torch.amp.autocast(device_type="cuda") + GradScaler`. ~2× speedup on GPU. Auto-disabled when `mixed_precision=false` (CPU training — no benefit, adds overhead).

### Early Stopping

Stops training if val macro F1 does not improve by `early_stopping_min_delta` for `early_stopping_patience` epochs. `best_model.pth` always saved at epoch with highest val macro F1.

### Output per Run — `experiments/<session>/run_N/`

```
config.yaml       — full config snapshot; reproduces this run with:
                    python train.py --config experiments/<session>/run_N/config.yaml
best_model.pth    — {"state_dict": ..., "backbone": "efficientnet_b2"}
metrics.json      — full metrics schema (see below)
tensorboard/      — TensorBoard event files
```

### `metrics.json` Schema

```json
{
  "experiment_name": "agent_run_2",
  "backbone": "efficientnet_b2",
  "epochs_trained": 10,
  "best_val_macro_f1": 0.763,
  "test": {
    "macro_f1": 0.781,
    "accuracy": 0.795,
    "weighted_f1": 0.790,
    "per_class": {
      "Shoes": {"f1-score": 0.85, "precision": 0.82, "recall": 0.88, "support": 120},
      "Accessories": {"f1-score": 0.71, ...}
    },
    "confusion_matrix": [[120, 5, 3, 2], ...]
  },
  "history": [
    {"epoch": 1, "train_loss": 1.42, "val_loss": 1.18, "val_macro_f1": 0.61, "lr": 0.0003}
  ]
}
```

---

## 7. Data Preparation Pipeline (`utils/data_prep.py`)

Manual data prep for the Human+Agent workflow. Run once before starting agent sessions.

### 11-Step Pipeline

| # | Step | Key Logic | Why It Matters |
|---|---|---|---|
| 1 | Load CSV | `pd.read_csv("articles.csv")`, filter to target classes | Removes irrelevant categories from label space |
| 2 | Label audit | Count by `product_type_name`, `garment_group_name` per class | H&M "Garment Upper body" includes swimwear, socks, t-shirts — noisy labels cap F1 ceiling |
| 3 | Resolve paths | `article_id[:3]/article_id.jpg`, drop rows with no file | No FileNotFoundError mid-epoch |
| 4 | Validate + min size | `PIL.Image.verify()` + reject `min(w,h) < 128px` | Corrupt files crash training; tiny thumbnails add noise |
| 5 | Near-dedup | dhash (8×8 difference hash), hamming distance ≤ 4 = duplicate | Catches reposted images with watermarks/crops; prevents train-to-test leakage |
| 6 | Cap per class | `random.sample(items, min(len, max_per_class))` | Balances class sizes; controls training time |
| 7 | Product-level split | Extract `product_id = article_id[:7]`, split by product groups | Same product in multiple angles inflates test F1 under row-level split — product split ensures zero leakage |
| 8 | Resize + pad | Aspect-ratio resize to fit 256×256, white-pad to square | H&M images are portrait (1166×1750); without padding, RandomResizedCrop sees <40% of image area |
| 9 | Class mapping + weights | `{0: "Shoes", 1: "Accessories", ...}` + inverse-frequency weights | Required by training engine for WeightedCrossEntropy and WeightedRandomSampler |
| 10 | Mean/std | Sample 2000 train images, compute per-channel stats | Saved as reference; training uses ImageNet stats which generalize well to fashion |
| 11 | Summary | Print per-class counts for each split + weights | Sanity check before committing to a long training run |

### Key Constants

```python
MIN_IMG_SIZE         = 128    # drop images smaller than this (pixels)
DEDUP_HAMMING_THRESH = 4      # 0 = exact match only; 4 catches near-dupes
RESIZE_PAD_SIZE      = 256    # store pre-padded at 256×256
MAX_PER_CLASS        = 500    # cap training samples per class
```

---

## 8. Full Agent: Data Prep Agent (`agents/data_prep_agent.py`)

Used only by the Full Agent workflow. Wraps `utils/data_prep_pipeline.py` with LLM decision-making at each step.

### Flow

```
1. Load articles.csv -> print full label distribution
2. Class selection:
   - If force_classes set: use them (skip LLM call)
   - Else: CLASS_DECISION_PROMPT -> LLM returns {"classes": [...], "rationale": "..."}
3. Filter DataFrame to selected classes
4. _analyze_data_stats(): count items, missing images, imbalance ratio (fast; file checks only)
5. _decide_pipeline_config(): PIPELINE_CONFIG_PROMPT -> LLM returns pipeline JSON
                              (falls back to DEFAULT_PIPELINE_CONFIG on parse error)
6. run_pipeline(): execute enabled steps: validate -> dedup -> resize_pad -> split
7. Write data_prep_notes.md: LLM-written dataset analysis
8. Return paths dict -> patched into training config in-memory (YAML never overwritten)
```

### Configurable Pipeline (`utils/data_prep_pipeline.py`)

```python
DEFAULT_PIPELINE_CONFIG = {
    "validate_images":     {"enabled": True,  "min_size": 128},
    "dedup":               {"enabled": True,  "hamming_thresh": 4},
    "resize_pad":          {"enabled": True,  "size": 256},
    "product_level_split": {"enabled": True},
    "compute_mean_std":    {"enabled": False, "sample_n": 2000},
    "max_train_per_class": 50,
    "eval_cap_ratio":      0.125,   # val/test cap = max_train * eval_cap_ratio
}
```

LLM can enable/disable steps and adjust parameters based on data quality. Saved to `pipeline_config.json` in output directory for transparency.

### Re-triggering Data Prep Mid-Session

When `improve` node changes `data_prep.max_train_per_class` or `data_prep.force_classes`, `needs_data_prep=True` is set. `init_iter` detects this and re-runs `prepare_data()` before the next training run — the agent can expand its training set without human intervention.

---

## 9. Evaluation — `evaluate.py`

### `evaluate()` — Called During Training

```python
@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names, use_amp):
    # Forward pass over all batches
    # Returns: loss, accuracy, macro_f1, weighted_f1, per_class dict, confusion_matrix
```

Used by `train.py` at every epoch for validation metrics and once at end of training for test metrics.

### `evaluate_checkpoint()` — Standalone

```python
metrics = evaluate_checkpoint(
    checkpoint_path="experiments/.../best_model.pth",
    config_path_or_dict="training_config.yaml",
    split="test"   # or "val"
)
print(metrics["macro_f1"])
```

Loads backbone name from `ckpt["backbone"]`, creates matching timm model, loads state dict, runs evaluation. Same metrics dict schema as `train.py` writes to `metrics.json`.

### CLI

```bash
python evaluate.py \
    --checkpoint experiments/20260715/run_2/best_model.pth \
    --config training_config.yaml \
    --split test \
    --output results/eval.json
```

---

## 10. Experiment Tracking & Logs

### Session Log — `<session_dir>/experiment_log.json`

One entry per run within a single agent session. Written after each `evaluate` node.

### Master Log — `experiments/master_log.json`

**Append-only.** Accumulates every run across all sessions ever. Never overwritten — safe to run multiple agent sessions concurrently. The dashboard reads this for cross-session comparisons.

```json
{
  "session_dir": "experiments/20260715_064118_local_fa",
  "timestamp": "2026-07-15 06:41:18",
  "workflow": "full_agent",
  "run": 2,
  "backbone": "efficientnet_b2",
  "macro_f1": 0.7812,
  "val_macro_f1": 0.7634,
  "accuracy": 0.7951,
  "epochs_trained": 10,
  "target_f1": 0.9,
  "gap_to_target": -0.1188,
  "is_best_in_session": true,
  "diff": {"optimizer.lr": 0.0001, "augmentations.mixup": true},
  "checkpoint": "experiments/20260715_064118_local_fa/run_2/best_model.pth",
  "data": {
    "data_dir": "data/local_fa",
    "classes": ["Garment_Upper_body", "Garment_Lower_body", "Accessories", "Shoes"],
    "split_counts": {"train": {"Shoes": 50}, "val": {}, "test": {}},
    "class_weights": {"Accessories": 1.12, "Shoes": 0.88}
  }
}
```

### Run Directory — `experiments/<session>/run_N/`

| File | Contents |
|---|---|
| `config.yaml` | Complete config snapshot — reproduces this run with `python train.py --config ...` |
| `best_model.pth` | `{"state_dict": ..., "backbone": "efficientnet_b2"}` — saved at best val macro F1 |
| `metrics.json` | Full test metrics, per-class F1, confusion matrix, epoch-by-epoch history |
| `notes.md` | LLM-written 3-section analysis (Changes Made / Results Analysis / Next Steps) |
| `tensorboard/` | TensorBoard event files — `tensorboard --logdir experiments/` |

---

## 11. Dashboard — `reports/dashboard.py`

```bash
streamlit run reports/dashboard.py
```

Auto-refreshes every 15 seconds (Streamlit cache TTL=15s). Color scheme: Human+Agent = blue (#2a78d6), Full Agent = green (#1baf7a).

### Panels

| Panel | Data Source | What it Shows |
|---|---|---|
| **KPIs** | `master_log.json` | Total sessions, total runs, best F1 ever, latest run F1 |
| **F1 Trend** | `experiment_log.json` | Line chart: macro F1 + val F1 per run, dashed target F1 line |
| **Run History** | `experiment_log.json` | Table: backbone, macro F1, accuracy, epochs, config diff |
| **Run Detail** | `run_N/metrics.json`, `run_N/notes.md` | Per-class F1 bars, epoch loss/F1 curve, LLM notes |
| **Cross-Session** | `master_log.json` | Bar chart: best F1 per session, color-coded by workflow |

---

## 12. Configuration System

### Single File, Three Sections

`training_config.yaml` is the only interface between the human/agent and the training engine.

```yaml
# SHARED — edit regardless of workflow
agent:
  experiment_name: local_fa
  max_iterations: 3
  target_f1: 0.9
  llm_model: gpt-4o-mini
  initial_checkpoint: null     # warm-start from a saved checkpoint

model:
  backbone: mobilenetv3_small_100
  pretrained: true
  checkpoint: null             # set by agent for run-to-run warm-start
  image_size: 224
  dropout: 0.3

training:
  epochs: 10
  batch_size: 16
  num_workers: 0
  mixed_precision: false
  early_stopping_patience: 10
  max_samples_per_class: 50    # null = use all data

optimizer:
  type: adamw
  lr: 0.0003
  weight_decay: 0.01
  momentum: 0.9                # sgd only

scheduler:
  type: cosine
  min_lr: 0.000001
  warmup_epochs: 0             # REQUIRED for transformer backbones

loss:
  type: weighted_ce
  label_smoothing: 0.0
  focal_gamma: 2.0
  use_class_weights: true

sampler:
  use_weighted: true

augmentations:
  # Advanced augmentations start OFF — agent enables them to improve F1
  random_resized_crop: true
  random_horizontal_flip: true
  color_jitter: true
  random_erasing: false        # agent may enable
  mixup: false                 # agent may enable
  cutmix: false                # agent may enable
  randaugment: false           # agent may enable
  autoaugment: false           # agent may enable (mutually exclusive with randaugment)

# HUMAN+AGENT only
paths:
  data_dir: data/local_sample
  class_weights: null
  class_mapping: null

# FULL AGENT only
data_prep:
  raw_data_dir: ../HM_Data/raw_data
  max_train_per_class: 50
  instructions: null
  force_classes:
    - "Garment_Upper_body"
    - "Garment_Lower_body"
    - "Garment_Full_body"
    - "Accessories"
    - "Shoes"
```

### Dot-Notation Diff Application

```python
def _apply_diff(cfg: dict, diff: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for key, val in diff.items():
        parts = key.split(".")        # "augmentations.mixup" -> ["augmentations", "mixup"]
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return cfg
```

Every change is logged in `last_diff` and written to `experiment_log.json`. Any run can be reproduced by loading its `config.yaml`.

---

## 13. Code File Reference

### `train.py` — Training Engine

**Purpose:** Config-driven PyTorch training pipeline. Agent never modifies this file.

**Key functions:**
- `build_transforms(cfg)` → `(train_transform, val_transform)` — builds full torchvision pipeline
- `mixup(x, y, alpha)` + `cutmix(x, y, alpha)` — batch-level augmentation functions
- `main(config_path)` — loads config, builds model/data/optimizer, runs training loop, writes `metrics.json`

**DataLoader:** `torchvision.datasets.ImageFolder` (class-per-folder: `<data_dir>/<split>/<class>/<image>.jpg`)

**Checkpoint format:** `{"state_dict": model.state_dict(), "backbone": backbone_name}`

---

### `evaluate.py` — Standalone Evaluator

**Purpose:** Load any checkpoint + config, evaluate on any split, return full metrics dict.

- `evaluate(model, loader, ...)` — inference loop: returns loss, accuracy, macro_f1, per_class, confusion_matrix
- `evaluate_checkpoint(checkpoint_path, config_path_or_dict, split)` — full pipeline

---

### `agents/training_agent.py` — LangGraph Agentic Loop

**Purpose:** Implements `train → evaluate → notes → improve` cycle as a LangGraph StateGraph.

**Key helpers:**
- `_apply_diff(cfg, diff)` — dot-notation config updater (deep-copy safe)
- `_fmt_per_class(metrics)` — formats per-class F1 table for LLM prompt
- `_append_master_log(entry)` — appends single entry to global master log

**Nodes:** `init_iter`, `run_train`, `evaluate`, `generate_notes`, `improve`

**Entry:** `run(config_path, max_iterations, target_f1, workflow, ...)` — builds graph, sets initial state, executes

---

### `agents/data_prep_agent.py` — Full Agent Data Preparation

**Purpose:** LLM-driven class selection + data pipeline. Used only by `run_full_agent.py`.

- `_analyze_data_stats(...)` — fast stats: file existence checks only (no image loading)
- `_decide_pipeline_config(stats, max_train, llm_model, ...)` — LLM decides pipeline; fallback to DEFAULT
- `prepare_data(raw_data_dir, output_dir, ...)` — full pipeline entry point; returns paths dict

---

### `agents/prompts.py` — All LLM Prompts

**Purpose:** Single source of truth for all LLM prompt strings.

**Exports:** `IMPROVE_PROMPT`, `NOTES_PROMPT`, `CLASS_DECISION_PROMPT`, `DATASET_ANALYSIS_PROMPT`, `PIPELINE_CONFIG_PROMPT`, `make_plateau_section()`, `make_warmstart_section()`, `make_data_prep_section()`

---

### `utils/llm_api.py` — OpenAI Wrapper

Thin wrapper around OpenAI Chat Completions. Temperature=0.3. Reads `OPENAI_API_KEY` from `.env`.

---

### `utils/data_prep.py` — Manual Dataset Preparation

11-step data prep for Human+Agent workflow. Validate, dedup, cap, product-level split, resize+pad, class mapping/weights, mean/std.

---

### `utils/data_prep_pipeline.py` — Configurable Pipeline (Full Agent)

Modular step-based pipeline used by `data_prep_agent.py`. Each step toggled by a config dict the LLM writes. Saves `pipeline_config.json` to output directory for transparency.

---

### `utils/compare_experiments.py` — Session Comparison

CLI: `python utils/compare_experiments.py experiments/sess_A experiments/sess_B`

---

### `reports/dashboard.py` — Streamlit Dashboard

`streamlit run reports/dashboard.py` — Altair charts, reads from `experiments/`, cache TTL=15s.

---

### `run_human_agent.py` + `run_full_agent.py` — Entry Points

Minimal entry points. `run_human_agent.py` calls `training_agent.run()`. `run_full_agent.py` calls `prepare_data()` first then `training_agent.run()` with `data_prep_config`.

---

## 14. Design Decisions & Tradeoffs

**Why config-diff instead of code rewriting?**
Config changes cover 95%+ of the optimization space. Rewriting `train.py` risks code bugs that corrupt the F1 signal and non-reversible changes. Config diffs are auditable, safe, and reproducible.

**Why LangGraph?**
Typed state (`TypedDict`), explicit conditional routing, extensible node graph. A plain loop works but offers no typed state and is hard to extend safely.

**Why timestamped session directories?**
Collision-free across multiple runs. `master_log.json` provides the cross-session view.

**Why rolling 3-note memory?**
3 notes = enough context to avoid repeating failures, short enough to stay within token budget.

**Why temperature 0.3?**
Low enough for grounded JSON (not hallucinating keys), high enough to explore different strategies.

**Why not modify images during preprocessing?**
Augmentations applied online give the agent full control. Pre-augmented images remove augmentation as a lever.

**Why stream subprocess stdout?**
`capture_output=True` hides progress until done. `Popen` with `-u` shows tqdm and epoch lines in real time.

**Why macro F1?**
Imbalanced dataset. Accuracy rewards majority-class prediction. Macro F1 forces optimization across all classes.

**Why product-level split?**
Multiple images per product. Row-level split leaks the same product into train and test, inflating test F1.

**Why baseline augmentations OFF?**
Gives the agent clear headroom to improve. If all augmentations are already ON, the agent plateaus immediately.

**Why not Bayesian optimization / Optuna?**
HPO explores blindly. The LLM reads per-class F1, reasons about *why* a class underperforms, adapts based on notes, and can switch backbone entirely. LLM ML knowledge guides exploration far more efficiently for a given compute budget.

---

## 15. Dataset Details — H&M Fashion

**Source:** H&M Personalized Fashion Recommendations (Kaggle)

| File | Records | Used? |
|---|---|---|
| `articles.csv` | 105,542 articles with product metadata | YES — labels + IDs |
| `images/` | ~105k product images (.jpg) | YES — training images |
| `customers.csv` | 1.37M customers | No |
| `transactions_train.csv` | 31.8M purchases | No |

**Image naming:** `images/<article_id[:3]>/<article_id>.jpg`

**Label column:** `product_group_name` — **ID column:** `article_id`

### Class Distribution

| Class | Raw count | Visual challenge |
|---|---|---|
| Garment Upper body | ~35k | Mixed: t-shirts + sweaters + swimwear + socks |
| Garment Lower body | ~20k | Mixed: trousers + skirts + shorts |
| Garment Full body | ~8k | Dresses + jumpsuits |
| Accessories | ~12k | Bags + belts + scarves + hats |
| Shoes | ~5k | Most visually distinct |

**Data challenges the agent must handle:**
- Class imbalance (up to 7:1 ratio)
- Label noise (coarse product group names group visually dissimilar items)
- Missing images (some article IDs have no image file)
- Near-duplicate images (same product, multiple angles)

---

## 16. Metric: Macro F1 Explained

### Why Not Accuracy?

`Accuracy = (TP + TN) / Total`. For imbalanced classes, a model always predicting the majority class achieves high accuracy while providing zero value for minority classes.

Example: 80% Shoes, 20% Accessories → model always predicts "Shoes" → 80% accuracy, but Accessories F1 = 0, Macro F1 ≈ 0.4.

### F1 Calculation

```
Precision = TP / (TP + FP)   — of all predicted as class C, how many were correct?
Recall    = TP / (TP + FN)   — of all actual class C items, how many did we find?
F1        = 2 * (P * R) / (P + R)   — harmonic mean
Macro F1  = mean(F1_class_1, F1_class_2, ..., F1_class_N)   — unweighted
```

### Aggregation Comparison

| Aggregation | How | Effect |
|---|---|---|
| **Micro F1** | Pool all TP/FP/FN, compute one F1 | Dominated by the largest class |
| **Macro F1** | F1 per class, then unweighted average | Every class counts equally |
| **Weighted F1** | F1 per class, weighted by class size | Similar to micro — large classes still dominate |

### How the Agent Uses Macro F1

- **Target:** `agent.target_f1` — agent stops early when test macro F1 ≥ this value
- **Best checkpoint:** `train.py` saves `best_model.pth` at epoch with highest *validation* macro F1
- **Per-class insight:** `metrics.json` per-class breakdown lets agent target specific underperforming classes
- **Plateau detection:** improvement < 0.005 for 2 consecutive runs triggers bold-change mode

---

## 17. Supported Model Backbones

All from the `timm` library. The agent picks from this enumerated list in the prompt.

| Backbone | Params | Best for |
|---|---|---|
| `mobilenetv3_small_100` | ~2.5M | Local dev / prototyping (fastest CPU) |
| `mobilenetv3_large_100` | ~5.5M | CPU-friendly; good accuracy/speed tradeoff |
| `efficientnet_b0` | ~5M | Small datasets (<2k imgs/class) |
| `efficientnet_b2` | ~9M | Good default for medium datasets |
| `efficientnet_b4` | ~19M | High accuracy; needs 1k+ imgs/class |
| `resnet50` | ~25M | Reliable baseline; widely understood |
| `convnext_tiny` | ~28M | Modern CNN; outperforms ResNet50 |
| `convnext_small` | ~50M | Good for 2k+ imgs/class (GPU recommended) |
| `convnext_base` | ~88M | Top CNN accuracy (GPU required) |
| `swin_tiny_patch4_window7_224` | ~28M | Transformer; strong on texture/pattern |
| `swin_small_patch4_window7_224` | ~50M | Better than swin_tiny (GPU recommended) |
| `vit_base_patch16_224` | ~86M | Pure transformer; best with 5k+ imgs/class (GPU required) |

**Warm-start behavior:** when backbone stays the same, agent sets `model.checkpoint` to previous best. When backbone changes, checkpoint is set to null (incompatible architecture → ImageNet pretrained).

---

## 18. Security & Safety Constraints

- **No code execution from LLM output:** agent only applies JSON config diffs; never executes LLM-generated code
- **No arbitrary file writes:** LLM cannot specify output paths — all paths are framework-determined
- **API key in `.env`:** never committed to version control
- **Grounded prompts:** explicit enumeration of valid config keys — LLM cannot invent new keys
- **JSON-only output:** `IMPROVE_PROMPT` requires only valid JSON with no prose — minimizes injection surface
- **Subprocess isolation:** `train.py` cannot influence agent state except through `metrics.json` read after training completes

---

## 19. Jury Q&A

### "What is the main innovation here?"

The agent loop architecture: a structured LangGraph state machine that orchestrates a real PyTorch training pipeline using an LLM as the reasoning engine. The LLM acts as a knowledgeable ML engineer — reads metrics, identifies what's underperforming, makes principled changes. The framework is general enough to apply to any ML training task with a YAML config.

### "How does the agent decide what to change?"

The `improve` node sends `IMPROVE_PROMPT` to the LLM containing: full config, all run metrics, per-class F1, last 3 experiment notes, and a **complete enumeration of all valid config keys** with valid ranges and conditional constraints. The LLM outputs a JSON diff of 2–5 changes. It cannot invent keys or values outside specified ranges.

### "Why not just use Bayesian optimization or Optuna?"

Traditional HPO explores blindly. The LLM agent:
1. Reads per-class F1 and identifies which specific classes underperform
2. Reasons about *why* (e.g., "Accessories has low recall → try focal loss")
3. Adapts strategy based on accumulated notes (no repeated failed approaches)
4. Can switch backbone entirely when stuck on a plateau

LLM's pre-trained ML knowledge guides exploration far more efficiently than random/Bayesian search.

### "What if the LLM hallucinates a non-existent config key?"

`_apply_diff()` uses `dict.setdefault()` — unknown keys create new sub-dicts but `train.py` only reads explicitly declared keys. Unknown keys are silently ignored. The prompt is also carefully grounded with exact valid keys — in practice, the LLM very rarely invents keys given an explicit enumeration.

### "How do you prevent the agent from getting stuck?"

1. **Plateau detection:** improvement < 0.005 for 2 runs → `plateau_count >= 2` → bold-change mode
2. **Rolling notes history:** last 3 notes tell the LLM what failed, preventing repeated attempts
3. **Budget:** `max_iterations` provides a hard stop

### "How is any experiment fully reproducible?"

Every run writes a complete config snapshot to `run_N/config.yaml`:
```bash
python train.py --config experiments/<session>/run_N/config.yaml
```
The backbone name is also embedded in `best_model.pth` — `evaluate_checkpoint()` can load any checkpoint without needing to know the backbone upfront.

### "What is the difference between the two workflows?"

- **Human+Agent:** you prepare the dataset (class selection, split, preprocessing), agent handles optimization. Better with domain knowledge or a pre-existing dataset.
- **Full Agent:** agent handles everything from raw CSV + images. `data_prep_agent.py` uses LLM to select classes, analyze quality, decide preprocessing steps. Requires only `data_prep.raw_data_dir`.

### "Can the agent change the dataset during training?"

Yes, in Full Agent mode. If `improve` sets `data_prep.max_train_per_class: 200`, `init_iter` detects `needs_data_prep=True` and re-runs `prepare_data()` before the next run. The agent can expand its training set mid-session without human intervention.

### "What LLM models are supported?"

Any OpenAI model: `gpt-4o-mini` (default — fast/cheap), `gpt-4o` (more capable), `gpt-4-turbo`. Configured via `agent.llm_model`. Temperature fixed at 0.3.

### "How do you evaluate the agent's quality (not just the model's F1)?"

- **F1 improvement curve:** run_1 F1 vs run_N F1 — did the agent improve?
- **Decision quality:** read `notes.md` — are changes principled and well-reasoned?
- **Plateau recovery:** did the agent escape a plateau with appropriate bold changes?
- **Baseline delta:** `initial_checkpoint` (run_0) shows starting point; all subsequent runs show agent improvement
- **Dashboard F1 trend:** visualizes full trajectory across all runs

### "What are the failure modes?"

1. **LLM API failure:** raises on API error; agent sets `done=True` with error message
2. **Training crash:** `train.py` non-zero exit → agent captures last 40 stdout lines, sets `done=True`
3. **No improvement:** if target F1 is too ambitious, agent iterates until `max_iterations` and reports best checkpoint found
4. **JSON parse failure:** `_decide_pipeline_config()` falls back to `DEFAULT_PIPELINE_CONFIG` — no crash

### "What does the agent produce at the end?"

1. `experiments/master_log.json` — full record of all runs
2. `experiments/<session>/run_N/best_model.pth` — best model checkpoint
3. `experiments/<session>/run_N/notes.md` — LLM-written analysis for every run
4. `experiments/<session>/run_N/config.yaml` — exact reproducible config for every run
5. `experiments/<session>/run_N/metrics.json` — full metrics + confusion matrix
6. `data/<name>/data_prep_notes.md` (Full Agent) — LLM-written data analysis report

### "How is class imbalance handled?"

Three independent layers:
1. **`class_weights.json`:** inverse-frequency weights at data prep time (minority class weight > 1)
2. **`WeightedCrossEntropy`:** loss weights each sample by its class weight
3. **`WeightedRandomSampler`:** oversamples minority classes in each training batch

All three independently enabled/disabled and tuned by the agent via config changes.

### "Why use LangGraph instead of a plain Python loop?"

LangGraph provides typed state (`TypedDict`), explicit conditional routing, and extensible node graph. A plain loop works for the current flow but offers no typed state and is hard to extend safely.

### "How does the warm-start mechanism work?"

When `agent.initial_checkpoint` is set:
1. Agent evaluates checkpoint on val+test → logged as `run_0` (baseline)
2. `model.checkpoint = initial_checkpoint_path` set in working config
3. Agent iterates from run_1, warm-starting each run from previous best checkpoint (when backbone unchanged)

This lets the agent improve an already-trained model rather than starting from ImageNet pretrained weights every session.


### Full Agent — End-to-End Autonomous (`run_full_agent.py`)

```
Agent:   raw_data/articles.csv + raw_data/images/
              │
         [data_prep_agent.py]
         LLM decides: which classes, how many, how to split, class weights
         Writes: data_prep_notes.md (dataset analysis + quality report)
              │
         data/auto/  ←  agent-created structure
              │
         Same agentic loop as Human+Agent
         train → evaluate → notes → improve → repeat
```

Agent receives only raw CSV + images. `data_prep_agent.py` handles everything data-related; then the identical training loop runs.

Edit `training_config.yaml` → **FULL AGENT** section (`data_prep.*`). Optional: set `data_prep.instructions` to guide class selection.

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

# ══ HUMAN+AGENT ══
paths:
  data_dir, class_mapping, class_weights, experiment_dir

# ══ FULL AGENT ══
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

data/auto/              # Full Agent output (agent-created)
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

## 11. Data Prep Agent (`agents/data_prep_agent.py`) — Full Agent only

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
| `run_human_agent.py` | Human+Agent entry point. |
| `run_full_agent.py` | Full Agent entry point (autonomous data prep + training). |
| `agents/training_agent.py` | LangGraph agent — full loop logic. |
| `agents/data_prep_agent.py` | LLM-driven data prep for Full Agent only. |
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
