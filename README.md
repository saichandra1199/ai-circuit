# AI Circuit — Autonomous ML Engineer

An AI agent that autonomously improves an image classifier through iterative experimentation. The real artifact is the **agentic loop** that trains, evaluates, reflects, and adjusts hyperparameters without human input.

> **The classifier is intentionally simple. The project is judged on AI-driven experimentation, reflection, decision making, and autonomous ML engineering — not on building the most sophisticated vision model.**

---

## Two Workflows

| | Workflow 1 | Workflow 2 |
|---|---|---|
| **Run** | `python run_human_agent.py` | `python run_full_agent.py` |
| **Human does** | Prepares dataset (`data/` structure, splits, class weights) | Sets config only |
| **Agent does** | Train → evaluate → reflect → improve | Pick classes + prep data → train → evaluate → reflect → improve |
| **Config section** | `paths` | `data_prep` |

Both workflows use the same agentic loop: **train → evaluate → reflect → improve → repeat**.

---

## How the Agent Works

1. Runs `train.py` with the current config — shows live batch progress + ETA
2. Reads results: macro F1, per-class F1, confusion matrix
3. Uses an LLM to write experiment notes (what worked, what failed, what to try next)
4. Proposes config changes as a JSON diff — e.g. `{"optimizer.lr": 0.0001, "augmentations.mixup": true}`
5. Repeats until target F1 is reached or iterations exhausted

The agent never rewrites Python code — only modifies the YAML config via dot-notation diffs. Every change is recorded in `experiments/master_log.json`.

---

## Setup

```bash
pip install -r requirements.txt
```

Create `.env`:
```
OPENAI_API_KEY=sk-...
```

---

## Quick Start

### Workflow 1 — You prepare data, agent trains and optimizes

```bash
python run_human_agent.py
```

Edit `training_config.yaml` → **WORKFLOW 1** section to point `paths.data_dir` at your dataset.

### Workflow 2 — Fully autonomous (raw data → trained model)

```bash
python run_full_agent.py
```

Edit `training_config.yaml` → **WORKFLOW 2** section: set `data_prep.raw_data_dir` to your raw dataset path, optionally add `data_prep.instructions`.

### Manual single run (no agent)

```bash
python train.py --config training_config.yaml
```

### Monitor with TensorBoard

```bash
tensorboard --logdir experiments/
```

---

## Config (`training_config.yaml`)

One file, three clearly marked sections:

```
# ══ SHARED — edit for both workflows ══
agent:
  max_iterations, target_f1, llm_model

model, training, optimizer, scheduler, loss, augmentations

# ══ WORKFLOW 1 ══
paths:
  data_dir, class_mapping, class_weights

# ══ WORKFLOW 2 ══
data_prep:
  raw_data_dir, max_train_per_class, instructions
```

---

## Directory Structure

```
AI_circuit/
├── train.py                      # training pipeline — fixed, agent never modifies
├── training_config.yaml          # single config for both workflows
├── run_human_agent.py            # Workflow 1 entry point
├── run_full_agent.py             # Workflow 2 entry point
├── requirements.txt
├── .env                          # OPENAI_API_KEY
│
├── agents/
│   ├── training_agent.py         # LangGraph loop: init→train→evaluate→notes→improve
│   ├── data_prep_agent.py        # LLM-driven data prep (Workflow 2 only)
│   └── prompts.py                # IMPROVE_PROMPT + NOTES_PROMPT templates
│
├── utils/
│   ├── llm_api.py                # OpenAI chat wrapper
│   └── create_sample_data.py     # creates data/sample/ from raw data
│
├── data/
│   ├── sample/                   # 500 train / ~63 val / ~63 test per class
│   ├── class_weights.json
│   └── class_mapping.json
│
├── docs/
│   └── Project.md                # full architecture reference
│
└── experiments/
    ├── master_log.json           # cross-session master record (all runs ever)
    └── 20240709_143022/          # one timestamped dir per agent session
        ├── experiment_log.json   # all runs in this session
        ├── run_1/
        │   ├── config.yaml       # exact config used for this run
        │   ├── best_model.pth    # best checkpoint (by val macro F1)
        │   ├── metrics.json      # full metrics + confusion matrix + history
        │   ├── notes.md          # LLM-written analysis of this run
        │   └── tensorboard/
        └── run_2/
            └── ...
```

---

## Experiment Logs

**`experiments/master_log.json`** — accumulates every run across all sessions. Use this to compare models and pick the best checkpoint.

```json
{
  "session_dir": "experiments/20240709_143022",
  "timestamp": "2024-07-09 14:30:22",
  "workflow": "human+agent",
  "run": 1,
  "backbone": "convnext_base.fb_in22k_ft_in1k",
  "macro_f1": 0.7234,
  "gap_to_target": -0.0266,
  "epochs_trained": 15,
  "data": {
    "data_dir": "data/sample",
    "classes": ["Accessories", "Garment Lower body", ...],
    "split_counts": { "train": {...}, "val": {...}, "test": {...} },
    "class_weights": {...}
  },
  "diff": { "optimizer.lr": 0.0001, "augmentations.mixup": true },
  "checkpoint": "experiments/20240709_143022/run_1/best_model.pth"
}
```

---

## What the Agent Can Change

Agent starts from your baseline config and tunes these every iteration (2–5 changes per run):

| Category | Keys |
|---|---|
| Model | `backbone`, `dropout`, `image_size`, `checkpoint` |
| Optimizer | `type`, `lr`, `weight_decay`, `momentum` |
| Scheduler | `type`, `min_lr`, `step_size`, `gamma` |
| Training | `epochs`, `batch_size`, `early_stopping_patience` |
| Loss | `type`, `focal_gamma`, `label_smoothing`, `use_class_weights` |
| Augmentations | `mixup`, `cutmix`, `randaugment`, `random_erasing`, `color_jitter`, `autoaugment` |
| Sampler | `use_weighted` |

Baseline has advanced augmentations OFF — gives the agent room to improve.

---

## Dataset

Works with any image classification dataset. Workflow 2 requires a CSV with a label column and an ID column matching image filenames. Configure `label_col`/`id_col` in `prepare_data()` for non-default schemas.

Stratified 80/10/10 split. Class imbalance handled via `WeightedRandomSampler` + `WeightedCrossEntropy`.

---

## Why Macro F1?

Dataset is typically imbalanced. Accuracy rewards majority-class prediction. Macro F1 treats every class equally — the right signal for imbalanced classification.

See [docs/Project.md](docs/Project.md) for full architecture details.
