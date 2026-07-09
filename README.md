# AI Circuit — Autonomous ML Engineer (Loop Engineering)

An AI agent that autonomously improves an image classifier through iterative experimentation. The task (H&M fashion classification) is a benchmark; the real artifact is the **agentic loop** that trains, evaluates, reflects, and adjusts hyperparameters without human input.

---

## What It Does

1. Runs a training experiment using `train.py` (code is fixed — agent never touches it)
2. Reads results: macro F1, per-class F1, confusion matrix
3. Uses an LLM to write experiment notes (what worked, what to try next)
4. Proposes config changes as a JSON diff
5. Repeats until target F1 is reached or iterations exhausted

---

## Two Modes

| Mode | Entry Point | What the agent controls |
|------|------------|------------------------|
| **Workflow 1** | `run_agent.py` | Hyperparameters only — data already prepared by human |
| **Workflow 2** | `run_workflow2.py` | Everything — data prep (class selection, splits) + hyperparameters |

---

## Setup

```bash
pip install -r requirements.txt
```

Create `.env` with your OpenAI key:
```
OPENAI_API_KEY=sk-...
```

---

## Quick Start

### Workflow 1 — Human data, agent trains

```bash
# default: 5 iterations, target 0.75 macro F1, uses data/sample/
python run_agent.py

# custom
python run_agent.py --max-iterations 8 --target-f1 0.80
```

### Workflow 2 — Fully autonomous (raw data → trained model)

```bash
python run_workflow2.py --raw-data-dir ../HM_Data/raw_data --max-iterations 5 --target-f1 0.75
```

The agent reads `articles.csv` + images, picks classes, splits data, then runs the training loop.

### Manual single training run

```bash
python train.py --config training_config.yaml
```

### Monitor with TensorBoard

```bash
tensorboard --logdir experiments/
```

---

## Directory Structure

```
AI_circuit/
├── train.py                    # training pipeline — fixed, agent does not modify
├── training_config.yaml        # baseline config (agent starts from this)
├── run_agent.py                # Workflow 1 entry point
├── run_workflow2.py            # Workflow 2 entry point
├── requirements.txt
├── .env                        # OPENAI_API_KEY
│
├── agents/
│   ├── hm_training_agent.py    # LangGraph loop: init→train→evaluate→notes→improve
│   ├── hm_data_prep_agent.py   # LLM-driven data prep (Workflow 2 only)
│   └── prompts.py              # IMPROVE_PROMPT + NOTES_PROMPT templates
│
├── utils/
│   ├── llm_api.py              # OpenAI chat wrapper (gpt-4o-mini)
│   └── create_sample_data.py   # creates data/sample/ from processed_data/
│
├── data/
│   ├── sample/                 # 500 train / 63 val / 63 test per class
│   ├── class_weights.json      # inverse-frequency weights (5 classes)
│   └── class_mapping.json
│
├── docs/
│   └── Project.md              # full architecture reference
│
└── experiments/                # auto-created; one subdir per agent run
    ├── run_1/
    │   ├── config.yaml         # exact config used for this run
    │   ├── best_model.pth      # best checkpoint (by val macro F1)
    │   ├── metrics.json        # full metrics + confusion matrix + history
    │   ├── notes.md            # LLM-written analysis of this run
    │   └── tensorboard/
    └── experiment_log.json     # cross-run summary (all diffs + F1 scores)
```

---

## Config: What the Agent Can Change

| Key | Default | Options |
|-----|---------|---------|
| `model.backbone` | `convnext_base` | any [timm](https://github.com/huggingface/pytorch-image-models) model |
| `model.checkpoint` | `null` | path to `.pth` for warm-starting |
| `optimizer.lr` | `0.0003` | 1e-5 to 5e-4 |
| `optimizer.type` | `adamw` | `adamw`, `adam`, `sgd` |
| `scheduler.type` | `cosine` | `cosine`, `step`, `onecycle`, `plateau` |
| `loss.type` | `weighted_ce` | `weighted_ce`, `focal` |
| `augmentations.mixup` | `false` | `true/false` |
| `augmentations.cutmix` | `false` | `true/false` |
| `augmentations.randaugment` | `false` | `true/false` |
| `training.max_samples_per_class` | `null` | integer or `null` |

Changes are expressed as JSON diffs (e.g. `{"optimizer.lr": 0.0001, "augmentations.mixup": true}`) and applied via dot-notation — the agent never rewrites Python code.

---

## Dataset

H&M Personalized Fashion (Kaggle) — 5 classes from `product_group_name`, ~91K images, stratified 80/10/10 split.

| Class | Train (full) | Class Weight |
|-------|-------------|--------------|
| Garment Upper body | 34,144 | 0.43 |
| Garment Lower body | 15,816 | 0.93 |
| Garment Full body | 10,620 | 1.38 |
| Accessories | 8,804 | 1.67 |
| Shoes | 4,125 | 3.56 |

Class weights are used in both `WeightedCrossEntropy` loss and `WeightedRandomSampler` to handle imbalance.

---

## Switching to Full Dataset

Edit `training_config.yaml`:
```yaml
paths:
  data_dir: data/full
training:
  max_samples_per_class: null   # or cap at e.g. 2000 for faster full-data runs
```

See [docs/Project.md](docs/Project.md) for full architecture details.
