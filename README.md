# AI Circuit — Autonomous ML Engineer (Loop Engineering)

An AI agent that autonomously improves an image classifier through iterative experimentation. The task (H&M fashion classification) is a benchmark; the real artifact is the **agentic loop** that trains, evaluates, reflects, and adjusts hyperparameters without human input.

> **The classifier is intentionally simple. The project is judged on AI-driven experimentation, reflection, decision making, and autonomous ML engineering — not on building the most sophisticated vision model.**

---

## Two Workflows

| | Workflow 1 | Workflow 2 |
|------|----------|-----------|
| **File** | `run_human_agent.py` | `run_full_agent.py` |
| **Human does** | Data prep (splits, class weights, `data/` structure) | Nothing |
| **Agent does** | Train → evaluate → reflect → improve | Pick classes + prep data + train → evaluate → reflect → improve |

**Workflow 1** — Human handles class selection, splitting, class weights, directory structure. Agent's job: receive a working pipeline, improve macro F1 through iterative config changes.

**Workflow 2** — Agent receives only raw CSV + images. LLM decides which classes to classify, how to split data, how to handle imbalance. Then runs the same optimization loop as Workflow 1.

Both workflows use the same agentic loop: **train → evaluate → reflect → improve → repeat**.

---

## How the Agent Works

1. Runs a training experiment using `train.py` (code is fixed — agent never touches it)
2. Reads results: macro F1, per-class F1, confusion matrix
3. Uses an LLM to write experiment notes (what worked, what failed, what to try next)
4. Proposes config changes as a JSON diff (e.g. `{"optimizer.lr": 0.0001, "augmentations.mixup": true}`)
5. Repeats until target F1 is reached or iterations exhausted

The agent never rewrites Python code — only modifies the YAML config via dot-notation diffs. Every change is auditable in `experiments/experiment_log.json`.

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

### Workflow 1 — Human data, agent trains and optimizes

```bash
# default: 5 iterations, target 0.75 macro F1, uses data/sample/
python run_human_agent.py

# custom
python run_human_agent.py --max-iterations 8 --target-f1 0.80
```

Data must already exist at `data/sample/` (or `data/full/`). See [Switching to full dataset](#switching-to-full-dataset) below.

### Workflow 2 — Fully autonomous (raw data → trained model)

```bash
python run_full_agent.py --raw-data-dir ../HM_Data/raw_data --max-iterations 5 --target-f1 0.75
```

Agent reads `articles.csv` + images, picks classes (LLM decision), splits data, generates class weights, then runs the full optimization loop.

### Manual single training run (no agent)

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
├── run_human_agent.py                # Workflow 1 entry point
├── run_full_agent.py            # Workflow 2 entry point
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

The baseline config reserves these for the agent — they start off so the agent has room to improve:

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
| `augmentations.random_erasing` | `false` | `true/false` |
| `loss.label_smoothing` | `0.0` | 0.0–0.2 |
| `training.max_samples_per_class` | `null` | integer or `null` |

---

## Dataset

H&M Personalized Fashion (Kaggle) — 5 classes from `product_group_name`, ~92K images, stratified 80/10/10 split.

| Class | Train (full) | Class Weight |
|-------|-------------|--------------|
| Garment Upper body | 34,144 | 0.43 |
| Garment Lower body | 15,816 | 0.93 |
| Garment Full body | 10,620 | 1.38 |
| Accessories | 8,804 | 1.67 |
| Shoes | 4,125 | 3.56 |

Class imbalance (~8:1) handled via `WeightedRandomSampler` + `WeightedCrossEntropy`. No augmented copies generated — the agent controls augmentation as part of optimization.

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
