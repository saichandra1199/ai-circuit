# AI Circuit — Autonomous ML Engineer

An LLM-powered autonomous ML engineering agent that **trains, evaluates, reflects, and iteratively improves** a PyTorch image classifier — without human intervention in the loop.

> **The classifier is intentionally simple. The project is judged on AI-driven experimentation, reflection, decision making, and autonomous ML engineering — not on building the most sophisticated vision model.**

---

## What This Project Does

AI Circuit is an **agentic ML pipeline** where a GPT-4o-powered LLM acts as a machine learning engineer. Given a dataset of images, the agent:

1. **Selects the right classes** to train on (Full Agent mode)
2. **Prepares the dataset** — validates images, deduplicates, resizes, splits into train/val/test (Full Agent mode)
3. **Trains a PyTorch image classifier** using a configurable backbone from the `timm` library
4. **Evaluates the model** — macro F1, per-class F1, confusion matrix
5. **Writes experiment notes** — what worked, what failed, what to try next
6. **Proposes config changes** as a structured JSON diff and applies them
7. **Repeats** until a target F1 is reached or the iteration budget is exhausted

The agent never modifies Python source code — all decisions are expressed as YAML config changes applied through dot-notation diffs (e.g. `{"optimizer.lr": 0.0001, "augmentations.mixup": true}`).

**Dataset**: H&M fashion product images with product group labels (Garment Upper Body, Shoes, Accessories, etc.)

---

## Two Workflows

| | Human+Agent | Full Agent |
|---|---|---|
| **Run** | `python run_human_agent.py` | `python run_full_agent.py` |
| **Human does** | Prepares dataset, optionally provides a pre-trained checkpoint | Sets config only |
| **Agent does** | Train → evaluate → reflect → improve | Pick classes + prep data → train → evaluate → reflect → improve |
| **Config section** | `paths` | `data_prep` |

Both workflows share the same agentic loop: **train → evaluate → reflect → improve → repeat**.

---

## How the Agent Works

1. Runs `train.py` with the current config (live epoch progress + ETA)
2. Reads results: macro F1, per-class F1, confusion matrix
3. LLM writes experiment notes — what worked, what failed, what to try next
4. LLM proposes config changes as a JSON diff, e.g. `{"optimizer.lr": 0.0001, "augmentations.mixup": true}`
5. Repeats until target F1 is reached or max iterations exhausted

The agent never rewrites Python code — only modifies YAML via dot-notation diffs. Every change is recorded in `experiments/master_log.json`.

---

## Setup

```bash
pip install -r requirements.txt
```

Create `.env` in project root:
```
OPENAI_API_KEY=sk-...
```

---

## Local Dev Quick Start (no GPU, no large dataset needed)

`data/sample/` ships with the repo — 500 images/class, 5 classes. The local config caps to 200/class and uses the smallest backbone (~2.5M params), so a full agent run takes about 2–3 minutes on CPU.

```bash
# single training run — no agent, no API key needed
python train.py --config training_config.yaml

# human+agent loop (needs OPENAI_API_KEY)
python run_human_agent.py --config training_config.yaml

# evaluate a saved checkpoint
python evaluate.py --checkpoint experiments/<session>/run_1/best_model.pth \
                   --config training_config.yaml --split test
```

Key settings in `training_config.yaml`:

| Setting | Value | Why |
|---|---|---|
| `model.backbone` | `mobilenetv3_small_100` | 2.5M params, fast CPU |
| `training.max_samples_per_class` | `200` | 200 imgs/class → fast iterations |
| `training.epochs` | `3` | enough to see learning |
| `training.mixed_precision` | `false` | CPU has no AMP |
| `training.num_workers` | `0` | safe on WSL2/Windows |

---

## Full Dataset Quick Start

```bash
# Human+Agent — you prepared the dataset
python run_human_agent.py

# Full Agent — fully autonomous
python run_full_agent.py
```

Edit `training_config.yaml` sections marked `HUMAN+AGENT` or `FULL AGENT` as needed.

---

## Human+Agent: Warm-Start from Your Own Checkpoint

If you already trained a model and want the agent to improve from it (instead of starting from scratch), set `initial_checkpoint` in the config:

```yaml
agent:
  initial_checkpoint: experiments/20240709_143022/run_3/best_model.pth
```

What happens:
1. Agent evaluates the checkpoint on val + test — logs it as **run_0 baseline**
2. Agent starts iterating from that checkpoint (warm-start)
3. Dashboard shows baseline F1 so you can see whether the agent is improving on it

---

## Standalone Evaluation

Evaluate any saved checkpoint without running the full agent:

```bash
python evaluate.py --checkpoint path/to/best_model.pth \
                   --config training_config.yaml \
                   --split test \
                   --output results/eval_out.json
```

Or import in code:

```python
from evaluate import evaluate_checkpoint
metrics = evaluate_checkpoint("path/to/best_model.pth", "training_config.yaml", split="test")
print(metrics["macro_f1"])
```

---

## Dashboard

View experiment logs, F1 trends, per-class metrics, and cross-session comparisons:

```bash
streamlit run reports/dashboard.py
```

Dashboard reads:
- `experiments/master_log.json` — all runs across sessions
- `experiments/<session>/experiment_log.json` — per-session run history
- `experiments/<session>/run_N/metrics.json` — full metrics + epoch history
- `experiments/<session>/run_N/notes.md` — LLM-written analysis

Panels:
- **KPIs** — total sessions, total runs, best F1, latest run F1
- **F1 trend** — macro F1 and val F1 per run, with target F1 line
- **Run history table** — backbone, F1, accuracy, epochs
- **Run detail** — per-class F1 bars, epoch loss curve, config diff, agent notes
- **Cross-session comparison** — best F1 per session with workflow color coding

---

## Compare Two Sessions

```bash
python utils/compare_experiments.py experiments/20240709_143022 experiments/20240710_091500
```

---

## Config (`training_config.yaml`)

One file, three clearly marked sections:

```
# ══ SHARED ══
agent:      experiment_name, max_iterations, target_f1, llm_model, initial_checkpoint
model:      backbone, pretrained, checkpoint, image_size, dropout
training:   epochs, batch_size, num_workers, mixed_precision, max_samples_per_class
optimizer:  type, lr, weight_decay
scheduler:  type, min_lr
loss:       type, focal_gamma, label_smoothing, use_class_weights
augmentations: mixup, cutmix, randaugment, random_erasing, ...

# ══ HUMAN+AGENT ══
paths:      data_dir, class_mapping, class_weights

# ══ FULL AGENT ══
data_prep:  raw_data_dir, max_train_per_class, instructions, force_classes
```

---

## Directory Structure

```
AI_circuit/
├── train.py                      # training pipeline (agent never modifies this)
├── evaluate.py                   # standalone evaluation — checkpoint + config → metrics
├── training_config.yaml          # config
├── run_human_agent.py            # Human+Agent entry point
├── run_full_agent.py             # Full Agent entry point
├── requirements.txt
├── .env                          # OPENAI_API_KEY
│
├── agents/
│   ├── training_agent.py         # LangGraph loop: init→train→evaluate→notes→improve
│   ├── data_prep_agent.py        # LLM-driven data prep (Full Agent only)
│   └── prompts.py                # IMPROVE_PROMPT + NOTES_PROMPT templates
│
├── utils/
│   ├── llm_api.py                # OpenAI chat wrapper
│   ├── data_prep.py              # dataset preparation (CSV + images → splits)
│   ├── compare_experiments.py    # CLI to compare two session dirs side-by-side
│   └── create_sample_data.py     # build data/sample/ from raw data
│
├── reports/
│   └── dashboard.py              # Streamlit dashboard — run: streamlit run reports/dashboard.py
│
├── data/
│   └── sample/                   # 500 train / ~63 val / ~63 test per class (ships with repo)
│       ├── train/<class>/
│       ├── val/<class>/
│       ├── test/<class>/
│       ├── class_mapping.json
│       └── class_weights.json
│
└── experiments/                  # created on first run
    ├── master_log.json           # all runs across all sessions
    └── 20240709_143022_local_dev/
        ├── experiment_log.json   # all runs in this session
        ├── run_0/                # baseline (only if initial_checkpoint set)
        ├── run_1/
        │   ├── config.yaml       # exact config used
        │   ├── best_model.pth    # best checkpoint by val F1
        │   ├── metrics.json      # full metrics + confusion matrix + epoch history
        │   ├── notes.md          # LLM-written analysis
        │   └── tensorboard/
        └── run_2/
```

---

## What the Agent Can Change

Agent starts from your baseline config and proposes 2–5 changes per run:

| Category | Keys |
|---|---|
| Model | `backbone`, `dropout`, `image_size`, `checkpoint` |
| Optimizer | `type`, `lr`, `weight_decay`, `momentum` |
| Scheduler | `type`, `min_lr`, `step_size`, `gamma` |
| Training | `epochs`, `batch_size`, `early_stopping_patience` |
| Loss | `type`, `focal_gamma`, `label_smoothing`, `use_class_weights` |
| Augmentations | `mixup`, `cutmix`, `randaugment`, `random_erasing`, `color_jitter`, `autoaugment` |
| Sampler | `use_weighted` |

Baseline has advanced augmentations OFF — gives the agent headroom to improve.

---

## Experiment Logs

**`experiments/master_log.json`** — accumulates every run across all sessions.

```json
{
  "session_dir": "experiments/20240709_143022_local_dev",
  "timestamp": "2024-07-09 14:30:22",
  "workflow": "human+agent",
  "run": 1,
  "backbone": "mobilenetv3_small_100",
  "macro_f1": 0.7234,
  "val_macro_f1": 0.7100,
  "accuracy": 0.7412,
  "epochs_trained": 3,
  "gap_to_target": -0.0266,
  "diff": {"optimizer.lr": 0.0001, "augmentations.mixup": true},
  "checkpoint": "experiments/20240709_143022_local_dev/run_1/best_model.pth"
}
```

---

## Dataset Preparation (`utils/data_prep.py`)

Run once before training to build clean train/val/test splits from raw H&M data:

```bash
python -m utils.data_prep --raw ../HM_Data/raw_data --out data/prepared --max-per-class 500
```

### Preprocessing pipeline — 11 steps in order

| Step | What it does | Why it matters for training |
|---|---|---|
| **1. Load + filter** | Read `articles.csv`, keep only rows matching 5 target classes | Removes irrelevant categories from the label space |
| **2. Label audit** | Print top subcategories per class (`product_type_name`, `garment_group_name`) | H&M labels are noisy — swimwear and socks both appear as "Garment Upper body". Review output before training; mislabeled clusters cap your F1 ceiling |
| **3. Resolve paths** | Build image file path from `article_id`, drop rows with no file on disk | Prevents FileNotFoundError during dataloader iteration |
| **4. Validate + min size** | PIL `im.verify()` on every image + reject images smaller than 128×128px in either dimension | Corrupt files crash training mid-epoch. Tiny images (thumbnails, placeholders) add noise without signal |
| **5. Near-duplicate removal** | Perceptual hash (dhash) with **hamming distance ≤ 4** between hashes = duplicate | Exact-match dedup misses watermark variants and slightly cropped reposts of the same image. Hamming distance catches these. Without this, train/test leakage inflates F1 |
| **6. Cap per class** | Random sample up to `max_per_class` images per class | Balances class sizes so majority class doesn't dominate training |
| **7. Product-level split** | Extract `product_id = article_id[:7]`, split by product groups (not individual articles) | Same product in multiple angles would appear in both train and test under row-level split → inflated test F1. Product-level split prevents this leakage. Prints a leakage count (should be 0) |
| **8. Resize + pad** | Resize preserving aspect ratio to fit 256×256, then white-pad to square | H&M images are portrait (~1166×1750). Without this, random crop at training time cuts off ~60% of the image. Storing pre-padded 256×256 images means the model sees the full product every iteration |
| **9. Class mapping + weights** | Write `class_mapping.json` (int→class name) and `class_weights.json` (inverse-frequency per class) | Weights fed to `WeightedCrossEntropy` and `WeightedRandomSampler` — forces equal attention to minority classes during training |
| **10. Dataset mean/std** | Compute per-channel mean/std on 2000 random train images at 224×224 | Saved to `dataset_stats.json` for reference. Training currently uses ImageNet mean/std which is a good approximation for fashion imagery |
| **11. Summary** | Print per-class counts for each split + weights + mean/std | Sanity check — verify splits look balanced before committing to a multi-hour training run |

### Output structure

```
data/prepared/
├── train/<class>/   # 80% of products
├── val/<class>/     # 10% of products
├── test/<class>/    # 10% of products
├── class_mapping.json    # {"0": "Accessories", "1": "Garment Full body", ...}
├── class_weights.json    # {"Accessories": 1.2, "Shoes": 0.8, ...}
└── dataset_stats.json    # mean, std, resize info
```

### Key config constants (top of `utils/data_prep.py`)

```python
MIN_IMG_SIZE         = 128   # drop images smaller than this (px)
DEDUP_HAMMING_THRESH = 4     # near-duplicate threshold (0 = exact match only)
RESIZE_PAD_SIZE      = 256   # store images at this square size
MAX_PER_CLASS        = 500   # cap per class for balance
```

---

## Dataset

Works with any image classification dataset. Full Agent workflow requires a CSV with a label column and an ID column matching image filenames. Configure `LABEL_COL`/`ID_COL` constants in `utils/data_prep.py` for non-default schemas.

Product-level stratified 80/10/10 split. Class imbalance handled via `WeightedRandomSampler` + `WeightedCrossEntropy`.

---

## Why Macro F1?

Dataset is typically imbalanced. Accuracy rewards majority-class prediction. Macro F1 treats every class equally — the right signal for imbalanced multi-class classification.

See [docs/Project.md](docs/Project.md) for full architecture details.
