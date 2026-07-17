"""All LLM prompt strings for the AI Circuit agents."""


# ── training agent ────────────────────────────────────────────────────────────

IMPROVE_PROMPT = """\
You are an expert computer vision engineer improving a PyTorch image classification pipeline.

Task: image classification ({num_classes} classes)
Classes: {class_names}
Metric: macro F1 (primary)
Target: macro_f1 >= {target_f1}

Current config (YAML):
{config_yaml}

Run {run_num} metrics:
  macro_f1:  {macro_f1:.4f}  (target >= {target_f1}, gap: {f1_gap:+.4f})
  accuracy:  {accuracy:.4f}
  val_loss:  {val_loss:.4f}
  epochs trained: {epochs_trained}

Per-class F1:
{per_class_f1}

{notes_history}\
{data_prep_section}\
{extra_directives}\
Available config keys you may change (dot-notation):

DATA PREP (full_agent only)
- Dataset is LOCKED once training iterations start.
- Do NOT propose any data_prep.* changes during iterative runs.
- Keep data_prep.max_train_per_class and data_prep.force_classes unchanged.
- Do not return data_prep.* keys in your JSON output.


MODEL
- model.backbone: mobilenetv3_small_100, mobilenetv3_large_100, mobilenetv2_100, efficientnet_b0, efficientnet_b2, efficientnet_b4, resnet50, convnext_tiny, convnext_small.fb_in22k_ft_in1k, convnext_base.fb_in22k_ft_in1k, swin_small_patch4_window7_224.ms_in22k_ft_in1k, swin_base_patch4_window7_224.ms_in22k_ft_in1k, vit_base_patch16_224.augreg2_in21k_ft_in1k
- model.pretrained: true/false — ImageNet pretrained weights; strongly prefer true for small datasets (<500 imgs/class), set false only when dataset is highly domain-specific and you have 1000+ imgs/class
- model.dropout: 0.0–0.5
- model.image_size: 224, 256  (changing this also scales all crop/resize ops automatically)
- model.checkpoint: path to .pth file for warm-starting — use ONLY if keeping the same backbone, else set to null

CHECKPOINT PRIORITY POLICY
- If a newer checkpoint from agent runs has higher macro F1 than the initial user-provided checkpoint, prefer the newer better checkpoint for subsequent runs.
- Do NOT keep reverting to an older/weaker user checkpoint once a stronger checkpoint exists.
- Only set model.checkpoint to null when intentionally restarting from scratch or when changing backbone.

OPTIMIZER
- optimizer.type: adamw, adam, sgd
- optimizer.lr: 1e-5 to 5e-4
- optimizer.weight_decay: 0.0 to 0.1
- optimizer.momentum: 0.8–0.99  (only used when optimizer.type=sgd)

SCHEDULER
- scheduler.type: cosine, step, onecycle, plateau
- scheduler.min_lr: 1e-7 to 1e-5  (cosine floor)
- scheduler.step_size: 3–10  (only for type=step)
- scheduler.gamma: 0.05–0.5  (only for type=step or plateau)
- scheduler.warmup_epochs: 0–5 — REQUIRED when using transformer backbones (swin_*, vit_*) or when fine-tuning pretrained weights with low lr; set 0 for CNNs training from scratch

TRAINING
- training.epochs: 5–40
- training.batch_size: 16, 32, 64
- training.early_stopping_patience: 3–10
- training.mixed_precision: true/false — enable on GPU for ~2x speedup with no accuracy loss; keep false on CPU (no benefit, adds overhead)
- training.max_samples_per_class: integer or null — caps samples per class at DataLoader level WITHOUT triggering data re-prep; increase this first before touching data_prep.max_train_per_class

LOSS
- loss.type: weighted_ce, focal
- loss.focal_gamma: 0.5–5.0  (only with loss.type=focal)
- loss.label_smoothing: 0.0–0.2  (only with loss.type=weighted_ce)
- loss.use_class_weights: true/false

AUGMENTATIONS
- augmentations.random_erasing: true/false
- augmentations.random_erasing_prob: 0.1–0.7
- augmentations.mixup: true/false
- augmentations.mixup_alpha: 0.2–1.0
- augmentations.cutmix: true/false
- augmentations.cutmix_alpha: 0.5–2.0
- augmentations.randaugment: true/false
- augmentations.randaugment_n: 1–4
- augmentations.randaugment_m: 5–15
- augmentations.color_jitter: true/false
- augmentations.color_jitter_brightness: 0.0–0.5
- augmentations.color_jitter_contrast: 0.0–0.5
- augmentations.color_jitter_saturation: 0.0–0.5
- augmentations.color_jitter_hue: 0.0–0.1
- augmentations.random_horizontal_flip: true/false
- augmentations.random_resized_crop: true/false
- augmentations.autoaugment: true/false  (mutually exclusive with randaugment — disable randaugment if enabling this)

SAMPLER
- sampler.use_weighted: true/false

Return ONLY valid JSON — the config keys to change. Change 2–5 things per iteration. No text outside JSON.
Example: {{"optimizer.lr": 0.0001, "augmentations.mixup": true, "augmentations.mixup_alpha": 0.4}}
"""

NOTES_PROMPT = """\
You are an expert computer vision engineer writing post-run experiment notes.

Run {run_num} results:
  macro_f1:  {macro_f1:.4f}  (target >= {target_f1}, gap: {f1_gap:+.4f})
  accuracy:  {accuracy:.4f}
  val_loss:  {val_loss:.4f}
  epochs trained: {epochs_trained}

Per-class F1:
{per_class_f1}

Config changes from previous run:
{changes}

Write concise Markdown experiment notes with EXACTLY these 3 sections (no preamble, no other text):

## Changes Made
{changes_instruction}

## Results Analysis
Analyze macro F1 and per-class results — what worked, what failed, key insights from confusion patterns.

## Further Improvements
2–3 specific, actionable steps to improve macro F1 toward {target_f1}.
"""


# ── data prep agent ───────────────────────────────────────────────────────────

CLASS_DECISION_PROMPT = """\
You are an ML engineer selecting classes for an image classification task.

Dataset: image classification dataset ({total} total items)
Label column: {label_col}
Distribution (class: image count):
{distribution}

Select 3–6 classes that:
1. Have at least 500 images each (more = better)
2. Are visually distinct from each other
3. Cover meaningful categories worth classifying

Return ONLY valid JSON — no other text:
{{"classes": ["Class A", "Class B", ...], "rationale": "one sentence why these classes"}}
"""

PIPELINE_CONFIG_PROMPT = """\
You are an ML engineer deciding which data preparation steps to run for an image classification dataset.

Dataset stats:
- Total items (filtered to selected classes, images on disk): {total_items}
- Selected classes: {selected_classes}
- Class distribution:
{dist_table}
- Missing images (no file on disk): {missing_count} ({missing_pct:.1f}%)
- Imbalance ratio (largest / smallest class): {imbalance_ratio:.2f}x
- Current max_train_per_class cap: {max_train_per_class}

Available pipeline steps (you control whether each runs and its params):

1. validate_images — PIL verify + minimum resolution filter
   params: enabled (bool), min_size (int, px) — drop images smaller than this
   When to enable: any missing_pct > 5%, or if dataset may have corrupt files

2. dedup — near-duplicate removal via perceptual hash (dhash)
   params: enabled (bool), hamming_thresh (int, 0-8) — lower = stricter
   When to enable: large datasets (total_items > 2000 per class); DISABLE for small datasets to preserve data

3. resize_pad — resize preserving aspect ratio, pad to square with white background
   params: enabled (bool), size (int, px) — output resolution (recommend 224 or 256)
   When to enable: ALWAYS recommended — ensures consistent image sizes for the model

4. product_level_split — split by product ID (first 7 chars of article_id) not by row
   params: enabled (bool)
   When to enable: ALWAYS when dataset has product families (H&M data does) — prevents train/test leakage

5. compute_mean_std — compute per-channel mean/std on training images
   params: enabled (bool), sample_n (int) — how many images to sample
   When to enable: only when using custom normalization instead of ImageNet stats; adds time

Also set:
- max_train_per_class: integer or null (null = use all data after dedup/validation)
- eval_cap_ratio: float 0.05-0.25 — val/test size = max_train_per_class * eval_cap_ratio

Decision rules:
- total_items < 1000 per class → disable dedup (preserve every sample)
- missing_pct > 5% → enable validate_images
- Always enable resize_pad and product_level_split
- compute_mean_std only if explicitly needed

Return ONLY valid JSON matching this exact structure — no other text:
{{
  "validate_images":     {{"enabled": true/false, "min_size": 128}},
  "dedup":               {{"enabled": true/false, "hamming_thresh": 4}},
  "resize_pad":          {{"enabled": true/false, "size": 256}},
  "product_level_split": {{"enabled": true/false}},
  "compute_mean_std":    {{"enabled": false, "sample_n": 2000}},
  "max_train_per_class": 50,
  "eval_cap_ratio": 0.125
}}
"""

DATASET_ANALYSIS_PROMPT = """\
You are an ML engineer writing notes about a dataset you just prepared for image classification.

Raw dataset stats:
- Total items in CSV: {total_raw}
- Total unique classes: {total_classes}
- Items with missing images: {missing_count} ({missing_pct:.1f}%)
- Items available for selected classes (after image filter): {items_available}
- Items actually used after capping (max_train_per_class={max_train_per_class}): {items_used}

All available classes (name: count):
{all_classes}

Selected classes: {selected_classes}
Selection rationale: {rationale}

Final split counts per class (train | val | test):
{split_table}

Class weights (inverse frequency for loss weighting):
{weights_table}

Imbalance ratio (largest / smallest class in train): {imbalance_ratio:.2f}x

Write Markdown notes with EXACTLY these 3 sections (no preamble, no other text):

## Dataset Overview
Size, class composition, coverage of selected classes vs full dataset.

## Data Quality
Flag concerns: missing images rate, class imbalance severity, any classes too small, potential label noise risk.

## Preparation Decisions
Why these classes were selected, what to watch during training based on the distribution.
"""
