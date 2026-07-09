IMPROVE_PROMPT = """\
You are an expert computer vision engineer improving a PyTorch image classification pipeline.

Task: 5-class H&M fashion image classification
Classes: Garment Upper body, Garment Lower body, Garment Full body, Accessories, Shoes
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
{plateau_section}\
Available config keys you may change (dot-notation):

MODEL
- model.backbone: convnext_base.fb_in22k_ft_in1k, convnext_small.fb_in22k_ft_in1k, convnext_large.fb_in22k_ft_in1k, swin_base_patch4_window7_224.ms_in22k_ft_in1k, swin_small_patch4_window7_224.ms_in22k_ft_in1k, tf_efficientnetv2_m.in21k_ft_in1k, tf_efficientnetv2_s.in21k_ft_in1k, vit_base_patch16_224.augreg2_in21k_ft_in1k, efficientnet_b4, efficientnet_b2, resnet50, convnext_tiny
- model.dropout: 0.0–0.5
- model.image_size: 224, 256  (changing this also scales all crop/resize ops automatically)
- model.checkpoint: path to .pth file for warm-starting — use ONLY if keeping the same backbone, else set to null

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

TRAINING
- training.epochs: 5–40
- training.batch_size: 16, 32, 64
- training.early_stopping_patience: 3–10

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
