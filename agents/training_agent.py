import copy
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import yaml
from langgraph.graph import END, StateGraph

from utils.llm_api import chat as _dial_chat
from agents.prompts import IMPROVE_PROMPT, NOTES_PROMPT


# ── state ─────────────────────────────────────────────────────────────────────

class TrainingState(TypedDict):
    session_dir: str
    llm_model: str
    workflow: str
    run_num: int
    base_config: dict
    current_config: dict
    last_diff: dict           # changes applied to reach current_config
    last_metrics: dict        # metrics.json from last completed run
    notes_history: list       # last 3 notes strings
    experiment_log: list      # all runs summary
    best_macro_f1: float
    best_checkpoint_path: str | None  # path to best_model.pth from best run so far
    plateau_count: int
    max_iterations: int
    target_f1: float
    done: bool
    error: str | None
    data_prep_config: dict         # full_agent only — current data prep settings
    data_prep_output_dir: str      # where prepare_data writes (for re-prep)
    needs_data_prep: bool          # if True, init_iter re-preps before training


# ── helpers ───────────────────────────────────────────────────────────────────

def _apply_diff(cfg: dict, diff: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for key, val in diff.items():
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return cfg


def _fmt_per_class(metrics: dict) -> str:
    lines = []
    for cls, vals in metrics.get("per_class", {}).items():
        lines.append(f"  {cls:<25} f1={vals['f1-score']:.4f}  prec={vals['precision']:.4f}  rec={vals['recall']:.4f}")
    return "\n".join(lines)


def _save_log(log: list, path: str = "experiments/experiment_log.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


def _append_master_log(entry: dict, path: str = "experiments/master_log.json"):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(p.read_text()) if p.exists() else []
    existing.append(entry)
    p.write_text(json.dumps(existing, indent=2))


def make_plateau_section(plateau_count: int) -> str:
    if plateau_count < 2:
        return ""
    return (
        f"⚠ PLATEAU DETECTED: F1 has not improved for {plateau_count} consecutive runs. "
        "Make a bolder change — try a different backbone, enable mixup/cutmix, or change the loss function.\n\n"
    )


def make_warmstart_section(best_ckpt: str | None, best_f1: float) -> str:
    if not best_ckpt:
        return ""
    return (
        f"Best checkpoint so far: {best_ckpt} (macro F1 = {best_f1:.4f})\n"
        "This is the strongest checkpoint seen so far (user baseline or agent-produced). "
        "Warm-start policy: when this checkpoint is better than the initial user checkpoint, prefer THIS checkpoint for subsequent runs. "
        "Include exactly this path via \"model.checkpoint\": \"<best_ckpt_path>\" unless you intentionally reset training. "
        "Do not keep pointing to an older/weaker user checkpoint once a better run checkpoint exists. "
        "You may still omit it / set null only when you intentionally want to train from scratch. "
        "If switching backbone, you MUST set checkpoint to null (architecture mismatch will crash).\n\n"
    )


def make_data_prep_section(dp_config: dict) -> str:
    if not dp_config:
        return ""
    classes = dp_config.get("force_classes")
    classes_str = json.dumps(classes) if classes else "null (LLM decides)"
    return (
        "Current data prep config (full_agent — READ-ONLY during iterative training):\n"
        f"  data_prep.max_train_per_class: {dp_config.get('max_train_per_class', 'null')}\n"
        f"  data_prep.force_classes: {classes_str}\n"
        "Dataset is locked once training has started.\n"
        "Do NOT propose any data_prep.* keys in this run.\n"
        "Keep data prep fixed and tune only model/training pipeline settings.\n\n"
    )


# ── nodes ─────────────────────────────────────────────────────────────────────

def init_iter(state: TrainingState) -> dict:
    updates: dict = {"needs_data_prep": False}

    # re-prep data if agent requested it
    if state.get("needs_data_prep") and state.get("data_prep_config"):
        from agents.data_prep_agent import prepare_data
        dp = state["data_prep_config"]
        print("\n[Re-preparing data with updated config...]")
        data_paths = prepare_data(
            raw_data_dir=dp["raw_data_dir"],
            output_dir=state.get("data_prep_output_dir", "data/auto"),
            max_train_per_class=dp.get("max_train_per_class"),
            llm_model=state["llm_model"],
            instructions=dp.get("instructions"),
            force_classes=dp.get("force_classes"),
        )
        cfg = copy.deepcopy(state["current_config"])
        cfg.setdefault("paths", {}).update({
            "data_dir": data_paths["data_dir"],
            "class_mapping": data_paths["class_mapping"],
            "class_weights": data_paths["class_weights"],
        })
        updates["current_config"] = cfg

    run_num = state["run_num"] + 1
    cfg = copy.deepcopy(updates.get("current_config", state["current_config"]))
    run_dir = Path(state["session_dir"]) / f"run_{run_num}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg.setdefault("paths", {})["output_dir"] = str(run_dir)
    cfg.setdefault("experiment", {})["name"] = f"agent_run_{run_num}"

    config_path = run_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*60}")
    print(f"Run {run_num} | config → {config_path}")
    if state["last_diff"]:
        print(f"Changes: {json.dumps(state['last_diff'], indent=2)}")
    print(f"{'='*60}")

    updates.update({"run_num": run_num, "current_config": cfg, "error": None})
    return updates


def run_train(state: TrainingState) -> dict:
    run_dir = Path(state["session_dir"]) / f"run_{state['run_num']}"
    config_path = run_dir / "config.yaml"

    t0 = time.time()
    # -u: unbuffered stdout so epoch/batch prints appear immediately
    proc = subprocess.Popen(
        [sys.executable, "-u", "train.py", "--config", str(config_path)],
        stdout=subprocess.PIPE, stderr=None,  # stderr (tqdm) goes direct to terminal
        text=True, bufsize=1,
    )
    tail_lines: list[str] = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail_lines.append(line)
    proc.wait()
    elapsed = time.time() - t0

    if proc.returncode != 0:
        return {"error": "".join(tail_lines[-40:]) or "train.py exited non-zero", "done": True}

    print(f"\nTraining completed in {elapsed:.0f}s")
    return {"error": None}


def evaluate(state: TrainingState) -> dict:
    run_dir = Path(state["session_dir"]) / f"run_{state['run_num']}"
    metrics_path = run_dir / "metrics.json"

    with open(metrics_path) as f:
        m = json.load(f)

    test_m = m["test"]
    macro_f1 = test_m["macro_f1"]
    best = max(state["best_macro_f1"], macro_f1)

    # plateau: F1 didn't improve by >0.005 from best
    plateau_count = state["plateau_count"]
    if macro_f1 < state["best_macro_f1"] + 0.005:
        plateau_count += 1
    else:
        plateau_count = 0

    ckpt_path = str(run_dir / "best_model.pth")
    best_ckpt = state["best_checkpoint_path"]
    if macro_f1 >= state["best_macro_f1"]:
        best_ckpt = ckpt_path

    log_entry = {
        "run": state["run_num"],
        "backbone": state["current_config"].get("model", {}).get("backbone", "unknown"),
        "macro_f1": macro_f1,
        "accuracy": test_m["accuracy"],
        "val_macro_f1": m["best_val_macro_f1"],
        "epochs_trained": m["epochs_trained"],
        "diff": state["last_diff"],
        "checkpoint": ckpt_path,
    }
    log = state["experiment_log"] + [log_entry]
    _save_log(log, str(Path(state["session_dir"]) / "experiment_log.json"))

    # collect data details from config
    cfg_paths = state["current_config"].get("paths", {})
    data_dir = Path(cfg_paths.get("data_dir", ""))
    mapping_path = cfg_paths.get("class_mapping", "")
    weights_path = cfg_paths.get("class_weights", "")

    class_mapping = {}
    if mapping_path and Path(mapping_path).exists():
        with open(mapping_path) as _f:
            class_mapping = json.load(_f)
    class_names = list(class_mapping.values())

    class_weights_data = {}
    if weights_path and Path(weights_path).exists():
        with open(weights_path) as _f:
            class_weights_data = json.load(_f)

    split_counts = {}
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if split_dir.exists():
            split_counts[split] = {
                cls: len(list((split_dir / cls).glob("*")))
                for cls in class_names
                if (split_dir / cls).exists()
            }

    _append_master_log({
        "session_dir": state["session_dir"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "workflow": state["workflow"],
        "run": state["run_num"],
        "backbone": log_entry["backbone"],
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(test_m["accuracy"], 4),
        "val_macro_f1": round(m["best_val_macro_f1"], 4),
        "epochs_trained": m["epochs_trained"],
        "target_f1": state["target_f1"],
        "gap_to_target": round(macro_f1 - state["target_f1"], 4),
        "is_best_in_session": macro_f1 >= state["best_macro_f1"],
        "diff": state["last_diff"],
        "checkpoint": ckpt_path,
        "data": {
            "data_dir": str(data_dir),
            "classes": class_names,
            "split_counts": split_counts,
            "class_weights": {k: round(v, 4) for k, v in class_weights_data.items()},
        },
    })

    done = (macro_f1 >= state["target_f1"]) or (state["run_num"] >= state["max_iterations"])

    if done:
        print(f"\nDone. Best macro F1: {best:.4f} (target {state['target_f1']})")

    return {
        "last_metrics": m,
        "best_macro_f1": best,
        "best_checkpoint_path": best_ckpt,
        "plateau_count": plateau_count,
        "experiment_log": log,
        "done": done,
    }


def generate_notes(state: TrainingState) -> dict:
    m = state["last_metrics"]
    test_m = m["test"]
    macro_f1 = test_m["macro_f1"]
    target_f1 = state["target_f1"]

    diff = state["last_diff"]
    changes_str = json.dumps(diff, indent=2) if diff else "baseline (first run)"
    changes_instruction = (
        "List each config change made from the previous run and the reasoning."
        if diff else "This is the baseline run — describe the initial config choices."
    )

    prompt = NOTES_PROMPT.format(
        run_num=state["run_num"],
        macro_f1=macro_f1,
        target_f1=target_f1,
        f1_gap=macro_f1 - target_f1,
        accuracy=test_m["accuracy"],
        val_loss=test_m["loss"],
        epochs_trained=m["epochs_trained"],
        per_class_f1=_fmt_per_class(test_m),
        changes=changes_str,
        changes_instruction=changes_instruction,
    )

    notes = _dial_chat(prompt, model=state["llm_model"])
    notes_history = (state["notes_history"] + [notes])[-3:]

    notes_path = Path(state["session_dir"]) / f"run_{state['run_num']}" / "notes.md"
    notes_path.write_text(notes)
    print(f"\n[Notes saved → {notes_path}]")

    return {"notes_history": notes_history}


def improve(state: TrainingState) -> dict:
    m = state["last_metrics"]
    test_m = m["test"]
    macro_f1 = test_m["macro_f1"]
    target_f1 = state["target_f1"]

    notes_section = ""
    if state["notes_history"]:
        notes_section = "Previous experiment notes (most recent last):\n"
        for i, note in enumerate(state["notes_history"]):
            notes_section += f"\n--- Run {state['run_num'] - len(state['notes_history']) + i + 1} notes ---\n{note}\n"
        notes_section += "\n"

    extra_directives = (
        make_plateau_section(state["plateau_count"])
        + make_warmstart_section(state.get("best_checkpoint_path"), state["best_macro_f1"])
    )
    data_prep_section = make_data_prep_section(state.get("data_prep_config") or {})

    class_names_list = list(test_m.get("per_class", {}).keys())
    prompt = IMPROVE_PROMPT.format(
        run_num=state["run_num"],
        target_f1=target_f1,
        num_classes=len(class_names_list),
        class_names=", ".join(class_names_list) if class_names_list else "see per-class F1 below",
        config_yaml=yaml.dump(state["current_config"], default_flow_style=False, sort_keys=False),
        macro_f1=macro_f1,
        f1_gap=macro_f1 - target_f1,
        accuracy=test_m["accuracy"],
        val_loss=test_m["loss"],
        epochs_trained=m["epochs_trained"],
        per_class_f1=_fmt_per_class(test_m),
        notes_history=notes_section,
        data_prep_section=data_prep_section,
        extra_directives=extra_directives,
    )

    raw = _dial_chat(prompt, model=state["llm_model"])

    # strip markdown fences if present
    if "```" in raw:
        match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        raw = match.group(1).strip() if match else raw

    try:
        diff = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[WARN] Could not parse LLM diff response:\n{raw[:500]}")
        diff = {}

    print(f"\n[Improve] Proposed changes: {json.dumps(diff, indent=2)}")

    # split data_prep.* keys from training keys
    dp_diff = {k: v for k, v in diff.items() if k.startswith("data_prep.")}
    train_diff = {k: v for k, v in diff.items() if not k.startswith("data_prep.")}

    new_dp_config = copy.deepcopy(state.get("data_prep_config") or {})
    needs_data_prep = False
    if dp_diff:
        needs_data_prep = True
        for k, v in dp_diff.items():
            subkey = k.split(".", 1)[1]
            new_dp_config[subkey] = v

    new_config = _apply_diff(state["current_config"], train_diff)
    # if force_classes changed, class mapping changes — must clear checkpoint
    if "data_prep.force_classes" in dp_diff:
        new_config.setdefault("model", {})["checkpoint"] = None
        print("[INFO] force_classes changed — checkpoint cleared (class mapping will change)")
    new_config.get("paths", {}).pop("output_dir", None)

    return {
        "current_config": new_config,
        "last_diff": diff,
        "data_prep_config": new_dp_config,
        "needs_data_prep": needs_data_prep,
    }


# ── graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(TrainingState)
    g.add_node("init_iter", init_iter)
    g.add_node("run_train", run_train)
    g.add_node("evaluate", evaluate)
    g.add_node("generate_notes", generate_notes)
    g.add_node("improve", improve)

    g.set_entry_point("init_iter")
    g.add_edge("init_iter", "run_train")
    g.add_conditional_edges("run_train", lambda s: "end" if s.get("error") else "ok",
                            {"end": END, "ok": "evaluate"})
    g.add_conditional_edges("evaluate", lambda s: "end" if s["done"] or s.get("error") else "continue",
                            {"end": END, "continue": "generate_notes"})
    g.add_edge("generate_notes", "improve")
    g.add_edge("improve", "init_iter")

    return g.compile()


def _eval_baseline(checkpoint_path: str, base_cfg: dict, session_dir: str) -> tuple[dict, list]:
    """Evaluate user-provided checkpoint as run_0 baseline. Returns (updated_state_fields, experiment_log)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from evaluate import evaluate_checkpoint

    print(f"\n{'='*60}")
    print(f"Evaluating baseline checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    test_m = evaluate_checkpoint(checkpoint_path, base_cfg, split="test")
    val_m  = evaluate_checkpoint(checkpoint_path, base_cfg, split="val")
    macro_f1 = test_m["macro_f1"]

    ckpt = __import__("torch").load(checkpoint_path, map_location="cpu", weights_only=True)
    backbone = ckpt.get("backbone", base_cfg.get("model", {}).get("backbone", "unknown")) if isinstance(ckpt, dict) else "unknown"

    log_entry = {
        "run": 0,
        "backbone": backbone,
        "macro_f1": macro_f1,
        "accuracy": test_m["accuracy"],
        "val_macro_f1": val_m["macro_f1"],
        "epochs_trained": "baseline",
        "diff": {},
        "checkpoint": checkpoint_path,
        "note": "human-provided baseline",
    }

    Path(session_dir).mkdir(parents=True, exist_ok=True)
    experiment_log = [log_entry]
    _save_log(experiment_log, str(Path(session_dir) / "experiment_log.json"))

    _append_master_log({
        "session_dir": session_dir,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "workflow": "human+agent",
        "run": 0,
        "backbone": backbone,
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(test_m["accuracy"], 4),
        "val_macro_f1": round(val_m["macro_f1"], 4),
        "epochs_trained": "baseline",
        "note": "human-provided baseline",
        "diff": {},
        "checkpoint": checkpoint_path,
    })

    print(f"\nBaseline | acc={test_m['accuracy']:.4f}  macro_f1={macro_f1:.4f}")
    return macro_f1, checkpoint_path, experiment_log


def run(config_path: str | dict = "training_config.yaml", max_iterations: int = 5, target_f1: float = 0.75, workflow: str = "human+agent", data_prep_config: dict | None = None, data_prep_output_dir: str = "data/auto"):
    if isinstance(config_path, dict):
        base_cfg = config_path
    else:
        with open(config_path) as f:
            base_cfg = yaml.safe_load(f)

    # remove output_dir if present in base config (agent manages it)
    base_cfg.get("paths", {}).pop("output_dir", None)

    data_dir = base_cfg.get("paths", {}).get("data_dir", "")
    dataset_name = Path(data_dir).name if data_dir else "unknown"
    exp_name = base_cfg.get("agent", {}).get("experiment_name") or ""
    session_label = exp_name.replace(" ", "_") if exp_name else dataset_name
    session_dir = str(Path("experiments") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session_label}")
    llm_model = base_cfg.get("agent", {}).get("llm_model", "gpt-4o-mini")
    print(f"Session dir: {session_dir}")
    print(f"LLM model:   {llm_model}")
    print(f"Workflow:    {workflow}")

    # human+agent: evaluate initial checkpoint as run_0 baseline
    best_macro_f1 = 0.0
    best_checkpoint_path = None
    experiment_log: list = []

    initial_checkpoint = base_cfg.get("agent", {}).get("initial_checkpoint") or base_cfg.get("model", {}).get("initial_checkpoint")
    if workflow == "human+agent" and initial_checkpoint and Path(initial_checkpoint).exists():
        best_macro_f1, best_checkpoint_path, experiment_log = _eval_baseline(
            initial_checkpoint, base_cfg, session_dir
        )
        # seed agent to warm-start from this checkpoint
        base_cfg.setdefault("model", {})["checkpoint"] = initial_checkpoint
    elif initial_checkpoint:
        print(f"[WARN] initial_checkpoint not found: {initial_checkpoint}")

    initial_state: TrainingState = {
        "session_dir": session_dir,
        "llm_model": llm_model,
        "workflow": workflow,
        "run_num": 0,
        "base_config": base_cfg,
        "current_config": copy.deepcopy(base_cfg),
        "last_diff": {},
        "last_metrics": {},
        "notes_history": [],
        "experiment_log": experiment_log,
        "best_macro_f1": best_macro_f1,
        "best_checkpoint_path": best_checkpoint_path,
        "plateau_count": 0,
        "max_iterations": max_iterations,
        "target_f1": target_f1,
        "done": False,
        "error": None,
        "data_prep_config": data_prep_config or {},
        "data_prep_output_dir": data_prep_output_dir,
        "needs_data_prep": False,
    }

    graph = build_graph()
    final = graph.invoke(initial_state)

    print(f"\n{'='*60}")
    print(f"Agent finished. Best macro F1: {final['best_macro_f1']:.4f}")
    print(f"Total runs:     {final['run_num']}")
    print(f"Session log:    {final['session_dir']}/experiment_log.json")
    print(f"Master log:     experiments/master_log.json")
    return final
