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


# ── nodes ─────────────────────────────────────────────────────────────────────

def init_iter(state: TrainingState) -> dict:
    run_num = state["run_num"] + 1
    cfg = copy.deepcopy(state["current_config"])
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

    return {"run_num": run_num, "current_config": cfg, "error": None}


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

    plateau_section = ""
    if state["plateau_count"] >= 2:
        plateau_section = (
            f"⚠ PLATEAU DETECTED: F1 has not improved for {state['plateau_count']} consecutive runs. "
            "Make a bolder change — try a different backbone, enable mixup/cutmix, or change the loss function.\n\n"
        )

    best_ckpt = state.get("best_checkpoint_path")
    if best_ckpt and plateau_section == "":
        plateau_section = (
            f"Best checkpoint available at: {best_ckpt}\n"
            "You may warm-start the next run by including \"model.checkpoint\": \"<path>\" in your JSON, "
            "but ONLY if keeping the same backbone — warm-starting with a different backbone will crash.\n\n"
        )

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
        plateau_section=plateau_section,
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

    new_config = _apply_diff(state["current_config"], diff)
    # remove output_dir so init_iter sets it fresh each run
    new_config.get("paths", {}).pop("output_dir", None)

    return {"current_config": new_config, "last_diff": diff}


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


def run(config_path: str | dict = "training_config.yaml", max_iterations: int = 5, target_f1: float = 0.75, workflow: str = "human+agent"):
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
    exp_suffix = f"_{exp_name.replace(' ', '_')}" if exp_name else ""
    session_dir = str(Path("experiments") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{dataset_name}{exp_suffix}")
    llm_model = base_cfg.get("agent", {}).get("llm_model", "gpt-4o-mini")
    print(f"Session dir: {session_dir}")
    print(f"LLM model:   {llm_model}")
    print(f"Workflow:    {workflow}")

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
        "experiment_log": [],
        "best_macro_f1": 0.0,
        "best_checkpoint_path": None,
        "plateau_count": 0,
        "max_iterations": max_iterations,
        "target_f1": target_f1,
        "done": False,
        "error": None,
    }

    graph = build_graph()
    final = graph.invoke(initial_state)

    print(f"\n{'='*60}")
    print(f"Agent finished. Best macro F1: {final['best_macro_f1']:.4f}")
    print(f"Total runs:     {final['run_num']}")
    print(f"Session log:    {final['session_dir']}/experiment_log.json")
    print(f"Master log:     experiments/master_log.json")
    return final
