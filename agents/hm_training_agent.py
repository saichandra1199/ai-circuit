import copy
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import TypedDict

import yaml
from langgraph.graph import END, StateGraph

from utils.llm_api import chat as _dial_chat
from agents.prompts import IMPROVE_PROMPT, NOTES_PROMPT


# ── state ─────────────────────────────────────────────────────────────────────

class HMTrainingState(TypedDict):
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


def _cfg_to_yaml(cfg: dict) -> str:
    return yaml.dump(cfg, default_flow_style=False, sort_keys=False)


def _save_log(log: list, path: str = "experiments/experiment_log.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


# ── nodes ─────────────────────────────────────────────────────────────────────

def init_iter(state: HMTrainingState) -> dict:
    run_num = state["run_num"] + 1
    cfg = copy.deepcopy(state["current_config"])
    run_dir = Path("experiments") / f"run_{run_num}"
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


def run_train(state: HMTrainingState) -> dict:
    run_dir = Path("experiments") / f"run_{state['run_num']}"
    config_path = run_dir / "config.yaml"

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "train.py", "--config", str(config_path)],
        capture_output=True, text=True
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"[TRAIN ERROR]\n{result.stderr[-2000:]}")
        return {"error": result.stderr[-2000:], "done": True}

    print(result.stdout[-1500:])
    print(f"Training completed in {elapsed:.0f}s")
    return {"error": None}


def evaluate(state: HMTrainingState) -> dict:
    run_dir = Path("experiments") / f"run_{state['run_num']}"
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

    run_dir = Path("experiments") / f"run_{state['run_num']}"
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
    _save_log(log)

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


def generate_notes(state: HMTrainingState) -> dict:
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

    notes = _dial_chat(prompt)
    notes_history = (state["notes_history"] + [notes])[-3:]

    notes_path = Path("experiments") / f"run_{state['run_num']}" / "notes.md"
    notes_path.write_text(notes)
    print(f"\n[Notes saved → {notes_path}]")

    return {"notes_history": notes_history}


def improve(state: HMTrainingState) -> dict:
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

    prompt = IMPROVE_PROMPT.format(
        run_num=state["run_num"],
        target_f1=target_f1,
        config_yaml=_cfg_to_yaml(state["current_config"]),
        macro_f1=macro_f1,
        f1_gap=macro_f1 - target_f1,
        accuracy=test_m["accuracy"],
        val_loss=test_m["loss"],
        epochs_trained=m["epochs_trained"],
        per_class_f1=_fmt_per_class(test_m),
        notes_history=notes_section,
        plateau_section=plateau_section,
    )

    raw = _dial_chat(prompt)

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


# ── routing ───────────────────────────────────────────────────────────────────

def route_after_eval(state: HMTrainingState) -> str:
    if state["done"] or state.get("error"):
        return "end"
    return "continue"


# ── graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(HMTrainingState)
    g.add_node("init_iter", init_iter)
    g.add_node("run_train", run_train)
    g.add_node("evaluate", evaluate)
    g.add_node("generate_notes", generate_notes)
    g.add_node("improve", improve)

    g.set_entry_point("init_iter")
    g.add_edge("init_iter", "run_train")
    g.add_conditional_edges("run_train", lambda s: "end" if s.get("error") else "ok",
                            {"end": END, "ok": "evaluate"})
    g.add_conditional_edges("evaluate", route_after_eval, {"end": END, "continue": "generate_notes"})
    g.add_edge("generate_notes", "improve")
    g.add_edge("improve", "init_iter")

    return g.compile()


def run(config_path: str = "training_config.yaml", max_iterations: int = 5, target_f1: float = 0.75):
    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)

    # remove output_dir if present in base config (agent manages it)
    base_cfg.get("paths", {}).pop("output_dir", None)

    initial_state: HMTrainingState = {
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
    print(f"Total runs: {final['run_num']}")
    print(f"Experiment log: experiments/experiment_log.json")
    return final
