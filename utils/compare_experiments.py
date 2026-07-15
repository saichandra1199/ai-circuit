"""
Compare two or more experiment sessions.

Usage:
    python utils/compare_experiments.py experiments/20240709_143022 experiments/20240710_091500
    python utils/compare_experiments.py experiments/20240709_143022  # single session summary
"""
import argparse
import json
from pathlib import Path


def load_session(session_path: str) -> dict:
    p = Path(session_path)
    log_path = p / "experiment_log.json"
    if not log_path.exists():
        raise FileNotFoundError(f"No experiment_log.json in {session_path}")
    runs = json.loads(log_path.read_text())

    # infer workflow from master log if available
    workflow = "unknown"
    master = Path("experiments/master_log.json")
    if master.exists():
        for entry in json.loads(master.read_text()):
            if entry.get("session_dir", "").endswith(p.name):
                workflow = entry.get("workflow", "unknown")
                break

    return {
        "session": p.name,
        "workflow": workflow,
        "runs": runs,
        "best_run": max(runs, key=lambda r: r.get("macro_f1", 0)),
        "best_f1": max(r.get("macro_f1", 0) for r in runs),
    }


def _bar(val: float, width: int = 20) -> str:
    filled = round(val * width)
    return "█" * filled + "░" * (width - filled)


def _workflow_label(w: str) -> str:
    return {"full_agent": "[Full Agent]", "human+agent": "[Human+Agent]"}.get(w, f"[{w}]")


def print_session(s: dict) -> None:
    label = _workflow_label(s["workflow"])
    print(f"\n{'═'*60}")
    print(f"  {s['session']}  {label}")
    print(f"{'═'*60}")
    print(f"  Runs: {len(s['runs'])}   Best macro F1: {s['best_f1']:.4f}")
    print()

    for r in s["runs"]:
        f1 = r.get("macro_f1", 0)
        val_f1 = r.get("val_macro_f1", 0)
        marker = " ★" if r == s["best_run"] else "  "
        diff_keys = list(r.get("diff", {}).keys())
        diff_str = ", ".join(diff_keys[:4]) + ("…" if len(diff_keys) > 4 else "")
        print(f"  run {r['run']:>2}{marker}  test F1={f1:.4f} {_bar(f1)}  val F1={val_f1:.4f}  ep={r.get('epochs_trained', '?'):>2}")
        print(f"         backbone: {r.get('backbone', 'unknown')}")
        if diff_str:
            print(f"         changes:  {diff_str}")
        notes_path = Path(f"{s['session']}/run_{r['run']}/notes.md")
        # try relative to experiments/
        if not notes_path.exists():
            notes_path = Path("experiments") / s["session"] / f"run_{r['run']}" / "notes.md"
        if notes_path.exists():
            lines = notes_path.read_text().splitlines()
            # grab first non-empty line after "## Results Analysis"
            in_results = False
            for line in lines:
                if "## Results Analysis" in line:
                    in_results = True
                    continue
                if in_results and line.strip():
                    print(f"         note:     {line.strip()[:80]}")
                    break
        print()


def compare(sessions: list[dict]) -> None:
    if len(sessions) == 1:
        print_session(sessions[0])
        return

    # side-by-side winner table
    print(f"\n{'═'*70}")
    print("  COMPARISON SUMMARY")
    print(f"{'═'*70}")
    print(f"  {'Session':<30} {'Workflow':<16} {'Runs':>4} {'Best F1':>8} {'Best backbone'}")
    print(f"  {'-'*30} {'-'*16} {'-'*4} {'-'*8} {'-'*30}")
    winner = max(sessions, key=lambda s: s["best_f1"])
    for s in sessions:
        mark = " ★" if s is winner else "  "
        best_bb = s["best_run"].get("backbone", "unknown")
        print(f"  {s['session']:<30} {_workflow_label(s['workflow']):<16} {len(s['runs']):>4} {s['best_f1']:>8.4f}{mark}  {best_bb}")

    print(f"\n  Winner: {winner['session']} (macro F1 = {winner['best_f1']:.4f})")
    print(f"  Best checkpoint: {winner['best_run'].get('checkpoint', 'n/a')}")

    # per-session detail
    for s in sessions:
        print_session(s)


def main():
    ap = argparse.ArgumentParser(description="Compare experiment sessions")
    ap.add_argument("sessions", nargs="+", help="Paths to session directories")
    args = ap.parse_args()

    loaded = []
    for path in args.sessions:
        try:
            loaded.append(load_session(path))
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    if not loaded:
        print("No valid sessions found.")
        return

    compare(loaded)


if __name__ == "__main__":
    main()
