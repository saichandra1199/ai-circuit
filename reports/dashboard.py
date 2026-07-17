# Streamlit dashboard for AI Circuit experiment tracking.
#
# Usage:
#   streamlit run reports/dashboard.py

import json
import os
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(os.path.dirname(os.path.abspath(__file__))).parent))
from utils.compare_experiments import load_session

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent
EXPERIMENTS_DIR = ROOT / "experiments"
MASTER_LOG = EXPERIMENTS_DIR / "master_log.json"

GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
WORKFLOW_COLORS = {"human+agent": "#2a78d6", "full_agent": "#1baf7a", "unknown": "#898781"}
METRIC_COLORS = {"macro_f1": "#2a78d6", "val_macro_f1": "#eda100", "accuracy": "#1baf7a"}


# ── loaders ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=15)
def load_master_log() -> pd.DataFrame:
    if not MASTER_LOG.exists():
        return pd.DataFrame()
    entries = json.loads(MASTER_LOG.read_text())
    df = pd.DataFrame(entries)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["session_name"] = df["session_dir"].apply(lambda p: Path(p).name)
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data(ttl=15)
def load_session_log(session_name: str) -> pd.DataFrame:
    log_path = EXPERIMENTS_DIR / session_name / "experiment_log.json"
    if not log_path.exists():
        return pd.DataFrame()
    entries = json.loads(log_path.read_text())
    df = pd.DataFrame(entries)
    df["run_num"] = pd.to_numeric(df["run"], errors="coerce")
    df["run_label"] = df["run_num"].apply(lambda r: f"run_{int(r)}" if pd.notna(r) else "run_unknown")
    return df


@st.cache_data(ttl=60)
def load_user_checkpoint_baseline(session_name: str) -> dict | None:
    """Baseline is the user-provided checkpoint in config (not run_0)."""
    run_cfg = load_run_config(session_name, 1)
    if not run_cfg:
        return None

    model_cfg = run_cfg.get("model", {})
    agent_cfg = run_cfg.get("agent", {})
    checkpoint = model_cfg.get("checkpoint") or agent_cfg.get("initial_checkpoint")
    if not checkpoint:
        return None

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        return None

    try:
        from evaluate import evaluate_checkpoint
        test_m = evaluate_checkpoint(str(ckpt_path), run_cfg, split="test")
        return {
            "checkpoint": str(ckpt_path),
            "macro_f1": float(test_m.get("macro_f1", 0.0)),
            "accuracy": float(test_m.get("accuracy", 0.0)),
        }
    except Exception:
        return None


def load_run_metrics(session_name: str, run_num: int) -> dict | None:
    p = EXPERIMENTS_DIR / session_name / f"run_{run_num}" / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_run_config(session_name: str, run_num: int) -> dict | None:
    p = EXPERIMENTS_DIR / session_name / f"run_{run_num}" / "config.yaml"
    if not p.exists():
        return None
    import yaml
    return yaml.safe_load(p.read_text())


def load_run_notes(session_name: str, run_num: int) -> str | None:
    p = EXPERIMENTS_DIR / session_name / f"run_{run_num}" / "notes.md"
    return p.read_text() if p.exists() else None


def list_sessions() -> list[str]:
    if not EXPERIMENTS_DIR.exists():
        return []
    return sorted(
        [d.name for d in EXPERIMENTS_DIR.iterdir()
         if d.is_dir() and (d / "experiment_log.json").exists()],
        reverse=True,
    )


# ── chart helpers ─────────────────────────────────────────────────────────────

def _per_class_chart(per_class: dict) -> alt.Chart:
    pc_df = pd.DataFrame([
        {"class": cls, "F1": v["f1-score"], "Precision": v["precision"], "Recall": v["recall"]}
        for cls, v in per_class.items()
    ])
    pc_long = pc_df.melt(id_vars="class", value_vars=["F1", "Precision", "Recall"],
                         var_name="metric", value_name="value")
    return (
        alt.Chart(pc_long)
        .mark_bar()
        .encode(
            x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), title=None),
            y=alt.Y("class:N", sort="-x", title=None),
            color=alt.Color("metric:N", legend=alt.Legend(title=None)),
            row=alt.Row("metric:N", title=None, header=alt.Header(labelAngle=0)),
            tooltip=["class", "metric", alt.Tooltip("value:Q", format=".4f")],
        )
        .properties(width=350, height=80)
        .configure_axis(gridColor=GRIDLINE, domainColor=AXIS, labelColor=TEXT_SECONDARY)
        .configure_view(strokeWidth=0)
    )


def _confusion_chart(cm: list, class_names: list) -> alt.Chart:
    cm_arr = np.array(cm)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_norm = (cm_arr / row_sums.clip(min=1) * 100).round(1)
    rows = []
    for i, actual in enumerate(class_names):
        for j, pred in enumerate(class_names):
            rows.append({"Actual": actual, "Predicted": pred,
                         "count": cm_arr[i][j], "pct": cm_norm[i][j]})
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("Predicted:N", axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("Actual:N"),
            color=alt.Color("pct:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["Actual", "Predicted",
                     alt.Tooltip("count:Q", title="Count"),
                     alt.Tooltip("pct:Q", title="% of row", format=".1f")],
        )
        .properties(width=280, height=280, title="Confusion matrix (% of actual class)")
        .configure_axis(gridColor=GRIDLINE, labelColor=TEXT_SECONDARY)
        .configure_view(strokeWidth=0)
    )


def _epoch_chart(history: list) -> alt.Chart:
    hist_df = pd.DataFrame(history)
    cols = [c for c in ["train_loss", "val_loss"] if c in hist_df.columns]
    hist_long = hist_df.melt(id_vars="epoch", value_vars=cols, var_name="split", value_name="loss")
    return (
        alt.Chart(hist_long)
        .mark_line(strokeWidth=1.5, point=True)
        .encode(
            x=alt.X("epoch:O", title="Epoch"),
            y=alt.Y("loss:Q", scale=alt.Scale(zero=False), title="Loss"),
            color=alt.Color("split:N", legend=alt.Legend(title=None)),
            tooltip=["epoch", "split", alt.Tooltip("loss:Q", format=".4f")],
        )
        .properties(height=180, title="Training / val loss")
        .configure_axis(gridColor=GRIDLINE, domainColor=AXIS, labelColor=TEXT_SECONDARY)
        .configure_view(strokeWidth=0)
    )


def _epoch_perf_chart(history: list) -> alt.Chart:
    hist_df = pd.DataFrame(history)
    metric_map = {
        "train_acc": "train_acc",
        "val_acc": "val_acc",
        "val_macro_f1": "val_macro_f1",
    }
    cols = [c for c in metric_map if c in hist_df.columns]
    perf_long = hist_df.melt(id_vars="epoch", value_vars=cols, var_name="metric", value_name="value")
    perf_long["label"] = perf_long["metric"].map(metric_map)
    return (
        alt.Chart(perf_long)
        .mark_line(strokeWidth=1.8, point=True)
        .encode(
            x=alt.X("epoch:O", title="Epoch"),
            y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1]), title="Score"),
            color=alt.Color("label:N", legend=alt.Legend(title=None)),
            tooltip=["epoch", "label", alt.Tooltip("value:Q", format=".4f")],
        )
        .properties(height=180, title="Epoch performance")
        .configure_axis(gridColor=GRIDLINE, domainColor=AXIS, labelColor=TEXT_SECONDARY)
        .configure_view(strokeWidth=0)
    )


def _best_only_chart(session_df: pd.DataFrame) -> alt.Chart:
    df = session_df.copy()
    df["run_num"] = pd.to_numeric(df.get("run_num", df["run"]), errors="coerce")
    df = df[df["run_num"].notna()].sort_values("run_num").copy()
    df["best_so_far"] = df["macro_f1"].cummax()
    df["run_label"] = df["run_num"].apply(lambda r: f"run_{int(r)}")
    df["is_new_best"] = df["best_so_far"].diff().fillna(df["best_so_far"]).gt(0)

    line = (
        alt.Chart(df)
        .mark_line(strokeWidth=2.2, color="#1baf7a", point=alt.OverlayMarkDef(size=65, filled=True))
        .encode(
            x=alt.X("run_label:N", title="Run"),
            y=alt.Y("best_so_far:Q", title="Best macro F1 so far", scale=alt.Scale(zero=False)),
            tooltip=["run_label", alt.Tooltip("best_so_far:Q", format=".4f")],
        )
    )
    highlights = (
        alt.Chart(df[df["is_new_best"]])
        .mark_point(size=95, filled=True, color="#1baf7a", stroke="white", strokeWidth=1.2)
        .encode(
            x="run_label:N",
            y="best_so_far:Q",
            tooltip=["run_label", alt.Tooltip("best_so_far:Q", format=".4f")],
        )
    )
    return (
        (line + highlights)
        .properties(height=220, title="Best-model progression (new best checkpoints only)")
        .configure_axis(gridColor=GRIDLINE, domainColor=AXIS, labelColor=TEXT_SECONDARY, titleColor=TEXT_PRIMARY)
        .configure_view(strokeWidth=0)
    )


def _f1_trend_chart(plot_df: pd.DataFrame, x_field: str, x_title: str,
                    show_all: bool, target_f1: float | None,
                    baseline_f1: float | None = None,
                    baseline_x: str | int | None = None,
                    x_sort: list | None = None) -> alt.Chart:
    trend_long = plot_df.melt(
        id_vars=[x_field] + (["session_name"] if show_all else []),
        value_vars=["macro_f1", "val_macro_f1"],
        var_name="metric", value_name="value",
    )
    color_enc = (
        alt.Color("session_name:N", legend=alt.Legend(title="Session")) if show_all
        else alt.Color("metric:N",
                       scale=alt.Scale(domain=["macro_f1", "val_macro_f1"],
                                       range=[METRIC_COLORS["macro_f1"], METRIC_COLORS["val_macro_f1"]]),
                       legend=alt.Legend(title="Metric"))
    )
    chart = (
        alt.Chart(trend_long)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True))
        .encode(
            x=alt.X(f"{x_field}:O", title=x_title, sort=x_sort),
            y=alt.Y("value:Q", title="F1", scale=alt.Scale(zero=False)),
            color=color_enc,
            strokeDash=alt.StrokeDash("metric:N") if show_all else alt.Undefined,
            tooltip=[x_field, "metric", alt.Tooltip("value:Q", format=".4f")]
            + (["session_name"] if show_all else []),
        )
        .properties(height=300)
    )
    if target_f1:
        rule = (
            alt.Chart(pd.DataFrame([{"target": target_f1}]))
            .mark_rule(strokeDash=[6, 3], color="#d03b3b", strokeWidth=1.5)
            .encode(y="target:Q", tooltip=[alt.Tooltip("target:Q", title="Target F1")])
        )
        base = (chart + rule).resolve_scale(y="shared")
    else:
        base = chart

    if baseline_f1 is not None:
        last_x = plot_df[x_field].iloc[-1]
        bx = baseline_x if baseline_x is not None else last_x
        baseline_color = "#d97706"
        baseline_rule = (
            alt.Chart(pd.DataFrame([{"baseline": baseline_f1}]))
            .mark_rule(strokeDash=[2, 2], color=baseline_color, strokeWidth=1.6)
            .encode(y="baseline:Q", tooltip=[alt.Tooltip("baseline:Q", title="Initial checkpoint F1")])
        )
        baseline_label = (
            alt.Chart(pd.DataFrame([{"x": bx, "baseline": baseline_f1, "label": "baseline"}]))
            .mark_text(align="left", dx=6, dy=-8, fontWeight="bold", color=baseline_color)
            .encode(
                x=alt.X("x:O", title=None),
                y="baseline:Q",
                text="label:N",
                tooltip=[alt.Tooltip("baseline:Q", title="Initial checkpoint F1")],
            )
        )
        baseline_point = (
            alt.Chart(pd.DataFrame([{"x": bx, "baseline": baseline_f1}]))
            .mark_point(size=55, filled=True, color=baseline_color, stroke="white", strokeWidth=1)
            .encode(x=alt.X("x:O", title=None), y="baseline:Q")
        )
        baseline_star = (
            alt.Chart(pd.DataFrame([{"x": bx, "baseline": baseline_f1, "star": "★"}]))
            .mark_text(dy=-20, fontSize=16, fontWeight="bold", color=baseline_color)
            .encode(
                x=alt.X("x:O", title=None),
                y="baseline:Q",
                text="star:N",
                tooltip=[alt.Tooltip("baseline:Q", title="Initial checkpoint F1")],
            )
        )
        base = (base + baseline_rule + baseline_point + baseline_label + baseline_star).resolve_scale(y="shared")

    return (
        base
        .configure_axis(gridColor=GRIDLINE, domainColor=AXIS, labelColor=TEXT_SECONDARY, titleColor=TEXT_PRIMARY)
        .configure_view(strokeWidth=0)
    )


# ── run detail renderer ───────────────────────────────────────────────────────

def render_run_detail(session_name: str, run_row: pd.Series, is_best: bool):
    run_num = int(run_row.get("run_num", run_row["run"]))
    f1 = run_row.get("macro_f1", 0)
    val_f1 = run_row.get("val_macro_f1", 0)
    backbone = run_row.get("backbone", "unknown")
    epochs = run_row.get("epochs_trained", "?")
    label = run_row["run_label"]
    star = " ★ best" if is_best else ""

    with st.expander(f"**{label}**  |  {backbone}  |  test F1 `{f1:.4f}`  |  val F1 `{val_f1:.4f}`  |  {epochs} epochs{star}"):
        metrics = load_run_metrics(session_name, run_num)

        # ── top metrics strip ──────────────────────────────────────────────
        if metrics:
            t = metrics.get("test", {})
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Test macro F1", f"{t.get('macro_f1', f1):.4f}")
            m2.metric("Val macro F1", f"{val_f1:.4f}")
            m3.metric("Accuracy", f"{t.get('accuracy', 0):.4f}")
            m4.metric("Weighted F1", f"{t.get('weighted_f1', 0):.4f}")
            train_s = metrics.get("total_training_time_s")
            m5.metric("Train time", f"{train_s:.0f}s" if train_s else "—")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Test macro F1", f"{f1:.4f}")
            m2.metric("Val macro F1", f"{val_f1:.4f}")
            if "accuracy" in run_row:
                m3.metric("Accuracy", f"{run_row['accuracy']:.4f}")
            st.caption("Baseline checkpoint — no training history.")

        st.divider()

        # ── per-class + confusion matrix ───────────────────────────────────
        if metrics and "test" in metrics:
            per_class = metrics["test"].get("per_class", {})
            cm = metrics["test"].get("confusion_matrix")
            class_names = list(per_class.keys())

            col_pc, col_cm = st.columns([1.2, 1])
            with col_pc:
                st.markdown("**Per-class metrics**")
                if per_class:
                    pc_df = pd.DataFrame([
                        {"Class": cls,
                         "F1": round(v["f1-score"], 4),
                         "Precision": round(v["precision"], 4),
                         "Recall": round(v["recall"], 4)}
                        for cls, v in per_class.items()
                    ]).sort_values("F1", ascending=False)
                    st.dataframe(pc_df, hide_index=True, use_container_width=True)

                    bar = (
                        alt.Chart(pc_df)
                        .mark_bar(cornerRadiusEnd=3, color=METRIC_COLORS["macro_f1"])
                        .encode(
                            x=alt.X("F1:Q", scale=alt.Scale(domain=[0, 1])),
                            y=alt.Y("Class:N", sort="-x", title=None),
                            tooltip=["Class", "F1", "Precision", "Recall"],
                        )
                        .properties(height=max(150, len(per_class) * 30))
                        .configure_axis(gridColor=GRIDLINE, domainColor=AXIS, labelColor=TEXT_SECONDARY)
                    )
                    st.altair_chart(bar, use_container_width=True)

            with col_cm:
                st.markdown("**Confusion matrix**")
                if cm and class_names:
                    st.altair_chart(_confusion_chart(cm, class_names), use_container_width=False)

            st.divider()

        # ── epoch history ──────────────────────────────────────────────────
        if metrics and metrics.get("history"):
            st.markdown("**Epoch history**")
            h1, h2 = st.columns(2)
            with h1:
                st.altair_chart(_epoch_chart(metrics["history"]), use_container_width=True)
            with h2:
                st.altair_chart(_epoch_perf_chart(metrics["history"]), use_container_width=True)
            st.divider()

        # ── config diff + notes ────────────────────────────────────────────
        col_diff, col_notes = st.columns([1, 1.2])
        with col_diff:
            st.markdown("**Config changes from previous run**")
            diff = run_row.get("diff", {})
            if diff:
                st.json(diff)
            else:
                st.caption("No diff (baseline or first run).")

            with st.expander("Full config used"):
                run_cfg = load_run_config(session_name, run_num)
                if run_cfg:
                    st.json(run_cfg)
                elif metrics and metrics.get("config"):
                    st.json(metrics["config"])
                else:
                    st.caption("Config not available.")

        with col_notes:
            st.markdown("**Agent notes**")
            notes = load_run_notes(session_name, run_num)
            if notes:
                st.markdown(notes)
            else:
                st.caption("No notes for this run.")


# ── page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="AI Circuit — Experiment Dashboard", layout="wide")
st.title("AI Circuit — Experiment Dashboard")

master_df = load_master_log()
sessions = list_sessions()

if not sessions:
    st.info("No experiments found. Run `python run_human_agent.py` or `python run_full_agent.py` first.")
    st.stop()

tab_overview, tab_session, tab_compare = st.tabs(["Overview", "Session", "Compare"])

with st.sidebar:
    st.header("Session")
    selected_session = st.selectbox("Select session", sessions)
    st.divider()
    st.header("View options")
    show_all = st.checkbox("Show all sessions in F1 trend", value=False)

# ── overview tab ──────────────────────────────────────────────────────────────

with tab_overview:
    if not master_df.empty:
        best_row = master_df.loc[master_df["macro_f1"].idxmax()]
        latest_row = master_df.iloc[-1]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Sessions", master_df["session_name"].nunique())
        k2.metric("Total runs", len(master_df))
        k3.metric("Best macro F1", f"{best_row['macro_f1']:.4f}",
                  help=f"{best_row['session_name']} / {best_row.get('backbone','?')}")
        k4.metric("Latest run F1", f"{latest_row['macro_f1']:.4f}",
                  delta=f"{latest_row['macro_f1'] - latest_row.get('target_f1', latest_row['macro_f1']):.4f} vs target"
                  if "target_f1" in latest_row else None)

    st.divider()
    st.subheader("All sessions")

    best_per_session = (
        master_df.groupby("session_name", as_index=False)
        .agg(best_macro_f1=("macro_f1", "max"),
             workflow=("workflow", "last"),
             runs=("run", "count"),
             best_backbone=("backbone", "last"),
             last_run=("timestamp", "max"))
        .sort_values("best_macro_f1", ascending=False)
        .reset_index(drop=True)
    ) if not master_df.empty else pd.DataFrame()

    if not best_per_session.empty:
        winner = best_per_session.iloc[0]["session_name"]

        # session cards
        cols = st.columns(min(3, len(best_per_session)))
        for i, row in best_per_session.iterrows():
            with cols[i % 3]:
                badge = "★ Best" if row["session_name"] == winner else ""
                wf_color = WORKFLOW_COLORS.get(row["workflow"], WORKFLOW_COLORS["unknown"])
                st.markdown(
                    f"**{row['session_name']}** {badge}  \n"
                    f"`{row['workflow']}`  \n"
                    f"Best F1: **{row['best_macro_f1']:.4f}**  |  Runs: {row['runs']}  \n"
                    f"Backbone: `{row['best_backbone']}`"
                )
                st.divider()

        st.subheader("Best F1 per session")
        comp_chart = (
            alt.Chart(best_per_session)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("best_macro_f1:Q", title="Best macro F1", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("session_name:N", sort="-x", title=None),
                color=alt.Color("workflow:N",
                                scale=alt.Scale(domain=list(WORKFLOW_COLORS.keys()),
                                                range=list(WORKFLOW_COLORS.values()))),
                tooltip=["session_name", "workflow", "runs",
                         alt.Tooltip("best_macro_f1:Q", format=".4f"), "best_backbone"],
            )
            .properties(height=max(200, len(best_per_session) * 40))
            .configure_axis(gridColor=GRIDLINE, domainColor=AXIS, labelColor=TEXT_SECONDARY)
        )
        st.altair_chart(comp_chart, use_container_width=True)

        st.subheader("Session summary table")
        disp = best_per_session.copy()
        disp["best_macro_f1"] = disp["best_macro_f1"].apply(lambda x: f"{x:.4f}")
        disp["last_run"] = disp["last_run"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(disp, hide_index=True, use_container_width=True)

# ── session tab ───────────────────────────────────────────────────────────────

with tab_session:
    session_df = load_session_log(selected_session)

    if session_df.empty:
        st.warning(f"No run log found for {selected_session}")
        st.stop()

    session_rows = master_df[master_df["session_name"] == selected_session] if not master_df.empty else pd.DataFrame()
    workflow = session_rows["workflow"].iloc[-1] if not session_rows.empty else "unknown"
    target_f1 = session_rows["target_f1"].iloc[-1] if not session_rows.empty and "target_f1" in session_rows.columns else None

    st.subheader(f"`{selected_session}`")
    st.caption(f"Workflow: **{workflow}**" + (f"  |  Target F1: **{target_f1}**" if target_f1 else ""))

    baseline_info = load_user_checkpoint_baseline(selected_session)
    best_run_row = session_df.loc[session_df["macro_f1"].idxmax()]
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Runs", len(session_df))
    if baseline_info is not None:
        s2.metric("User checkpoint F1", f"{baseline_info['macro_f1']:.4f}",
                  help=baseline_info["checkpoint"])
        s3.metric(
            "Best macro F1",
            f"{best_run_row['macro_f1']:.4f}",
            delta=f"{best_run_row['macro_f1'] - baseline_info['macro_f1']:+.4f} vs user checkpoint",
            help=f"run_{int(best_run_row['run'])}",
        )
    else:
        s2.metric("Best macro F1", f"{best_run_row['macro_f1']:.4f}",
                  help=f"run_{int(best_run_row.get('run_num', best_run_row['run']))}")
        s3.metric("User checkpoint F1", "—")
        st.info("Baseline star is shown only when the session config includes a valid user checkpoint path.")
    s4.metric("Best val F1", f"{session_df['val_macro_f1'].max():.4f}")
    s5.metric("Best backbone", best_run_row.get("backbone", "unknown"))

    st.divider()
    st.subheader("F1 trend")

    plot_df = session_df.sort_values("run_num").reset_index(drop=True).copy()
    if show_all and not master_df.empty:
        plot_df = master_df[["run", "macro_f1", "val_macro_f1", "session_name"]].copy()
        plot_df["run_order"] = range(len(plot_df))
        x_field, x_title = "run_order", "Global run order"
        x_sort = None
        baseline_x = None
    else:
        plot_df["run_order"] = plot_df.index
        plot_df["run_label"] = plot_df["run_num"].apply(lambda r: f"run_{int(r)}")
        x_field, x_title = "run_label", "Run"
        x_sort = (["baseline"] if baseline_info is not None else []) + plot_df["run_label"].tolist()
        baseline_x = "baseline" if baseline_info is not None else None

    st.altair_chart(
        _f1_trend_chart(
            plot_df,
            x_field,
            x_title,
            show_all,
            target_f1,
            baseline_f1=None if baseline_info is None else float(baseline_info["macro_f1"]),
            baseline_x=baseline_x,
            x_sort=x_sort,
        ),
        use_container_width=True,
    )
    st.caption("val_macro_f1 = best val F1 during training.  macro_f1 = final test F1.")

    st.altair_chart(_best_only_chart(session_df), use_container_width=True)
    st.caption("This curve moves only when a run beats all previous runs (best checkpoint progression).")

    st.divider()
    st.subheader("Runs  (click to expand)")

    best_run_num = int(best_run_row.get("run_num", best_run_row["run"]))
    for _, row in session_df.sort_values("run_num").iterrows():
        render_run_detail(selected_session, row, is_best=(int(row.get("run_num", row["run"])) == best_run_num))

# ── compare tab ───────────────────────────────────────────────────────────────

with tab_compare:
    st.subheader("Compare two sessions")

    if len(sessions) < 2:
        st.info("Need at least 2 sessions to compare.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            session_a = st.selectbox("Session A", sessions, index=0, key="cmp_a")
        with c2:
            session_b = st.selectbox("Session B", sessions, index=1, key="cmp_b")

        if session_a == session_b:
            st.warning("Select two different sessions.")
        else:
            try:
                sa = load_session(str(EXPERIMENTS_DIR / session_a))
                sb = load_session(str(EXPERIMENTS_DIR / session_b))
            except FileNotFoundError as e:
                st.error(str(e))
                st.stop()

            winner = sa if sa["best_f1"] >= sb["best_f1"] else sb

            summary_df = pd.DataFrame([
                {
                    "session": s["session"],
                    "workflow": s["workflow"],
                    "runs": len(s["runs"]),
                    "best_macro_f1": round(s["best_f1"], 4),
                    "best_backbone": s["best_run"].get("backbone", "unknown"),
                    "best_checkpoint": s["best_run"].get("checkpoint", "n/a"),
                }
                for s in [sa, sb]
            ])
            st.markdown(f"**Winner: `{winner['session']}`** — macro F1 `{winner['best_f1']:.4f}`")
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

            st.divider()

            # F1 trend overlay
            rows = []
            for s in [sa, sb]:
                for r in s["runs"]:
                    rows.append({"session": s["session"], "run": r["run"],
                                 "macro_f1": r.get("macro_f1", 0),
                                 "val_macro_f1": r.get("val_macro_f1", 0)})
            cmp_df = pd.DataFrame(rows)
            cmp_long = cmp_df.melt(id_vars=["session", "run"],
                                   value_vars=["macro_f1", "val_macro_f1"],
                                   var_name="metric", value_name="value")
            cmp_chart = (
                alt.Chart(cmp_long)
                .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True))
                .encode(
                    x=alt.X("run:O", title="Run"),
                    y=alt.Y("value:Q", title="F1", scale=alt.Scale(zero=False)),
                    color=alt.Color("session:N", legend=alt.Legend(title="Session")),
                    strokeDash=alt.StrokeDash("metric:N", legend=alt.Legend(title="Metric")),
                    tooltip=["session", "run", "metric", alt.Tooltip("value:Q", format=".4f")],
                )
                .properties(height=320, title="F1 trend — A vs B")
                .configure_axis(gridColor=GRIDLINE, domainColor=AXIS,
                                labelColor=TEXT_SECONDARY, titleColor=TEXT_PRIMARY)
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(cmp_chart, use_container_width=True)

            # per-session run tables
            col_a, col_b = st.columns(2)
            for col, s in [(col_a, sa), (col_b, sb)]:
                with col:
                    st.markdown(f"**{s['session']}** (`{s['workflow']}`)")
                    runs_df = pd.DataFrame([
                        {
                            "run": r["run"],
                            "backbone": r.get("backbone", ""),
                            "macro_f1": f"{r.get('macro_f1', 0):.4f}",
                            "val_f1": f"{r.get('val_macro_f1', 0):.4f}",
                            "epochs": r.get("epochs_trained", "?"),
                            "changes": ", ".join(list(r.get("diff", {}).keys())[:3]),
                        }
                        for r in s["runs"]
                    ])
                    st.dataframe(runs_df, hide_index=True, use_container_width=True)
