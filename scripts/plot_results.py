"""
scripts/plot_results.py - Plotting and Metrics Aggregation Pipeline.

Generates publication-quality figures and summary Markdown tables from
experimental result files.
"""

import argparse
import glob
import json
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Clean publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

STRATEGY_COLORS = {
    "naive": "#e74c3c",       # Red
    "replay": "#3498db",      # Blue
    "ewc": "#9b59b6",         # Purple
    "parseval": "#e67e22",    # Orange
    "car_sr": "#2ecc71",      # Emerald Green
    "car_sr_no_proj": "#1abc9c",     # Teal
    "car_sr_no_parseval": "#f1c40f", # Yellow
}

STRATEGY_LABELS = {
    "naive": "Naive Fine-Tuning",
    "replay": "Experience Replay",
    "ewc": "EWC",
    "parseval": "Parseval Regularization",
    "car_sr": "CAR-SR (Proposed)",
    "car_sr_no_proj": "CAR-SR (w/o Projection)",
    "car_sr_no_parseval": "CAR-SR (w/o Parseval)",
}


def load_all_results(results_dir: str) -> List[Dict[str, Any]]:
    records = []
    pattern = os.path.join(results_dir, "**", "results.json")
    for file_path in glob.glob(pattern, recursive=True):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rec = {
                "strategy": data["history"]["strategy"],
                "sequence": data["history"]["sequence_name"],
                "seed": data["history"]["seed"],
                "avg_final_return": data["metrics"]["avg_final_return"],
                "forgetting": data["metrics"]["forgetting"],
                "plasticity": data["metrics"]["plasticity"],
                "backward_transfer": data["metrics"]["backward_transfer"],
                "forward_transfer": data["metrics"]["forward_transfer"],
                "return_matrix": data["return_matrix"],
                "file_path": file_path,
            }
            records.append(rec)
        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")
    return records


def generate_summary_table(df: pd.DataFrame, output_dir: str) -> str:
    grouped = df.groupby(["sequence", "strategy"]).agg({
        "avg_final_return": ["mean", "std", "count"],
        "forgetting": ["mean", "std"],
        "plasticity": ["mean", "std"],
        "forward_transfer": ["mean", "std"],
    }).reset_index()

    lines = []
    lines.append("| Sequence | Strategy | Avg Final Return (↑) | Forgetting (↓) | Plasticity (↑) | Forward Transfer (↑) | Runs |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")

    for seq in ["similar", "partial", "dissimilar"]:
        seq_df = df[df["sequence"] == seq]
        if seq_df.empty:
            continue
        for strat in ["naive", "replay", "ewc", "parseval", "car_sr", "car_sr_no_proj", "car_sr_no_parseval"]:
            sub = seq_df[seq_df["strategy"] == strat]
            if sub.empty:
                continue
            n = len(sub)
            afr_m, afr_s = sub["avg_final_return"].mean(), sub["avg_final_return"].std()
            f_m, f_s = sub["forgetting"].mean(), sub["forgetting"].std()
            p_m, p_s = sub["plasticity"].mean(), sub["plasticity"].std()
            fwt_m, fwt_s = sub["forward_transfer"].mean(), sub["forward_transfer"].std()

            f_str = f"{f_m:.1f} ± {0.0 if np.isnan(f_s) else f_s:.1f}"
            afr_str = f"**{afr_m:.1f} ± {0.0 if np.isnan(afr_s) else afr_s:.1f}**" if strat == "car_sr" else f"{afr_m:.1f} ± {0.0 if np.isnan(afr_s) else afr_s:.1f}"
            p_str = f"{p_m:.1f} ± {0.0 if np.isnan(p_s) else p_s:.1f}"
            fwt_str = f"{fwt_m:.1f} ± {0.0 if np.isnan(fwt_s) else fwt_s:.1f}"
            lines.append(f"| `{seq}` | **{STRATEGY_LABELS.get(strat, strat)}** | {afr_str} | {f_str} | {p_str} | {fwt_str} | {n} |")

    table_md = "\n".join(lines)
    with open(os.path.join(output_dir, "summary_table.md"), "w", encoding="utf-8") as f:
        f.write(table_md)
    return table_md


def plot_performance_and_forgetting(df: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sequences = ["similar", "partial", "dissimilar"]
    main_strats = ["naive", "replay", "ewc", "parseval", "car_sr"]

    x = np.arange(len(sequences))
    width = 0.15

    # 1. Final Average Return
    ax1 = axes[0]
    for idx, strat in enumerate(main_strats):
        means = []
        stds = []
        for seq in sequences:
            sub = df[(df["sequence"] == seq) & (df["strategy"] == strat)]
            if not sub.empty:
                means.append(sub["avg_final_return"].mean())
                stds.append(sub["avg_final_return"].std() if len(sub) > 1 else 0.0)
            else:
                means.append(0.0)
                stds.append(0.0)
        offset = (idx - 2) * width
        ax1.bar(x + offset, means, width, yerr=stds, capsize=4,
                label=STRATEGY_LABELS.get(strat, strat),
                color=STRATEGY_COLORS.get(strat, "#333333"), alpha=0.9)

    ax1.set_title("Average Final Return Across Task Sequences")
    ax1.set_ylabel("Average Final Return (Higher is Better)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Similar\n(No Conflict)", "Partial\n(Asymmetric Conflict)", "Dissimilar\n(Global Conflict)"])
    ax1.legend(loc="upper right", frameon=True)
    ax1.set_ylim(0, 520)

    # 2. Catastrophic Forgetting
    ax2 = axes[1]
    for idx, strat in enumerate(main_strats):
        means = []
        stds = []
        for seq in sequences:
            sub = df[(df["sequence"] == seq) & (df["strategy"] == strat)]
            if not sub.empty:
                means.append(sub["forgetting"].mean())
                stds.append(sub["forgetting"].std() if len(sub) > 1 else 0.0)
            else:
                means.append(0.0)
                stds.append(0.0)
        offset = (idx - 2) * width
        ax2.bar(x + offset, means, width, yerr=stds, capsize=4,
                label=STRATEGY_LABELS.get(strat, strat),
                color=STRATEGY_COLORS.get(strat, "#333333"), alpha=0.9)

    ax2.set_title("Catastrophic Forgetting Across Task Sequences")
    ax2.set_ylabel("Catastrophic Forgetting (Lower is Better)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Similar\n(No Conflict)", "Partial\n(Asymmetric Conflict)", "Dissimilar\n(Global Conflict)"])
    ax2.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "task_performance_bar.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_forgetting_vs_conflict(df: pd.DataFrame, output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sequences = ["similar", "partial", "dissimilar"]
    main_strats = ["naive", "replay", "ewc", "parseval", "car_sr"]

    for strat in main_strats:
        means = []
        for seq in sequences:
            sub = df[(df["sequence"] == seq) & (df["strategy"] == strat)]
            means.append(sub["forgetting"].mean() if not sub.empty else 0.0)
        ax.plot(sequences, means, marker="o", linewidth=2.2, markersize=7,
                label=STRATEGY_LABELS.get(strat, strat),
                color=STRATEGY_COLORS.get(strat, "#333333"))

    ax.set_title("Forgetting Trajectory as Task Conflict Increases")
    ax.set_xlabel("Task Sequence (Graduated Task Conflict)")
    ax.set_ylabel("Catastrophic Forgetting Score")
    ax.set_xticks(range(len(sequences)))
    ax.set_xticklabels(["Similar\n(Low Conflict)", "Partial\n(Medium Conflict)", "Dissimilar\n(High Conflict)"])
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "forgetting_vs_conflict.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_stability_plasticity_pareto(df: pd.DataFrame, output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    main_strats = ["naive", "replay", "ewc", "parseval", "car_sr"]

    for strat in main_strats:
        sub = df[df["strategy"] == strat]
        if sub.empty:
            continue
        # Retention = 500 - forgetting (higher retention is better)
        ax.scatter(sub["plasticity"], 500.0 - sub["forgetting"],
                   label=STRATEGY_LABELS.get(strat, strat),
                   color=STRATEGY_COLORS.get(strat, "#333333"),
                   s=80, alpha=0.85, edgecolors="black", linewidth=1.0)

    ax.set_title("Stability-Plasticity Trade-Off Across All Runs")
    ax.set_xlabel("Plasticity (Immediate Task Learning Ability ↑)")
    ax.set_ylabel("Retention Capacity (500 - Forgetting ↑)")
    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "stability_plasticity_pareto.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_ablation_comparison(df: pd.DataFrame, output_dir: str):
    """Generates an ablation breakdown figure comparing CAR-SR vs ablated variants."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ablation_strats = ["car_sr", "car_sr_no_proj", "car_sr_no_parseval"]
    labels = ["Full CAR-SR", "w/o Projection", "w/o Parseval"]
    colors = [STRATEGY_COLORS[s] for s in ablation_strats]

    # Focus on the 'partial' sequence where asymmetric conflict is active
    sub = df[df["sequence"] == "partial"]

    # 1. Catastrophic Forgetting
    ax1 = axes[0]
    f_means = [sub[sub["strategy"] == s]["forgetting"].mean() for s in ablation_strats]
    f_stds = [sub[sub["strategy"] == s]["forgetting"].std() for s in ablation_strats]
    bars1 = ax1.bar(labels, f_means, yerr=f_stds, capsize=5, color=colors, alpha=0.9, edgecolor="black", linewidth=1.2)
    ax1.set_title("Ablation: Forgetting on Partial Conflict\n(Lower is Better)")
    ax1.set_ylabel("Catastrophic Forgetting")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{yval:.1f}", ha="center", va="bottom", fontweight="bold")

    # 2. Forward Transfer
    ax2 = axes[1]
    fwt_means = [sub[sub["strategy"] == s]["forward_transfer"].mean() for s in ablation_strats]
    fwt_stds = [sub[sub["strategy"] == s]["forward_transfer"].std() for s in ablation_strats]
    bars2 = ax2.bar(labels, fwt_means, yerr=fwt_stds, capsize=5, color=colors, alpha=0.9, edgecolor="black", linewidth=1.2)
    ax2.set_title("Ablation: Forward Transfer on Partial Conflict\n(Higher is Better)")
    ax2.set_ylabel("Forward Transfer (FWT)")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "ablation_breakdown.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Continual RL Results")
    parser.add_argument("--input", type=str, default="results", help="Directory containing result JSON files")
    parser.add_argument("--output-dir", type=str, default="results/plots", help="Directory to save generated plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    records = load_all_results(args.input)
    if not records:
        print(f"No result files found in {args.input}")
        return

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} experiment runs across strategies and sequences.")

    table_md = generate_summary_table(df, args.output_dir)
    print("\nSummary Table:")
    print(table_md)

    plot_performance_and_forgetting(df, args.output_dir)
    plot_forgetting_vs_conflict(df, args.output_dir)
    plot_stability_plasticity_pareto(df, args.output_dir)
    plot_ablation_comparison(df, args.output_dir)


if __name__ == "__main__":
    main()
