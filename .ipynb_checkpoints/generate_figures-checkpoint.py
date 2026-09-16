"""
generate_figures.py — Generate all paper figures as publication-quality PDFs.

Usage:
    python generate_figures.py --data_dir data/processed --output_dir figures/
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from sklearn.cluster import KMeans

# ── Journal style ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "axes.grid": False,
})

def get_cmap(name, n):
    """Compatible with both old and new matplotlib."""
    try:
        return matplotlib.colormaps[name].resampled(n)
    except AttributeError:
        return plt.cm.get_cmap(name, n)


# ═══════════════════════════════════════════════════════════════════
# Fig 1: Multi-Receiver Structure Diagram
# ═══════════════════════════════════════════════════════════════════
def fig1_multi_receiver(output_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    sender = FancyBboxPatch((0.5, 2.5), 2, 1, boxstyle="round,pad=0.15",
                            facecolor="#4472C4", edgecolor="black", linewidth=1.2)
    ax.add_patch(sender)
    ax.text(1.5, 3.0, "Sender\nVehicle", ha="center", va="center",
            fontsize=10, fontweight="bold", color="white")

    ax.annotate("", xy=(3.8, 3.0), xytext=(2.7, 3.0),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.5))
    ax.text(3.25, 3.35, "BSM\nbroadcast", ha="center", va="bottom", fontsize=8, style="italic")

    positions = [(4.2, 4.5), (4.2, 2.5), (4.2, 0.5), (4.2, 1.5)]
    labels = ["Receiver 1", "Receiver 2", "Receiver 3", "Receiver N"]

    for i, ((x, y), lab) in enumerate(zip(positions, labels)):
        if i == 3:
            ax.text(x + 1.0, y + 0.55, "\u22EE", ha="center", va="center", fontsize=14)
        rec = FancyBboxPatch((x, y - 0.35), 2, 0.7, boxstyle="round,pad=0.1",
                            facecolor="#70AD47", edgecolor="black", linewidth=1)
        ax.add_patch(rec)
        ax.text(x + 1.0, y, lab, ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax.annotate("", xy=(x, y), xytext=(3.8, 3.0),
                    arrowprops=dict(arrowstyle="-|>", color="#999999", lw=0.8,
                                   connectionstyle="arc3,rad=0.1"))

    for i, ((x, y), lab) in enumerate(zip(positions, labels)):
        lx, ly = 7.0, y
        log = FancyBboxPatch((lx, ly - 0.3), 2.2, 0.6, boxstyle="round,pad=0.08",
                             facecolor="#FFC000", edgecolor="black", linewidth=0.8)
        ax.add_patch(log)
        ax.text(lx + 1.1, ly, f"Log file {i+1 if i < 3 else 'N'}", ha="center", va="center", fontsize=8)
        ax.annotate("", xy=(lx, ly), xytext=(x + 2, y),
                    arrowprops=dict(arrowstyle="-|>", color="#999999", lw=0.7))

    ax.text(8.1, 5.5, "Same BSM recorded\n14.8\u00D7 on average", ha="center", va="top",
            fontsize=9, style="italic", color="#C00000",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFE0E0", edgecolor="#C00000", linewidth=0.8))

    ax.set_title("Fig. 1. VeReMi NextGen multi-receiver log structure", fontsize=11, pad=10)
    fig.savefig(Path(output_dir) / "fig1_multi_receiver.pdf")
    fig.savefig(Path(output_dir) / "fig1_multi_receiver.png")
    plt.close(fig)
    print("  \u2713 fig1_multi_receiver")


# ═══════════════════════════════════════════════════════════════════
# Fig 2: Pipeline Flowchart
# ═══════════════════════════════════════════════════════════════════
def fig2_pipeline(output_dir):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    boxes = [
        (5, 9.2, "Raw per-attack simulation files\n(~3M rows each, multi-receiver)", "#D9E2F3"),
        (5, 7.8, "Stage 1: Receiver Aggregation\nmean_dist, n_receivers, tx_delay\nper (sender_id, messageID)", "#B4C6E7"),
        (5, 6.2, "Stage 2: Deduplication\n\u2192 ~200K unique messages per file", "#B4C6E7"),
        (5, 4.6, "Stage 3: Sequential Feature Engineering\nspeed_consistency, accel_consistency,\nheading_traj_consistency, etc.", "#B4C6E7"),
        (5, 3.0, "Combine all attack files + normals\n\u2192 Sender-level downsampling (25% attack)", "#E2EFDA"),
        (5, 1.6, "K-Means RSU Partitioning\n(k = 4, 8, 16)\n\u2192 Apply train/val/test splits per partition", "#E2EFDA"),
        (5, 0.3, "Federated Training\nXGB Ensemble | FedAvg | FedProx", "#FFF2CC"),
    ]

    for (cx, cy, text, color) in boxes:
        w, h = 4.2, 1.0
        box = FancyBboxPatch((cx - w/2, cy - h/2), w, h, boxstyle="round,pad=0.15",
                             facecolor=color, edgecolor="black", linewidth=1)
        ax.add_patch(box)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=8, linespacing=1.3)

    arrow_ys = [(8.7, 8.3), (7.3, 6.7), (5.7, 5.1), (4.1, 3.5), (2.5, 2.1), (1.1, 0.8)]
    for (y1, y2) in arrow_ys:
        ax.annotate("", xy=(5, y2), xytext=(5, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2))

    ax.text(8.0, 7.8, "Receiver features\ncomputed HERE\n(before dedup)", ha="center",
            fontsize=8, color="#C00000", style="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFE0E0", edgecolor="#C00000", lw=0.6))
    ax.annotate("", xy=(7.1, 7.8), xytext=(7.7, 7.8),
                arrowprops=dict(arrowstyle="-|>", color="#C00000", lw=0.8))

    ax.text(8.0, 4.6, "Scaler fit on\ntrain data only", ha="center",
            fontsize=8, color="#4472C4", style="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#D9E2F3", edgecolor="#4472C4", lw=0.6))
    ax.annotate("", xy=(7.1, 4.6), xytext=(7.7, 4.6),
                arrowprops=dict(arrowstyle="-|>", color="#4472C4", lw=0.8))

    ax.set_title("Fig. 2. Data processing pipeline", fontsize=11, pad=10)
    fig.savefig(Path(output_dir) / "fig2_pipeline.pdf")
    fig.savefig(Path(output_dir) / "fig2_pipeline.png")
    plt.close(fig)
    print("  \u2713 fig2_pipeline")


# ═══════════════════════════════════════════════════════════════════
# Fig 3: RSU Partition Heterogeneity
# ═══════════════════════════════════════════════════════════════════
def fig3_rsu_partitions(output_dir, data_dir):
    train_path = Path(data_dir) / "train.parquet"
    coords = None
    if train_path.exists():
        df = pd.read_parquet(train_path)
        if "sender_pos_x" in df.columns:
            coords = df[["sender_pos_x", "sender_pos_y"]].dropna().values
            print(f"  Loaded {len(coords)} positions from train.parquet")
        del df

    if coords is None:
        print("  No position data found.")
        return

    rng = np.random.RandomState(42)
    n_sample = min(15000, len(coords))
    sample = coords[rng.choice(len(coords), n_sample, replace=False)]

    for k in [4, 8, 16]:
        fig, ax = plt.subplots(figsize=(6, 5))
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(sample)
        cmap = get_cmap("tab20", k)

        for c in range(k):
            mask = labels == c
            ax.scatter(sample[mask, 0], sample[mask, 1],
                       s=2, alpha=0.5, c=[cmap(c)], label=f"RSU {c+1}", rasterized=True)

        ax.set_xlabel("Position X (m)")
        ax.set_ylabel("Position Y (m)")
        ax.set_title(f"Fig. 3. RSU spatial partitions (k = {k})", fontsize=11, pad=10)
        ax.legend(loc="best", fontsize=7, markerscale=4, ncol=2 if k > 8 else 1)
        fig.tight_layout()
        fig.savefig(Path(output_dir) / f"fig3_rsu_k{k}.pdf")
        fig.savefig(Path(output_dir) / f"fig3_rsu_k{k}.png")
        plt.close(fig)
        print(f"  ✓ fig3_rsu_k{k}")


# ═══════════════════════════════════════════════════════════════════
# Fig 4: Feature Importance Bar Chart
# ═══════════════════════════════════════════════════════════════════
def fig4_feature_importance(output_dir):
    features = [
        "speed_consistency", "accel_consistency", "heading_traj_consist.",
        "dist_to_road_edge", "distance", "sender_spd",
        "heading_change", "sender_acl", "n_receivers",
        "mean_sndr_rcvr_dist", "tx_delay", "time_delta"
    ]
    rf_imp = [0.264, 0.162, 0.125, 0.113, 0.096, 0.059, 0.054, 0.048, 0.033, 0.029, 0.017, 0.001]

    categories = ["Sequential", "Sequential", "Sequential", "Context", "Sequential",
                  "Raw", "Sequential", "Raw", "Receiver", "Receiver", "Receiver", "Sequential"]
    cat_colors = {"Sequential": "#4472C4", "Raw": "#70AD47", "Receiver": "#FFC000", "Context": "#C55A11"}
    colors = [cat_colors[c] for c in categories]

    fig, ax = plt.subplots(figsize=(7, 4))
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, rf_imp, color=colors, edgecolor="black", linewidth=0.4, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Random Forest)")
    ax.set_xlim(0, 0.30)

    for bar, val in zip(bars, rf_imp):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)

    handles = [mpatches.Patch(color=cat_colors[c], label=c) for c in ["Sequential", "Raw", "Receiver", "Context"]]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)

    ax.set_title("Fig. 4. Feature importance \u2014 Random Forest (binary classification)", fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "fig4_feature_importance.pdf")
    fig.savefig(Path(output_dir) / "fig4_feature_importance.png")
    plt.close(fig)
    print("  \u2713 fig4_feature_importance")


# ═══════════════════════════════════════════════════════════════════
# Fig 5: Per-Attack AUC Bar Chart
# ═══════════════════════════════════════════════════════════════════
def fig5_per_attack_auc(output_dir):
    attacks = [
        "zeroSpeedReport", "suddenStop", "feignedBraking",
        "randomSpeedOffset", "reversedHeading", "accelMultiplication",
        "suddenConstantSpeed", "randomPositionOffset", "dosAttack",
        "constantSpeedOffset", "trafficCongestionSybil",
        "constantPositionOffset", "dataReplay", "positionMirroring"
    ]
    aucs = [1.000, 1.000, 0.9999, 0.9998, 0.9997, 0.9996,
            0.9996, 0.9995, 0.9993, 0.9992, 0.9979,
            0.9147, 0.7751, 0.7089]

    categories = ["Speed", "Multi", "Accel", "Speed", "Heading", "Accel",
                  "Speed", "Position", "Multi", "Speed", "Multi",
                  "Position", "Multi", "Position"]
    cat_colors = {
        "Position": "#C55A11", "Speed": "#4472C4", "Heading": "#70AD47",
        "Accel": "#7030A0", "Multi": "#FFC000"
    }
    colors = [cat_colors[c] for c in categories]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = np.arange(len(attacks))
    bars = ax.barh(y_pos, aucs, color=colors, edgecolor="black", linewidth=0.4, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(attacks, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("AUC (Random Forest, per-attack binary)")
    ax.set_xlim(0.6, 1.02)

    ax.axvline(x=0.91, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(0.908, -0.5, "AUC = 0.91", color="red", fontsize=8, ha="right")

    for bar, val, atk in zip(bars, aucs, attacks):
        if val < 0.95:
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=8, color="#C00000", fontweight="bold")

    handles = [mpatches.Patch(color=cat_colors[c], label=c)
               for c in ["Position", "Speed", "Heading", "Accel", "Multi"]]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)

    ax.set_title("Fig. 5. Per-attack binary detection performance (AUC)", fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "fig5_per_attack_auc.pdf")
    fig.savefig(Path(output_dir) / "fig5_per_attack_auc.png")
    plt.close(fig)
    print("  \u2713 fig5_per_attack_auc")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output_dir", default="figures")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Generating paper figures...")
    fig1_multi_receiver(out)
    fig2_pipeline(out)
    fig3_rsu_partitions(out, args.data_dir)
    fig4_feature_importance(out)
    fig5_per_attack_auc(out)
    print(f"\nAll figures saved to {out}/")


if __name__ == "__main__":
    main()