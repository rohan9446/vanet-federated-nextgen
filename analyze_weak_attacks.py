"""
analyze_weak_attacks.py — Deep analysis of positionMirroring and dataReplay.

Usage:
    python analyze_weak_attacks.py --data_dir data/per_attack
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from preprocess import aggregate_and_dedup, engineer_sequential_features

FEATURES = [
    "sender_spd", "sender_acl", "time_delta", "distance", "heading_change",
    "speed_consistency", "accel_consistency", "heading_trajectory_consistency",
    "tx_delay", "mean_sender_receiver_dist", "n_receivers",
    "sender_dist_to_road_edge",
]


def analyze_attack(data_dir, attack_name):
    """Load, dedup, engineer features, then analyze attack vs normal."""

    print(f"\n{'='*70}")
    print(f"DEEP ANALYSIS: {attack_name}")
    print(f"{'='*70}")

    # Load and process
    df = pd.read_parquet(Path(data_dir) / "train" / f"{attack_name}.parquet")
    print(f"\n  Raw rows: {len(df):,}")

    df = aggregate_and_dedup(df, verbose=True)
    df = engineer_sequential_features(df, verbose=True)

    normal = df[df["attacker"] == 0]
    attack = df[df["attacker"] == 1]

    print(f"\n  After dedup: {len(df):,} rows")
    print(f"  Normal: {len(normal):,} | Attack: {len(attack):,}")

    # ---- 1. What exactly does this attack modify? ----
    print(f"\n  --- 1. RAW FIELD COMPARISON ---")

    # Load Normal baseline for ground truth comparison
    normal_file = pd.read_parquet(Path(data_dir) / "train" / "Normal.parquet",
                                   columns=["sender_id", "messageID",
                                            "sender_pos_x", "sender_pos_y",
                                            "sender_spd", "sender_acl",
                                            "sender_hed",
                                            "sender_dist_to_road_edge"])
    normal_file = normal_file.drop_duplicates(["sender_id", "messageID"])

    # Get attack messages and find their "true" values from Normal baseline
    atk_msgs = attack[["sender_id", "messageID",
                        "sender_pos_x", "sender_pos_y",
                        "sender_spd", "sender_acl", "sender_hed",
                        "sender_dist_to_road_edge"]].copy()

    merged = atk_msgs.merge(normal_file, on=["sender_id", "messageID"],
                             suffixes=("_atk", "_true"))

    if len(merged) > 0:
        print(f"  Matched {len(merged):,} attack messages to ground truth")

        for field in ["sender_pos_x", "sender_pos_y", "sender_spd",
                       "sender_acl", "sender_hed", "sender_dist_to_road_edge"]:
            atk_col = f"{field}_atk"
            true_col = f"{field}_true"
            if atk_col in merged.columns and true_col in merged.columns:
                diff = merged[atk_col] - merged[true_col]
                n_changed = (diff.abs() > 0.01).sum()
                pct = n_changed / len(merged) * 100

                print(f"\n    {field}:")
                print(f"      Changed: {n_changed:,} ({pct:.1f}%)")
                if n_changed > 0:
                    print(f"      Diff: mean={diff.mean():.4f} std={diff.std():.4f} "
                          f"min={diff.min():.4f} max={diff.max():.4f}")
                    print(f"      True: mean={merged[true_col].mean():.2f} "
                          f"std={merged[true_col].std():.2f}")
                    print(f"      Attack: mean={merged[atk_col].mean():.2f} "
                          f"std={merged[atk_col].std():.2f}")

        # Position displacement
        pos_diff = np.sqrt(
            (merged["sender_pos_x_atk"] - merged["sender_pos_x_true"])**2 +
            (merged["sender_pos_y_atk"] - merged["sender_pos_y_true"])**2
        )
        print(f"\n    Position displacement (Euclidean):")
        print(f"      Mean: {pos_diff.mean():.2f}m")
        print(f"      Std:  {pos_diff.std():.2f}m")
        print(f"      Min:  {pos_diff.min():.2f}m")
        print(f"      Max:  {pos_diff.max():.2f}m")
        print(f"      Zero: {(pos_diff < 0.01).sum():,} "
              f"({(pos_diff < 0.01).mean()*100:.1f}%)")
    else:
        print(f"  WARNING: No matching messages found")

    # ---- 2. Feature distributions ----
    print(f"\n  --- 2. FEATURE DISTRIBUTIONS ---")
    print(f"  {'Feature':<36} {'Normal_mean':>12} {'Attack_mean':>12} "
          f"{'Effect':>8} {'Overlap%':>10}")
    print(f"  {'-'*80}")

    for feat in FEATURES:
        if feat not in df.columns:
            continue
        n_vals = normal[feat].dropna()
        a_vals = attack[feat].dropna()

        nm, am = n_vals.mean(), a_vals.mean()
        ns = n_vals.std()
        effect = abs(am - nm) / ns if ns > 0 else 0

        # Distribution overlap (approximate via percentile comparison)
        # What % of attack values fall within normal's 5th-95th percentile?
        n_low, n_high = n_vals.quantile(0.05), n_vals.quantile(0.95)
        overlap = ((a_vals >= n_low) & (a_vals <= n_high)).mean() * 100

        marker = " ***" if effect > 0.5 else " **" if effect > 0.2 else ""
        print(f"  {feat:<36} {nm:>12.4f} {am:>12.4f} "
              f"{effect:>7.3f}{marker} {overlap:>9.1f}%")

    # ---- 3. Per-sender analysis ----
    print(f"\n  --- 3. PER-SENDER ANALYSIS ---")

    atk_senders = attack["sender_id"].unique()
    print(f"  Attack senders: {len(atk_senders)}")

    # How many messages per attack sender?
    msgs_per_sender = attack.groupby("sender_id").size()
    print(f"  Messages per attacker: mean={msgs_per_sender.mean():.1f} "
          f"min={msgs_per_sender.min()} max={msgs_per_sender.max()}")

    # Per-sender feature variance (are all attack messages similar?)
    print(f"\n  Per-sender feature consistency (std within each attacker):")
    for feat in ["speed_consistency", "sender_dist_to_road_edge",
                  "heading_trajectory_consistency", "mean_sender_receiver_dist"]:
        if feat in attack.columns:
            per_sender_std = attack.groupby("sender_id")[feat].std().mean()
            print(f"    {feat:40s} within-sender std = {per_sender_std:.4f}")

    # ---- 4. Subgroup analysis ----
    print(f"\n  --- 4. DETECTABLE vs UNDETECTABLE SUBGROUPS ---")

    # Which attack messages WOULD be caught by a simple threshold?
    # Use speed_consistency as primary detector
    if "speed_consistency" in attack.columns:
        sc_threshold = normal["speed_consistency"].quantile(0.99)
        caught_sc = (attack["speed_consistency"].abs() > abs(sc_threshold)).sum()
        print(f"  speed_consistency > {sc_threshold:.4f}: "
              f"catches {caught_sc:,} / {len(attack):,} "
              f"({caught_sc/len(attack)*100:.1f}%)")

    if "sender_dist_to_road_edge" in attack.columns:
        # Distance to road edge: negative means off-road
        off_road = (attack["sender_dist_to_road_edge"] < 0).sum()
        n_off_road = (normal["sender_dist_to_road_edge"] < 0).sum()
        print(f"  sender_dist_to_road_edge < 0 (off-road): "
              f"attack={off_road:,} ({off_road/len(attack)*100:.1f}%) "
              f"vs normal={n_off_road:,} ({n_off_road/len(normal)*100:.1f}%)")

    if "heading_trajectory_consistency" in attack.columns:
        htc_threshold = normal["heading_trajectory_consistency"].quantile(0.99)
        caught_htc = (attack["heading_trajectory_consistency"] > htc_threshold).sum()
        print(f"  heading_traj_consistency > {htc_threshold:.2f}: "
              f"catches {caught_htc:,} / {len(attack):,} "
              f"({caught_htc/len(attack)*100:.1f}%)")

    # ---- 5. Scenario breakdown ----
    print(f"\n  --- 5. PER-SCENARIO DETECTION ---")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    for scenario in sorted(df["scenario"].unique()):
        s_normal = normal[normal["scenario"] == scenario]
        s_attack = attack[attack["scenario"] == scenario]

        if len(s_attack) < 10 or len(s_normal) < 10:
            continue

        s_df = pd.concat([s_normal, s_attack])
        X = s_df[FEATURES].fillna(0).values
        y = s_df["attacker"].values

        sc = StandardScaler().fit(X)
        m = RandomForestClassifier(n_estimators=50, class_weight="balanced",
                                    random_state=42, n_jobs=-1)
        m.fit(sc.transform(X), y)
        auc = roc_auc_score(y, m.predict_proba(sc.transform(X))[:, 1])

        print(f"    {scenario:20s} normal={len(s_normal):>6,} "
              f"attack={len(s_attack):>6,} | AUC={auc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    analyze_attack(args.data_dir, "positionMirroring")
    analyze_attack(args.data_dir, "dataReplay")

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()