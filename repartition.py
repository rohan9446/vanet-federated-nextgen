"""
repartition.py — Create partitions with natural heterogeneity.

Downsample attacks by SENDER before partitioning (not per-type-balanced).
This preserves spatial clustering of attackers → different RSUs see
different attack rates and attack type mixes.

Usage:
    python repartition.py
"""

import gc
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from preprocess import (
    CLASSIFICATION_FEATURES, PARTITION_FEATURES, LABEL_COLS,
    aggregate_and_dedup, engineer_sequential_features,
    partition_kmeans, save_partitions,
)

EXCLUDE = {"Normal", "timeDelayAttack"}
SEED = 42

data_dir = Path("data/per_attack/train")
output_dir = Path("data/processed")

# ---------------------------------------------------------------
# Step 1: Process all attack files
# ---------------------------------------------------------------
print("=" * 60)
print("Step 1: Process attack files")
print("=" * 60)

attack_files = sorted([f for f in data_dir.glob("*.parquet") if f.stem not in EXCLUDE])

normal_collected = False
all_normals = None
all_attacks = []

for pf in attack_files:
    print(f"  {pf.stem}...", end=" ")
    df = pd.read_parquet(pf)
    df = aggregate_and_dedup(df, verbose=False)
    df = engineer_sequential_features(df, verbose=False)

    df.loc[df["attacker"] == 0, "attack_type"] = "Normal"
    df.loc[df["attacker"] == 0, "attack_category"] = "Normal"
    df.loc[df["attacker"] == 0, "attack_triclass"] = "Normal"

    if not normal_collected:
        all_normals = df[df["attacker"] == 0].copy()
        normal_collected = True

    atk = df[df["attacker"] == 1].copy()
    if len(atk) > 0:
        all_attacks.append(atk)
        print(f"{len(atk):,} attacks")
    else:
        print("no attacks")

    del df
    gc.collect()

combined = pd.concat([all_normals] + all_attacks, ignore_index=True)
del all_normals, all_attacks
gc.collect()

n = len(combined)
na = int(combined["attacker"].sum())
print(f"\nBefore downsampling: {n:,} rows | "
      f"normal={n-na:,} ({(n-na)/n:.1%}) | attack={na:,} ({na/n:.1%})")

# ---------------------------------------------------------------
# Step 2: Sender-level downsampling (NOT per-type balanced)
# ---------------------------------------------------------------
print(f"\n{'='*60}")
print("Step 2: Sender-level downsampling (natural distribution)")
print("=" * 60)

rng = np.random.RandomState(SEED)

normal = combined[combined["attacker"] == 0]
attack = combined[combined["attacker"] == 1]

n_normal = len(normal)
target_rate = 0.25
n_target_attack = int(n_normal * target_rate / (1 - target_rate))

# Get attack senders with their row counts and positions
atk_sender_stats = attack.groupby("sender_id").agg(
    count=("attacker", "size"),
    mean_x=("sender_pos_x", "mean"),
    mean_y=("sender_pos_y", "mean"),
    attack_type=("attack_type", "first"),
    scenario=("scenario", "first"),
).reset_index()

print(f"  Total attack senders: {len(atk_sender_stats):,}")
print(f"  Total attack rows:    {len(attack):,}")
print(f"  Target attack rows:   {n_target_attack:,}")

# Shuffle senders randomly and take until we reach target
atk_sender_stats = atk_sender_stats.sample(frac=1, random_state=rng).reset_index(drop=True)
cumulative = atk_sender_stats["count"].cumsum()
n_keep = (cumulative <= n_target_attack).sum()
n_keep = max(n_keep, 1)

keep_senders = set(atk_sender_stats["sender_id"].iloc[:n_keep])
attack_sampled = attack[attack["sender_id"].isin(keep_senders)]

combined = pd.concat([normal, attack_sampled], ignore_index=True)
combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

n = len(combined)
na = int(combined["attacker"].sum())
print(f"\n  After downsampling: {n:,} rows | "
      f"normal={n-na:,} ({(n-na)/n:.1%}) | attack={na:,} ({na/n:.1%})")

# Show natural distribution (NOT forced equal)
print(f"\n  Attack type distribution (natural, not forced equal):")
atk_dist = combined[combined["attacker"] == 1]["attack_type"].value_counts()
for at, cnt in atk_dist.sort_index().items():
    print(f"    {at:40s} {cnt:>8,}")

print(f"\n  Attack scenario distribution:")
atk_sc = combined[combined["attacker"] == 1]["scenario"].value_counts()
for sc, cnt in atk_sc.sort_index().items():
    print(f"    {sc:20s} {cnt:>8,} ({cnt/na:.1%})")

# ---------------------------------------------------------------
# Step 3: K-Means partition (natural heterogeneity)
# ---------------------------------------------------------------
print(f"\n{'='*60}")
print("Step 3: K-Means partitioning")
print("=" * 60)

for k in [4, 8, 16]:
    print(f"\n  --- k={k} ---")
    asn, km = partition_kmeans(combined, k, seed=SEED)
    save_partitions(combined, asn, km, k, str(output_dir))

    # Detailed heterogeneity report
    print(f"\n  Heterogeneity analysis:")
    rates = []
    for c in range(k):
        mask = asn == c
        part = combined[mask]
        n_p = len(part)
        na_p = int(part["attacker"].sum())
        rate = na_p / n_p if n_p > 0 else 0
        rates.append(rate)

        # Attack type mix
        if na_p > 0:
            top_types = part[part["attacker"] == 1]["attack_type"].value_counts().head(3)
            types_str = ", ".join([f"{t}:{c}" for t, c in top_types.items()])
        else:
            types_str = "none"

        print(f"    C{c:>2}: {n_p:>8,} rows | atk: {rate:>5.1%} | "
              f"top attacks: {types_str}")

    print(f"\n  Attack rate range: [{min(rates):.1%}, {max(rates):.1%}] | "
          f"std: {np.std(rates):.3f}")

print(f"\n{'='*60}")
print("Done. Partitions updated with natural heterogeneity.")
print("Centralized train/val/test.parquet are UNTOUCHED.")
print("=" * 60)