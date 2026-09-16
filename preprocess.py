"""
preprocess.py — VeReMi NextGen preprocessing pipeline (v4 - FINAL).

Pipeline:
  1. Load per-attack file (complete simulation)
  2. Aggregate receiver copies → mean_sender_receiver_dist, n_receivers, tx_delay
  3. Deduplicate by (sender_id, messageID) → ~200K rows per file
  4. Engineer sequential features (groupby scenario+sender, sort sendTime)
  5. Keep all rows (attacker=0 + attacker=1)
  6. Combine: normals from one file + attacks from all files
  7. Report distribution, then downsample by SENDER (not row)
  8. K-Means partition

Features (12 classification + 2 partition):
  Raw (2):        sender_spd, sender_acl
  Sequential (6): time_delta, distance, heading_change, speed_consistency,
                   accel_consistency, heading_trajectory_consistency
  Receiver (3):   tx_delay, mean_sender_receiver_dist, n_receivers
  Context (1):    sender_dist_to_road_edge
  Partition only: sender_pos_x, sender_pos_y

Usage:
    python preprocess.py --data_dir data/per_attack --output_dir data/processed
"""

import argparse
import gc
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
NS_TO_SEC = 1e-9
MIN_TIME_DELTA = 0.05

CLASSIFICATION_FEATURES = [
    "sender_spd", "sender_acl",
    "time_delta", "distance", "heading_change",
    "speed_consistency", "accel_consistency",
    "heading_trajectory_consistency",
    "tx_delay", "mean_sender_receiver_dist", "n_receivers",
    "sender_dist_to_road_edge",
]

PARTITION_FEATURES = ["sender_pos_x", "sender_pos_y"]

ALL_OUTPUT_COLS = CLASSIFICATION_FEATURES + PARTITION_FEATURES

LABEL_COLS = ["attacker", "attack_type", "attack_category",
              "attack_triclass", "scenario", "sender_id"]

ATTACK_CATEGORY = {
    "Normal":                     "Normal",
    "timeDelayAttack":            "Time",
    "constantPositionOffset":     "Position",
    "randomPositionOffset":       "Position",
    "positionMirroring":          "Position",
    "constantSpeedOffset":        "Speed",
    "randomSpeedOffset":          "Speed",
    "zeroSpeedReport":            "Speed",
    "suddenConstantSpeed":        "Speed",
    "reversedHeading":            "Heading",
    "feignedBraking":             "Acceleration",
    "accelerationMultiplication": "Acceleration",
    "suddenStop":                 "Multi",
    "dosAttack":                  "Multi",
    "trafficCongestionSybil":     "Multi",
    "dataReplay":                 "Multi",
}

ATTACK_TRICLASS = {
    "Normal":                     "Normal",
    "constantPositionOffset":     "Single-param",
    "randomPositionOffset":       "Single-param",
    "positionMirroring":          "Single-param",
    "constantSpeedOffset":        "Single-param",
    "randomSpeedOffset":          "Single-param",
    "zeroSpeedReport":            "Single-param",
    "suddenConstantSpeed":        "Single-param",
    "reversedHeading":            "Single-param",
    "feignedBraking":             "Single-param",
    "accelerationMultiplication": "Single-param",
    "timeDelayAttack":            "Single-param",
    "suddenStop":                 "Multi-param",
    "dosAttack":                  "Multi-param",
    "trafficCongestionSybil":     "Multi-param",
    "dataReplay":                 "Multi-param",
}

# File used for normal messages (avoid Sybil which adds phantom receivers)
NORMAL_SOURCE_FILE = "accelerationMultiplication"


# -----------------------------------------------------------------------
# Step 1+2: Aggregate receiver copies → deduplicate
# -----------------------------------------------------------------------

def aggregate_and_dedup(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """
    Aggregate per-receiver copies, then collapse to one row per message.
    
    Before dedup:  ~3M rows (14.8 copies per message)
    After dedup:   ~200K rows (one per unique sender message)
    """
    t0 = time.time()
    n_before = len(df)

    # Compute per-copy features BEFORE dedup
    df["_sr_dist"] = np.sqrt(
        (df["sender_pos_x"] - df["receiver_pos_x"]) ** 2 +
        (df["sender_pos_y"] - df["receiver_pos_y"]) ** 2
    )
    df["_tx_delay"] = ((df["rcvTime"] - df["sendTime"]) * NS_TO_SEC).clip(lower=0)

    # Aggregate across copies of same message
    agg = df.groupby(["sender_id", "messageID"]).agg(
        mean_sender_receiver_dist=("_sr_dist", "mean"),
        n_receivers=("_sr_dist", "count"),
        tx_delay=("_tx_delay", "mean"),
    ).reset_index()

    # Deduplicate: keep first row per (sender_id, messageID) for sender fields
    sender_cols = [
        "sender_id", "messageID", "sendTime", "attacker", "attack_type",
        "scenario", "sender_pos_x", "sender_pos_y",
        "sender_spd", "sender_acl", "sender_hed",
        "sender_dist_to_road_edge",
    ]
    # Only keep columns that exist
    sender_cols = [c for c in sender_cols if c in df.columns]
    
    deduped = df[sender_cols].drop_duplicates(
        subset=["sender_id", "messageID"], keep="first"
    )

    # Merge aggregated features
    result = deduped.merge(agg, on=["sender_id", "messageID"], how="left")

    n_after = len(result)
    if verbose:
        print(f"      Dedup: {n_before:,} -> {n_after:,} rows "
              f"({n_before/n_after:.1f}x reduction)")

    del df, agg, deduped
    gc.collect()

    return result


# -----------------------------------------------------------------------
# Step 3: Sequential feature engineering
# -----------------------------------------------------------------------

def engineer_sequential_features(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """
    Compute sequential features on DEDUPLICATED data.
    Groups by (scenario, sender_id), sorted by sendTime.
    """
    t0 = time.time()

    df = df.sort_values(["scenario", "sender_id", "sendTime"]).reset_index(drop=True)
    grp = df.groupby(["scenario", "sender_id"], sort=False)

    prev_px = grp["sender_pos_x"].shift(1)
    prev_py = grp["sender_pos_y"].shift(1)
    prev_spd = grp["sender_spd"].shift(1)
    prev_hed = grp["sender_hed"].shift(1)
    prev_st = grp["sendTime"].shift(1)

    # time_delta (seconds, from sendTime)
    df["time_delta"] = ((df["sendTime"] - prev_st) * NS_TO_SEC).clip(lower=0)
    reliable = df["time_delta"] >= MIN_TIME_DELTA

    # distance
    dx = df["sender_pos_x"] - prev_px
    dy = df["sender_pos_y"] - prev_py
    df["distance"] = np.sqrt(dx**2 + dy**2)

    # speed_consistency
    derived_spd = np.where(reliable, df["distance"] / df["time_delta"], 0.0)
    df["speed_consistency"] = np.where(
        reliable, df["sender_spd"] - derived_spd, 0.0)

    # accel_consistency
    spd_change = df["sender_spd"] - prev_spd
    derived_acl = np.where(reliable, spd_change / df["time_delta"], 0.0)
    df["accel_consistency"] = np.where(
        reliable, df["sender_acl"] - derived_acl, 0.0)

    # heading_trajectory_consistency (SUMO: 0=North, 90=East, CW)
    atan2_deg = np.degrees(np.arctan2(dy, dx))
    move_hed = (90.0 - atan2_deg) % 360.0
    rep_hed = df["sender_hed"] % 360.0
    adiff = np.abs(rep_hed - move_hed)
    df["heading_trajectory_consistency"] = np.minimum(adiff, 360.0 - adiff)
    df.loc[df["distance"] < 0.1, "heading_trajectory_consistency"] = 0.0

    # heading_change
    h1 = np.deg2rad(df["sender_hed"])
    h2 = np.deg2rad(prev_hed)
    cs = np.cos(h1) * np.cos(h2) + np.sin(h1) * np.sin(h2)
    df["heading_change"] = np.degrees(np.arccos(np.clip(cs, -1.0, 1.0)))

    # Fill NaN for first message per sender
    fill = ["time_delta", "distance", "speed_consistency", "accel_consistency",
            "heading_trajectory_consistency", "heading_change"]
    df[fill] = df[fill].fillna(0.0)

    # Labels
    df["attack_category"] = df["attack_type"].map(ATTACK_CATEGORY).fillna("Unknown")
    df["attack_triclass"] = df["attack_type"].map(ATTACK_TRICLASS).fillna("Unknown")

    # Fill remaining nulls
    df["sender_dist_to_road_edge"] = df["sender_dist_to_road_edge"].fillna(0.0)
    df["mean_sender_receiver_dist"] = df["mean_sender_receiver_dist"].fillna(0.0)
    df["n_receivers"] = df["n_receivers"].fillna(1)

    if verbose:
        print(f"      Sequential features: {time.time()-t0:.1f}s")

    return df


# -----------------------------------------------------------------------
# Process one attack file (Steps 1-4)
# -----------------------------------------------------------------------

def process_attack_file(parquet_path, attack_name, verbose=True):
    """
    Full pipeline for one attack file:
    1. Load
    2. Aggregate receiver copies
    3. Deduplicate
    4. Engineer sequential features
    Returns (normal_df, attack_df)
    """
    df = pd.read_parquet(parquet_path)
    n_total = len(df)
    n_atk = int(df["attacker"].sum())

    if verbose:
        print(f"\n    {attack_name}: {n_total:,} rows -> ", end="")

    # Aggregate + dedup
    df = aggregate_and_dedup(df, verbose=verbose)

    # Sequential features
    df = engineer_sequential_features(df, verbose=verbose)

    # Keep output columns
    keep = [c for c in ALL_OUTPUT_COLS + LABEL_COLS if c in df.columns]
    df = df[keep]

    normal_df = df[df["attacker"] == 0].reset_index(drop=True)
    attack_df = df[df["attacker"] == 1].reset_index(drop=True)

    if verbose:
        print(f"      Result: {len(normal_df):,} normal + "
              f"{len(attack_df):,} attack")

    del df
    gc.collect()

    return normal_df, attack_df


# -----------------------------------------------------------------------
# Sender-level downsampling
# -----------------------------------------------------------------------

def downsample_by_sender(df, target_rate=0.25, seed=42, verbose=True):
    """
    Downsample attacks by removing entire SENDERS (preserving timelines).
    """
    rng = np.random.RandomState(seed)

    normal = df[df["attacker"] == 0]
    attack = df[df["attacker"] == 1]

    n_normal = len(normal)
    n_attack = len(attack)
    current_rate = n_attack / len(df)

    if verbose:
        print(f"    Before: {len(df):,} rows | "
              f"normal={n_normal:,} | attack={n_attack:,} | "
              f"rate={current_rate:.1%}")

    if current_rate <= target_rate:
        if verbose:
            print(f"    Already at or below target. No downsampling needed.")
        return df

    n_target_attack = int(n_normal * target_rate / (1 - target_rate))
    types = sorted(attack["attack_type"].unique())
    per_type = n_target_attack // len(types)

    if verbose:
        print(f"    Target: {n_target_attack:,} attack rows | "
              f"Per type ({len(types)}): ~{per_type:,}")

    sampled_attacks = []
    for at in types:
        at_df = attack[attack["attack_type"] == at]
        at_senders = at_df.groupby("sender_id").size().reset_index(name="count")

        # Shuffle senders and take until we reach per_type rows
        at_senders = at_senders.sample(frac=1, random_state=rng)
        cumulative = at_senders["count"].cumsum()
        n_senders_to_keep = (cumulative <= per_type).sum()
        # Keep at least 1 sender
        n_senders_to_keep = max(1, n_senders_to_keep)

        keep_senders = set(at_senders["sender_id"].iloc[:n_senders_to_keep])
        kept = at_df[at_df["sender_id"].isin(keep_senders)]
        sampled_attacks.append(kept)

        if verbose:
            print(f"      {at:40s} senders: {len(at_senders):>5,} -> "
                  f"{n_senders_to_keep:>5,} | rows: {len(at_df):>8,} -> "
                  f"{len(kept):>8,}")

    result = pd.concat([normal] + sampled_attacks, ignore_index=True)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)

    actual_rate = result["attacker"].mean()
    if verbose:
        print(f"    After: {len(result):,} rows | rate={actual_rate:.1%}")

    return result


# -----------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------

def validate_features(df, split_name):
    print(f"\n    Validation ({split_name}):")
    issues = 0

    for col in CLASSIFICATION_FEATURES:
        if col not in df.columns:
            continue
        n_inf = np.isinf(df[col]).sum()
        n_null = df[col].isna().sum()
        if n_inf > 0:
            print(f"      [FAIL] {col}: {n_inf:,} inf")
            issues += 1
        if n_null > 0:
            print(f"      [WARN] {col}: {n_null:,} nulls")
            issues += 1

    for col in ["speed_consistency", "accel_consistency"]:
        if col in df.columns:
            p99 = df[col].abs().quantile(0.99)
            mx = df[col].abs().max()
            print(f"      [INFO] {col}: P99={p99:.3f}, max={mx:.3f}")
            if mx > 10000:
                issues += 1

    if issues == 0:
        print(f"      All checks passed")
    return issues


def compute_feature_stats(df):
    stats = {}
    for col in CLASSIFICATION_FEATURES:
        if col not in df.columns:
            continue
        v = df[col].dropna()
        stats[col] = {
            "mean": round(float(v.mean()), 6),
            "std": round(float(v.std()), 6),
            "min": round(float(v.min()), 6),
            "max": round(float(v.max()), 6),
            "P99_abs": round(float(v.abs().quantile(0.99)), 6),
        }
    return stats


# -----------------------------------------------------------------------
# K-Means
# -----------------------------------------------------------------------

def partition_kmeans(train_df, n_clusters, seed=42, verbose=True):
    t0 = time.time()
    coords = train_df[PARTITION_FEATURES].values.astype(np.float64)
    valid = ~np.isnan(coords).any(axis=1)

    if valid.sum() > 1_000_000:
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed,
                                 n_init=10, max_iter=300, batch_size=10000)
    else:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed,
                        n_init=10, max_iter=300)

    kmeans.fit(coords[valid])
    assignments = np.zeros(len(train_df), dtype=np.int32)
    assignments[valid] = kmeans.labels_

    if verbose:
        print(f"    k={n_clusters}: {time.time()-t0:.1f}s")
        for k in range(n_clusters):
            m = assignments == k
            cnt = m.sum()
            ar = train_df.loc[m, "attacker"].mean() * 100
            ctr = kmeans.cluster_centers_[k]
            sc = train_df.loc[m, "scenario"].value_counts()
            print(f"      C{k:>2}: {cnt:>8,} ({cnt/len(train_df)*100:>5.1f}%) | "
                  f"atk: {ar:>5.1f}% | ({ctr[0]:>.0f}, {ctr[1]:>.0f}) | "
                  f"{sc.index[0] if len(sc)>0 else '?'}")

    return assignments, kmeans


def save_partitions(train_df, assignments, kmeans, n_clusters, output_dir):
    part_dir = Path(output_dir) / "partitions" / f"k{n_clusters}"
    part_dir.mkdir(parents=True, exist_ok=True)

    save_cols = [c for c in CLASSIFICATION_FEATURES + LABEL_COLS
                 if c in train_df.columns]

    meta = {"n_clusters": n_clusters,
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "classification_features": CLASSIFICATION_FEATURES,
            "partitions": {}}

    for k in range(n_clusters):
        m = assignments == k
        pdf = train_df.loc[m, save_cols].reset_index(drop=True)
        pdf.to_parquet(part_dir / f"partition_{k}.parquet", index=False)

        n, na = len(pdf), int(pdf["attacker"].sum())
        p = na / n if n > 0 else 0
        ent = -(p*np.log2(p) + (1-p)*np.log2(1-p)) if 0 < p < 1 else 0

        meta["partitions"][str(k)] = {
            "n_samples": n, "n_attack": na, "n_normal": n-na,
            "attack_rate": round(p, 4), "label_entropy": round(ent, 4),
            "attack_distribution": pdf[pdf["attacker"]==1][
                "attack_type"].value_counts().to_dict() if na > 0 else {},
            "scenario_distribution": pdf["scenario"].value_counts().to_dict(),
        }

    with open(part_dir / "partition_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(part_dir / "kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    print(f"    Saved {n_clusters} partitions -> {part_dir}/")


# -----------------------------------------------------------------------
# Convenience
# -----------------------------------------------------------------------

def get_Xy(df, classification="binary"):
    feat_cols = [c for c in CLASSIFICATION_FEATURES if c in df.columns]
    X = df[feat_cols].fillna(0.0).values.astype(np.float32)
    if classification == "binary":
        y = df["attacker"].values.astype(np.int64)
    elif classification == "triclass":
        y = df["attack_triclass"].astype("category").cat.codes.values.astype(np.int64)
    elif classification == "7class":
        y = df["attack_category"].astype("category").cat.codes.values.astype(np.int64)
    else:
        raise ValueError(classification)
    return X, y, feat_cols


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_clusters", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--attack_rate", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data_dir = Path(args.data_dir)

    # =================================================================
    # STEP 1-4: Process per-attack files
    # =================================================================
    print("=" * 70)
    print("STEP 1-4: Aggregate, dedup, engineer features per-attack-file")
    print("=" * 70)

    processed = {}

    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"  [ERROR] {split_dir} not found")
            return

        EXCLUDE_ATTACKS = {"Normal", "timeDelayAttack"}
        attack_files = sorted([
                f for f in split_dir.glob("*.parquet") if f.stem not in EXCLUDE_ATTACKS
                ])
        print(f"\n  === {split}: {len(attack_files)} attack files ===")

        normal_collected = False
        all_normals = None
        all_attacks = []

        for pf in attack_files:
            attack_name = pf.stem
            normal_df, attack_df = process_attack_file(
                str(pf), attack_name, verbose=True)

            # Normals from one file only
            if not normal_collected and attack_name == NORMAL_SOURCE_FILE:
                all_normals = normal_df
                normal_collected = True
                print(f"      ** Normals taken from {attack_name}: "
                      f"{len(normal_df):,} **")
            elif not normal_collected and pf == attack_files[-1]:
                # Fallback: use last file if source not found
                all_normals = normal_df
                normal_collected = True
                print(f"      ** Normals taken from {attack_name} (fallback): "
                      f"{len(normal_df):,} **")
            else:
                del normal_df

            if len(attack_df) > 0:
                all_attacks.append(attack_df)

            del attack_df
            gc.collect()

        attacks_combined = pd.concat(all_attacks, ignore_index=True)
        del all_attacks
        gc.collect()

        df = pd.concat([all_normals, attacks_combined], ignore_index=True)
        del all_normals, attacks_combined
        gc.collect()

        # Fix: normal messages inherit attack_type from source file — override to "Normal"
        df.loc[df["attacker"] == 0, "attack_type"] = "Normal"
        df.loc[df["attacker"] == 0, "attack_category"] = "Normal"
        df.loc[df["attacker"] == 0, "attack_triclass"] = "Normal"

        n = len(df)
        na = int(df["attacker"].sum())
        print(f"\n  {split} combined: {n:,} rows | "
              f"normal={n-na:,} ({(n-na)/n:.1%}) | "
              f"attack={na:,} ({na/n:.1%})")

        processed[split] = df

    # =================================================================
    # STEP 5: Report distribution BEFORE downsampling
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 5: Distribution BEFORE downsampling")
    print("=" * 70)

    for split, df in processed.items():
        n = len(df)
        na = int(df["attacker"].sum())
        nn = n - na
        rate = na / n

        print(f"\n  {split}:")
        print(f"    Total: {n:,} | Normal: {nn:,} | Attack: {na:,} | "
              f"Rate: {rate:.1%}")
        print(f"    Normal senders: {df[df['attacker']==0]['sender_id'].nunique():,}")
        print(f"    Attack senders: {df[df['attacker']==1]['sender_id'].nunique():,}")

        if split == "train":
            print(f"\n    Per-attack-type:")
            atk = df[df["attacker"] == 1]
            for at, cnt in atk["attack_type"].value_counts().sort_index().items():
                n_senders = atk[atk["attack_type"]==at]["sender_id"].nunique()
                print(f"      {at:40s} {cnt:>8,} rows | {n_senders:>5,} senders")

    # =================================================================
    # STEP 6: Downsample by sender
    # =================================================================
    print(f"\n{'='*70}")
    print(f"STEP 6: Downsample to ~{args.attack_rate:.0%} (sender-level)")
    print("=" * 70)

    for split in ["train", "val", "test"]:
        print(f"\n  {split}:")
        processed[split] = downsample_by_sender(
            processed[split], target_rate=args.attack_rate, seed=args.seed)

    # =================================================================
    # STEP 7: Validation
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 7: Validation")
    print("=" * 70)

    for split, df in processed.items():
        validate_features(df, split)

    # =================================================================
    # STEP 8: Feature statistics
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 8: Feature statistics (train)")
    print("=" * 70)

    stats = compute_feature_stats(processed["train"])
    print(f"\n  {'Feature':<34} {'Mean':>10} {'Std':>10} "
          f"{'Min':>10} {'Max':>10} {'P99abs':>10}")
    print(f"  {'-'*86}")
    for f, s in stats.items():
        print(f"  {f:<34} {s['mean']:>10.3f} {s['std']:>10.3f} "
              f"{s['min']:>10.3f} {s['max']:>10.3f} {s['P99_abs']:>10.3f}")

    all_stats = {s: compute_feature_stats(processed[s]) for s in processed}
    with open(Path(args.output_dir) / "feature_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    # =================================================================
    # STEP 9: Final label distribution
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 9: Final label distribution")
    print("=" * 70)

    for split, df in processed.items():
        n = len(df)
        na = int(df["attacker"].sum())
        print(f"\n  {split}: {n:,} | normal={n-na:,} ({(n-na)/n:.1%}) | "
              f"attack={na:,} ({na/n:.1%})")

    tr = processed["train"]
    atk = tr[tr["attacker"] == 1]

    print(f"\n  Train per-type:")
    for at, cnt in atk["attack_type"].value_counts().sort_index().items():
        ns = atk[atk["attack_type"]==at]["sender_id"].nunique()
        print(f"    {at:40s} {cnt:>8,} rows | {ns:>5,} senders")

    print(f"\n  Train 7-class:")
    for c, cnt in tr["attack_category"].value_counts().sort_index().items():
        print(f"    {c:20s} {cnt:>8,}")

    print(f"\n  Train tri-class:")
    for c, cnt in tr["attack_triclass"].value_counts().sort_index().items():
        print(f"    {c:20s} {cnt:>8,}")

    # =================================================================
    # STEP 10: Save
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 10: Save")
    print("=" * 70)

    for split, df in processed.items():
        out = Path(args.output_dir) / f"{split}.parquet"
        df.to_parquet(out, index=False)
        mb = out.stat().st_size / 1e6
        print(f"  {split}: {len(df):,} rows -> {out.name} ({mb:.1f} MB)")

    # =================================================================
    # STEP 11: Partitioning
    # =================================================================
    print(f"\n{'='*70}")
    print("STEP 11: RSU partitioning")
    print("=" * 70)

    for k in args.n_clusters:
        print(f"\n  --- k={k} ---")
        asn, km = partition_kmeans(processed["train"], k, seed=args.seed)
        save_partitions(processed["train"], asn, km, k, args.output_dir)

    # Done
    print(f"\n{'='*70}")
    print("DONE")
    print("=" * 70)
    for f in sorted(Path(args.output_dir).rglob("*")):
        if f.is_file():
            sz = f.stat().st_size
            label = f"{sz/1e6:.1f} MB" if sz > 1e6 else f"{sz/1e3:.0f} KB"
            print(f"  {f.relative_to(args.output_dir)}  ({label})")

    print(f"\n  Features (12): {CLASSIFICATION_FEATURES}")
    print(f"  Partition (2): {PARTITION_FEATURES}")


if __name__ == "__main__":
    main()