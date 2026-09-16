"""
explore_data.py — Deep dive into VeReMi NextGen data structure.

Understand the data completely before building the pipeline.

Usage:
    python explore_data.py --file data/per_attack/train/constantPositionOffset.parquet
    python explore_data.py --file data/per_attack/train/constantPositionOffset.parquet --compare-normal data/per_attack/train/Normal.parquet
"""

import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="One per_attack parquet file")
    parser.add_argument("--compare-normal", default=None,
                        help="Optional: Normal.parquet for comparison")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit rows loaded (for memory)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"EXPLORING: {args.file}")
    print("=" * 70)

    df = pd.read_parquet(args.file)
    if args.max_rows:
        df = df.head(args.max_rows)

    # =================================================================
    # 1. BASIC STRUCTURE
    # =================================================================
    print(f"\n{'='*70}")
    print("1. BASIC STRUCTURE")
    print(f"{'='*70}")
    print(f"  Rows:    {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"\n  Column types:")
    for col in df.columns:
        dtype = df[col].dtype
        nulls = df[col].isna().sum()
        sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else "N/A"
        print(f"    {col:35s} {str(dtype):>10} | nulls={nulls:>8,} | "
              f"sample={str(sample)[:40]}")

    # =================================================================
    # 2. MESSAGE DUPLICATION ANALYSIS
    # =================================================================
    print(f"\n{'='*70}")
    print("2. MESSAGE DUPLICATION (same sender message, multiple receivers)")
    print(f"{'='*70}")

    msg_counts = df.groupby(["sender_id", "messageID"]).size()
    unique_msgs = len(msg_counts)
    total_rows = len(df)

    print(f"  Unique messages (sender_id + messageID): {unique_msgs:,}")
    print(f"  Total rows:                              {total_rows:,}")
    print(f"  Inflation factor:                        {total_rows/unique_msgs:.1f}x")
    print(f"\n  Copies per message:")
    print(f"    Min:    {msg_counts.min()}")
    print(f"    Mean:   {msg_counts.mean():.1f}")
    print(f"    Median: {msg_counts.median():.0f}")
    print(f"    Max:    {msg_counts.max()}")
    print(f"    Std:    {msg_counts.std():.1f}")

    # Distribution of copy counts
    copy_dist = msg_counts.value_counts().sort_index()
    print(f"\n  Copy count distribution (top 10):")
    for n_copies, count in copy_dist.head(10).items():
        print(f"    {n_copies:>3} copies: {count:>8,} messages")

    # =================================================================
    # 3. WHAT CHANGES ACROSS COPIES OF SAME MESSAGE?
    # =================================================================
    print(f"\n{'='*70}")
    print("3. FIELD VARIATION ACROSS COPIES OF SAME MESSAGE")
    print(f"{'='*70}")

    # Take a sample of messages with multiple copies
    multi_copy_msgs = msg_counts[msg_counts > 1].head(100).index
    if len(multi_copy_msgs) > 0:
        sample_msgs = df[df.set_index(["sender_id", "messageID"]).index.isin(multi_copy_msgs)]
        
        print(f"\n  Checking {len(multi_copy_msgs)} messages with multiple copies...")
        print(f"  Fields that STAY CONSTANT across copies (same message):")
        print(f"  Fields that CHANGE across copies (receiver-dependent):")
        
        constant_cols = []
        varying_cols = []
        
        for col in df.columns:
            try:
                n_unique_per_msg = sample_msgs.groupby(
                    ["sender_id", "messageID"])[col].nunique()
                if n_unique_per_msg.max() <= 1:
                    constant_cols.append(col)
                else:
                    pct_varying = (n_unique_per_msg > 1).mean() * 100
                    varying_cols.append((col, pct_varying))
            except Exception:
                varying_cols.append((col, -1))

        print(f"\n  CONSTANT (sender-side, same across all receivers):")
        for col in constant_cols:
            print(f"    {col}")
        
        print(f"\n  VARYING (receiver-dependent or per-observation):")
        for col, pct in varying_cols:
            print(f"    {col:35s} (varies in {pct:.0f}% of multi-copy messages)")

    # =================================================================
    # 4. RECEIVER IDENTITY
    # =================================================================
    print(f"\n{'='*70}")
    print("4. RECEIVER IDENTITY")
    print(f"{'='*70}")

    # No receiver_id column, but can we derive from receiver_pos?
    if "receiver_pos_x" in df.columns:
        # Check if (receiver_pos_x, receiver_pos_y) identifies unique receivers
        # For a given (sender_id, messageID), each copy has a different receiver
        sample = df[df.set_index(["sender_id", "messageID"]).index.isin(
            multi_copy_msgs[:10])] if len(multi_copy_msgs) > 0 else df.head(100)
        
        # Unique receiver positions
        rx = df["receiver_pos_x"]
        ry = df["receiver_pos_y"]
        unique_recv_pos = df.groupby(
            [rx.round(2), ry.round(2)]).ngroups
        print(f"  Unique receiver positions (rounded 2dp): {unique_recv_pos:,}")
        
        # Check: is rcvTime unique per copy?
        rcv_unique = df.groupby(["sender_id", "messageID"])["rcvTime"].nunique()
        print(f"  rcvTime unique per message copy: "
              f"{(rcv_unique > 1).mean()*100:.1f}% have varying rcvTime")

    # =================================================================
    # 5. SENDER ANALYSIS
    # =================================================================
    print(f"\n{'='*70}")
    print("5. SENDER ANALYSIS")
    print(f"{'='*70}")

    unique_senders = df["sender_id"].nunique()
    print(f"  Unique senders: {unique_senders:,}")

    # Senders by attacker label
    sender_labels = df.groupby("sender_id")["attacker"].agg(["mean", "count", "sum"])
    pure_normal = (sender_labels["mean"] == 0).sum()
    pure_attack = (sender_labels["mean"] == 1).sum()
    mixed = ((sender_labels["mean"] > 0) & (sender_labels["mean"] < 1)).sum()

    print(f"  Pure normal senders (all attacker=0):  {pure_normal:,}")
    print(f"  Pure attack senders (all attacker=1):  {pure_attack:,}")
    print(f"  Mixed senders (both 0 and 1):          {mixed:,}")

    # Messages per sender (unique messages, not copies)
    msgs_per_sender = df.groupby("sender_id")["messageID"].nunique()
    print(f"\n  Unique messages per sender:")
    print(f"    Min:    {msgs_per_sender.min()}")
    print(f"    Mean:   {msgs_per_sender.mean():.1f}")
    print(f"    Median: {msgs_per_sender.median():.0f}")
    print(f"    Max:    {msgs_per_sender.max()}")

    # Attacker sender stats
    atk_senders = sender_labels[sender_labels["mean"] == 1].index
    if len(atk_senders) > 0:
        atk_msgs = msgs_per_sender[atk_senders]
        print(f"\n  Attack senders ({len(atk_senders):,}):")
        print(f"    Messages per attacker: mean={atk_msgs.mean():.1f}, "
              f"median={atk_msgs.median():.0f}, max={atk_msgs.max()}")

    # =================================================================
    # 6. TEMPORAL ANALYSIS
    # =================================================================
    print(f"\n{'='*70}")
    print("6. TEMPORAL ANALYSIS")
    print(f"{'='*70}")

    # Convert to seconds for readability
    NS = 1e-9

    # sendTime range
    st_min = df["sendTime"].min() * NS
    st_max = df["sendTime"].max() * NS
    print(f"  sendTime range: {st_min:.1f}s to {st_max:.1f}s "
          f"(duration: {st_max - st_min:.1f}s)")

    # rcvTime range
    rt_min = df["rcvTime"].min() * NS
    rt_max = df["rcvTime"].max() * NS
    print(f"  rcvTime range:  {rt_min:.1f}s to {rt_max:.1f}s")

    # tx_delay distribution
    tx_delay = (df["rcvTime"] - df["sendTime"]) * NS
    print(f"\n  tx_delay (rcvTime - sendTime):")
    print(f"    Mean:   {tx_delay.mean():.6f}s")
    print(f"    Std:    {tx_delay.std():.6f}s")
    print(f"    Min:    {tx_delay.min():.6f}s")
    print(f"    Max:    {tx_delay.max():.6f}s")

    # Time between consecutive unique messages per sender
    deduped = df.drop_duplicates(subset=["sender_id", "messageID"]).copy()
    deduped = deduped.sort_values(["sender_id", "sendTime"])
    deduped["time_diff"] = deduped.groupby("sender_id")["sendTime"].diff() * NS
    td = deduped["time_diff"].dropna()

    print(f"\n  Time between consecutive messages (per sender, deduplicated):")
    print(f"    Mean:   {td.mean():.4f}s")
    print(f"    Median: {td.median():.4f}s")
    print(f"    Min:    {td.min():.6f}s")
    print(f"    Max:    {td.max():.1f}s")
    print(f"    Std:    {td.std():.4f}s")

    # =================================================================
    # 7. SPATIAL ANALYSIS
    # =================================================================
    print(f"\n{'='*70}")
    print("7. SPATIAL ANALYSIS")
    print(f"{'='*70}")

    for prefix, label in [("sender", "Sender"), ("receiver", "Receiver")]:
        px = f"{prefix}_pos_x"
        py = f"{prefix}_pos_y"
        if px in df.columns:
            print(f"\n  {label} positions:")
            print(f"    X: [{df[px].min():.1f}, {df[px].max():.1f}] "
                  f"(range: {df[px].max()-df[px].min():.1f}m)")
            print(f"    Y: [{df[py].min():.1f}, {df[py].max():.1f}] "
                  f"(range: {df[py].max()-df[py].min():.1f}m)")

    # sender_receiver_dist
    dist = np.sqrt(
        (df["sender_pos_x"] - df["receiver_pos_x"])**2 +
        (df["sender_pos_y"] - df["receiver_pos_y"])**2
    )
    print(f"\n  Sender-receiver distance:")
    print(f"    Mean:   {dist.mean():.1f}m")
    print(f"    Median: {dist.median():.1f}m")
    print(f"    Max:    {dist.max():.1f}m")
    print(f"    Min:    {dist.min():.1f}m")

    # Per scenario
    print(f"\n  Per scenario:")
    for sc, sdf in df.groupby("scenario"):
        print(f"    {sc}: {len(sdf):,} rows, "
              f"X=[{sdf['sender_pos_x'].min():.0f}, {sdf['sender_pos_x'].max():.0f}], "
              f"Y=[{sdf['sender_pos_y'].min():.0f}, {sdf['sender_pos_y'].max():.0f}]")

    # =================================================================
    # 8. ATTACK ANALYSIS (what differs between attacker=0 and attacker=1)
    # =================================================================
    print(f"\n{'='*70}")
    print("8. ATTACK vs NORMAL COMPARISON")
    print(f"{'='*70}")

    n0 = df[df["attacker"] == 0]
    n1 = df[df["attacker"] == 1]

    if len(n1) > 0:
        print(f"\n  Normal messages:  {len(n0):,}")
        print(f"  Attack messages:  {len(n1):,}")

        compare_cols = ["sender_spd", "sender_acl", "sender_hed",
                        "sender_pos_x", "sender_pos_y",
                        "sender_dist_to_road_edge"]

        print(f"\n  {'Field':<30} {'Normal_mean':>14} {'Attack_mean':>14} {'Diff':>10}")
        print(f"  {'-'*70}")
        for col in compare_cols:
            if col in df.columns:
                nm = n0[col].mean()
                am = n1[col].mean()
                diff = am - nm
                print(f"  {col:<30} {nm:>14.4f} {am:>14.4f} {diff:>10.4f}")

        # Deduplicated comparison (per unique message)
        print(f"\n  After deduplication (unique messages only):")
        dedup0 = n0.drop_duplicates(["sender_id", "messageID"])
        dedup1 = n1.drop_duplicates(["sender_id", "messageID"])
        print(f"    Normal: {len(n0):,} -> {len(dedup0):,} unique")
        print(f"    Attack: {len(n1):,} -> {len(dedup1):,} unique")

        # Compute sequential features on deduplicated data
        print(f"\n  Sequential features on DEDUPLICATED data:")
        for label, ddf in [("Normal", dedup0), ("Attack", dedup1)]:
            ddf = ddf.sort_values(["scenario", "sender_id", "sendTime"]).copy()
            grp = ddf.groupby(["scenario", "sender_id"])

            prev_px = grp["sender_pos_x"].shift(1)
            prev_py = grp["sender_pos_y"].shift(1)
            prev_spd = grp["sender_spd"].shift(1)
            prev_st = grp["sendTime"].shift(1)

            dt = ((ddf["sendTime"] - prev_st) * NS).clip(lower=0)
            dx = ddf["sender_pos_x"] - prev_px
            dy = ddf["sender_pos_y"] - prev_py
            dist_consec = np.sqrt(dx**2 + dy**2)

            reliable = dt >= 0.05
            derived_spd = np.where(reliable, dist_consec / dt, np.nan)
            spd_consistency = np.where(reliable,
                                        ddf["sender_spd"].values - derived_spd,
                                        np.nan)

            print(f"\n    {label}:")
            print(f"      time_delta: mean={dt.mean():.4f}s, "
                  f"median={dt.median():.4f}s")
            print(f"      distance:   mean={dist_consec.mean():.4f}m, "
                  f"median={np.nanmedian(dist_consec):.4f}m")

            sc_valid = spd_consistency[~np.isnan(spd_consistency)]
            if len(sc_valid) > 0:
                print(f"      speed_consistency: mean={np.mean(sc_valid):.4f}, "
                      f"std={np.std(sc_valid):.4f}, "
                      f"P99={np.percentile(np.abs(sc_valid), 99):.4f}")

    # =================================================================
    # 9. SCENARIO BREAKDOWN
    # =================================================================
    print(f"\n{'='*70}")
    print("9. SCENARIO BREAKDOWN")
    print(f"{'='*70}")

    for sc, sdf in df.groupby("scenario"):
        n = len(sdf)
        na = sdf["attacker"].sum()
        ns = sdf["sender_id"].nunique()
        um = sdf.groupby(["sender_id", "messageID"]).ngroups
        print(f"\n  {sc}:")
        print(f"    Rows: {n:,} | Unique msgs: {um:,} | "
              f"Inflation: {n/um:.1f}x")
        print(f"    Senders: {ns:,} | Attack rows: {na:,} ({na/n:.1%})")

    # =================================================================
    # 10. RECEIVER FILE ORIGIN (from build_dataset)
    # =================================================================
    print(f"\n{'='*70}")
    print("10. RECEIVER ANALYSIS")
    print(f"{'='*70}")

    # Each original JSON was one receiver's log
    # receiver_pos identifies the receiver
    # Group by approximate receiver position to count unique receivers
    if "receiver_pos_x" in df.columns:
        # Round to nearest meter for grouping
        recv_groups = df.groupby([
            df["receiver_pos_x"].round(0),
            df["receiver_pos_y"].round(0)
        ]).agg(
            n_rows=("sender_id", "size"),
            n_unique_senders=("sender_id", "nunique"),
            n_unique_msgs=("messageID", "nunique"),
        )
        
        print(f"  Approximate unique receiver positions: {len(recv_groups):,}")
        print(f"  (Receivers are vehicles, so positions change over time)")
        
        # More stable: count unique (sender_id, messageID) combos per receiver
        # Actually, receiver is identified by the original JSON file
        # Since receiver moves, let's check if rcvTime groups identify a receiver
        
        # Better approach: for a given (sender_id, messageID), 
        # each row is a different receiver
        sample_msg = df.groupby(["sender_id", "messageID"]).first().index[0]
        msg_copies = df[(df["sender_id"] == sample_msg[0]) & 
                        (df["messageID"] == sample_msg[1])]
        
        print(f"\n  Example: message from {sample_msg[0]}, ID={sample_msg[1]}")
        print(f"    Copies: {len(msg_copies)}")
        print(f"    Receiver positions:")
        for _, row in msg_copies.head(5).iterrows():
            print(f"      ({row['receiver_pos_x']:.1f}, {row['receiver_pos_y']:.1f}) | "
                  f"dist={np.sqrt((row['sender_pos_x']-row['receiver_pos_x'])**2 + (row['sender_pos_y']-row['receiver_pos_y'])**2):.1f}m | "
                  f"rcvTime={row['rcvTime']}")

    print(f"\n{'='*70}")
    print("EXPLORATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()