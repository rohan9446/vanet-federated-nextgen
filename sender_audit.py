"""
sender_audit.py — Check sender overlap across Normal and attack files.

Run BEFORE rebuilding the pipeline to understand data impact.

Usage:
    python sender_audit.py --data_dir data/per_attack
"""

import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Path to data/per_attack (has train/ val/ test/)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        print(f"\n{'='*70}")
        print(f"SPLIT: {split}")
        print(f"{'='*70}")

        # 1. Collect senders per attack type
        normal_senders = set()
        attack_senders_by_type = {}  # attack_type -> set of sender_ids
        attack_rows_by_type = {}    # attack_type -> count of attacker=1 rows
        normal_rows = 0

        parquet_files = sorted(split_dir.glob("*.parquet"))
        print(f"\n  Files: {len(parquet_files)}")

        for pf in parquet_files:
            attack_name = pf.stem
            df = pd.read_parquet(pf, columns=["sender_id", "attacker"])

            if attack_name == "Normal":
                normal_senders = set(df["sender_id"].unique())
                normal_rows = len(df)
                print(f"  Normal: {len(normal_senders):,} unique senders, "
                      f"{normal_rows:,} rows")
            else:
                atk_df = df[df["attacker"] == 1]
                atk_senders = set(atk_df["sender_id"].unique())
                attack_senders_by_type[attack_name] = atk_senders
                attack_rows_by_type[attack_name] = len(atk_df)
                print(f"  {attack_name:40s} {len(atk_senders):>6,} attack senders, "
                      f"{len(atk_df):>8,} attack rows")

        # 2. All attack senders (union across all attack types)
        all_attack_senders = set()
        for s in attack_senders_by_type.values():
            all_attack_senders |= s

        # 3. Overlap: senders in both Normal and any attack
        overlap = normal_senders & all_attack_senders
        clean_normal = normal_senders - all_attack_senders
        attack_only = all_attack_senders - normal_senders

        print(f"\n  --- Sender Overlap Analysis ---")
        print(f"  Total unique normal senders:    {len(normal_senders):>8,}")
        print(f"  Total unique attack senders:    {len(all_attack_senders):>8,}")
        print(f"  Overlap (in both):              {len(overlap):>8,}")
        print(f"  Clean normal (normal only):     {len(clean_normal):>8,}")
        print(f"  Attack only (never normal):     {len(attack_only):>8,}")

        # 4. Cross-attack overlap: senders appearing in multiple attack types
        sender_attack_types = defaultdict(set)
        for atype, senders in attack_senders_by_type.items():
            for s in senders:
                sender_attack_types[s].add(atype)

        multi_attack_senders = {s for s, types in sender_attack_types.items()
                                 if len(types) > 1}
        single_attack_senders = {s for s, types in sender_attack_types.items()
                                  if len(types) == 1}

        print(f"\n  --- Cross-Attack Overlap ---")
        print(f"  Senders in exactly 1 attack type:  {len(single_attack_senders):>8,}")
        print(f"  Senders in 2+ attack types:        {len(multi_attack_senders):>8,}")

        if multi_attack_senders:
            type_counts = defaultdict(int)
            for s in multi_attack_senders:
                n = len(sender_attack_types[s])
                type_counts[n] += 1
            for n, cnt in sorted(type_counts.items()):
                print(f"    In {n} attack types: {cnt:,} senders")

        # 5. Impact of removing overlap from Normal
        if normal_senders:
            # Load Normal file to count rows per sender
            normal_df = pd.read_parquet(split_dir / "Normal.parquet",
                                        columns=["sender_id"])
            rows_from_overlap = normal_df[
                normal_df["sender_id"].isin(overlap)].shape[0]
            rows_clean = normal_rows - rows_from_overlap
            pct_kept = rows_clean / normal_rows * 100

            print(f"\n  --- Impact on Normal Data ---")
            print(f"  Normal rows total:              {normal_rows:>10,}")
            print(f"  Rows from overlap senders:      {rows_from_overlap:>10,} "
                  f"(would be removed)")
            print(f"  Clean normal rows remaining:    {rows_clean:>10,} "
                  f"({pct_kept:.1f}% kept)")

        # 6. Impact on attack data if sender-disjoint across attack types
        if multi_attack_senders:
            print(f"\n  --- Impact if Sender-Disjoint Across Attacks ---")
            print(f"  Strategy: assign each multi-attack sender to ONE attack type")
            print(f"  (keep in the type where they have most messages)")

            # For each multi-attack sender, find which type has most rows
            reassignment = {}
            for s in multi_attack_senders:
                best_type = None
                best_count = 0
                for atype in sender_attack_types[s]:
                    df = pd.read_parquet(
                        split_dir / f"{atype}.parquet",
                        columns=["sender_id", "attacker"])
                    cnt = ((df["sender_id"] == s) & (df["attacker"] == 1)).sum()
                    if cnt > best_count:
                        best_count = cnt
                        best_type = atype
                reassignment[s] = best_type

            # Count rows lost per attack type
            print(f"\n  Per-type impact:")
            for atype in sorted(attack_senders_by_type.keys()):
                original = attack_rows_by_type[atype]
                senders_removed = {s for s in multi_attack_senders
                                   if s in attack_senders_by_type[atype]
                                   and reassignment.get(s) != atype}
                # Estimate rows lost (proportional to senders removed)
                type_senders = attack_senders_by_type[atype]
                if len(type_senders) > 0:
                    frac_removed = len(senders_removed) / len(type_senders)
                    est_rows_lost = int(original * frac_removed)
                else:
                    est_rows_lost = 0
                remaining = original - est_rows_lost
                print(f"    {atype:40s} {original:>8,} -> ~{remaining:>8,} "
                      f"(lose ~{est_rows_lost:,})")

        # 7. Summary recommendation
        print(f"\n  --- Recommendation ---")
        if len(overlap) == 0:
            print(f"  No overlap! Senders are already disjoint.")
        elif len(clean_normal) > len(overlap):
            print(f"  Overlap is manageable. Removing {len(overlap):,} senders "
                  f"from Normal keeps {pct_kept:.1f}% of normal data.")
            print(f"  Proceed with sender-disjoint approach.")
        else:
            print(f"  WARNING: Overlap is large. Removing would lose "
                  f"{100-pct_kept:.1f}% of normal data.")
            print(f"  Consider alternative: add scenario prefix to sender_id "
                  f"to disambiguate.")


if __name__ == "__main__":
    main()