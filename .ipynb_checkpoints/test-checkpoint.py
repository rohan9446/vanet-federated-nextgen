# check_timedelay.py
import pandas as pd
import numpy as np

df = pd.read_parquet("data/per_attack/train/timeDelayAttack.parquet")

# Deduplicate first
dedup = df.drop_duplicates(subset=["sender_id", "messageID"], keep="first")

normal = dedup[dedup["attacker"] == 0]
attack = dedup[dedup["attacker"] == 1]

NS = 1e-9

print("=== RAW TIMESTAMP ANALYSIS ===")
print(f"\nNormal messages: {len(normal):,}")
print(f"Attack messages: {len(attack):,}")

# tx_delay = rcvTime - sendTime
n_delay = (normal["rcvTime"] - normal["sendTime"]) * NS
a_delay = (attack["rcvTime"] - attack["sendTime"]) * NS

print(f"\ntx_delay (seconds):")
print(f"  Normal: mean={n_delay.mean():.6f} std={n_delay.std():.6f} "
      f"min={n_delay.min():.6f} max={n_delay.max():.6f}")
print(f"  Attack: mean={a_delay.mean():.6f} std={a_delay.std():.6f} "
      f"min={a_delay.min():.6f} max={a_delay.max():.6f}")

# time_delta per sender
for label, subset in [("Normal", normal), ("Attack", attack)]:
    s = subset.sort_values(["scenario", "sender_id", "sendTime"])
    s["td"] = s.groupby(["scenario", "sender_id"])["sendTime"].diff() * NS
    td = s["td"].dropna()
    print(f"\ntime_delta ({label}):")
    print(f"  mean={td.mean():.6f} std={td.std():.6f} "
          f"min={td.min():.6f} max={td.max():.6f}")

# Check: does the attack modify sendTime itself?
# Compare attack sender sendTimes vs same sender in Normal.parquet
normal_file = pd.read_parquet("data/per_attack/train/Normal.parquet",
                               columns=["sender_id", "messageID", "sendTime"])
normal_file = normal_file.drop_duplicates(["sender_id", "messageID"])

# Find attack senders that also exist in Normal
atk_senders = attack["sender_id"].unique()
print(f"\nAttack senders: {len(atk_senders)}")

# Merge to compare sendTimes for same (sender, messageID)
merged = attack[["sender_id", "messageID", "sendTime"]].merge(
    normal_file, on=["sender_id", "messageID"], suffixes=("_attack", "_normal"))

if len(merged) > 0:
    diff = (merged["sendTime_attack"] - merged["sendTime_normal"]) * NS
    print(f"\nsendTime difference (attack - normal) for same messages:")
    print(f"  Count: {len(merged):,}")
    print(f"  Mean:  {diff.mean():.6f}s")
    print(f"  Std:   {diff.std():.6f}s")
    print(f"  Min:   {diff.min():.6f}s")
    print(f"  Max:   {diff.max():.6f}s")
    print(f"  Zero:  {(diff == 0).sum():,} ({(diff==0).mean()*100:.1f}%)")
    print(f"  >0:    {(diff > 0).sum():,}")
    print(f"  <0:    {(diff < 0).sum():,}")
else:
    print("\nNo matching messages found between attack and Normal files")