"""
per_attack_analysis.py — Binary detection per individual attack type.

For each of 14 attack types: train Normal-vs-Attack binary classifier.
Reports AUC, F1, Precision, Recall per attack.

Usage:
    python per_attack_analysis.py --data_dir data/processed --output_dir results/per_attack
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
)

CLASSIFICATION_FEATURES = [
    "sender_spd", "sender_acl",
    "time_delta", "distance", "heading_change",
    "speed_consistency", "accel_consistency",
    "heading_trajectory_consistency",
    "tx_delay", "mean_sender_receiver_dist", "n_receivers",
    "sender_dist_to_road_edge",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    train = pd.read_parquet(Path(args.data_dir) / "train.parquet")
    val = pd.read_parquet(Path(args.data_dir) / "val.parquet")
    test = pd.read_parquet(Path(args.data_dir) / "test.parquet")

    normal_tr = train[train["attacker"] == 0]
    normal_va = val[val["attacker"] == 0]
    normal_te = test[test["attacker"] == 0]

    attacks = sorted(train[train["attacker"] == 1]["attack_type"].unique())
    print(f"Attacks: {len(attacks)}")
    print(f"Normal: train={len(normal_tr):,} val={len(normal_va):,} test={len(normal_te):,}")

    all_results = []

    print(f"\n{'Attack':<40} {'Model':<5} {'AUC':>8} {'F1':>8} {'Prec':>8} {'Rec':>8}")
    print("-" * 82)

    for attack in attacks:
        atk_tr = train[(train["attack_type"] == attack) & (train["attacker"] == 1)]
        atk_va = val[(val["attack_type"] == attack) & (val["attacker"] == 1)]
        atk_te = test[(test["attack_type"] == attack) & (test["attacker"] == 1)]

        # Combine normal + this attack
        tr = pd.concat([normal_tr, atk_tr])
        va = pd.concat([normal_va, atk_va])
        te = pd.concat([normal_te, atk_te])

        X_tr = tr[CLASSIFICATION_FEATURES].fillna(0).values.astype(np.float32)
        y_tr = tr["attacker"].values
        X_va = va[CLASSIFICATION_FEATURES].fillna(0).values.astype(np.float32)
        y_va = va["attacker"].values
        X_te = te[CLASSIFICATION_FEATURES].fillna(0).values.astype(np.float32)
        y_te = te["attacker"].values

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_va_s = scaler.transform(X_va)
        X_te_s = scaler.transform(X_te)

        for model_name in ["rf", "xgb"]:
            seed_aucs, seed_f1s, seed_precs, seed_recs = [], [], [], []

            for seed in args.seeds:
                if model_name == "rf":
                    from sklearn.ensemble import RandomForestClassifier
                    m = RandomForestClassifier(
                        n_estimators=200, class_weight="balanced",
                        random_state=seed, n_jobs=-1)
                    m.fit(X_tr_s, y_tr)
                else:
                    import xgboost as xgb
                    sw = np.where(y_tr == 1,
                                  len(y_tr) / (2 * y_tr.sum()),
                                  len(y_tr) / (2 * (len(y_tr) - y_tr.sum())))
                    m = xgb.XGBClassifier(
                        n_estimators=200, max_depth=8, learning_rate=0.1,
                        tree_method="hist", random_state=seed, n_jobs=-1,
                        eval_metric="logloss")
                    m.fit(X_tr_s, y_tr, sample_weight=sw,
                          eval_set=[(X_va_s, y_va)], verbose=False)

                y_prob = m.predict_proba(X_te_s)[:, 1]
                y_pred = m.predict(X_te_s)

                seed_aucs.append(roc_auc_score(y_te, y_prob))
                seed_f1s.append(f1_score(y_te, y_pred, zero_division=0))
                seed_precs.append(precision_score(y_te, y_pred, zero_division=0))
                seed_recs.append(recall_score(y_te, y_pred, zero_division=0))

            auc_m, f1_m = np.mean(seed_aucs), np.mean(seed_f1s)
            prec_m, rec_m = np.mean(seed_precs), np.mean(seed_recs)

            print(f"{attack:<40} {model_name:<5} {auc_m:>8.4f} {f1_m:>8.4f} "
                  f"{prec_m:>8.4f} {rec_m:>8.4f}")

            all_results.append({
                "attack_type": attack,
                "model": model_name,
                "auc_mean": round(auc_m, 6),
                "auc_std": round(np.std(seed_aucs), 6),
                "f1_mean": round(f1_m, 6),
                "f1_std": round(np.std(seed_f1s), 6),
                "precision_mean": round(prec_m, 6),
                "recall_mean": round(rec_m, 6),
                "n_train_attack": len(atk_tr),
                "n_test_attack": len(atk_te),
            })

    # Summary by category
    print(f"\n{'='*60}")
    print("SUMMARY BY ATTACK CATEGORY (RF)")
    print(f"{'='*60}")

    df_res = pd.DataFrame(all_results)
    rf_res = df_res[df_res["model"] == "rf"]

    category_map = {
        "Position": ["constantPositionOffset", "randomPositionOffset", "positionMirroring"],
        "Speed": ["constantSpeedOffset", "randomSpeedOffset", "zeroSpeedReport", "suddenConstantSpeed"],
        "Heading": ["reversedHeading"],
        "Acceleration": ["feignedBraking", "accelerationMultiplication"],
        "Multi": ["suddenStop", "dosAttack", "trafficCongestionSybil", "dataReplay"],
    }

    print(f"\n  {'Category':<20} {'Avg AUC':>10} {'Avg F1':>10}")
    print(f"  {'-'*42}")
    for cat, atk_list in category_map.items():
        cat_rows = rf_res[rf_res["attack_type"].isin(atk_list)]
        if len(cat_rows) > 0:
            print(f"  {cat:<20} {cat_rows['auc_mean'].mean():>10.4f} "
                  f"{cat_rows['f1_mean'].mean():>10.4f}")

    # Overall
    print(f"\n  {'Overall':<20} {rf_res['auc_mean'].mean():>10.4f} "
          f"{rf_res['f1_mean'].mean():>10.4f}")

    # Save
    df_res.to_csv(Path(args.output_dir) / "per_attack_results.csv", index=False)
    with open(Path(args.output_dir) / "per_attack_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved -> {args.output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()