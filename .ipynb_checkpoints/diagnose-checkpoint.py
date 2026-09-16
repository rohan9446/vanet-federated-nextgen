# diagnose.py (update features to 12)
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

FEATURES = [
    "sender_spd", "sender_acl", "time_delta", "distance", "heading_change",
    "speed_consistency", "accel_consistency", "heading_trajectory_consistency",
    "tx_delay", "mean_sender_receiver_dist", "n_receivers",
    "sender_dist_to_road_edge",
]

train = pd.read_parquet("data/processed/train.parquet")
test = pd.read_parquet("data/processed/test.parquet")

normal_tr = train[train["attacker"]==0]
normal_te = test[test["attacker"]==0]

print(f"{'Attack':<40} {'AUC':>8} {'F1':>8}")
print("-"*60)

for attack in sorted(train[train["attacker"]==1]["attack_type"].unique()):
    atk_tr = train[(train["attack_type"]==attack) & (train["attacker"]==1)]
    atk_te = test[(test["attack_type"]==attack) & (test["attacker"]==1)]
    
    tr = pd.concat([normal_tr, atk_tr])
    te = pd.concat([normal_te, atk_te])
    
    X_tr = tr[FEATURES].fillna(0).values
    y_tr = tr["attacker"].values
    X_te = te[FEATURES].fillna(0).values
    y_te = te["attacker"].values
    
    sc = StandardScaler().fit(X_tr)
    m = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    m.fit(sc.transform(X_tr), y_tr)
    
    auc = roc_auc_score(y_te, m.predict_proba(sc.transform(X_te))[:,1])
    f1 = f1_score(y_te, m.predict(sc.transform(X_te)))
    print(f"{attack:<40} {auc:>8.4f} {f1:>8.4f}")