"""
train_baselines.py — Centralized baseline models (12 features).

Usage:
    python -m models.centralized.train_baselines --data_dir data/processed --output_dir results/centralized
    python -m models.centralized.train_baselines --data_dir data/processed --output_dir results/centralized --models xgb --seeds 42 --classifications binary
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    classification_report,
)

CLASSIFICATION_FEATURES = [
    "sender_spd", "sender_acl",
    "time_delta", "distance", "heading_change",
    "speed_consistency", "accel_consistency",
    "heading_trajectory_consistency",
    "tx_delay", "mean_sender_receiver_dist", "n_receivers",
    "sender_dist_to_road_edge",
]

def load_split(data_dir, split):
    path = Path(data_dir) / f"{split}.parquet"
    df = pd.read_parquet(path)
    print(f"  {split}: {len(df):,} rows")
    return df

def get_Xy(df, classification, label_encoder=None):
    feat_cols = [c for c in CLASSIFICATION_FEATURES if c in df.columns]
    X = df[feat_cols].fillna(0.0).values.astype(np.float32)
    if classification == "binary":
        y = df["attacker"].values.astype(np.int64)
        names = ["Normal", "Attack"]
        le = None
    elif classification == "triclass":
        if label_encoder is None:
            le = LabelEncoder(); le.fit(sorted(df["attack_triclass"].unique()))
        else: le = label_encoder
        y = le.transform(df["attack_triclass"]).astype(np.int64)
        names = le.classes_.tolist()
    elif classification == "7class":
        if label_encoder is None:
            le = LabelEncoder(); le.fit(sorted(df["attack_category"].unique()))
        else: le = label_encoder
        y = le.transform(df["attack_category"]).astype(np.int64)
        names = le.classes_.tolist()
    else: raise ValueError(classification)
    return X, y, feat_cols, names, le

def evaluate(y_true, y_pred, y_prob, class_names):
    nc = len(class_names)
    avg = "binary" if nc == 2 else "macro"
    f1 = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
    prec = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
    rec = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob[:, 1])) if nc == 2 else \
              float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except: auc = float("nan")
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    pcf1 = {n: round(report[n]["f1-score"], 4) for n in class_names if n in report}
    return {"auc": round(auc, 6), "f1": round(f1, 6),
            "precision": round(prec, 6), "recall": round(rec, 6),
            "per_class_f1": pcf1}

def _feat_imp(model):
    return dict(zip(CLASSIFICATION_FEATURES,
                    model.feature_importances_.round(6).tolist()))

def train_rf(X_tr, y_tr, X_te, y_te, seed, names):
    from sklearn.ensemble import RandomForestClassifier
    m = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                               random_state=seed, n_jobs=-1)
    m.fit(X_tr, y_tr)
    met = evaluate(y_te, m.predict(X_te), m.predict_proba(X_te), names)
    met["feature_importance"] = _feat_imp(m)
    return met

def train_et(X_tr, y_tr, X_te, y_te, seed, names):
    from sklearn.ensemble import ExtraTreesClassifier
    m = ExtraTreesClassifier(n_estimators=200, class_weight="balanced",
                              random_state=seed, n_jobs=-1)
    m.fit(X_tr, y_tr)
    met = evaluate(y_te, m.predict(X_te), m.predict_proba(X_te), names)
    met["feature_importance"] = _feat_imp(m)
    return met

def train_xgb(X_tr, y_tr, X_va, y_va, X_te, y_te, seed, names):
    import xgboost as xgb
    nc = len(names)
    cls, cnts = np.unique(y_tr, return_counts=True)
    wmap = {c: len(y_tr)/(nc*n) for c, n in zip(cls, cnts)}
    sw = np.array([wmap[y] for y in y_tr], dtype=np.float32)
    params = {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.1,
              "tree_method": "hist", "random_state": seed, "n_jobs": -1,
              "eval_metric": "logloss" if nc==2 else "mlogloss"}
    if nc == 2: params["objective"] = "binary:logistic"
    else: params["objective"] = "multi:softprob"; params["num_class"] = nc
    m = xgb.XGBClassifier(**params)
    m.fit(X_tr, y_tr, sample_weight=sw, eval_set=[(X_va, y_va)], verbose=False)
    met = evaluate(y_te, m.predict(X_te), m.predict_proba(X_te), names)
    met["feature_importance"] = _feat_imp(m)
    return met

def train_dl(model_name, X_tr, y_tr, X_va, y_va, X_te, y_te, seed, names):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(seed); np.random.seed(seed)
    nf, nc = X_tr.shape[1], len(names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Device: {device}")
    cc = np.bincount(y_tr, minlength=nc).astype(float)
    cw = torch.FloatTensor(len(y_tr)/(nc*cc)).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    bidir = model_name == "bilstm"
    model = _make_lstm(nf, nc, bidir).to(device)
    bs = 1024
    mk = lambda X, y, s: DataLoader(TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
        batch_size=bs*(1 if s else 2), shuffle=s, num_workers=2, pin_memory=True)
    tr_dl, va_dl, te_dl = mk(X_tr, y_tr, True), mk(X_va, y_va, False), mk(X_te, y_te, False)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2, min_lr=1e-5)
    best_f1, best_state, pat = -1, None, 5
    avg = "binary" if nc == 2 else "macro"
    for ep in range(50):
        model.train(); tloss, nb = 0, 0
        for xb, yb in tr_dl:
            xb, yb = xb.unsqueeze(1).to(device), yb.to(device)
            opt.zero_grad(); loss = criterion(model(xb), yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tloss += loss.item(); nb += 1
        model.eval(); vp, vt = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                vp.extend(model(xb.unsqueeze(1).to(device)).argmax(-1).cpu().numpy())
                vt.extend(yb.numpy())
        vf1 = float(f1_score(vt, vp, average=avg, zero_division=0))
        sched.step(1-vf1)
        print(f"      Ep {ep+1:>2}: loss={tloss/nb:.4f} val_f1={vf1:.4f} lr={opt.param_groups[0]['lr']:.6f}")
        if vf1 > best_f1:
            best_f1, best_state = vf1, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_cnt = 0
        else:
            pat_cnt = pat_cnt + 1 if 'pat_cnt' in dir() else 1
            if pat_cnt >= pat: print(f"      Early stop ep {ep+1}"); break
    if best_state: model.load_state_dict(best_state)
    model.eval(); preds, probs = [], []
    with torch.no_grad():
        for xb, _ in te_dl:
            out = model(xb.unsqueeze(1).to(device))
            preds.extend(out.argmax(-1).cpu().numpy())
            probs.append(torch.softmax(out, -1).cpu().numpy())
    return evaluate(y_te, np.array(preds), np.vstack(probs), names)

def _make_lstm(nf, nc, bidir):
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(nf, 128, 2, dropout=0.3, batch_first=True, bidirectional=bidir)
            h = 256 if bidir else 128
            self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, nc))
        def forward(self, x):
            out, _ = self.lstm(x); return self.fc(out[:, -1, :])
    return M()

MODEL_NAMES = {"rf": "Random Forest", "et": "Extra Trees", "xgb": "XGBoost",
               "lstm": "LSTM", "bilstm": "BiLSTM"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES.keys()),
                        choices=list(MODEL_NAMES.keys()))
    parser.add_argument("--classifications", nargs="+", default=["binary", "triclass", "7class"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60); print("Loading data"); print("="*60)
    train_df = load_split(args.data_dir, "train")
    val_df = load_split(args.data_dir, "val")
    test_df = load_split(args.data_dir, "test")
    all_results = []

    for cl in args.classifications:
        print(f"\n{'='*60}\nClassification: {cl}\n{'='*60}")
        X_tr, y_tr, fc, names, le = get_Xy(train_df, cl)
        X_va, y_va, _, _, _ = get_Xy(val_df, cl, le)
        X_te, y_te, _, _, _ = get_Xy(test_df, cl, le)
        print(f"  Features ({len(fc)}): {fc}")
        print(f"  Classes: {names}")
        print(f"  Shapes: train={X_tr.shape} val={X_va.shape} test={X_te.shape}")
        print(f"  Train dist: {dict(zip(names, np.bincount(y_tr)))}")
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr).astype(np.float32)
        X_va_s = scaler.transform(X_va).astype(np.float32)
        X_te_s = scaler.transform(X_te).astype(np.float32)

        for mk in args.models:
            for seed in args.seeds:
                rid = f"{mk}__{cl}__seed{seed}"
                rp = Path(args.output_dir) / f"{rid}.json"
                if rp.exists():
                    print(f"\n  [SKIP] {rid}")
                    all_results.append(json.loads(rp.read_text())); continue
                print(f"\n  {MODEL_NAMES[mk]} | {cl} | seed={seed}")
                t0 = time.time()
                try:
                    if mk == "rf": met = train_rf(X_tr_s, y_tr, X_te_s, y_te, seed, names)
                    elif mk == "et": met = train_et(X_tr_s, y_tr, X_te_s, y_te, seed, names)
                    elif mk == "xgb": met = train_xgb(X_tr_s, y_tr, X_va_s, y_va, X_te_s, y_te, seed, names)
                    else: met = train_dl(mk, X_tr_s, y_tr, X_va_s, y_va, X_te_s, y_te, seed, names)
                    el = round(time.time()-t0, 1)
                    res = {"run_id": rid, "model": mk, "classification": cl,
                           "seed": seed, "runtime_s": el, "class_names": names, **met}
                    rp.write_text(json.dumps(res, indent=2)); all_results.append(res)
                    print(f"    AUC={met['auc']:.4f} F1={met['f1']:.4f} "
                          f"P={met['precision']:.4f} R={met['recall']:.4f} | {el}s")
                    print(f"    Per-class: {met.get('per_class_f1', {})}")
                except Exception as e:
                    print(f"    [FAIL] {e}"); import traceback; traceback.print_exc()
                gc.collect()

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    rows = []
    for cl in args.classifications:
        cr = [r for r in all_results if r.get("classification") == cl]
        print(f"\n  {cl}:")
        print(f"  {'Model':<12} {'AUC':>14} {'F1':>14} {'Prec':>14} {'Rec':>14}")
        print(f"  {'-'*70}")
        for mk in args.models:
            mr = [r for r in cr if r.get("model") == mk]
            if not mr: continue
            a = [r["auc"] for r in mr if not np.isnan(r.get("auc", float("nan")))]
            f = [r["f1"] for r in mr]; p = [r["precision"] for r in mr]; rc = [r["recall"] for r in mr]
            fmt = lambda v: f"{np.mean(v):.4f}±{np.std(v):.4f}" if len(v)>1 else f"{v[0]:.4f}"
            print(f"  {mk:<12} {fmt(a):>14} {fmt(f):>14} {fmt(p):>14} {fmt(rc):>14}")
            rows.append({"classification": cl, "model": mk,
                         "auc_mean": round(float(np.mean(a)),6) if a else None,
                         "f1_mean": round(float(np.mean(f)),6),
                         "f1_std": round(float(np.std(f)),6) if len(f)>1 else 0})
    pd.DataFrame(rows).to_csv(Path(args.output_dir)/"summary.csv", index=False)
    for mk in ["rf", "xgb"]:
        fi = [r for r in all_results if r.get("model")==mk and r.get("classification")=="binary"
              and "feature_importance" in r]
        if fi:
            imp = dict(sorted(fi[0]["feature_importance"].items(), key=lambda x: x[1], reverse=True))
            Path(args.output_dir, f"feature_importance_{mk}.json").write_text(json.dumps(imp, indent=2))
            print(f"\n  Feature importance ({mk}):")
            for feat, val in list(imp.items())[:5]:
                print(f"    {feat:34s} {val:.4f}")
    print("\nDone.")

if __name__ == "__main__":
    main()