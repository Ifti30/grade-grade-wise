#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fixed prediction script:
- Loads artifacts saved by train.py.
- Computes current GPA/CGPA, final CGPA prediction (ensemble) and next-sem GPA.
- Prints __RESULT__ JSON and writes a detailed JSON to --out-file.
"""

import argparse, os, sys, json, numpy as np, datetime, joblib
from pathlib import Path

# Force headless backend so plotting works in containers
import matplotlib
matplotlib.use("Agg")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns  # optional
    from matplotlib.lines import Line2D
except Exception:
    plt = None
    sns = None
    Line2D = None

def semester_gpa(sem, GP):
    pts = []
    for k, g in sem.items():
        if k != "attendancePercentage" and g in GP:
            pts.append(GP[g])
    return float(np.mean(pts)) if pts else None

def compute_cgpa(semesters, sem_nums, upto, GP):
    use = sem_nums[:upto]
    total_points = 0.0
    total_hours = 0.0
    for sem_no in use:
        sem = semesters.get(str(sem_no))
        if not isinstance(sem, dict):
            continue
        sem_gpa = semester_gpa(sem, GP)
        if sem_gpa is None:
            continue
        ch = sem.get("creditHours")
        if not isinstance(ch, (int, float)) or ch <= 0:
            continue
        total_points += sem_gpa * float(ch)
        total_hours += float(ch)
    if total_hours <= 0:
        return None
    return float(total_points / total_hours)

def build_features_for_final(student, GP, feature_order):
    semesters = student.get("semesters", {})
    if not semesters: return None, None
    sem_nums = sorted(map(int, semesters.keys()))
    if len(sem_nums) < 2: return None, None
    upto = sem_nums[-2]
    att, credit_hours = [], []
    for s in sem_nums:
        if s <= upto:
            sem = semesters[str(s)]
            if "attendancePercentage" in sem: att.append(sem["attendancePercentage"])
            ch = sem.get("creditHours")
            if isinstance(ch, (int, float)) and ch > 0:
                credit_hours.append(ch)
    avg_att = float(np.mean(att)) if att else 0.0
    s_count = len(sem_nums) - 1
    cgpa_1 = compute_cgpa(semesters, sem_nums, 1, GP)
    cgpa_s = compute_cgpa(semesters, sem_nums, s_count, GP)
    if cgpa_1 is None or cgpa_s is None:
        return None, None
    gpa_trend = 0.0 if s_count == 1 else float((cgpa_s - cgpa_1) / (s_count - 1))
    avg_ch = float(np.mean(credit_hours)) if credit_hours else 0.0
    feature_map = {
        "ssc_gpa": float(student.get("ssc_gpa", 0.0)),
        "hsc_gpa": float(student.get("hsc_gpa", 0.0)),
        "gender_bin": 1 if str(student.get("gender", "")).lower() == "female" else 0,
        "birth_year": int(student.get("birth_year", 0)),
        "avg_credit_hours": avg_ch,
        "avg_attendance": avg_att,
        "gpa_trend": float(gpa_trend)
    }
    X = [feature_map.get(name, 0.0) for name in feature_order]
    assert len(X) == len(feature_order)
    return X, None

def build_features_for_next(student, GP, feature_order):
    semesters = student.get("semesters", {})
    if not semesters: return None
    sem_nums = sorted(map(int, semesters.keys()))
    if len(sem_nums) < 2:
        return None
    use = sem_nums[:-1]
    att, credit_hours = [], []
    for s in use:
        sem = semesters[str(s)]
        if "attendancePercentage" in sem: att.append(sem["attendancePercentage"])
        ch = sem.get("creditHours")
        if isinstance(ch, (int, float)) and ch > 0:
            credit_hours.append(ch)
    avg_att = float(np.mean(att)) if att else 0.0
    s_count = len(use)
    cgpa_1 = compute_cgpa(semesters, sem_nums, 1, GP)
    cgpa_s = compute_cgpa(semesters, sem_nums, s_count, GP)
    if cgpa_1 is None or cgpa_s is None:
        return None
    gpa_trend = 0.0 if s_count == 1 else float((cgpa_s - cgpa_1) / (s_count - 1))
    avg_ch = float(np.mean(credit_hours)) if credit_hours else 0.0
    feature_map = {
        "ssc_gpa": float(student.get("ssc_gpa", 0.0)),
        "hsc_gpa": float(student.get("hsc_gpa", 0.0)),
        "gender_bin": 1 if str(student.get("gender", "")).lower() == "female" else 0,
        "birth_year": int(student.get("birth_year", 0)),
        "avg_credit_hours": avg_ch,
        "avg_attendance": avg_att,
        "gpa_trend": float(gpa_trend)
    }
    X = [feature_map.get(name, 0.0) for name in feature_order]
    assert len(X) == len(feature_order)
    return X

def compute_current(student, GP):
    sems = student.get("semesters", {})
    if not sems: return None, None, None
    sem_nums = sorted(map(int, sems.keys()))
    last = sem_nums[-1]
    sem = sems[str(last)]
    last_sem_gpa = semester_gpa(sem, GP)
    cgpa = compute_cgpa(sems, sem_nums, len(sem_nums), GP)
    return last_sem_gpa, cgpa, last

def average_course_load(student):
    semesters = student.get("semesters", {})
    if not semesters:
        return None
    loads = []
    for sem in semesters.values():
        if not isinstance(sem, dict):
            continue
        load = sum(1 for k in sem if k != "attendancePercentage")
        if load > 0:
            loads.append(load)
    if not loads:
        return None
    return float(np.mean(loads))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org-id", required=True)
    ap.add_argument("--student-json", required=True)
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--out-file", required=True)
    ap.add_argument("--credit-hours", type=float, default=None)
    args = ap.parse_args()

    org_id = args.org_id
    student = json.load(open(args.student_json,"r"))
    art_dir = Path(args.artifacts_dir)
    out_file = Path(args.out_file)

    meta = json.load(open(art_dir/"metadata.json", "r"))
    GP = meta.get("grade_points", {
        "A+":4.0,"A":3.75,"A-":3.5,"B+":3.25,"B":3.0,"B-":2.75,"C":2.5,"C-":2.25,"D":2.0,"F":0.0
    })
    max_gpa = float(meta.get("max_gpa", max(GP.values())))
    best_model = meta.get("best_model")
    baseline_course_load = meta.get("baseline_course_load")
    baseline_credit_hours = meta.get("baseline_credit_hours")
    credit_hours = args.credit_hours

    # Static asset helper
    storage_root = None
    for idx, part in enumerate(out_file.parts):
        if part == "storage":
            storage_root = Path(*out_file.parts[:idx + 1])
            break

    def to_static_path(path_obj: Path) -> str:
        if storage_root is not None:
            try:
                rel = path_obj.relative_to(storage_root)
                return "/static/" + rel.as_posix()
            except Exception:
                pass
        return str(path_obj)

    print(f"[INFO] org={org_id} student={student.get('student_id')} max_gpa={max_gpa}")

    enabled_models = set(meta.get("enabled_models") or [])

    # Load models
    models = {}
    for name in ["DecisionTree","RandomForest","LightGBM","SVR"]:
        p = art_dir/f"{name}.joblib"
        if p.exists() and (not enabled_models or name in enabled_models):
            models[name] = joblib.load(p)

    next_models = {}
    for name in ["DecisionTree","RandomForest","LightGBM","SVR"]:
        p = art_dir/f"{name}Next.joblib"
        if p.exists() and (not enabled_models or name in enabled_models):
            next_models[name] = joblib.load(p)
        elif name in models:
            next_models[name] = models[name]

    # MLP + scaler
    import torch, torch.nn as nn
    class MLP(nn.Module):
        def __init__(self, in_dim, hid=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hid), nn.ReLU(),
                nn.Linear(hid, hid), nn.ReLU(),
                nn.Linear(hid, 1)
            )
        def forward(self, x): return self.net(x)

    feat_order = meta["feature_order"]
    mlp_model = None
    mlp_scaler = None
    mlp_path = art_dir/"MLP.pt"
    mlp_scaler_path = art_dir/"MLP_Scaler.joblib"
    if mlp_path.exists() and mlp_scaler_path.exists() and (not enabled_models or "MLP" in enabled_models):
        mlp_model = MLP(in_dim=len(feat_order), hid=64)
        mlp_model.load_state_dict(torch.load(mlp_path, map_location="cpu"))
        mlp_model.eval()
        mlp_scaler = joblib.load(mlp_scaler_path)

    mlp_next_model = None
    mlp_next_scaler = None
    mlp_next_path = art_dir/"MLPNext.pt"
    mlp_next_scaler_path = art_dir/"MLPNext_Scaler.joblib"
    if mlp_next_path.exists() and mlp_next_scaler_path.exists() and (not enabled_models or "MLP" in enabled_models):
        mlp_next_model = MLP(in_dim=len(feat_order), hid=64)
        mlp_next_model.load_state_dict(torch.load(mlp_next_path, map_location="cpu"))
        mlp_next_model.eval()
        mlp_next_scaler = joblib.load(mlp_next_scaler_path)
    elif mlp_model is not None and mlp_scaler is not None:
        mlp_next_model = mlp_model
        mlp_next_scaler = mlp_scaler

    report_path = art_dir/"report.json"
    report = None
    if report_path.exists():
        try:
            report = json.load(open(report_path, "r"))
        except Exception:
            report = None

    # Features
    Xf_new, _ = build_features_for_final(student, GP, feat_order)
    Xn_new = build_features_for_next(student, GP, feat_order)

    # Current
    cur_sem_gpa, cur_cgpa, last_sem_idx = compute_current(student, GP)
    if cur_sem_gpa is not None:
        print(f"[INFO] current last-sem GPA (sem {last_sem_idx}) = {cur_sem_gpa:.2f}")
    if cur_cgpa is not None:
        print(f"[INFO] current CGPA (to sem {last_sem_idx}) = {cur_cgpa:.2f}")

    # Predict final CGPA (5 models)
    preds_final = {}
    if Xf_new is not None:
        Xf = np.array(Xf_new, float).reshape(1,-1)
        for name, model in models.items():
            try:
                preds_final[name] = float(model.predict(Xf)[0])
            except Exception as e:
                print(f"[WARN] {name} prediction failed: {e}")
        if mlp_model is not None and mlp_scaler is not None:
            Xs = mlp_scaler.transform(Xf)
            with torch.no_grad():
                preds_final["MLP"] = float(mlp_model(torch.tensor(Xs, dtype=torch.float32)).numpy().reshape(-1)[0])
    else:
        print("[WARN] Not enough semesters for final CGPA prediction (needs ≥ 2).")

    # Predict next semester GPA (RF+LGBM+SVR)
    preds_next = {}
    if Xn_new is not None:
        Xn = np.array(Xn_new, float).reshape(1,-1)
        for name, model in next_models.items():
            try:
                preds_next[name] = float(model.predict(Xn)[0])
            except Exception as e:
                print(f"[WARN] {name} next-sem prediction failed: {e}")
        if mlp_next_model is not None and mlp_next_scaler is not None:
            Xn_s = mlp_next_scaler.transform(Xn)
            with torch.no_grad():
                preds_next["MLP"] = float(mlp_next_model(torch.tensor(Xn_s, dtype=torch.float32)).numpy().reshape(-1)[0])

    # Ensemble mean
    ens_final = float(np.mean([v for v in preds_final.values()])) if preds_final else None
    ens_next  = float(np.mean([v for v in preds_next.values()])) if preds_next else None

    def clamp_gpa(value):
        return max(0.0, min(max_gpa, float(value)))

    load_adjusted = None
    course_load = None
    if credit_hours is not None:
        try:
            if credit_hours > 0:
                course_load = credit_hours / 3.0
                course_load = max(1.0, min(7.0, float(course_load)))
        except Exception:
            course_load = None

    scale = None
    if course_load is not None:
        base_load = baseline_course_load
        if base_load is None:
            base_load = average_course_load(student)
        if base_load is None or base_load <= 0:
            base_load = 4.0
        coefficient = 0.10
        min_scale = 0.92
        max_scale = 1.08
        scale = 1.0 + coefficient * ((course_load - base_load) / base_load)
        scale = max(min_scale, min(max_scale, float(scale)))

    if scale is not None:
        adjusted_final = {name: clamp_gpa(value * scale) for name, value in preds_final.items()}
        adjusted_next = {name: clamp_gpa(value * scale) for name, value in preds_next.items()}
        adj_final_mean = float(np.mean(list(adjusted_final.values()))) if adjusted_final else None
        adj_next_mean = float(np.mean(list(adjusted_next.values()))) if adjusted_next else None

        load_adjusted = {
            "final_cgpa": adjusted_final,
            "next_sem_gpa": adjusted_next,
            "ensemble": {
                "final_cgpa_mean": adj_final_mean,
                "next_sem_gpa_mean": adj_next_mean
            },
            "context": {
                "baseline_course_load": float(base_load),
                "baseline_credit_hours": float(baseline_credit_hours) if baseline_credit_hours is not None else None,
                "requested_course_load": float(course_load),
                "requested_credit_hours": float(credit_hours) if credit_hours is not None else None,
                "coefficient": coefficient,
                "min_scale": min_scale,
                "max_scale": max_scale,
                "scale": scale
            },
            "delta": {
                "final_cgpa_mean": (adj_final_mean - ens_final) if (adj_final_mean is not None and ens_final is not None) else None,
                "next_sem_gpa_mean": (adj_next_mean - ens_next) if (adj_next_mean is not None and ens_next is not None) else None
            }
        }

    # Risk from predicted next_sem_cgpa thresholds
    risk_label = "Unknown"
    thresholds = meta.get("risk_thresholds")
    if not thresholds and report:
        thresholds = report.get("classification", {}).get("thresholds")
    high_max = thresholds.get("high_max") if isinstance(thresholds, dict) else None
    med_max = thresholds.get("med_max") if isinstance(thresholds, dict) else None
    if ens_next is not None and isinstance(high_max, (int, float)) and isinstance(med_max, (int, float)):
        if ens_next <= high_max:
            risk_label = "High"
        elif ens_next <= med_max:
            risk_label = "Medium"
        else:
            risk_label = "Low"

    plots = {}
    if plt is not None:
        try:
            if preds_final:
                fig, ax = plt.subplots(figsize=(6, 4))
                models = list(preds_final.keys())
                values = [preds_final[m] for m in models]
                colors = ['#0ea5e9'] * len(models)
                if best_model:
                    for i, model in enumerate(models):
                        if model.lower() == str(best_model).lower():
                            colors[i] = '#22c55e'
                ax.bar(models, values, color=colors)
                ax.set_ylabel("Predicted Final CGPA")
                ax.set_title("Final CGPA Predictions by Model")
                ax.tick_params(axis='x', rotation=30)
                if best_model and Line2D is not None:
                    ax.legend(handles=[Line2D([0], [0], color='#22c55e', lw=8, label='Best Model')])
                plt.tight_layout()
                final_plot = out_file.with_name(out_file.stem + "_final_cgpa.png")
                plt.savefig(final_plot)
                plt.close(fig)
                plots["final_cgpa_comparison"] = to_static_path(final_plot)

            if preds_next:
                fig, ax = plt.subplots(figsize=(6, 4))
                models = list(preds_next.keys())
                values = [preds_next[m] for m in models]
                ax.bar(models, values, color='#6366f1')
                ax.set_ylabel("Predicted Next Sem GPA")
                ax.set_title("Next Semester GPA Predictions")
                ax.tick_params(axis='x', rotation=30)
                plt.tight_layout()
                next_plot = out_file.with_name(out_file.stem + "_next_sem_gpa.png")
                plt.savefig(next_plot)
                plt.close(fig)
                plots["next_sem_gpa_comparison"] = to_static_path(next_plot)
        except Exception as e:
            print(f"[WARN] prediction plotting failed: {e}")

    s_used = len(student.get("semesters", {})) - 1
    result_payload = {
        "student_id": student.get("student_id"),
        "s_used": s_used,
        "current": {"last_sem_index": last_sem_idx, "last_sem_gpa": cur_sem_gpa, "current_cgpa": cur_cgpa},
        "predictions": {
            "final_cgpa": preds_final,
            "next_sem_gpa": preds_next,
            "ensemble": {"final_cgpa_mean": ens_final, "next_sem_gpa_mean": ens_next}
        },
        "credit_hours": credit_hours,
        "course_load": course_load,
        "load_adjusted": load_adjusted,
        "risk": risk_label,
        "created_at": datetime.datetime.utcnow().isoformat()+"Z",
        "grade_points_used": GP,
        "max_gpa": max_gpa,
        "best_model": best_model,
        "plots": plots
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(result_payload, f, indent=2)
    print(f"[INFO] wrote {out_file}")

    hist_path = Path(art_dir) / "analysis_history.json"
    try: hist = json.load(open(hist_path, "r"))
    except Exception: hist = []
    hist.append(result_payload)
    with open(hist_path, "w") as f:
        json.dump(hist, f, indent=2)
    print(f"[INFO] appended to {hist_path}")

    print("__RESULT__" + json.dumps({
        "status": "ok",
        "predictions": result_payload["predictions"],
        "loadAdjusted": result_payload["load_adjusted"],
        "creditHours": result_payload["credit_hours"],
        "courseLoad": result_payload["course_load"],
        "risk": risk_label,
        "sUsed": s_used,
        "current": result_payload["current"],
        "outFile": str(out_file),
        "max_gpa": max_gpa,
        "plots": plots,
        "bestModel": best_model
    }))
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        print("__RESULT__" + json.dumps({"status":"error","error":str(e)}))
        sys.exit(1)
