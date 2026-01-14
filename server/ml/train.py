#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fixed training script:
- Trains DT, RF, (optional) SVR, LightGBM, and a PyTorch MLP.
- Trains a risk classifier with SMOTE.
- **Always saves artifacts** even if plotting fails.
- Writes metadata.json and prints __RESULT__ JSON used by the Node backend.
"""

import argparse, os, sys, json, datetime, math
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd

# Threading hints
os.environ.setdefault("OMP_NUM_THREADS", "4")

# Optional plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns  # optional
except Exception:
    plt = None
    sns = None

# ------------------------- Bounds -------------------------
BOUNDS = {
    "DT_MAX_DEPTH": (1, 50),
    "DT_MIN_SAMPLES_LEAF": (1, 50),
    "RF_TREES": (50, 1000),
    "RF_MAX_DEPTH": (1, 50),
    "RF_MIN_SAMPLES_LEAF": (1, 50),
    "LGBM_N_ESTIMATORS": (200, 4000),
    "LGBM_REG_ALPHA": (0.0, 10.0),
    "LGBM_REG_LAMBDA": (0.0, 10.0),
    "MLP_HIDDEN": (16, 256),
    "MLP_EPOCHS": (50, 600),
    "MLP_PATIENCE": (10, 100),
    "SVR_C": (0.1, 100.0),
    "SVR_EPSILON": (0.001, 1.0),
    "TEST_SIZE": (0.10, 0.30),
    "THREADS": (2, 8),
}

SCHEMA_VERSION = "v2_option2_nextcgpa_risk"

# ------------------------- Data cleaning -------------------------
# Normalize input records to a consistent schema:
# - coerce types, clamp impossible values, dedupe repeated semesters/courses
# - keep counters for diagnostics without hard-failing the run
def coerce_float(value):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        num = float(value)
    except Exception:
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num

def coerce_int(value):
    num = coerce_float(value)
    if num is None:
        return None
    return int(num)

def is_missing(value):
    if value is None or value == "":
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False

def clamp_value(value, min_value=None, max_value=None):
    if value is None:
        return None
    if min_value is not None and value < min_value:
        return min_value
    if max_value is not None and value > max_value:
        return max_value
    return value

def sanitize_semester(sem, grade_points, max_gpa, stats):
    """Clean a single semester dict (attendance, credits, per-course grades)."""
    if not isinstance(sem, dict):
        stats["invalid_semesters"] += 1
        return None
    cleaned = {}

    att_val = coerce_float(sem.get("attendancePercentage"))
    if att_val is not None:
        if att_val < 0 or att_val > 100:
            stats["attendance_clamped"] += 1
        att_val = clamp_value(att_val, 0, 100)
        cleaned["attendancePercentage"] = att_val
    else:
        stats["missing_attendance"] += 1

    credit_val = coerce_float(sem.get("creditHours"))
    if credit_val is not None and credit_val > 0:
        cleaned["creditHours"] = credit_val
    else:
        if credit_val is not None:
            stats["invalid_credit_hours"] += 1
        stats["missing_credit_hours"] += 1

    course_map = {}
    courses = sem.get("courses")
    if isinstance(courses, list):
        for course in courses:
            if not isinstance(course, dict):
                continue
            code = (
                course.get("code")
                or course.get("course_code")
                or course.get("courseId")
                or course.get("id")
                or course.get("name")
            )
            grade = course.get("grade") or course.get("letterGrade") or course.get("result")
            if code is None:
                continue
            key = str(code)
            if key in course_map:
                stats["duplicate_courses"] += 1
            course_map[key] = grade

    for key, value in sem.items():
        if key in (
            "attendancePercentage",
            "creditHours",
            "courses",
            "semester",
            "semester_no",
            "sem_no",
            "semesterNumber",
            "term"
        ):
            continue
        if value is None or value == "":
            continue
        if key in course_map:
            stats["duplicate_courses"] += 1
        course_map[str(key)] = value

    for key, grade in course_map.items():
        if isinstance(grade, (int, float)):
            if math.isnan(float(grade)):
                stats["invalid_grades"] += 1
                continue
            val = clamp_value(float(grade), 0, max_gpa)
            if val != float(grade):
                stats["grade_clamped"] += 1
                stats["grade_clamped_by_field"]["course_numeric"] += 1
            cleaned[key] = val
        elif grade in grade_points:
            cleaned[key] = grade
        else:
            stats["invalid_grades"] += 1

    return cleaned

def merge_semesters(existing, incoming, stats):
    """Merge two semester dicts, preferring existing values and de-duping courses."""
    if existing is None:
        return incoming
    merged = dict(existing)

    if "attendancePercentage" not in merged and "attendancePercentage" in incoming:
        merged["attendancePercentage"] = incoming["attendancePercentage"]
    if "creditHours" not in merged and "creditHours" in incoming:
        merged["creditHours"] = incoming["creditHours"]

    for key, value in incoming.items():
        if key in ("attendancePercentage", "creditHours"):
            continue
        if key in merged and merged[key] != value:
            stats["duplicate_courses"] += 1
        merged[key] = value
    return merged

def sanitize_semesters(semesters, grade_points, max_gpa, stats):
    """Normalize semesters from dict or list into a clean dict keyed by semester number."""
    cleaned = {}
    if isinstance(semesters, dict):
        items = semesters.items()
    elif isinstance(semesters, list):
        items = []
        for entry in semesters:
            if not isinstance(entry, dict):
                continue
            sem_no = (
                entry.get("semester")
                or entry.get("semester_no")
                or entry.get("sem_no")
                or entry.get("semesterNumber")
                or entry.get("term")
            )
            if sem_no is None:
                continue
            items.append((sem_no, entry))
    else:
        stats["invalid_semesters"] += 1
        return cleaned

    for sem_no, sem in items:
        sem_id = coerce_int(sem_no)
        if sem_id is None:
            continue
        key = str(sem_id)
        cleaned_sem = sanitize_semester(sem, grade_points, max_gpa, stats)
        if cleaned_sem is None:
            continue
        if key in cleaned:
            stats["duplicate_semesters"] += 1
            cleaned[key] = merge_semesters(cleaned[key], cleaned_sem, stats)
        else:
            cleaned[key] = cleaned_sem
    return cleaned

def sanitize_student(raw, grade_points, max_gpa, stats):
    """Clean a student record and its nested semesters."""
    if not isinstance(raw, dict):
        stats["invalid_students"] += 1
        return None
    student = dict(raw)

    raw_ssc = coerce_float(student.get("ssc_gpa"))
    if raw_ssc is None:
        stats["missing_demographics"] += 1
        raw_ssc = 0.0
    ssc = clamp_value(raw_ssc, 0, max_gpa)
    if raw_ssc != ssc:
        stats["grade_clamped"] += 1
        stats["grade_clamped_by_field"]["ssc_gpa"] += 1
    student["ssc_gpa"] = float(ssc)

    raw_hsc = coerce_float(student.get("hsc_gpa"))
    if raw_hsc is None:
        stats["missing_demographics"] += 1
        raw_hsc = 0.0
    hsc = clamp_value(raw_hsc, 0, max_gpa)
    if raw_hsc != hsc:
        stats["grade_clamped"] += 1
        stats["grade_clamped_by_field"]["hsc_gpa"] += 1
    student["hsc_gpa"] = float(hsc)

    birth_year = coerce_int(student.get("birth_year"))
    if birth_year is None:
        stats["missing_demographics"] += 1
        birth_year = 0
    student["birth_year"] = int(birth_year)

    student["gender"] = str(student.get("gender") or "")

    semesters = sanitize_semesters(student.get("semesters", {}), grade_points, max_gpa, stats)
    student["semesters"] = semesters
    return student

def merge_students(existing, incoming, stats):
    """Merge duplicate student records by filling missing demographics and semesters."""
    merged = dict(existing)
    for key in ("ssc_gpa", "hsc_gpa", "birth_year", "gender"):
        if is_missing(merged.get(key)) and not is_missing(incoming.get(key)):
            merged[key] = incoming.get(key)
    existing_sems = merged.get("semesters", {})
    incoming_sems = incoming.get("semesters", {})
    if isinstance(existing_sems, dict) and isinstance(incoming_sems, dict):
        for sem_key, sem_value in incoming_sems.items():
            if sem_key in existing_sems:
                stats["duplicate_semesters"] += 1
                existing_sems[sem_key] = merge_semesters(existing_sems[sem_key], sem_value, stats)
            else:
                existing_sems[sem_key] = sem_value
        merged["semesters"] = existing_sems
    return merged
    return merged

# ------------------------- GPA helpers -------------------------
# Compute semester/CGPA values in a resilient way (skip invalid/missing data).
def validate_grade_points(grade_points: dict):
    if not isinstance(grade_points, dict) or len(grade_points) < 2 or len(grade_points) > 30:
        raise ValueError("GRADE_POINTS must be an object with 2..30 entries")
    vals = []
    for k, v in grade_points.items():
        if not isinstance(k, str): raise ValueError("GRADE_POINTS keys must be strings")
        if not isinstance(v, (int, float)): raise ValueError("GRADE_POINTS values must be numbers")
        if v < 0 or v > 10: raise ValueError("GRADE_POINTS values must be within [0,10]")
        vals.append(float(v))
    return float(max(vals))

def compute_semester_gpa(sem, GP):
    if not isinstance(sem, dict):
        return None
    max_gpa = float(max(GP.values())) if GP else 4.0
    pts = []
    for k, g in sem.items():
        if k in ("attendancePercentage", "creditHours", "courses"):
            continue
        if g in GP:
            pts.append(GP[g])
        elif isinstance(g, (int, float)):
            val = float(g)
            if math.isnan(val) or math.isinf(val):
                continue
            val = clamp_value(val, 0, max_gpa)
            pts.append(val)
    return float(np.mean(pts)) if pts else None

def semester_gpa(sem, GP):
    return compute_semester_gpa(sem, GP)

def compute_cgpa(semesters, sem_nums, upto, GP):
    if not isinstance(semesters, dict):
        return None
    use = sem_nums[:upto]
    total_points = 0.0
    total_hours = 0.0
    for sem_no in use:
        sem = semesters.get(str(sem_no))
        if not isinstance(sem, dict):
            continue
        sem_gpa = compute_semester_gpa(sem, GP)
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

def cumulative_cgpa(semesters, GP):
    if not isinstance(semesters, dict):
        return None
    sem_nums = sorted(int(k) for k in semesters.keys() if str(k).isdigit())
    if not sem_nums:
        return None
    return compute_cgpa(semesters, sem_nums, len(sem_nums), GP)

def build_features_for_final(student, GP):
    """Features/label for final CGPA prediction from all but last semester."""
    semesters = student.get("semesters", {})
    if not semesters: return None, None
    sem_nums = sorted(map(int, semesters.keys()))
    if len(sem_nums) < 2: return None, None
    y = compute_cgpa(semesters, sem_nums, len(sem_nums), GP)
    if y is None: return None, None

    upto = sem_nums[-2]
    att = []
    for s in sem_nums:
        if s <= upto:
            sem = semesters[str(s)]
            if not isinstance(sem, dict):
                continue
            if "attendancePercentage" in sem: att.append(sem["attendancePercentage"])
    avg_att = float(np.mean(att)) if att else 0.0
    s_count = len(sem_nums) - 1
    cgpa_1 = compute_cgpa(semesters, sem_nums, 1, GP)
    cgpa_s = compute_cgpa(semesters, sem_nums, s_count, GP)
    if cgpa_1 is None or cgpa_s is None:
        return None, None
    gpa_trend = 0.0 if s_count == 1 else float((cgpa_s - cgpa_1) / (s_count - 1))
    avg_ch = average_credit_hours(student)

    X = [
        float(student.get("ssc_gpa", 0.0)),
        float(student.get("hsc_gpa", 0.0)),
        1 if str(student.get("gender","")).lower()=="female" else 0,
        int(student.get("birth_year", 0)),
        float(avg_ch) if avg_ch is not None else 0.0,
        avg_att,
        gpa_trend
    ]

    return X, float(y)

def build_features_for_next(student, GP):
    """Features for next-semester CGPA prediction using all completed semesters."""
    semesters = student.get("semesters", {})
    if not semesters: return None
    sem_nums = sorted(map(int, semesters.keys()))
    att = []
    for s in sem_nums:
        sem = semesters[str(s)]
        if not isinstance(sem, dict):
            continue
        if "attendancePercentage" in sem: att.append(sem["attendancePercentage"])
    avg_att = float(np.mean(att)) if att else 0.0
    s_count = len(sem_nums)
    cgpa_1 = compute_cgpa(semesters, sem_nums, 1, GP)
    cgpa_s = compute_cgpa(semesters, sem_nums, s_count, GP)
    if cgpa_1 is None or cgpa_s is None:
        return None
    gpa_trend = 0.0 if s_count == 1 else float((cgpa_s - cgpa_1) / (s_count - 1))
    avg_ch = average_credit_hours(student)

    X = [
        float(student.get("ssc_gpa", 0.0)),
        float(student.get("hsc_gpa", 0.0)),
        1 if str(student.get("gender","")).lower()=="female" else 0,
        int(student.get("birth_year", 0)),
        float(avg_ch) if avg_ch is not None else 0.0,
        avg_att, gpa_trend
    ]
    return X

def build_features_for_next_label(student, GP):
    """Features/label for next-semester CGPA (train-time windows)."""
    semesters = student.get("semesters", {})
    if not semesters: return None, None
    sem_nums = sorted(map(int, semesters.keys()))
    if len(sem_nums) < 2: return None, None
    y = compute_cgpa(semesters, sem_nums, len(sem_nums), GP)
    if y is None: return None, None

    upto = sem_nums[-2]
    att = []
    for s in sem_nums:
        if s <= upto:
            sem = semesters[str(s)]
            if not isinstance(sem, dict):
                continue
            if "attendancePercentage" in sem: att.append(sem["attendancePercentage"])
    avg_att = float(np.mean(att)) if att else 0.0
    s_count = len(sem_nums) - 1
    cgpa_1 = compute_cgpa(semesters, sem_nums, 1, GP)
    cgpa_s = compute_cgpa(semesters, sem_nums, s_count, GP)
    if cgpa_1 is None or cgpa_s is None:
        return None, None
    gpa_trend = 0.0 if s_count == 1 else float((cgpa_s - cgpa_1) / (s_count - 1))
    avg_ch = average_credit_hours(student)

    X = [
        float(student.get("ssc_gpa", 0.0)),
        float(student.get("hsc_gpa", 0.0)),
        1 if str(student.get("gender","")).lower()=="female" else 0,
        int(student.get("birth_year", 0)),
        float(avg_ch) if avg_ch is not None else 0.0,
        avg_att, gpa_trend
    ]
    return X, float(y)

def average_credit_hours(student):
    """Average credit hours across valid semesters."""
    semesters = student.get("semesters", {})
    if not semesters:
        return None
    loads = []
    for sem in semesters.values():
        if not isinstance(sem, dict):
            continue
        ch = sem.get("creditHours", None)
        if ch is not None and ch > 0:
            loads.append(ch)
    if not loads:
        return None
    return float(np.mean(loads))

def iter_student_windows(student):
    """Yield rolling window sizes for training (1..n-1)."""
    semesters = student.get("semesters", {})
    if not isinstance(semesters, dict):
        return []
    sem_nums = sorted(int(k) for k in semesters.keys() if str(k).isdigit())
    if len(sem_nums) < 2:
        return []
    return list(range(1, len(sem_nums)))

def build_features_upto_s(student, GP, s, gpa_trend):
    """Build feature vector for a student using semesters up to s."""
    semesters = student.get("semesters", {})
    if not isinstance(semesters, dict):
        return None
    sem_nums = sorted(int(k) for k in semesters.keys() if str(k).isdigit())
    if s < 1 or s > len(sem_nums):
        return None
    use = sem_nums[:s]
    att, credit_hours = [], []
    for sem_no in use:
        sem = semesters.get(str(sem_no))
        if not isinstance(sem, dict):
            continue
        if "attendancePercentage" in sem:
            att.append(sem["attendancePercentage"])
        ch = sem.get("creditHours")
        if isinstance(ch, (int, float)) and ch > 0:
            credit_hours.append(ch)
    avg_att = float(np.mean(att)) if att else 0.0
    avg_ch = float(np.mean(credit_hours)) if credit_hours else 0.0
    return [
        float(student.get("ssc_gpa", 0.0)),
        float(student.get("hsc_gpa", 0.0)),
        1 if str(student.get("gender", "")).lower() == "female" else 0,
        int(student.get("birth_year", 0)),
        avg_ch,
        avg_att,
        float(gpa_trend)
    ]

def predict_single_student(
    student: dict,
    models_next: dict,
    best_model_name: str,
    grade_points: dict,
    thresholds: dict,
    feat_names: list,
    explicit_s: Optional[int] = None
):
    """Reference inference flow for debugging (mirrors training feature logic)."""
    semesters = student.get("semesters", {})
    if not isinstance(semesters, dict):
        return None
    sem_nums = sorted(int(k) for k in semesters.keys() if str(k).isdigit())
    if len(sem_nums) < 2:
        return None
    n = len(sem_nums)
    if explicit_s is None:
        s = n - 1
    else:
        s = max(1, min(int(explicit_s), n - 1))

    cgpa_1 = compute_cgpa(semesters, sem_nums, 1, grade_points)
    cgpa_s = compute_cgpa(semesters, sem_nums, s, grade_points)
    if cgpa_1 is None or cgpa_s is None:
        return None
    gpa_trend = 0.0 if s == 1 else float((cgpa_s - cgpa_1) / (s - 1))

    use = sem_nums[:s]
    att, credit_hours = [], []
    for sem_no in use:
        sem = semesters.get(str(sem_no))
        if not isinstance(sem, dict):
            continue
        if "attendancePercentage" in sem:
            att.append(sem["attendancePercentage"])
        ch = sem.get("creditHours")
        if isinstance(ch, (int, float)) and ch > 0:
            credit_hours.append(ch)
    avg_att = float(np.mean(att)) if att else 0.0
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
    X = [feature_map.get(name, 0.0) for name in feat_names]
    X_arr = np.array([X], float)
    assert X_arr.shape[1] == len(feat_names)

    model = models_next.get(best_model_name)
    if model is None:
        return None
    pred_value = float(model.predict(X_arr)[0])

    high_max = thresholds.get("high_max") if isinstance(thresholds, dict) else None
    med_max = thresholds.get("med_max") if isinstance(thresholds, dict) else None
    if isinstance(high_max, (int, float)) and pred_value <= high_max:
        risk = "High"
    elif isinstance(med_max, (int, float)) and pred_value <= med_max:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "student_id": student.get("student_id"),
        "s_used": s,
        "pred_next_sem_cgpa": pred_value,
        "predicted_risk": risk
    }

def emit_progress(**data):
    """Emit progress events consumed by the Node SSE logger."""
    print(
        "__PROGRESS__" +
        json.dumps(data, separators=(",", ":")),
        flush=True
    )
def emit_result(payload: dict):
    """
    Emits the final result as a single-line JSON payload.
    REQUIRED by Node state machine.
    """
    print(
        "__RESULT__" +
        json.dumps(payload, separators=(",", ":")),
        flush=True
    )
    
# ------------------------- Main -------------------------
def main():
    # Pipeline: load config/data -> clean -> build features -> train -> evaluate -> save artifacts.
    import warnings; warnings.filterwarnings("ignore")

    ap = argparse.ArgumentParser()
    ap.add_argument("--org-id", required=True)
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--config-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()


    org_id = args.org_id
    out_dir = Path(args.out_dir); plots_dir = out_dir/"plots"
    out_dir.mkdir(parents=True, exist_ok=True); plots_dir.mkdir(parents=True, exist_ok=True)

    # Load config & clamp bounds
    cfg = json.load(open(args.config_json))
    def clamp(name, default):
        lo, hi = BOUNDS.get(name, (None,None))
        v = cfg.get(name, default)
        try: v = float(v)
        except Exception: v = default
        if lo is not None: v = max(lo, v)
        if hi is not None: v = min(hi, v)
        return v
    def clamp_int(name, default):
        return int(round(clamp(name, default)))
    def parse_depth(name, default=0):
        try:
            v = int(cfg.get(name, default))
        except Exception:
            v = default
        if v <= 0:
            return None
        return int(clamp(name, v))
    def parse_bool(name, default=True):
        v = cfg.get(name, default)
        if isinstance(v, str):
            return v.lower() == "true"
        return bool(v)

    RANDOM_SEED = int(cfg.get("RANDOM_SEED", 42))
    THREADS     = int(clamp("THREADS", 4))
    TEST_SIZE   = float(clamp("TEST_SIZE", 0.2))
    DT_ENABLE   = parse_bool("DT_ENABLE", True)
    DT_MAX_DEPTH = parse_depth("DT_MAX_DEPTH", 0)
    DT_MIN_SAMPLES_LEAF = clamp_int("DT_MIN_SAMPLES_LEAF", 1)
    RF_ENABLE   = parse_bool("RF_ENABLE", True)
    RF_TREES    = int(clamp("RF_TREES", 400))
    RF_MAX_DEPTH = parse_depth("RF_MAX_DEPTH", 0)
    RF_MIN_SAMPLES_LEAF = clamp_int("RF_MIN_SAMPLES_LEAF", 1)
    LGBM_ENABLE = parse_bool("LGBM_ENABLE", True)
    LGBM_N_EST  = int(clamp("LGBM_N_ESTIMATORS", 2000))
    LGBM_REG_ALPHA = float(clamp("LGBM_REG_ALPHA", 0.0))
    LGBM_REG_LAMBDA = float(clamp("LGBM_REG_LAMBDA", 0.0))
    MLP_ENABLE  = parse_bool("MLP_ENABLE", True)
    MLP_HIDDEN  = int(clamp("MLP_HIDDEN", 64))
    MLP_EPOCHS  = int(clamp("MLP_EPOCHS", 300))
    MLP_PATIENCE= int(clamp("MLP_PATIENCE", 40))
    SVR_ENABLE  = parse_bool("SVR_ENABLE", True)
    SVR_C       = float(clamp("SVR_C", 10.0))
    SVR_EPSILON = float(clamp("SVR_EPSILON", 0.1))
    RISK_HIGH_MAX = float(cfg.get("RISK_HIGH_MAX", 3.30))
    RISK_MED_MAX  = float(cfg.get("RISK_MED_MAX", 3.50))
    GRADE_POINTS = cfg.get("GRADE_POINTS", {
        "A+":4.0,"A":3.75,"A-":3.5,"B+":3.25,"B":3.0,"B-":2.75,"C+":2.5,"C":2.25,"D":2.0,"F":0.0
    })
    max_gpa = validate_grade_points(GRADE_POINTS)

    # Optional: set torch threads
    try:
        import torch; torch.set_num_threads(THREADS)
    except Exception:
        pass

    # Load raw data and normalize shape
    train_path = args.train_json
    assert os.path.exists(train_path), f"Training file not found: {train_path}"
    payload = json.load(open(train_path))
    if isinstance(payload, dict) and "students" in payload:
        data = payload.get("students") or []
    else:
        data = payload
    if not isinstance(data, list):
        raise ValueError("Training data must be a list of students or an object with a students list.")
    stats = {
        "invalid_students": 0,
        "invalid_semesters": 0,
        "duplicate_students": 0,
        "duplicate_semesters": 0,
        "duplicate_courses": 0,
        "missing_demographics": 0,
        "missing_attendance": 0,
        "missing_credit_hours": 0,
        "invalid_credit_hours": 0,
        "attendance_clamped": 0,
        "grade_clamped": 0,
        "invalid_grades": 0,
        "grade_clamped_by_field": {
            "ssc_gpa": 0,
            "hsc_gpa": 0,
            "course_numeric": 0
        }
    }
    cleaned_students = {}
    for idx, student in enumerate(data):
        if not isinstance(student, dict):
            stats["invalid_students"] += 1
            continue
        sid = student.get("student_id")
        if sid is None or sid == "":
            sid = f"stu_{idx}"
        student["student_id"] = sid
        cleaned = sanitize_student(student, GRADE_POINTS, max_gpa, stats)
        if cleaned is None:
            continue
        if sid in cleaned_students:
            stats["duplicate_students"] += 1
            cleaned_students[sid] = merge_students(cleaned_students[sid], cleaned, stats)
        else:
            cleaned_students[sid] = cleaned
    data = list(cleaned_students.values())
    print(f"[RUN_START] runId={args.model_id} org={org_id} time={datetime.datetime.utcnow().isoformat()}Z")
    print(f"[INFO] org={org_id} students={len(data)} max_gpa={max_gpa}")
    print(
        "[INFO] preprocessing " +
        "invalid_students={invalid_students} invalid_semesters={invalid_semesters} "
        "duplicate_students={duplicate_students} duplicate_semesters={duplicate_semesters} "
        "duplicate_courses={duplicate_courses} missing_demographics={missing_demographics} "
        "missing_attendance={missing_attendance} missing_credit_hours={missing_credit_hours} "
        "invalid_credit_hours={invalid_credit_hours} attendance_clamped={attendance_clamped} "
        "grade_clamped={grade_clamped} invalid_grades={invalid_grades}".format(**stats)
    )
    print(
        "[INFO] grade_clamped_by_field " +
        "ssc_gpa={ssc_gpa} hsc_gpa={hsc_gpa} course_numeric={course_numeric}".format(
            **stats["grade_clamped_by_field"]
        )
    )

    # Build feature/label datasets (final CGPA and next-sem CGPA)
    X_final, y_final = [], []
    X_next, y_next = [], []
    sid_final = []
    sid_next = []
    avg_course_loads = []
    semester_gpa_sums = {}
    semester_gpa_counts = {}
    total_gpa_sum = 0.0
    total_gpa_count = 0

    for student in data:
        if not isinstance(student, dict):
            continue
        semesters = student.get("semesters", {})
        if not isinstance(semesters, dict):
            continue
        sem_nums = sorted(int(k) for k in semesters.keys() if str(k).isdigit())
        if len(sem_nums) < 2:
            continue
        cgpa_final = compute_cgpa(semesters, sem_nums, len(sem_nums), GRADE_POINTS)
        if cgpa_final is None:
            continue
        for sem in semesters.values():
            if not isinstance(sem, dict):
                continue
            ch = sem.get("creditHours", None)
            if ch is None or ch <= 0:
                continue
            sem_gpa = compute_semester_gpa(sem, GRADE_POINTS)
            if sem_gpa is None:
                continue
            bucket = int(round(ch))  # Use rounded credit hours as bucket
            semester_gpa_sums[bucket] = semester_gpa_sums.get(bucket, 0.0) + sem_gpa
            semester_gpa_counts[bucket] = semester_gpa_counts.get(bucket, 0) + 1
            total_gpa_sum += sem_gpa
            total_gpa_count += 1

        # Build rolling windows: for each prefix semester count, predict next-sem CGPA.

        cgpa_1 = compute_cgpa(semesters, sem_nums, 1, GRADE_POINTS)
        if cgpa_1 is None:
            continue
        for s in iter_student_windows(student):
            cgpa_s = compute_cgpa(semesters, sem_nums, s, GRADE_POINTS)
            if cgpa_s is None:
                continue
            cgpa_next = compute_cgpa(semesters, sem_nums, s + 1, GRADE_POINTS)
            if cgpa_next is None:
                continue
            gpa_trend = 0.0 if s == 1 else float((cgpa_s - cgpa_1) / (s - 1))
            X = build_features_upto_s(student, GRADE_POINTS, s, gpa_trend)
            if X is None:
                continue
            X_final.append(X)
            y_final.append(float(cgpa_final))
            sid_final.append(student["student_id"])
            X_next.append(X)
            y_next.append(float(cgpa_next))
            sid_next.append(student["student_id"])

        avg_ch = average_credit_hours(student)
        if avg_ch is not None:
            avg_course_loads.append(avg_ch)

    # Aggregate cohort stats used in metadata/reporting.
    semester_gpa_by_load = {
        str(k): float(semester_gpa_sums[k]/semester_gpa_counts[k])
        for k in semester_gpa_sums
    }


    feat_names = ["ssc_gpa","hsc_gpa","gender_bin","birth_year","avg_credit_hours","avg_attendance","gpa_trend"]
    feature_count = len(feat_names)
    X_final = np.array(X_final, float) if len(X_final) else np.empty((0, feature_count))
    y_final = np.array(y_final, float) if len(y_final) else np.empty((0,))
    X_next  = np.array(X_next,  float) if len(X_next)  else np.empty((0, feature_count))
    y_next  = np.array(y_next,  float) if len(y_next)  else np.empty((0,))

    emit_progress(
        phase="data_ready",
        samplesFinal=len(X_final),
        samplesNext=len(X_next),
        uniqueStudentsFinal=len(set(sid_final)),
        uniqueStudentsNext=len(set(sid_next))
    )

    overall_semester_gpa = float(total_gpa_sum / total_gpa_count) if total_gpa_count else None

    final_cgpa_sums = {}
    final_cgpa_counts = {}

    for student, final_gpa in zip(data, y_final):
        avg_ch = average_credit_hours(student)
        if avg_ch is None:
            continue
        bucket = int(round(avg_ch))  # Round average credit hours for bucket
        final_cgpa_sums[bucket] = final_cgpa_sums.get(bucket, 0.0) + final_gpa
        final_cgpa_counts[bucket] = final_cgpa_counts.get(bucket, 0) + 1

    final_cgpa_by_load = {
        str(k): float(final_cgpa_sums[k]/final_cgpa_counts[k])
        for k in final_cgpa_sums
    }

    overall_final_cgpa = None
    if len(y_final):
        overall_final_cgpa = float(np.mean(y_final))

    baseline_course_load = float(np.mean(avg_course_loads)) if avg_course_loads else None

    # Models
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    from sklearn.metrics import (
        r2_score,
        mean_squared_error,
        accuracy_score,
        mean_absolute_error,
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report
    )

    emit_progress(
        phase="init",
        modelId=args.model_id,
        orgId=org_id
    )
    print(
        "[INFO] enabled_models " +
        f"DecisionTree={DT_ENABLE} RandomForest={RF_ENABLE} SVR={SVR_ENABLE} "
        f"LightGBM={LGBM_ENABLE} MLP={MLP_ENABLE}"
    )

    # Split for evaluation
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.model_selection import train_test_split as _tts

    # MLP
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

    def train_mlp(Xtr, ytr, Xval, yval, epochs=300, lr=1e-3, patience=40, hid=64):
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr); Xval_s = scaler.transform(Xval)
        xt = torch.tensor(Xtr_s, dtype=torch.float32)
        yt = torch.tensor(ytr.reshape(-1,1), dtype=torch.float32)
        xv = torch.tensor(Xval_s, dtype=torch.float32)
        yv = torch.tensor(yval.reshape(-1,1), dtype=torch.float32)
        model = MLP(in_dim=Xtr.shape[1], hid=hid)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        best = math.inf; best_state = None; patience_ctr = 0
        history = {"train": [], "valid": []}
        checkpoint_path = out_dir/"MLP_state.pt"
        start_epoch = 0
        resumed = False

        if args.resume and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            scaler = checkpoint["scaler"]
            model.load_state_dict(checkpoint["model"])
            opt.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"] + 1
            resumed = True

            emit_progress(
                phase="resume",
                model="MLP",
                startEpoch=start_epoch
            )

        emit_progress(
            phase="training_start",
            model="MLP",
            samples=len(Xtr)
        )
        for ep in range(start_epoch, epochs):
            def save_checkpoint(epoch):
                torch.save(
                    {
                        "epoch": epoch,
                        "scaler": scaler,
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict()
                    },
                    checkpoint_path
                )
            
            model.train(); opt.zero_grad()
            pred = model(xt); loss = loss_fn(pred, yt); loss.backward(); opt.step()
            model.eval()
            save_checkpoint(ep)
            with torch.no_grad():
                vloss = loss_fn(model(xv), yv).item()
            emit_progress(
                phase="training",
                model="MLP",
                epoch=ep + 1,
                totalEpochs=epochs,
                valLoss=float(vloss)
            )
            history["train"].append(float(loss.item()))
            history["valid"].append(float(vloss))
            if vloss < best - 1e-6:
                best = vloss; best_state = {k:v.clone() for k,v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, scaler, history, resumed

    # Evaluate
    def eval_model(m, Xtr, ytr, Xte, yte, name):
        yhat_tr = m.predict(Xtr); yhat_te = m.predict(Xte)
        if isinstance(yhat_tr, (list, tuple)): yhat_tr = np.array(yhat_tr)
        if isinstance(yhat_te, (list, tuple)): yhat_te = np.array(yhat_te)
        r2_tr = r2_score(ytr, yhat_tr); r2_te = r2_score(yte, yhat_te)
        # Older sklearn may not support squared=False; take sqrt manually
        rmse_tr = math.sqrt(mean_squared_error(ytr, yhat_tr))
        rmse_te = math.sqrt(mean_squared_error(yte, yhat_te))
        mae_tr = mean_absolute_error(ytr, yhat_tr)
        mae_te = mean_absolute_error(yte, yhat_te)
        return {
            "name": name,
            "r2_tr": r2_tr,
            "r2_te": r2_te,
            "rmse_tr": rmse_tr,
            "rmse_te": rmse_te,
            "mae_tr": mae_tr,
            "mae_te": mae_te,
            "yhat_te": yhat_te
        }

    def eval_mlp(model, scaler, Xtr, ytr, Xte, yte):
        Xtr_s = scaler.transform(Xtr); Xte_s = scaler.transform(Xte)
        import torch
        with torch.no_grad():
            yhat_tr = model(torch.tensor(Xtr_s, dtype=torch.float32)).numpy().reshape(-1)
            yhat_te = model(torch.tensor(Xte_s, dtype=torch.float32)).numpy().reshape(-1)
        r2_tr = r2_score(ytr, yhat_tr); r2_te = r2_score(yte, yhat_te)
        rmse_tr = math.sqrt(mean_squared_error(ytr, yhat_tr))
        rmse_te = math.sqrt(mean_squared_error(yte, yhat_te))
        mae_tr = mean_absolute_error(ytr, yhat_tr)
        mae_te = mean_absolute_error(yte, yhat_te)
        return {
            "name": "MLP",
            "r2_tr": r2_tr,
            "r2_te": r2_te,
            "rmse_tr": rmse_tr,
            "rmse_te": rmse_te,
            "mae_tr": mae_tr,
            "mae_te": mae_te,
            "yhat_te": yhat_te
        }

    class MLPWrapper:
        def __init__(self, model, scaler):
            self.model = model
            self.scaler = scaler

        def predict(self, X):
            Xs = self.scaler.transform(X)
            with torch.no_grad():
                return self.model(torch.tensor(Xs, dtype=torch.float32)).numpy().reshape(-1)

    def sample_predictions(y_true, y_pred, limit=200):
        if len(y_true) == 0:
            return []
        total = len(y_true)
        size = min(limit, total)
        idx = np.linspace(0, total - 1, num=size, dtype=int)
        return [{"actual": float(y_true[i]), "predicted": float(y_pred[i])} for i in idx]

    def sample_residuals(y_true, y_pred, limit=200):
        if len(y_true) == 0:
            return []
        total = len(y_true)
        size = min(limit, total)
        idx = np.linspace(0, total - 1, num=size, dtype=int)
        return [
            {
                "actual": float(y_true[i]),
                "predicted": float(y_pred[i]),
                "residual": float(y_true[i] - y_pred[i])
            }
            for i in idx
        ]

    def make_histogram(values, bins=20):
        if values is None or len(values) == 0:
            return []
        counts, edges = np.histogram(values, bins=bins)
        return [
            {
                "binStart": float(edges[i]),
                "binEnd": float(edges[i + 1]),
                "count": int(counts[i])
            }
            for i in range(len(counts))
        ]
    
    def mlp_permutation_importance(model, scaler, X, y, n_repeats=5):
        """
        Manual permutation importance for PyTorch MLP
        """
        Xs = scaler.transform(X)

        import torch
        with torch.no_grad():
            baseline_preds = model(torch.tensor(Xs, dtype=torch.float32)).numpy().ravel()
        baseline_rmse = math.sqrt(mean_squared_error(y, baseline_preds))

        rng = np.random.RandomState(42)
        importances = []

        for j in range(Xs.shape[1]):
            rmses = []
            for _ in range(n_repeats):
                Xp = Xs.copy()
                rng.shuffle(Xp[:, j])

                with torch.no_grad():
                    preds = model(torch.tensor(Xp, dtype=torch.float32)).numpy().ravel()

                rmse = math.sqrt(mean_squared_error(y, preds))
                rmses.append(rmse)

            importances.append(float(np.mean(rmses) - baseline_rmse))

        return [
            {"feature": feat_names[i], "importance": importances[i]}
            for i in range(len(feat_names))
        ] 

    def compute_feature_importance_for_model(name, model, X, y):
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            else:
                from sklearn.inspection import permutation_importance
                if len(X) > 300:
                    rng = np.random.RandomState(RANDOM_SEED)
                    idx = rng.choice(len(X), size=300, replace=False)
                    X = X[idx]
                    y = y[idx]
                result = permutation_importance(
                    model,
                    X,
                    y,
                    n_repeats=5,
                    random_state=RANDOM_SEED,
                    scoring="neg_root_mean_squared_error"
                )
                importances = result.importances_mean
            items = []
            for idx, feature in enumerate(feat_names):
                value = float(importances[idx]) if idx < len(importances) else 0.0
                items.append({"feature": feature, "importance": value})
            items.sort(key=lambda item: abs(item["importance"]), reverse=True)
            return items
        except Exception as e:
            print(f"[WARN] feature importance failed for {name}: {e}")
            return []

    def compute_feature_importance_map(models, mlp_model, mlp_scaler, Xte, yte):
        importance = {}
        for name, model in models.items():
            importance[name] = compute_feature_importance_for_model(name, model, Xte, yte)
        if mlp_model is not None and mlp_scaler is not None:
            importance["MLP"] = mlp_permutation_importance(
                mlp_model,
                mlp_scaler,
                Xte,
                yte
            )
        return importance

    def build_dataset_metrics(suite):
        if suite is None:
            return None
        yte = suite["test"]["y"]
        metrics = {}
        predictions = {}
        for result in suite["results"]:
            metrics[result["name"]] = {
                "rmse": float(result["rmse_te"]),
                "r2": float(result["r2_te"]),
                "mae": float(result["mae_te"])
            }
            predictions[result["name"]] = sample_predictions(yte, result["yhat_te"])
        return {
            "models": metrics,
            "predictions": predictions,
            "featureImportance": suite["feature_importance"],
            "learningCurves": suite["learning_curves"],
            "testSize": int(len(yte))
        }

    def build_regression_report(suite):
        if suite is None:
            return None
        yte = suite["test"]["y"]
        results = suite["results"]
        if not results:
            return None
        best = min(results, key=lambda r: (r["rmse_te"], r["rmse_tr"]))
        yhat_map = {r["name"]: r["yhat_te"] for r in results}
        residuals = []
        if best["name"] in yhat_map:
            residuals = sample_residuals(yte, yhat_map[best["name"]])
        return {
            "bestModel": str(best["name"]),
            "metrics": build_dataset_metrics(suite),
            "residualSamples": residuals,
            "split": suite.get("split")
        }

    def train_suite(X, y, groups, label):
        if len(X) < 2 or len(y) < 2:
            raise ValueError(f"Not enough samples to train {label} models.")
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        train_idx, test_idx = next(gss.split(X, y, groups))
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        groups_tr = groups[train_idx]
        groups_te = groups[test_idx]
        assert set(groups_tr).isdisjoint(set(groups_te))

        models = {}
        learning_curves = {}

        if DT_ENABLE:
            emit_progress(
                phase="model_start",
                model="DecisionTree",
                label=label
            )
            dt = DecisionTreeRegressor(
                random_state=RANDOM_SEED,
                max_depth=DT_MAX_DEPTH,
                min_samples_leaf=DT_MIN_SAMPLES_LEAF
            )
            dt.fit(X_tr, y_tr)
            emit_progress(
                phase="model_trained",
                model="DecisionTree",
                label=label
            )
            models["DecisionTree"] = dt

        if RF_ENABLE:
            emit_progress(
                phase="model_start",
                model="RandomForest",
                label=label
            )
            rf = RandomForestRegressor(
                n_estimators=RF_TREES,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                max_depth=RF_MAX_DEPTH,
                min_samples_leaf=RF_MIN_SAMPLES_LEAF
            )
            rf.fit(X_tr, y_tr)
            emit_progress(
                phase="model_trained",
                model="RandomForest",
                label=label
            )            
            models["RandomForest"] = rf

        if SVR_ENABLE:
            emit_progress(
                phase="model_start",
                model="SVR",
                label=label
            )
            svr = SVR(kernel="rbf", C=SVR_C, epsilon=SVR_EPSILON, gamma="scale")
            svr.fit(X_tr, y_tr)
            emit_progress(
                phase="model_trained",
                model="SVR",
                label=label
            )
            models["SVR"] = svr

        if LGBM_ENABLE:
            emit_progress(
                phase="model_start",
                model="LightGBM",
                label=label
            )
            lgbm = lgb.LGBMRegressor(
                n_estimators=LGBM_N_EST,
                learning_rate=0.03,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=LGBM_REG_ALPHA,
                reg_lambda=LGBM_REG_LAMBDA,
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
            evals_result = {}
            Xtr_lgb, Xval_lgb, ytr_lgb, yval_lgb = _tts(X_tr, y_tr, test_size=0.2, random_state=RANDOM_SEED)
            lgbm.fit(
                Xtr_lgb, ytr_lgb,
                eval_set=[(Xtr_lgb, ytr_lgb), (Xval_lgb, yval_lgb)],
                eval_names=["train", "valid"],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False),
                           lgb.record_evaluation(evals_result)]
            )
            emit_progress(
                phase="model_trained",
                model="LightGBM",
                label=label
            )
            train_curve = evals_result.get("train", {}).get("rmse", [])
            valid_curve = evals_result.get("valid", {}).get("rmse", [])
            if train_curve or valid_curve:
                learning_curves["LightGBM"] = {
                    "train": [float(v) for v in train_curve],
                    "valid": [float(v) for v in valid_curve]
                }
            models["LightGBM"] = lgbm

        mlp_model = None
        mlp_scaler = None
        if MLP_ENABLE:
            emit_progress(
                phase="model_start",
                model="MLP",
                label=label
            )
            Xtr_mlp, Xval_mlp, ytr_mlp, yval_mlp = _tts(X_tr, y_tr, test_size=0.2, random_state=RANDOM_SEED)
            mlp_model, mlp_scaler, mlp_history, mlp_resumed = train_mlp(
                Xtr_mlp,
                ytr_mlp,
                Xval_mlp,
                yval_mlp,
                epochs=MLP_EPOCHS,
                patience=MLP_PATIENCE,
                hid=MLP_HIDDEN
            )
            emit_progress(
                phase="model_trained",
                model="MLP",
                label=label
            )
            learning_curves["MLP"] = mlp_history

        results = []
        for name, model in models.items():
            results.append(eval_model(model, X_tr, y_tr, X_te, y_te, name))
        if mlp_model is not None and mlp_scaler is not None:
            results.append(eval_mlp(mlp_model, mlp_scaler, X_tr, y_tr, X_te, y_te))

        feature_importance = compute_feature_importance_map(models, mlp_model, mlp_scaler, X_te, y_te)

        return {
            "models": models,
            "mlp_model": mlp_model,
            "mlp_scaler": mlp_scaler,
            "results": results,
            "learning_curves": learning_curves,
            "feature_importance": feature_importance,
            "test": {"X": X_te, "y": y_te},
            "split": {"train_size": int(len(y_tr)), "test_size": int(len(y_te))}
        }

    final_suite = train_suite(X_final, y_final, np.array(sid_final), "final_cgpa")
    next_suite = train_suite(X_next, y_next, np.array(sid_next), "next_sem_cgpa") if len(y_next) > 1 else None
    if not final_suite["results"]:
        raise ValueError("No models are enabled for training.")

    # Rank
    final_results = final_suite["results"]
    rank_df = pd.DataFrame(final_results).sort_values(by=["rmse_te","rmse_tr"], ascending=[True,True]).reset_index(drop=True)
    best_name = rank_df.iloc[0]["name"]
    emit_progress(
        phase="completed",
        bestModel=best_name
    )
    
    # --------- SAVE ARTIFACTS FIRST (so plotting errors won't break saving) ---------
    import joblib, torch
    final_models = final_suite["models"]
    if "DecisionTree" in final_models:
        joblib.dump(final_models["DecisionTree"], out_dir/"DecisionTree.joblib")
    if "RandomForest" in final_models:
        joblib.dump(final_models["RandomForest"], out_dir/"RandomForest.joblib")
    if "SVR" in final_models:
        joblib.dump(final_models["SVR"], out_dir/"SVR.joblib")
    if "LightGBM" in final_models:
        joblib.dump(final_models["LightGBM"], out_dir/"LightGBM.joblib")
    if final_suite["mlp_model"] is not None and final_suite["mlp_scaler"] is not None:
        torch.save(final_suite["mlp_model"].state_dict(), out_dir/"MLP.pt")
        joblib.dump(final_suite["mlp_scaler"], out_dir/"MLP_Scaler.joblib")

    if next_suite is not None:
        next_models = next_suite["models"]
        if "DecisionTree" in next_models:
            joblib.dump(next_models["DecisionTree"], out_dir/"DecisionTreeNext.joblib")
        if "RandomForest" in next_models:
            joblib.dump(next_models["RandomForest"], out_dir/"RandomForestNext.joblib")
        if "SVR" in next_models:
            joblib.dump(next_models["SVR"], out_dir/"SVRNext.joblib")
        if "LightGBM" in next_models:
            joblib.dump(next_models["LightGBM"], out_dir/"LightGBMNext.joblib")
        if next_suite["mlp_model"] is not None and next_suite["mlp_scaler"] is not None:
            torch.save(next_suite["mlp_model"].state_dict(), out_dir/"MLPNext.pt")
            joblib.dump(next_suite["mlp_scaler"], out_dir/"MLPNext_Scaler.joblib")

    # Risk classifier
    # Label by training-fold quantiles on next-semester CGPA
    def label_risk_by_thresholds(cgpa, high_max, med_max):
        if cgpa <= high_max:
            return "High"
        if cgpa <= med_max:
            return "Medium"
        return "Low"
    risk_values = y_next
    ypred_risk = None
    yc_te = None
    from sklearn.ensemble import RandomForestClassifier
    if len(risk_values) >= 2:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        groups_all = np.array(sid_next)
        train_idx, test_idx = next(gss.split(X_next, risk_values, groups_all))
        Xc_tr, Xc_te = X_next[train_idx], X_next[test_idx]
        yv_tr, yv_te = risk_values[train_idx], risk_values[test_idx]
        groups_tr = groups_all[train_idx]
        groups_te = groups_all[test_idx]
        assert set(groups_tr).isdisjoint(set(groups_te))
        q_high = float(np.quantile(yv_tr, 0.30))
        q_med = float(np.quantile(yv_tr, 0.70))
        risk_thresholds = {"high_max": q_high, "med_max": q_med}
        yc_tr = np.array([label_risk_by_thresholds(v, q_high, q_med) for v in yv_tr])
        yc_te = np.array([label_risk_by_thresholds(v, q_high, q_med) for v in yv_te])
        unique_risk = np.unique(yc_tr)
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=RANDOM_SEED)
        Xc_tr_res, yc_tr_res = sm.fit_resample(Xc_tr, yc_tr)
        risk_clf = RandomForestClassifier(n_estimators=250, random_state=RANDOM_SEED, n_jobs=-1)
        risk_clf.fit(Xc_tr_res, yc_tr_res)
        ypred_risk = risk_clf.predict(Xc_te)
        risk_accuracy = accuracy_score(yc_te, ypred_risk)
    else:
        from sklearn.dummy import DummyClassifier
        # Fall back to constant classifier if we have only one risk label
        constant_label = "Low"
        risk_clf = DummyClassifier(strategy="constant", constant=constant_label)
        risk_clf.fit(X_next, np.array([constant_label] * len(risk_values)))
        risk_accuracy = 1.0
        risk_thresholds = {"high_max": None, "med_max": None}

    joblib.dump(risk_clf, out_dir/"RiskClassifier.joblib")
    if yc_te is None or ypred_risk is None:
        yv_all = risk_values
        if risk_thresholds["high_max"] is None or risk_thresholds["med_max"] is None:
            yc_te = np.array([constant_label] * len(yv_all))
        else:
            yc_te = np.array([
                label_risk_by_thresholds(v, risk_thresholds["high_max"], risk_thresholds["med_max"])
                for v in yv_all
            ])
        ypred_risk = risk_clf.predict(X_next)
    labels = [lbl for lbl in ["High", "Medium", "Low"] if lbl in set(yc_te)]
    if not labels:
        labels = sorted(set(yc_te))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        yc_te,
        ypred_risk,
        average="macro",
        zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        yc_te,
        ypred_risk,
        average="weighted",
        zero_division=0
    )
    cm = confusion_matrix(yc_te, ypred_risk, labels=labels).tolist()

    # Determine storage root so we can emit static URLs
    storage_root = None
    for idx, part in enumerate(out_dir.parts):
        if part == "storage":
            storage_root = Path(*out_dir.parts[:idx + 1])
            break

    def to_static_path(path_obj: Path) -> str:
        if storage_root is not None:
            try:
                rel = path_obj.relative_to(storage_root)
                return "/static/" + rel.as_posix()
            except Exception:
                pass
        return str(path_obj)

    dataset_stats = {
        "students_total": len(data),
        "unique_students_final": len(set(sid_final)),
        "unique_students_next": len(set(sid_next)),
        "rows_final": len(sid_final),
        "rows_next": len(sid_next),
        "avg_rows_per_student": (len(sid_next) / len(set(sid_next))) if len(set(sid_next)) else 0.0
    }
    report_created_at = datetime.datetime.utcnow().isoformat() + "Z"
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": report_created_at,
        "splitting": {
            "method": "GroupShuffleSplit",
            "test_size": float(TEST_SIZE)
        },
        "regression": {
            "next_sem_cgpa": build_regression_report(next_suite),
            "final_cgpa": build_regression_report(final_suite)
        },
        "dataset": {
            "stats": dataset_stats,
            "final_cgpa_hist": make_histogram(y_final),
            "next_sem_cgpa_hist": make_histogram(y_next)
        },
        "classification": {
            "risk_target": "next_sem_cgpa",
            "thresholds": risk_thresholds,
            "labels": labels,
            "accuracy": float(risk_accuracy),
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(prec_w),
            "recall_weighted": float(rec_w),
            "f1_weighted": float(f1_w),
            "confusion_matrix": cm
        }
    }
    with open(out_dir/"report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Metadata
    meta = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.datetime.utcnow().isoformat()+"Z",
        "feature_order": feat_names,
        "grade_points": GRADE_POINTS,
        "max_gpa": max_gpa,
        "best_model": str(best_name),
        "baseline_course_load": baseline_course_load,
        "baseline_credit_hours": float(baseline_course_load * 3.0) if baseline_course_load is not None else None,
        "models": {
            r["name"]: {
                "r2_tr": r["r2_tr"],
                "r2_te": r["r2_te"],
                "rmse_tr": r["rmse_tr"],
                "rmse_te": r["rmse_te"],
                "mae_tr": r["mae_tr"],
                "mae_te": r["mae_te"]
            }
            for r in final_results
        },
        "risk_accuracy": float(risk_accuracy),
        "enabled_models": [r["name"] for r in final_results],
        "next_models": [r["name"] for r in next_suite["results"]] if next_suite else [],
        "course_load_stats": {
            "semester_gpa_by_load": semester_gpa_by_load,
            "final_cgpa_by_load": final_cgpa_by_load,
            "overall_semester_gpa": overall_semester_gpa,
            "overall_final_cgpa": overall_final_cgpa
        },
        "dataset_stats": dataset_stats,
        "risk_thresholds": risk_thresholds,
        "report_path": to_static_path(out_dir/"report.json"),
        "mlp_hidden": int(MLP_HIDDEN)
    }
    with open(out_dir/"metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # --------- Plotting (best-effort, won't crash training) ---------
    saved_plots = {}
    if plt is not None:
        try:
            # Residuals for best model
            yhat_map = {r["name"]: r["yhat_te"] for r in final_results}
            yf_te = final_suite["test"]["y"]
            resid = yf_te - yhat_map[best_name]
            fig, ax = plt.subplots(1,2, figsize=(10,4))
            ax[0].scatter(yhat_map[best_name], resid, alpha=0.5); ax[0].axhline(0, color='k', lw=1)
            ax[0].set_title(f"{best_name}: Residuals vs Prediction"); ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Residual")
            if sns is not None:
                sns.histplot(resid, kde=True, ax=ax[1]); ax[1].set_title(f"{best_name}: Residual Distribution")
            else:
                ax[1].hist(resid, bins=30); ax[1].set_title(f"{best_name}: Residual Distribution")
            plt.tight_layout(); p = plots_dir/"best_model_residuals.png"; plt.savefig(p); plt.close(fig)
            saved_plots["best_model_residuals"] = to_static_path(p)

            # Feature importances for RF / LGBM
            rf = final_models.get("RandomForest")
            lgbm = final_models.get("LightGBM")
            if rf is not None:
                try:
                    fig = plt.figure(figsize=(6,4))
                    (sns.barplot(x=pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False).values,
                                 y=pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False).index)
                     if sns is not None else plt.barh(feat_names, rf.feature_importances_))
                    plt.title("RF Feature Importance"); plt.tight_layout()
                    p = plots_dir/"rf_importance.png"
                    plt.savefig(p); plt.close()
                    saved_plots["rf_feature_importance"] = to_static_path(p)
                except Exception as e:
                    print(f"[WARN] RF plot failed: {e}")

            if lgbm is not None:
                try:
                    fig = plt.figure(figsize=(6,4))
                    (sns.barplot(x=pd.Series(lgbm.feature_importances_, index=feat_names).sort_values(ascending=False).values,
                                 y=pd.Series(lgbm.feature_importances_, index=feat_names).sort_values(ascending=False).index)
                     if sns is not None else plt.barh(feat_names, lgbm.feature_importances_))
                    plt.title("LGBM Feature Importance"); plt.tight_layout()
                    p = plots_dir/"lgbm_importance.png"
                    plt.savefig(p); plt.close()
                    saved_plots["lgbm_feature_importance"] = to_static_path(p)
                except Exception as e:
                    print(f"[WARN] LGBM plot failed: {e}")

            # Risk confusion matrix
            try:
                from sklearn.metrics import ConfusionMatrixDisplay
                if yc_te is not None and ypred_risk is not None:
                    ConfusionMatrixDisplay.from_predictions(yc_te, ypred_risk)
                    plt.title("Risk Classifier Confusion Matrix"); plt.tight_layout()
                    p = plots_dir/"risk_confusion_matrix.png"
                    plt.savefig(p); plt.close()
                    saved_plots["risk_confusion_matrix"] = to_static_path(p)
            except Exception as e:
                print(f"[WARN] Risk CM plot failed: {e}")

            # LightGBM learning curve
            try:
                lgbm_curve = final_suite["learning_curves"].get("LightGBM", {})
                train_curve = lgbm_curve.get("train", [])
                valid_curve = lgbm_curve.get("valid", [])
                if train_curve or valid_curve:
                    fig = plt.figure(figsize=(6,4))
                    if train_curve:
                        plt.plot(train_curve, label="Train RMSE")
                    if valid_curve:
                        plt.plot(valid_curve, label="Validation RMSE")
                    plt.xlabel("Iteration"); plt.ylabel("RMSE"); plt.title("LightGBM Learning Curve")
                    plt.legend(); plt.tight_layout()
                    p = plots_dir/"lightgbm_learning_curve.png"
                    plt.savefig(p); plt.close()
                    saved_plots["lightgbm_learning_curve"] = to_static_path(p)
            except Exception as e:
                print(f"[WARN] Learning curve plot failed: {e}")

        except Exception as e:
            print(f"[WARN] plotting failed: {e}")

    # ------------------------- Final result -------------------------

    best_metrics = rank_df.iloc[0]
    metrics_summary = {
        "rmse": float(best_metrics["rmse_te"]),
        "r2": float(best_metrics["r2_te"]),
        "accuracy": float(risk_accuracy)
    }

    metrics_payload = {
        "summary": metrics_summary,
        "bestModel": str(best_name),
        "enabledModels": [r["name"] for r in final_results],
        "final": build_dataset_metrics(final_suite),
        "next": build_dataset_metrics(next_suite) if next_suite else None
    }

    result = {
        "status": "ok",
        "bestModel": meta["best_model"],
        "artifactsDir": str(out_dir),
        "plots": saved_plots,
        "gradePoints": GRADE_POINTS,
        "metrics": metrics_payload
    }
    resumed_training = bool(args.resume) and bool(MLP_ENABLE) and bool(locals().get("mlp_resumed", False))
    emit_result({
        "status": "ok",
        "modelId": args.model_id,
        "schemaVersion": SCHEMA_VERSION,
        "bestModel": meta["best_model"],
        "rmse": float(best_metrics["rmse_te"]),
        "r2": float(best_metrics["r2_te"]),
        "riskAccuracy": float(risk_accuracy),
        "enabledModels": [r["name"] for r in final_results],
        "artifactsDir": str(out_dir),
        "plots": saved_plots,
        "metrics": metrics_payload,
        "resumed": resumed_training
    })
    return 0

if __name__ == "__main__":
    try:
        import time; sys.exit(main())
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        print("__RESULT__" + json.dumps({"status":"error","error":str(e)}, separators=(",", ":")), flush=True)
        sys.exit(1)

if __name__ == "__main__" and False:
    # Example usage of predict_single_student
    # This block must NEVER run by default
    pass
