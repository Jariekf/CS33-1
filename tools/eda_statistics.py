"""
Deterministic EDA statistics: descriptive tables, correlations, collinearity,
quick RandomForest feature importance, and train/test distribution drift.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

# Re-use same heuristics as eda_charts
def infer_target_column(df: pd.DataFrame) -> str | None:
    if df.empty or df.shape[1] == 0:
        return None
    semantic_exact = {
        "target", "label", "labels", "outcome", "response",
        "dependent", "y", "class",
    }
    semantic_contains = ("target", "label", "outcome", "response", "dependent")
    lowered = {c: str(c).strip().lower().replace(" ", "_") for c in df.columns}
    exact_hits = [c for c, lc in lowered.items() if lc in semantic_exact]
    contains_hits = [c for c, lc in lowered.items() if any(tok in lc for tok in semantic_contains)]
    if exact_hits:
        return exact_hits[0]
    if contains_hits:
        return contains_hits[0]
    return str(df.columns[-1])


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def _top_corr_pairs(corr: pd.DataFrame, k: int = 20) -> list[dict[str, Any]]:
    cols = corr.columns.tolist()
    pairs: list[tuple[float, str, str]] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            v = corr.loc[a, b]
            if pd.notna(v):
                pairs.append((abs(float(v)), a, b, float(v)))
    pairs.sort(key=lambda x: -x[0])
    out = []
    for _, a, b, coeff in pairs[:k]:
        out.append({"col_a": a, "col_b": b, "coeff": _json_safe(coeff)})
    return out


def _collinear_pairs(corr: pd.DataFrame, threshold: float = 0.95) -> list[dict[str, Any]]:
    cols = corr.columns.tolist()
    out: list[dict[str, Any]] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            v = corr.loc[a, b]
            if pd.notna(v) and abs(float(v)) > threshold:
                out.append({"col_a": a, "col_b": b, "coeff": float(v)})
    return out


def _detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    found: list[str] = []
    for c in df.columns:
        lc = str(c).lower()
        if any(t in lc for t in ("date", "time", "timestamp", "year", "month", "day", "period")):
            found.append(str(c))
    return found


def _quick_rf_importance(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[list[dict[str, Any]], str]:
    """Returns (top15 list of {feature, importance}), error_message or ''."""
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder
    except ImportError as e:
        return [], f"sklearn unavailable: {e}"

    if target_col not in df.columns:
        return [], "target column not in dataframe"

    y_raw = df[target_col]
    X_df = df.drop(columns=[target_col])
    if X_df.shape[1] == 0:
        return [], "no feature columns after dropping target"

    X_num = X_df.select_dtypes(include=[np.number]).copy()
    for c in X_num.columns:
        X_num[c] = pd.to_numeric(X_num[c], errors="coerce")
        med = X_num[c].median()
        X_num[c] = X_num[c].fillna(med)

    X_cat = X_df.select_dtypes(include=["object", "category"]).copy()
    for c in list(X_cat.columns):
        if X_cat[c].nunique(dropna=True) > 30:
            X_cat = X_cat.drop(columns=[c], errors="ignore")

    if X_cat.shape[1] > 0:
        X_enc = pd.get_dummies(X_cat, dummy_na=True)
        X_all = pd.concat([X_num, X_enc], axis=1)
    else:
        X_all = X_num

    X_all = X_all.fillna(0)
    nuniq = X_all.nunique(dropna=False)
    X_all = X_all.loc[:, nuniq > 1]

    if X_all.shape[1] == 0:
        return [], "all features constant after encoding"

    nu = int(y_raw.nunique(dropna=True))
    is_obj = y_raw.dtype == object or str(y_raw.dtype) == "category"
    # High-cardinality numeric target → regression
    if pd.api.types.is_numeric_dtype(y_raw) and nu > 25:
        use_classifier = False
    else:
        is_int_low_card = pd.api.types.is_integer_dtype(y_raw) and nu <= 25
        is_float_low_card = pd.api.types.is_float_dtype(y_raw) and nu <= 20
        use_classifier = is_obj or is_int_low_card or is_float_low_card

    if use_classifier:
        le = LabelEncoder()
        y_enc = le.fit_transform(y_raw.astype(str).fillna("__na__"))
        if len(np.unique(y_enc)) < 2:
            return [], "target has a single class after encoding"
        clf = RandomForestClassifier(
            n_estimators=50, max_depth=6, random_state=42, n_jobs=-1,
        )
        clf.fit(X_all, y_enc)
        imp = clf.feature_importances_
    else:
        y_num = pd.to_numeric(y_raw, errors="coerce")
        mask = y_num.notna()
        if mask.sum() < 15:
            return [], "too few non-null numeric target rows"
        reg = RandomForestRegressor(
            n_estimators=50, max_depth=6, random_state=42, n_jobs=-1,
        )
        reg.fit(X_all.loc[mask], y_num.loc[mask])
        imp = reg.feature_importances_

    names = X_all.columns.tolist()
    pairs = sorted(zip(names, imp), key=lambda x: -x[1])[:15]
    top = [{"feature": str(a), "importance": float(b)} for a, b in pairs]
    return top, ""


def _train_test_drift(
    train: pd.DataFrame,
    test: pd.DataFrame,
    max_cols: int = 40,
) -> list[dict[str, Any]]:
    """KS for numeric, chi2 for categorical; skip low-sample columns."""
    out: list[dict[str, Any]] = []
    common = [c for c in train.columns if c in test.columns]
    if not common:
        return out

    from scipy.stats import chi2_contingency, ks_2samp

    count = 0
    for col in common:
        if count >= max_cols:
            break
        tr = train[col]
        te = test[col]
        if pd.api.types.is_numeric_dtype(tr) and pd.api.types.is_numeric_dtype(te):
            trv = pd.to_numeric(tr, errors="coerce").dropna()
            tev = pd.to_numeric(te, errors="coerce").dropna()
            if len(trv) < 10 or len(tev) < 10:
                continue
            try:
                stat, p = ks_2samp(trv, tev)
                out.append({
                    "col": col,
                    "test": "ks_2samp",
                    "statistic": float(stat),
                    "p_value": float(p),
                    "drift_warning": bool(p < 0.05),
                })
                count += 1
            except Exception as exc:
                logger.debug("KS skip %s: %s", col, exc)
        elif tr.dtype == object or str(tr.dtype) == "category":
            try:
                vc_tr = tr.astype(str).value_counts()
                vc_te = te.astype(str).value_counts()
                cats = sorted(set(vc_tr.index) | set(vc_te.index))
                if len(cats) > 50:
                    cats = cats[:50]
                row_tr = [vc_tr.get(c, 0) for c in cats]
                row_te = [vc_te.get(c, 0) for c in cats]
                table = np.array([row_tr, row_te])
                if table.sum() == 0:
                    continue
                chi2, p, _, _ = chi2_contingency(table)
                out.append({
                    "col": col,
                    "test": "chi2_contingency",
                    "statistic": float(chi2),
                    "p_value": float(p),
                    "drift_warning": bool(p < 0.05),
                })
                count += 1
            except Exception as exc:
                logger.debug("chi2 skip %s: %s", col, exc)
    return out


def compute_eda_statistics(
    train_path: str,
    test_path: str | None = None,
    target_col: str | None = None,
) -> dict[str, Any]:
    """
    Load train (and optional test) CSV; return JSON-serializable EDA statistics dict.
    """
    result: dict[str, Any] = {
        "error": None,
        "train_path": train_path,
        "test_path": test_path,
        "inferred_target": None,
        "target_type_hint": "unknown",
        "descriptive_stats": {},
        "skewness": {},
        "category_summary": {},
        "sample_sufficiency": {},
        "pearson_corr_top": [],
        "spearman_corr_top": [],
        "collinear_pairs": [],
        "target_correlation": [],
        "class_balance": {},
        "rf_importance_top15": [],
        "rf_error": "",
        "datetime_columns": [],
        "train_test_drift": [],
    }

    if not os.path.isfile(train_path):
        result["error"] = f"train file not found: {train_path}"
        return result

    try:
        df = pd.read_csv(train_path)
    except Exception as exc:
        result["error"] = f"read_csv failed: {exc}"
        return result

    if df.empty:
        result["error"] = "empty dataframe"
        return result

    inferred = infer_target_column(df) if target_col is None else target_col
    if inferred is not None and inferred not in df.columns:
        inferred = None
    if target_col is not None and target_col in df.columns:
        inferred = target_col

    result["inferred_target"] = inferred

    n_rows, n_feats = df.shape[0], df.shape[1]
    rpf = (n_rows / max(n_feats, 1)) if n_feats else 0.0
    result["sample_sufficiency"] = {
        "n_rows": int(n_rows),
        "n_features": int(n_feats),
        "rows_per_feature": round(float(rpf), 4),
        "warning_low_samples_per_feature": bool(rpf < 10 and n_feats > 5),
    }

    # --- descriptive ---
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    desc: dict[str, Any] = {}
    if num_cols:
        desc_num = df[num_cols].describe().to_dict()
        for col in num_cols:
            if col in desc_num:
                desc[col] = {k: _json_safe(v) for k, v in desc_num[col].items()}
    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in obj_cols[:30]:
        vc = df[col].astype(str).value_counts().head(5)
        desc[col] = {
            "dtype": "object",
            "unique": int(df[col].nunique(dropna=True)),
            "top_values": vc.to_dict(),
        }
    result["descriptive_stats"] = desc

    skew: dict[str, Any] = {}
    for c in num_cols:
        try:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s) < 3:
                continue
            sk = float(s.skew())
            direction = "symmetric"
            if sk > 0.5:
                direction = "right_skewed"
            elif sk < -0.5:
                direction = "left_skewed"
            skew[str(c)] = {"skewness": sk, "direction": direction}
        except Exception:
            continue
    result["skewness"] = skew

    cat_sum: dict[str, Any] = {}
    for col in obj_cols[:40]:
        nu = int(df[col].nunique(dropna=True))
        vc = df[col].value_counts(normalize=True, dropna=True)
        top3 = vc.head(3).to_dict()
        mx = float(vc.max()) if len(vc) else 0.0
        cat_sum[str(col)] = {
            "nunique": nu,
            "top3_share": {str(k): float(v) for k, v in top3.items()},
            "is_highly_imbalanced": bool(mx > 0.9 and nu > 1),
        }
    result["category_summary"] = cat_sum

    # --- correlations (numeric only) ---
    num_df = df.select_dtypes(include=[np.number]).copy()
    if inferred and inferred in num_df.columns:
        num_df_feat = num_df.drop(columns=[inferred])
    else:
        num_df_feat = num_df

    if num_df_feat.shape[1] >= 2:
        pear = num_df_feat.corr(method="pearson")
        result["pearson_corr_top"] = _top_corr_pairs(pear, 20)
        result["collinear_pairs"] = _collinear_pairs(pear, 0.95)
        spear = num_df_feat.corr(method="spearman")
        result["spearman_corr_top"] = _top_corr_pairs(spear, 20)
    else:
        result["pearson_corr_top"] = []
        result["spearman_corr_top"] = []
        result["collinear_pairs"] = []

    # --- target correlation ---
    tgt_corr: list[dict[str, Any]] = []
    if inferred is not None:
        y = df[inferred]
        y_num = pd.to_numeric(y, errors="coerce")
        if y_num.notna().sum() >= 10 and y_num.nunique() > 1:
            for c in num_cols:
                if c == inferred:
                    continue
                x = pd.to_numeric(df[c], errors="coerce")
                m = x.notna() & y_num.notna()
                if m.sum() < 10:
                    continue
                r = x[m].corr(y_num[m])
                if pd.notna(r):
                    tgt_corr.append({"feature": str(c), "pearson_with_target": float(r)})
            tgt_corr.sort(key=lambda d: -abs(d["pearson_with_target"]))
            tgt_corr = tgt_corr[:15]
    result["target_correlation"] = tgt_corr

    # --- class balance ---
    if inferred is not None:
        y = df[inferred]
        if not pd.api.types.is_numeric_dtype(y) or y.nunique(dropna=True) <= 25:
            vc = y.astype(str).value_counts(normalize=True, dropna=True)
            result["class_balance"] = {str(k): float(v) for k, v in vc.items()}
        else:
            result["class_balance"] = {}

    # target type hint
    if inferred is not None:
        y = df[inferred]
        nu = y.nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(y) and nu > 25:
            result["target_type_hint"] = "regression"
        elif nu <= 20:
            result["target_type_hint"] = "classification"
        else:
            result["target_type_hint"] = "regression"
    else:
        result["target_type_hint"] = "clustering_or_none"

    result["datetime_columns"] = _detect_datetime_columns(df)

    # --- RF importance ---
    if inferred:
        rf_top, rf_err = _quick_rf_importance(df, inferred)
        result["rf_importance_top15"] = rf_top
        result["rf_error"] = rf_err
    else:
        result["rf_importance_top15"] = []
        result["rf_error"] = "no target column; skipped RF"

    # --- train/test drift ---
    if test_path and os.path.isfile(test_path):
        try:
            dte = pd.read_csv(test_path)
            result["train_test_drift"] = _train_test_drift(df, dte)
        except Exception as exc:
            result["train_test_drift"] = []
            logger.warning("train_test drift failed: %s", exc)
    else:
        result["train_test_drift"] = []

    return _json_safe(result)


def stats_dict_to_json_for_prompt(stats: dict[str, Any], max_chars: int = 12000) -> str:
    """Compact JSON for LLM context (truncate if huge)."""
    s = json.dumps(stats, ensure_ascii=False, indent=2)
    if len(s) <= max_chars:
        return s
    return s[: max_chars // 2] + "\n...[truncated]...\n" + s[-(max_chars // 2) :]
