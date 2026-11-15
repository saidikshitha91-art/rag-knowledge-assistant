"""Structured-data analysis: profiling + anomaly detection."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        info = {"column": c, "dtype": str(s.dtype), "missing": int(s.isna().sum()),
                "unique": int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s):
            info.update(mean=round(s.mean(), 3), std=round(s.std(), 3),
                        min=s.min(), max=s.max())
        else:
            top = s.mode(dropna=True)
            info["top_value"] = top.iloc[0] if len(top) else None
        rows.append(info)
    return pd.DataFrame(rows)


def profile_text(df: pd.DataFrame, max_rows: int = 5) -> str:
    return (f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n"
            f"Column profile:\n{profile(df).to_string(index=False)}\n\n"
            f"Sample rows:\n{df.head(max_rows).to_string(index=False)}")


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05, z_thresh: float = 3.0):
    """Isolation Forest on numeric columns + per-column z-score explanations."""
    num = df.select_dtypes(include="number")
    if num.shape[1] == 0 or len(num) < 10:
        return pd.DataFrame(), "Need at least 10 rows and one numeric column."

    X = num.fillna(num.median())
    model = IsolationForest(contamination=contamination, random_state=42)
    labels = model.fit_predict(X)
    scores = -model.score_samples(X)          # higher = more anomalous

    z = (X - X.mean()) / X.std(ddof=0).replace(0, np.nan)
    reasons = z.abs().apply(
        lambda r: ", ".join(f"{c} (z={z.loc[r.name, c]:.1f})" for c in r.index if r[c] > z_thresh),
        axis=1,
    )
    out = df.copy()
    out["anomaly_score"] = scores.round(4)
    out["flagged_columns"] = reasons
    out = out[labels == -1].sort_values("anomaly_score", ascending=False)
    return out, f"{len(out)} of {len(df)} rows flagged as anomalous."
