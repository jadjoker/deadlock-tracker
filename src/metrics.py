from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REQUIRED_MATCH_COLUMNS = [
    "account_id",
    "match_id",
    "start_time",
    "match_duration_s",
]


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add common derived fields used by dashboard and reports."""
    out = df.copy()

    out["duration_min"] = (out["match_duration_s"] / 60.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["kda"] = (out["player_kills"] + out["player_assists"]) / out["player_deaths"].replace(0, 1)
    out["is_win"] = out["match_result"] == 1

    out["cs"] = out["last_hits"] + out["denies"]
    out["cs_per_min"] = (out["cs"] / out["duration_min"].replace(0, np.nan)).fillna(0.0)

    out["souls"] = out["net_worth"]
    out["souls_per_min"] = (out["souls"] / out["duration_min"].replace(0, np.nan)).fillna(0.0)

    out["deaths_per_min"] = (out["player_deaths"] / out["duration_min"].replace(0, np.nan)).fillna(0.0)
    out["assist_ratio"] = (
        out["player_assists"] / (out["player_kills"] + out["player_assists"]).replace(0, np.nan)
    ).fillna(0.0)

    return out


def match_data_quality_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Return quick data quality counts for the loaded match rows."""
    required_missing_columns = [c for c in REQUIRED_MATCH_COLUMNS if c not in df.columns]

    total_rows = len(df)
    if total_rows == 0:
        return {
            "total_rows": 0,
            "valid_rows": 0,
            "invalid_rows": 0,
            "invalid_pct": 0.0,
            "required_missing_columns": required_missing_columns,
        }

    bad_mask = pd.Series(False, index=df.index)

    for c in REQUIRED_MATCH_COLUMNS:
        if c not in df.columns:
            continue
        bad_mask = bad_mask | df[c].isna()

    if "match_id" in df.columns:
        bad_mask = bad_mask | (df["match_id"] <= 0)
    if "account_id" in df.columns:
        bad_mask = bad_mask | (df["account_id"] <= 0)
    if "start_time" in df.columns:
        bad_mask = bad_mask | (df["start_time"] <= 0)
    if "match_duration_s" in df.columns:
        bad_mask = bad_mask | (df["match_duration_s"] <= 0)

    invalid_rows = int(bad_mask.sum())
    valid_rows = int(total_rows - invalid_rows)

    return {
        "total_rows": int(total_rows),
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "invalid_pct": round((invalid_rows / total_rows) * 100.0, 2),
        "required_missing_columns": required_missing_columns,
    }
