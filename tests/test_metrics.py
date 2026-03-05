import pandas as pd

from metrics import add_derived_metrics, match_data_quality_summary


def test_add_derived_metrics_basic_values():
    df = pd.DataFrame(
        [
            {
                "player_kills": 10,
                "player_assists": 5,
                "player_deaths": 5,
                "match_result": 1,
                "last_hits": 50,
                "denies": 10,
                "net_worth": 12000.0,
                "match_duration_s": 1200,
            }
        ]
    )

    out = add_derived_metrics(df)
    row = out.iloc[0]

    assert row["duration_min"] == 20
    assert row["kda"] == 3
    assert row["is_win"] is True
    assert row["cs"] == 60
    assert row["cs_per_min"] == 3
    assert row["souls"] == 12000.0
    assert row["souls_per_min"] == 600
    assert row["deaths_per_min"] == 0.25
    assert row["assist_ratio"] == (5 / 15)


def test_add_derived_metrics_handles_zero_duration_and_zero_kills_assists():
    df = pd.DataFrame(
        [
            {
                "player_kills": 0,
                "player_assists": 0,
                "player_deaths": 0,
                "match_result": 0,
                "last_hits": 10,
                "denies": 5,
                "net_worth": 1000.0,
                "match_duration_s": 0,
            }
        ]
    )

    out = add_derived_metrics(df)
    row = out.iloc[0]

    assert row["duration_min"] == 0
    assert row["kda"] == 0
    assert row["cs_per_min"] == 0
    assert row["souls_per_min"] == 0
    assert row["deaths_per_min"] == 0
    assert row["assist_ratio"] == 0


def test_match_data_quality_summary_counts_invalid_rows():
    df = pd.DataFrame(
        [
            {"account_id": 1, "match_id": 2, "start_time": 3, "match_duration_s": 60},
            {"account_id": 0, "match_id": 5, "start_time": 10, "match_duration_s": 60},
            {"account_id": 10, "match_id": -1, "start_time": 10, "match_duration_s": 60},
            {"account_id": 10, "match_id": 11, "start_time": 0, "match_duration_s": 60},
            {"account_id": 10, "match_id": 12, "start_time": 20, "match_duration_s": 0},
        ]
    )

    summary = match_data_quality_summary(df)

    assert summary["total_rows"] == 5
    assert summary["valid_rows"] == 1
    assert summary["invalid_rows"] == 4
    assert summary["required_missing_columns"] == []


def test_match_data_quality_summary_missing_columns_reports_all_required():
    df = pd.DataFrame([{"foo": 1}])
    summary = match_data_quality_summary(df)
    assert set(summary["required_missing_columns"]) == {
        "account_id",
        "match_id",
        "start_time",
        "match_duration_s",
    }
