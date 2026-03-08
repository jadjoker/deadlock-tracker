# src/app.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import streamlit as st

from metrics import add_derived_metrics

RAW_DIR = Path("data/raw")
FRIENDS_PATH = Path("data/friends.json")
PLAYER_STATS_GLOBS = [
    "player_hero_stats_*_*.json",  # preferred naming: player_hero_stats_<Name>_<account_id>.json
    "player_stats*.json",          # backwards-compatible fallback
]

HEROES_JSON = Path("data/heroes.json")
HEROES_PARQUET = Path("data/heroes.parquet")  # optional fallback

RAW_FILE_RE = re.compile(r"^matches_(?P<name>.+)_(?P<account_id>\d+)\.json$", re.IGNORECASE)


# ---------------------------
# Helpers
# ---------------------------

def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def first_present(d: dict[str, Any], keys: list[str]) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].map(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x
            )
            out[col] = out[col].fillna("").astype(str)
    return out


def _collect_account_records(obj: Any) -> list[dict[str, Any]]:
    """Recursively collect dict objects that contain an account_id."""
    out: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        if "account_id" in obj:
            out.append(obj)
        for v in obj.values():
            out.extend(_collect_account_records(v))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_collect_account_records(item))
    return out


def _flatten_to_rows(obj: Any, prefix: str = "") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            rows.extend(_flatten_to_rows(v, key))
    elif isinstance(obj, list):
        if obj and all(not isinstance(x, (dict, list)) for x in obj):
            rows.append({"field": prefix or "value", "value": ", ".join(map(str, obj))})
        else:
            rows.append({"field": prefix or "value", "value": json.dumps(obj, ensure_ascii=False)})
    else:
        rows.append({"field": prefix or "value", "value": "" if obj is None else str(obj)})
    return rows


def _collect_hero_records(obj: Any, current_account_id: int | None = None) -> list[dict[str, Any]]:
    """Recursively collect dict records that contain hero_id, carrying account_id context."""
    out: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        if "account_id" in obj:
            current_account_id = _safe_int(obj.get("account_id"), default=current_account_id or 0)

        if "hero_id" in obj:
            rec = dict(obj)
            if "account_id" not in rec and current_account_id is not None:
                rec["account_id"] = current_account_id
            out.append(rec)

        for v in obj.values():
            out.extend(_collect_hero_records(v, current_account_id=current_account_id))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_collect_hero_records(item, current_account_id=current_account_id))
    return out


def _first_existing(cols: list[str], candidates: list[str]) -> str | None:
    cset = set(cols)
    for c in candidates:
        if c in cset:
            return c
    return None


def _state_key(player_label: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", str(player_label)).strip("_")
    return f"hero_filter__{safe}"


# ---------------------------
# Cached loaders
# ---------------------------

@st.cache_data(show_spinner=False)
def load_friends_map() -> dict[int, str]:
    """
    Optional: map account_id -> label from data/friends.json.
    If file missing, we fall back to filename-derived name.
    """
    if not FRIENDS_PATH.exists():
        return {}
    data = json.loads(FRIENDS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return {}
    out: dict[int, str] = {}
    for f in data:
        if isinstance(f, dict) and "account_id" in f:
            aid = _safe_int(f.get("account_id"))
            label = str(f.get("label") or aid)
            out[aid] = label
    return out


@st.cache_data(show_spinner=False)
def load_hero_dict() -> pd.DataFrame:
    if HEROES_JSON.exists():
        raw = json.loads(HEROES_JSON.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            rows: list[dict[str, Any]] = []
            for h in raw:
                if not isinstance(h, dict):
                    continue
                hid = _safe_int(h.get("id"), default=-1)
                if hid <= 0:
                    continue

                hero_name = str(h.get("name") or hid)

                images = h.get("images") if isinstance(h.get("images"), dict) else {}
                if not isinstance(images, dict):
                    images = {}

                hero_icon_small = first_present(images, [
                    "icon_image_small_webp",
                    "icon_image_small",
                    "icon_hero_card_webp",
                    "icon_hero_card",
                ])
                hero_card = first_present(images, [
                    "card_image_webp",
                    "card_image",
                    "hero_card_webp",
                    "hero_card",
                    "portrait_image_webp",
                    "portrait_image",
                ])
                hero_portrait = first_present(images, [
                    "portrait_image_webp",
                    "portrait_image",
                    "full_portrait_webp",
                    "full_portrait",
                ])

                meta: dict[str, Any] = {}
                prioritized_keys = [
                    "role", "roles", "difficulty", "class", "type",
                    "faction", "description", "tagline",
                    "primary_attribute", "attributes",
                    "lore", "abilities", "weapon", "stats", "release_date",
                ]
                for key in prioritized_keys:
                    if key in h and h.get(key) not in (None, "", [], {}):
                        meta[key] = h.get(key)

                # Include any other useful top-level hero fields (excluding IDs, display name, and images).
                for key, value in h.items():
                    if key in {"id", "name", "images"}:
                        continue
                    if key in meta:
                        continue
                    if value in (None, "", [], {}):
                        continue
                    meta[key] = value

                rows.append({
                    "hero_id": hid,
                    "hero_name": hero_name,
                    "hero_icon_small": hero_icon_small,
                    "hero_card": hero_card,
                    "hero_portrait": hero_portrait,
                    "hero_meta_json": json.dumps(meta, ensure_ascii=False),
                })

            heroes = pd.DataFrame(rows)
            if not heroes.empty:
                heroes = heroes.drop_duplicates(subset=["hero_id"]).reset_index(drop=True)
                heroes["hero_id"] = heroes["hero_id"].astype(int)
                for c in ["hero_name", "hero_icon_small", "hero_card", "hero_portrait", "hero_meta_json"]:
                    heroes[c] = heroes[c].fillna("").astype(str)
                return heroes

    if HEROES_PARQUET.exists():
        heroes = pd.read_parquet(HEROES_PARQUET)
        if "id" in heroes.columns and "name" in heroes.columns:
            heroes = heroes.rename(columns={"id": "hero_id", "name": "hero_name"}).copy()
            heroes["hero_id"] = heroes["hero_id"].apply(_safe_int).astype(int)
            heroes["hero_name"] = heroes["hero_name"].astype(str)
            heroes["hero_icon_small"] = ""
            heroes["hero_card"] = ""
            heroes["hero_portrait"] = ""
            heroes["hero_meta_json"] = "{}"
            heroes = heroes.drop_duplicates(subset=["hero_id"]).reset_index(drop=True)
            return heroes[["hero_id", "hero_name", "hero_icon_small", "hero_card", "hero_portrait", "hero_meta_json"]]

    return pd.DataFrame(columns=["hero_id", "hero_name", "hero_icon_small", "hero_card", "hero_portrait", "hero_meta_json"])


@st.cache_data(show_spinner=True)
def load_all_matches(raw_dir: str) -> pd.DataFrame:
    base_dir = Path(raw_dir)
    raw_files = []
    for p in sorted(base_dir.glob("matches_*_*.json")):
        m = RAW_FILE_RE.match(p.name)
        if not m:
            continue
        raw_files.append({
            "name": m.group("name"),
            "account_id": int(m.group("account_id")),
            "path": p,
        })
    if not raw_files:
        return pd.DataFrame()

    rows: List[dict] = []
    for rf in raw_files:
        name = rf["name"]
        account_id = rf["account_id"]
        path: Path = rf["path"]

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(data, list):
            continue

        for r in data:
            if not isinstance(r, dict):
                continue
            r2 = dict(r)
            r2["_source_file"] = path.name
            r2["_file_name"] = name
            r2["_file_account_id"] = account_id
            rows.append(r2)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Ensure expected raw columns exist
    expected_cols = [
        "account_id", "match_id", "hero_id", "hero_level", "start_time",
        "game_mode", "match_mode", "player_team",
        "player_kills", "player_deaths", "player_assists",
        "denies", "net_worth", "last_hits",
        "abandoned_time_s",
        "match_duration_s", "match_result",
        "username",
        "team_abandoned",
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None

    # Types
    for c in [
        "account_id", "match_id", "hero_id", "hero_level", "start_time",
        "game_mode", "match_mode",
        "player_kills", "player_deaths", "player_assists",
        "denies", "last_hits",
        "abandoned_time_s", "match_duration_s", "match_result",
    ]:
        df[c] = df[c].apply(_safe_int)

    # team_abandoned sometimes exists as bool-ish; keep it simple
    if "team_abandoned" in df.columns:
        df["team_abandoned"] = df["team_abandoned"].map(lambda x: bool(x) if x in (True, False) else False)

    df["net_worth"] = df["net_worth"].apply(_safe_float)

    # Labeling
    friends_map = load_friends_map()
    df["player_label"] = df["account_id"].map(friends_map).fillna(df["_file_name"].astype(str))

    # Numeric game mode mapping (raw)
    def map_game_mode(x: int) -> str:
        if x == 1:
            return "Deadlock"
        if x == 4:
            return "Street Brawl"
        return f"Mode {x}"

    df["game_mode_display"] = df["game_mode"].apply(map_game_mode)

    # Derived fields
    df["start_dt"] = pd.to_datetime(df["start_time"], unit="s", utc=True).dt.tz_convert("America/New_York")
    df = add_derived_metrics(df)

    # Heroes
    heroes_df = load_hero_dict()
    if heroes_df.empty:
        df["hero_display"] = df["hero_id"].astype(str)
        df["hero_icon_small"] = ""
        df["hero_card"] = ""
        df["hero_portrait"] = ""
        df["hero_meta_json"] = "{}"
    else:
        df = df.merge(heroes_df, on="hero_id", how="left")
        df["hero_display"] = df["hero_name"].fillna(df["hero_id"].astype(str))
        df["hero_icon_small"] = df["hero_icon_small"].fillna("")
        df["hero_card"] = df["hero_card"].fillna("")
        df["hero_portrait"] = df["hero_portrait"].fillna("")
        df["hero_meta_json"] = df["hero_meta_json"].fillna("{}")

    return df


@st.cache_data(show_spinner=False)
def load_player_stats(raw_dir: str) -> pd.DataFrame:
    """
    Loads optional deeper player stats JSON files from data/raw matching PLAYER_STATS_GLOBS.
    Expected to include account_id somewhere in each record.
    """
    base_dir = Path(raw_dir)
    files: list[Path] = []
    for pattern in PLAYER_STATS_GLOBS:
        files.extend(base_dir.glob(pattern))
    files = sorted(set(files))
    if not files:
        return pd.DataFrame(columns=["account_id", "player_stats_json", "player_stats_source"])

    rows: list[dict[str, Any]] = []
    for p in files:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        recs = _collect_account_records(payload)
        for rec in recs:
            aid = _safe_int(rec.get("account_id"), default=0)
            if aid <= 0:
                continue
            rows.append(
                {
                    "account_id": aid,
                    "player_stats_json": json.dumps(rec, ensure_ascii=False),
                    "player_stats_source": p.name,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["account_id", "player_stats_json", "player_stats_source"])

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["account_id"], keep="last").reset_index(drop=True)
    out["account_id"] = out["account_id"].astype(int)
    out["player_stats_json"] = out["player_stats_json"].astype(str)
    out["player_stats_source"] = out["player_stats_source"].astype(str)
    return out


@st.cache_data(show_spinner=False)
def load_player_hero_stats(raw_dir: str) -> pd.DataFrame:
    """Load hero-level records from local player stats JSON files."""
    base_dir = Path(raw_dir)
    files: list[Path] = []
    for pattern in PLAYER_STATS_GLOBS:
        files.extend(base_dir.glob(pattern))
    files = sorted(set(files))
    if not files:
        return pd.DataFrame(columns=["account_id", "hero_id", "_stats_source"])

    rows: list[dict[str, Any]] = []
    for pth in files:
        try:
            payload = json.loads(pth.read_text(encoding="utf-8"))
        except Exception:
            continue

        for rec in _collect_hero_records(payload):
            aid = _safe_int(rec.get("account_id"), default=0)
            hid = _safe_int(rec.get("hero_id"), default=0)
            if aid <= 0 or hid <= 0:
                continue

            out_row = dict(rec)
            out_row["account_id"] = aid
            out_row["hero_id"] = hid
            out_row["_stats_source"] = pth.name
            rows.append(out_row)

    if not rows:
        return pd.DataFrame(columns=["account_id", "hero_id", "_stats_source"])

    return pd.DataFrame(rows)


def build_player_hero_metrics(player_hero_stats: pd.DataFrame, heroes_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-account per-hero metrics from local hero stats JSON rows."""
    if player_hero_stats.empty:
        return pd.DataFrame()

    dfh = player_hero_stats.copy()
    exclude = {"account_id", "hero_id", "_stats_source"}

    numeric_cols: list[str] = []
    for c in dfh.columns:
        if c in exclude:
            continue
        num = pd.to_numeric(dfh[c], errors="coerce")
        if num.notna().any():
            dfh[c] = num
            numeric_cols.append(c)

    aggs: dict[str, str] = {"_stats_source": "last"}
    for c in numeric_cols:
        aggs[c] = "sum"

    grouped = (
        dfh.groupby(["account_id", "hero_id"], as_index=False)
           .agg(aggs)
           .rename(columns={"_stats_source": "player_stats_source"})
    )

    if heroes_df is not None and not heroes_df.empty:
        grouped = grouped.merge(
            heroes_df[["hero_id", "hero_name", "hero_icon_small", "hero_card", "hero_portrait"]],
            on="hero_id",
            how="left",
        )
        grouped["hero_display"] = grouped["hero_name"].fillna(grouped["hero_id"].astype(str))
    else:
        grouped["hero_display"] = grouped["hero_id"].astype(str)
        grouped["hero_icon_small"] = ""
        grouped["hero_card"] = ""
        grouped["hero_portrait"] = ""

    cols = grouped.columns.tolist()
    wins_col = _first_existing(cols, ["wins", "win_count", "matches_won"])
    losses_col = _first_existing(cols, ["losses", "loss_count", "matches_lost"])
    matches_col = _first_existing(cols, ["matches", "games_played", "match_count", "games"])
    kills_col = _first_existing(cols, ["kills", "player_kills", "hero_kills"])
    deaths_col = _first_existing(cols, ["deaths", "player_deaths", "hero_deaths"])
    assists_col = _first_existing(cols, ["assists", "player_assists", "hero_assists"])

    if matches_col is None and wins_col and losses_col:
        grouped["matches"] = grouped[wins_col].fillna(0) + grouped[losses_col].fillna(0)
        matches_col = "matches"
    elif matches_col:
        grouped["matches"] = grouped[matches_col].fillna(0)

    if wins_col and "matches" in grouped.columns:
        grouped["winrate"] = grouped[wins_col] / grouped["matches"].replace(0, np.nan)
        grouped["winrate"] = grouped["winrate"].fillna(0.0)

    if kills_col and deaths_col and assists_col:
        grouped["kda"] = (grouped[kills_col] + grouped[assists_col]) / grouped[deaths_col].replace(0, 1)

    display_first = [
        "account_id", "hero_id", "hero_display", "player_stats_source", "matches", "winrate", "kda",
        "hero_icon_small", "hero_card", "hero_portrait",
    ]
    ordered = [c for c in display_first if c in grouped.columns] + [c for c in grouped.columns if c not in set(display_first)]
    return grouped[ordered]



def hero_icon_path(hero_id: int) -> str:
    """Return path to hero icon if it exists."""
    p = Path(f"assets/heroes/{hero_id}.png")
    if p.exists():
        return str(p)
    return ""

def hero_image_for_row(row: pd.Series) -> str:
    for key in ("hero_portrait", "hero_card", "hero_icon_small"):
        v = row.get(key, "")
        if isinstance(v, str) and v:
            return v
    return ""


def short_label(name: str, max_len: int = 14) -> str:
    """Shorten long hero names for tight button layouts."""
    s = str(name)
    return s if len(s) <= max_len else (s[: max_len - 1] + "…")


# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title="Deadcock Tracker", layout="wide")
st.title("Deadcock Tracker")


st.markdown(
    """<style>
    /* Compact hero filter/button rows so more fit on one line */
    .stButton > button {
        padding: 0.20rem 0.45rem;
        font-size: 0.78rem;
        line-height: 1.05;
        margin-top: 0.15rem;
    }
    /* Slightly tighter captions */
    .stCaption {
        margin-top: 0.10rem;
    }
    </style>""",
    unsafe_allow_html=True,
)


df = load_all_matches(str(RAW_DIR))
player_hero_stats_raw_df = load_player_hero_stats(str(RAW_DIR))
if df.empty:
    st.warning("No match JSON files found in data/raw (expected matches_*_*.json).")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
all_players = sorted(df["player_label"].unique().tolist())
selected_players = st.sidebar.multiselect("Players", all_players, default=all_players, key="sidebar_players")

all_modes = sorted(df["game_mode_display"].unique().tolist())
selected_modes = st.sidebar.multiselect("Game Modes", all_modes, default=all_modes, key="sidebar_modes")

if st.sidebar.button("Apply filters", key="apply_filters"):
    st.rerun()

fdf = df[df["player_label"].isin(selected_players) & df["game_mode_display"].isin(selected_modes)].copy()
players_filtered = sorted(fdf["player_label"].unique().tolist())

tabs = st.tabs(["Leaderboard", "Player Drilldown", "Hero Meta", "Hero Browser", "Player Hero Stats"])


# ---------------------------
# Leaderboard
# ---------------------------
with tabs[0]:
    st.subheader("Leaderboard")
    st.caption("Metric tips: Winrate = wins / matches • KDA = (kills + assists) / max(1, deaths) • CS/min = (last_hits + denies) / match_minutes • Souls/min = net_worth / match_minutes • Assist Ratio = assists / (kills + assists)")

    summary = (
        fdf.groupby(["player_label"], as_index=False)
           .agg(
               matches=("match_id", "nunique"),
               wins=("is_win", "sum"),
               winrate=("is_win", "mean"),
               avg_kda=("kda", "mean"),
               avg_cs_per_min=("cs_per_min", "mean"),
               avg_souls_per_min=("souls_per_min", "mean"),
               avg_deaths_per_min=("deaths_per_min", "mean"),
               avg_assist_ratio=("assist_ratio", "mean"),
           )
    )

    summary = summary.sort_values(["winrate", "matches"], ascending=[False, False]).reset_index(drop=True)
    summary.insert(0, "rank", summary.index + 1)

    st.dataframe(make_arrow_safe(summary), width="stretch", hide_index=True)


# ---------------------------
# Player Drilldown
# ---------------------------
with tabs[1]:
    st.subheader("Player Drilldown")

    if not players_filtered:
        st.info("No players available in current filters.")
    else:
        player = st.selectbox("Select Player", players_filtered, key="player_select")
        pdf_all = fdf[fdf["player_label"] == player].sort_values("start_time", ascending=False).copy()

        # Build hero summary once (for top 3 + filter buttons + table)
        hero_summary = (
            pdf_all.groupby(["hero_id", "hero_display"], as_index=False)
               .agg(
                   matches=("match_id", "nunique"),
                   wins=("is_win", "sum"),
                   winrate=("is_win", "mean"),
                   avg_kda=("kda", "mean"),
                   avg_cs_per_min=("cs_per_min", "mean"),
                   avg_souls_per_min=("souls_per_min", "mean"),
                   avg_deaths_per_min=("deaths_per_min", "mean"),
                   avg_assist_ratio=("assist_ratio", "mean"),
                   hero_icon_small=("hero_icon_small", "first"),
                   hero_card=("hero_card", "first"),
                   hero_portrait=("hero_portrait", "first"),
               )
        )
        hero_summary["winrate_pct"] = (hero_summary["winrate"] * 100.0).round(2)
        hero_summary["avg_kda"] = hero_summary["avg_kda"].round(2)
        hero_summary["avg_cs_per_min"] = hero_summary["avg_cs_per_min"].round(2)
        hero_summary["avg_souls_per_min"] = hero_summary["avg_souls_per_min"].round(2)
        if "avg_deaths_per_min" in hero_summary.columns:
            hero_summary["avg_deaths_per_min"] = hero_summary["avg_deaths_per_min"].round(2)
        if "avg_assist_ratio" in hero_summary.columns:
            hero_summary["avg_assist_ratio"] = hero_summary["avg_assist_ratio"].round(2)
        hero_summary = hero_summary.sort_values(["matches", "winrate_pct"], ascending=[False, False]).reset_index(drop=True)

        # Selected hero state
        key = _state_key(player)
        if key not in st.session_state:
            st.session_state[key] = "(All heroes)"
        selected_hero = st.session_state[key]

        # --- Profile header: metrics + top 3 heroes ---
        # Apply hero filter for the "profile" metrics and match list
        if selected_hero == "(All heroes)":
            pdf = pdf_all.copy()
        else:
            pdf = pdf_all[pdf_all["hero_display"] == selected_hero].copy()

        wins = int(pdf["is_win"].sum())
        matches = int(pdf["match_id"].nunique())
        winrate = (wins / matches) if matches else 0.0

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Matches", matches)
        c2.metric("Wins", wins)
        c3.metric("Winrate", f"{winrate:.3f}")
        c4.metric("Avg KDA", f"{pdf['kda'].mean():.2f}")
        c5.metric("Avg CS/min", f"{pdf['cs_per_min'].mean():.2f}")
        c6.metric("Avg Souls/min", f"{pdf['souls_per_min'].mean():.2f}")

        # Additional metrics (new)
        d1, d2 = st.columns(2)
        d1.metric("Avg Deaths/min", f"{pdf['deaths_per_min'].mean():.2f}")
        d2.metric("Avg Assist Ratio", f"{pdf['assist_ratio'].mean():.2f}")
        st.caption("Metric tips: KDA = (kills + assists) / max(1, deaths) • CS/min = (last_hits + denies) / match_minutes • Souls/min = net_worth / match_minutes • Deaths/min = deaths / match_minutes • Assist Ratio = assists / (kills + assists)")

        st.markdown("#### Win rate by game length")

        # Buckets based on match duration (minutes)
        bins = [0, 15, 25, 35, np.inf]
        labels = ["<15m", "15–25m", "25–35m", "35m+"]
        tmp = pdf.copy()
        tmp["length_bucket"] = pd.cut(tmp["duration_min"], bins=bins, labels=labels, right=False, include_lowest=True)

        wr_len = (
            tmp.groupby("length_bucket", as_index=False)
               .agg(matches=("match_id", "nunique"), winrate=("is_win", "mean"))
        )
        wr_len["winrate_pct"] = (wr_len["winrate"] * 100.0).round(2)
        wr_len = wr_len.drop(columns=["winrate"])
        st.dataframe(make_arrow_safe(wr_len), width="stretch", hide_index=True)

        st.markdown("#### Top 3 most played heroes")
        top3 = hero_summary.head(3).copy()

        top_cols = st.columns(3, gap="large")
        for i in range(3):
            with top_cols[i]:
                if i >= len(top3):
                    st.write("")
                    continue
                r = top3.iloc[i]
                img = hero_image_for_row(r)
                if img:
                    st.image(img, width=56)
                st.markdown(f"**{i+1}. {r['hero_display']}**")
                st.caption(f"{int(r['matches'])} matches • {float(r['winrate_pct']):.2f}% WR")

        st.markdown("#### Filter by hero")
        # Button grid (acts like clicking the portrait)
        # Row 1: All heroes button
        btn_cols = st.columns(10, gap="small")
        with btn_cols[0]:
            if st.button("All", use_container_width=True, key=f"{key}_all"):
                st.session_state[key] = "(All heroes)"
                st.rerun()

        # Show up to first 11 heroes as quick buttons (most played first)
        visible = hero_summary
        for idx, r in visible.iterrows():
            col = btn_cols[(idx % 8) + 1]  # use cols 1..8 for heroes
            with col:
                img = hero_icon_path(int(r["hero_id"]))
                if img:
                    st.image(img, width=30)
                label = short_label(r["hero_display"], max_len=14)
                if st.button(label, use_container_width=True, key=f"{key}_btn_{int(r['hero_id'])}"):
                    st.session_state[key] = r["hero_display"]
                    st.rerun()

        st.divider()

        # --- Full Match History ---
        st.markdown("### Full Match History")

        default_cols = [
        "start_dt",
        "match_id",
        "game_mode_display",
        "hero_display",
        "match_result",
        "player_kills",
        "player_deaths",
        "deaths_per_min",
        "player_assists",
        "assist_ratio",
        "cs_per_min",
        "souls_per_min",
        "souls",
        "match_duration_s",
        ]

        cols = st.multiselect(
        "Columns",
        options=list(pdf.columns),
        default=[c for c in default_cols if c in pdf.columns],
        key=f"player_match_cols_{player}",
        )

        st.dataframe(make_arrow_safe(pdf[cols]), width="stretch", hide_index=True)

        # --- Hero Breakdown moved to bottom (table + dropdown for the rest) ---
        st.divider()
        st.markdown("### Hero Breakdown (all heroes for this player)")

        table_cols = ["hero_display", "matches", "wins", "winrate_pct", "avg_kda", "avg_cs_per_min", "avg_souls_per_min", "avg_deaths_per_min", "avg_assist_ratio"]
        table_cols = [c for c in table_cols if c in hero_summary.columns]
        st.dataframe(make_arrow_safe(hero_summary[table_cols]), width="stretch", hide_index=True)



# ---------------------------
# Hero Meta
# ---------------------------
with tabs[2]:
    st.subheader("Hero Meta (group)")
    st.caption("Averages are per-match means across the currently filtered players and modes.")

    meta = (
        fdf.groupby(["hero_id", "hero_display"], as_index=False)
           .agg(
               matches=("match_id", "nunique"),
               winrate=("is_win", "mean"),
               avg_kda=("kda", "mean"),
               avg_cs_per_min=("cs_per_min", "mean"),
               avg_souls_per_min=("souls_per_min", "mean"),
               avg_deaths_per_min=("deaths_per_min", "mean"),
               avg_assist_ratio=("assist_ratio", "mean"),
           )
           .sort_values(["matches"], ascending=False)
           .reset_index(drop=True)
    )

    st.dataframe(make_arrow_safe(meta), width="stretch", hide_index=True)


# ---------------------------
# Hero Browser
# ---------------------------
with tabs[3]:
    st.subheader("Hero Browser")
    st.caption("Browse hero art + a cleaned metadata view from data/heroes.json.")

    heroes = (
        df[["hero_id", "hero_display", "hero_icon_small", "hero_card", "hero_portrait", "hero_meta_json"]]
        .drop_duplicates(subset=["hero_id"])
        .sort_values("hero_display")
        .reset_index(drop=True)
    )

    hero_choice = st.selectbox("Select a hero", heroes["hero_display"].tolist(), key="hero_browser_select")
    hrow = heroes[heroes["hero_display"] == hero_choice].iloc[0]

    colA, colB = st.columns([1, 2], gap="large")
    with colA:
        img = ""
        for candidate in [hrow.get("hero_icon_small", ""), hrow.get("hero_card", ""), hrow.get("hero_portrait", "")]:
            if isinstance(candidate, str) and candidate:
                img = candidate
                break

        with st.container(border=True):
            if img:
                st.image(img, width=120)
            else:
                st.write("No hero image available")
            st.markdown(f"**{hrow['hero_display']}**")
            st.caption(f"Hero ID: {int(hrow['hero_id'])}")

        image_rows = []
        for label, url in [
            ("Icon", hrow.get("hero_icon_small", "")),
            ("Card", hrow.get("hero_card", "")),
            ("Portrait", hrow.get("hero_portrait", "")),
        ]:
            if isinstance(url, str) and url:
                image_rows.append({"image_type": label, "url": url})

        if image_rows:
            st.markdown("#### Image URLs")
            st.dataframe(make_arrow_safe(pd.DataFrame(image_rows)), width="stretch", hide_index=True)

    with colB:
        st.markdown("### Metadata")
        try:
            meta_obj = json.loads(hrow.get("hero_meta_json", "{}"))
        except Exception:
            meta_obj = {}

        if meta_obj:
            preferred_order = [
                "role", "roles", "difficulty", "class", "type", "faction",
                "primary_attribute", "description", "tagline",
            ]

            summary_items = []
            detail_items = []
            for k, v in meta_obj.items():
                target = summary_items if k in preferred_order else detail_items
                target.append({"field": k, "value": v})

            if summary_items:
                st.markdown("#### Core hero details")
                st.dataframe(make_arrow_safe(pd.DataFrame(summary_items)), width="stretch", hide_index=True)

            if detail_items:
                st.markdown("#### Additional hero fields")
                st.dataframe(make_arrow_safe(pd.DataFrame(detail_items)), width="stretch", hide_index=True)

            with st.expander("View raw hero metadata JSON"):
                st.json(meta_obj)
        else:
            st.write("No hero metadata available.")


# ---------------------------
# Player Hero Stats (from JSON)
# ---------------------------
with tabs[4]:
    st.subheader("Player Hero Stats")
    st.caption("Derived from local player hero stats JSON files, linked by account_id + hero_id.")

    if player_hero_stats_raw_df.empty:
        st.info(
            f"No hero stats records found in {RAW_DIR} for patterns {PLAYER_STATS_GLOBS}. "
            "Add files like player_hero_stats_Jake_105260527.json with hero_id/account_id fields."
        )
    else:
        heroes_lookup = load_hero_dict()
        hero_metrics_df = build_player_hero_metrics(player_hero_stats_raw_df, heroes_lookup)

        all_labels = sorted(fdf["player_label"].unique().tolist())
        selected_label = st.selectbox("Select Player", all_labels, key="player_hero_stats_select")

        selected_accounts = sorted(
            {
                _safe_int(x)
                for x in fdf[fdf["player_label"] == selected_label]["account_id"].unique().tolist()
                if _safe_int(x) > 0
            }
        )

        hdf = hero_metrics_df[hero_metrics_df["account_id"].isin(selected_accounts)].copy()
        if hdf.empty:
            st.warning("No linked hero stats found for this player label/account_id in local stats JSON.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Heroes tracked", int(hdf["hero_id"].nunique()))
            if "matches" in hdf.columns:
                m2.metric("Matches (from JSON)", int(pd.to_numeric(hdf["matches"], errors="coerce").fillna(0).sum()))
            else:
                m2.metric("Matches (from JSON)", "n/a")
            if "winrate" in hdf.columns:
                m3.metric("Avg Winrate", f"{float(pd.to_numeric(hdf['winrate'], errors='coerce').fillna(0).mean())*100:.2f}%")
            else:
                m3.metric("Avg Winrate", "n/a")

            sort_cols = [c for c in ["matches", "hero_display"] if c in hdf.columns]
            if sort_cols:
                asc = [False if c == "matches" else True for c in sort_cols]
                hdf = hdf.sort_values(sort_cols, ascending=asc)

            top = hdf.head(6).copy()
            st.markdown("#### Top heroes from JSON stats")
            top_cols = st.columns(6)
            for i in range(6):
                with top_cols[i]:
                    if i >= len(top):
                        st.write("")
                        continue
                    row = top.iloc[i]
                    img = ""
                    for c in ["hero_icon_small", "hero_card", "hero_portrait"]:
                        v = row.get(c, "")
                        if isinstance(v, str) and v:
                            img = v
                            break
                    if img:
                        st.image(img, width=42)
                    st.caption(str(row.get("hero_display", row.get("hero_id", "?"))))
                    if "matches" in top.columns:
                        st.caption(f"matches: {float(row.get('matches', 0)):.0f}")
                    if "winrate" in top.columns:
                        st.caption(f"wr: {float(row.get('winrate', 0))*100:.1f}%")

            show_cols = ["hero_display", "hero_id", "matches", "winrate", "kda", "player_stats_source"]
            show_cols = [c for c in show_cols if c in hdf.columns]

            extra_numeric = [
                c for c in hdf.columns
                if c not in set(show_cols + ["account_id", "hero_icon_small", "hero_card", "hero_portrait", "hero_name"])
                and pd.api.types.is_numeric_dtype(hdf[c])
            ]
            show_cols.extend(extra_numeric[:8])

            out = hdf[show_cols].copy()
            if "winrate" in out.columns:
                out["winrate"] = (out["winrate"] * 100.0).round(2)
            st.dataframe(make_arrow_safe(out), width="stretch", hide_index=True)
