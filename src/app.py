# src/app.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import streamlit as st

RAW_DIR = Path("data/raw")
FRIENDS_PATH = Path("data/friends.json")

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


def scan_raw_files() -> List[dict]:
    out: List[dict] = []
    for p in sorted(RAW_DIR.glob("matches_*_*.json")):
        m = RAW_FILE_RE.match(p.name)
        if not m:
            continue
        out.append({
            "name": m.group("name"),
            "account_id": int(m.group("account_id")),
            "path": p,
        })
    return out


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
                for key in [
                    "role", "roles", "difficulty", "class", "type",
                    "faction", "description", "tagline",
                    "primary_attribute", "attributes",
                ]:
                    if key in h and h.get(key) not in (None, "", [], {}):
                        meta[key] = h.get(key)

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
    df["duration_min"] = (df["match_duration_s"] / 60.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["kda"] = (df["player_kills"] + df["player_assists"]) / df["player_deaths"].replace(0, 1)
    df["is_win"] = df["match_result"] == 1

    df["cs"] = df["last_hits"] + df["denies"]
    df["cs_per_min"] = (df["cs"] / df["duration_min"].replace(0, np.nan)).fillna(0.0)

    df["souls"] = df["net_worth"]
    df["souls_per_min"] = (df["souls"] / df["duration_min"].replace(0, np.nan)).fillna(0.0)

    # New derived metrics
    df["deaths_per_min"] = (df["player_deaths"] / df["duration_min"].replace(0, np.nan)).fillna(0.0)
    df["assist_ratio"] = (df["player_assists"] / (df["player_kills"] + df["player_assists"]).replace(0, np.nan)).fillna(0.0)

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

st.set_page_config(page_title="Deadlock Friend Tracker", layout="wide")
st.title("Deadlock Friend Tracker")


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
if df.empty:
    st.warning("No match JSON files found in data/raw (expected matches_*_*.json).")
    st.stop()

# Status line
status_bits = []
status_bits.append(f"Raw files: {len(scan_raw_files())}")
status_bits.append("Heroes ✅" if HEROES_JSON.exists() else ("Heroes ⚠️" if HEROES_PARQUET.exists() else "Heroes ❌"))
st.caption(" • ".join(status_bits))

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

tabs = st.tabs(["Leaderboard", "Player Drilldown", "Hero Meta", "Hero Browser"])


# ---------------------------
# Leaderboard
# ---------------------------
with tabs[0]:
    st.subheader("Leaderboard")

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

        st.markdown("#### Pick a hero from the table")
        hero_options = ["(All heroes)"] + hero_summary["hero_display"].tolist()
        picked = st.selectbox(
            "Jump to hero",
            hero_options,
            index=0 if selected_hero == "(All heroes)" else (hero_options.index(selected_hero) if selected_hero in hero_options else 0),
            key=f"{key}_picker",
        )
        if picked != selected_hero:
            st.session_state[key] = picked
            st.rerun()


# ---------------------------
# Hero Meta
# ---------------------------
with tabs[2]:
    st.subheader("Hero Meta (group)")

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
        for candidate in [hrow.get("hero_card", ""), hrow.get("hero_portrait", ""), hrow.get("hero_icon_small", "")]:
            if isinstance(candidate, str) and candidate:
                img = candidate
                break
        if img:
            st.image(img, use_container_width=True)
        st.markdown(f"**Hero ID:** {int(hrow['hero_id'])}")

    with colB:
        st.markdown("### Metadata")
        try:
            meta_obj = json.loads(hrow.get("hero_meta_json", "{}"))
        except Exception:
            meta_obj = {}
        if meta_obj:
            meta_items = [{"field": k, "value": meta_obj[k]} for k in meta_obj.keys()]
            st.dataframe(make_arrow_safe(pd.DataFrame(meta_items)), width="stretch", hide_index=True)
        else:
            st.write("No hero metadata available.")
