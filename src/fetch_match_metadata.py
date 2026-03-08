from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from http_client import get_json

FRIENDS_PATH = Path("data/friends.json")

RAW_DIR = Path("data/raw")     # your existing match-history downloads
META_DIR = Path("data/meta")   # NEW: per-friend metadata caches
META_DIR.mkdir(parents=True, exist_ok=True)

# ---- Configure endpoint here ----
# You told me match history endpoint is:
#   /v1/players/{account_id}/match-history
#
# For metadata, you need the match-details endpoint from Scalar.
# Common patterns are like:
#   /v1/matches/{match_id}
#   /v1/matches/{match_id}/metadata
#
# Put your actual Scalar path here:
MATCH_META_ENDPOINT_TEMPLATE = "/v1/matches/{match_id}/metadata"


# ------------------------
# Helpers
# ------------------------

def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def load_friends() -> List[dict]:
    data = json.loads(FRIENDS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError("data/friends.json must be a JSON list of {account_id,label}")
    out = []
    for f in data:
        if isinstance(f, dict) and "account_id" in f:
            out.append({
                "account_id": safe_int(f.get("account_id")),
                "label": str(f.get("label") or f.get("account_id")),
            })
    return out


def friend_raw_file(friend_label: str, account_id: int) -> Path:
    # Match your existing naming convention if different
    # e.g. Jake_105260527.json
    return RAW_DIR / f"matches_{friend_label}_{account_id}.json"


def friend_meta_file(friend_label: str, account_id: int) -> Path:
    return META_DIR / f"metadata_{friend_label}_{account_id}.json"


def load_match_history_ids(path: Path) -> List[int]:
    """
    Reads a friend's raw match-history JSON list and extracts match_id.
    """
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    ids = []
    for r in data:
        if isinstance(r, dict) and "match_id" in r:
            mid = safe_int(r.get("match_id"))
            if mid > 0:
                ids.append(mid)
    # unique, stable order
    return sorted(set(ids))


def load_existing_meta(path: Path) -> Dict[str, Any]:
    """
    Metadata cache format: dict { "<match_id>": { ...metadata... }, ... }
    Using string keys so JSON stays consistent.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(data, dict):
        return data
    # If you previously stored list, convert to dict keyed by match_id
    if isinstance(data, list):
        out: Dict[str, Any] = {}
        for item in data:
            if isinstance(item, dict) and "match_id" in item:
                out[str(safe_int(item["match_id"]))] = item
        return out
    return {}


def save_meta(path: Path, meta: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def normalize_meta_response(match_id: int, payload: Any) -> dict:
    """
    Keep this conservative. Store match_id and the raw payload
    (or pick specific fields once we know exact schema).
    """
    if isinstance(payload, dict):
        out = dict(payload)
        out["match_id"] = match_id
        return out
    return {"match_id": match_id, "payload": payload}


def fetch_one_meta(match_id: int) -> Optional[dict]:
    """
    Fetch match metadata with basic backoff handling for 429 and transient errors.
    """
    endpoint = MATCH_META_ENDPOINT_TEMPLATE.format(match_id=match_id)

    # get_json() already includes retry/backoff for transient failures and 429s.
    try:
        data = get_json(endpoint)
        return normalize_meta_response(match_id, data)
    except Exception as e:
        print(f"  !! Failed match_id={match_id} after HTTP client retries: {e}")
        return None


# ------------------------
# Main
# ------------------------

def main() -> None:
    friends = load_friends()

    for f in friends:
        account_id = int(f["account_id"])
        label = f["label"]

        raw_path = friend_raw_file(label, account_id)
        meta_path = friend_meta_file(label, account_id)

        print(f"\nFetch metadata for {label} ({account_id})")

        match_ids = load_match_history_ids(raw_path)
        if not match_ids:
            print(f"  No raw match history file found or empty: {raw_path}")
            continue

        existing = load_existing_meta(meta_path)
        existing_ids = set(existing.keys())

        missing = [mid for mid in match_ids if str(mid) not in existing_ids]

        print(f"  Raw unique matches: {len(match_ids)}")
        print(f"  Meta already cached: {len(existing_ids)}")
        print(f"  Missing metadata: {len(missing)}")

        if not missing:
            print("  Nothing to do.")
            continue

        # Fetch missing, incrementally save every N to avoid losing progress
        save_every = 25
        fetched = 0

        for i, mid in enumerate(missing, start=1):
            meta_obj = fetch_one_meta(mid)
            if meta_obj is not None:
                existing[str(mid)] = meta_obj
                fetched += 1

            if (i % save_every) == 0:
                save_meta(meta_path, existing)
                print(f"  .. saved progress ({i}/{len(missing)})")

        save_meta(meta_path, existing)
        print(f"  Done. Fetched {fetched}/{len(missing)} new metadata.")
        print(f"  Wrote: {meta_path}")

    print("\nAll friends complete.")


if __name__ == "__main__":
    main()
