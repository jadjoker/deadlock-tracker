# src/fetch_matches.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich import print

from http_client import get_json

FRIENDS_PATH = Path("data/friends.json")
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def load_friends() -> list[dict[str, Any]]:
    """
    Expects data/friends.json as a JSON list like:
    [
      {"label":"Jake","account_id":105260527},
      {"label":"Mark","account_id":121318900}
    ]
    """
    if not FRIENDS_PATH.exists():
        raise RuntimeError("Missing data/friends.json (create it first).")

    data = json.loads(FRIENDS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(
            "data/friends.json must be a JSON list, e.g. "
            "[{\"label\":\"Jake\",\"account_id\":123}]"
        )

    out: list[dict[str, Any]] = []
    for i, f in enumerate(data):
        if not isinstance(f, dict):
            raise RuntimeError(f"friends.json item #{i} is not an object")
        if "account_id" not in f:
            raise RuntimeError(f"friends.json item #{i} missing account_id")

        label = str(f.get("label") or f["account_id"])
        # Make labels filesystem-safe-ish (avoid slashes etc.)
        label = "".join(ch for ch in label if ch.isalnum() or ch in ("-", "_")).strip() or str(f["account_id"])

        account_id = int(f["account_id"])
        out.append({"label": label, "account_id": account_id})

    return out


def fetch_one(label: str, account_id: int) -> Path:
    """
    Fetch match history and MERGE only new matches into the existing file (by match_id).

    Behavior:
    - Always calls the API (so it can discover new matches)
    - If the file doesn't exist: saves the full response
    - If it exists: appends only matches with match_id not already present
    - Sorts the merged file by start_time (descending)
    """
    out = RAW_DIR / f"matches_{label}_{account_id}.json"
    path = f"/v1/players/{account_id}/match-history"

    print(f"[cyan]Fetch[/cyan] {label} ({account_id})")
    print(f"[cyan]Requesting[/cyan] {path} ...")

    new_data = get_json(path)

    print(f"[cyan]Received[/cyan] {label} ({account_id}) response")

    if not isinstance(new_data, list):
        raise RuntimeError(f"Expected list response from {path}, got {type(new_data)}")

    # First run: just save
    if not out.exists():
        out.write_text(json.dumps(new_data, indent=2), encoding="utf-8")
        print(f"[green]Saved new file[/green] {out.name} ({len(new_data)} matches)")
        return out

    # Load existing
    existing_data = json.loads(out.read_text(encoding="utf-8"))
    if not isinstance(existing_data, list):
        raise RuntimeError(f"Existing file {out.name} is not a JSON list (corrupted?)")

    existing_ids = {
        m.get("match_id")
        for m in existing_data
        if isinstance(m, dict) and m.get("match_id") is not None
    }

    truly_new = [
        m for m in new_data
        if isinstance(m, dict) and m.get("match_id") not in existing_ids
    ]

    if not truly_new:
        print(f"[yellow]No new matches[/yellow] for {label}")
        return out

    print(f"[green]+ {len(truly_new)} new matches[/green] for {label}")

    merged = existing_data + truly_new
    merged.sort(key=lambda m: (m.get("start_time") or 0), reverse=True)

    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"[green]Updated[/green] {out.name} ({len(merged)} total matches)")

    return out


def main() -> None:
    friends = load_friends()
    for f in friends:
        fetch_one(f["label"], f["account_id"])


if __name__ == "__main__":
    main()