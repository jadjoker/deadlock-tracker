from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()

RAW_DIR = Path("data/raw")
FRIENDS_PATH = Path("data/friends.json")


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


@dataclass
class Summary:
    label: str
    account_id: int
    matches: int
    winrate: float
    avg_kda: float
    farm_per_min: float
    nw_per_min: float
    score: float


def load_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path.name} did not contain a JSON list")
    return [r for r in data if isinstance(r, dict)]


def summarize(label: str, account_id: int, rows: list[dict[str, Any]]) -> Summary:
    n = len(rows)
    wins = sum(1 for r in rows if r.get("match_result") == 1)
    winrate = safe_div(wins, n)

    kdas: list[float] = []
    farm_pm: list[float] = []
    nw_pm: list[float] = []

    for r in rows:
        k = int(r.get("player_kills") or 0)
        d = int(r.get("player_deaths") or 0)
        a = int(r.get("player_assists") or 0)

        dur_s = int(r.get("match_duration_s") or 0)
        minutes = dur_s / 60.0 if dur_s else 0.0

        kdas.append(safe_div(k + a, max(1, d)))

        lh = int(r.get("last_hits") or 0)
        dn = int(r.get("denies") or 0)
        farm_pm.append(safe_div(lh + dn, minutes))

        nw = float(r.get("net_worth") or 0.0)
        nw_pm.append(safe_div(nw, minutes))

    avg_kda = mean(kdas) if kdas else 0.0
    farm_per_min = mean(farm_pm) if farm_pm else 0.0
    nw_per_min = mean(nw_pm) if nw_pm else 0.0

    # Simple, stable scoring (tweak later)
    # - winrate is already 0..1
    # - KDA term saturates at 4.0
    # - farm/min saturates at 20
    # - NW/min saturates at 1200 (tune if needed)
    kda_term = min(avg_kda / 4.0, 1.0)
    farm_term = min(farm_per_min / 20.0, 1.0)
    nw_term = min(nw_per_min / 1200.0, 1.0)

    score = (0.50 * winrate) + (0.30 * kda_term) + (0.10 * farm_term) + (0.10 * nw_term)

    return Summary(
        label=label,
        account_id=account_id,
        matches=n,
        winrate=winrate,
        avg_kda=avg_kda,
        farm_per_min=farm_per_min,
        nw_per_min=nw_per_min,
        score=score,
    )


def main() -> None:
    friends = json.loads(FRIENDS_PATH.read_text(encoding="utf-8"))
    summaries: list[Summary] = []

    for f in friends:
        label = str(f["label"])
        account_id = int(f["account_id"])
        # Find file written by fetch script:
        candidates = sorted(RAW_DIR.glob(f"matches_{label}_{account_id}.json"))
        if not candidates:
            console.print(f"[red]Missing raw file for {label} ({account_id}). Run fetch_matches.py[/red]")
            continue

        rows = load_json_list(candidates[0])
        summaries.append(summarize(label, account_id, rows))

    summaries.sort(key=lambda s: (s.score, s.matches), reverse=True)

    table = Table(title="Deadlock Friend Leaderboard")
    table.add_column("Rank", justify="right")
    table.add_column("Player")
    table.add_column("Matches", justify="right")
    table.add_column("Winrate", justify="right")
    table.add_column("Avg KDA", justify="right")
    table.add_column("Farm/min", justify="right")
    table.add_column("NW/min", justify="right")
    table.add_column("Score", justify="right")

    for i, s in enumerate(summaries, start=1):
        table.add_row(
            str(i),
            f"{s.label} ({s.account_id})",
            str(s.matches),
            f"{s.winrate:.3f}",
            f"{s.avg_kda:.2f}",
            f"{s.farm_per_min:.2f}",
            f"{s.nw_per_min:.2f}",
            f"{s.score:.3f}",
        )

    console.print(table)


if __name__ == "__main__":
    main()