import json
import math
import os
from datetime import datetime, timezone


RECENCY_HOURS = 24.0
PLACES = 10
PRIOR_MASS = 0.05


def parse_time(value):
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_input(input_dir):
    with open(os.path.join(input_dir, "task.json"), encoding="utf-8") as handle:
        task = json.load(handle)
    reads = []
    with open(os.path.join(input_dir, "history.ndjson"), encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                reads.append(json.loads(line))
    return task, reads


def place_rows(place, weighted_games):
    selected = sorted(weighted_games, key=lambda item: (-item[1], item[0]))[:100]
    total = sum(weight for _, weight in selected)
    if total <= 0:
        return []

    nonzero = [
        (universe_id, weight / total)
        for universe_id, weight in selected
        if round(weight / total, 6) > 0
    ]
    nonzero_total = sum(probability for _, probability in nonzero)
    if nonzero_total <= 0:
        return []

    rounded = [
        {
            "universe_id": universe_id,
            "place": place,
            "p": round(probability / nonzero_total, 10),
        }
        for universe_id, probability in nonzero
    ]
    rounded[0]["p"] = round(
        rounded[0]["p"] + 1.0 - sum(row["p"] for row in rounded), 10
    )
    return rounded


def fallback_forecast(reads):
    latest = sorted(
        reads[-1].get("up_and_coming", []),
        key=lambda row: int(row.get("rank", PLACES + 1)),
    )
    games = []
    seen = set()
    for row in latest:
        universe_id = int(row["universe_id"])
        if universe_id not in seen:
            games.append((universe_id, int(row.get("rank", len(games) + 1))))
            seen.add(universe_id)
        if len(games) == PLACES:
            break

    probabilities = []
    for place in range(1, PLACES + 1):
        weighted_games = [
            (universe_id, 0.6 if own_place == place else 0.4 / 9.0)
            for universe_id, own_place in games
        ]
        probabilities.extend(place_rows(place, weighted_games))
    return {
        "probabilities": probabilities,
        "top10": [universe_id for universe_id, _ in games],
    }


def forecast(task, reads):
    history_until = parse_time(task["history_until"])
    if len(reads) < 3:
        return fallback_forecast(reads)

    counts = {}
    games_seen = set()
    for read in reads:
        rows = read.get("up_and_coming", [])
        if not rows:
            continue
        at = parse_time(read["at"])
        hours_since_read = (history_until - at).total_seconds() / 3600.0
        weight = math.exp(-hours_since_read / RECENCY_HOURS)
        for row in rows:
            universe_id = int(row["universe_id"])
            games_seen.add(universe_id)
            place = int(row["rank"])
            if 1 <= place <= PLACES:
                place_counts = counts.setdefault(place, {})
                place_counts[universe_id] = (
                    place_counts.get(universe_id, 0.0) + weight
                )

    if not games_seen:
        return {"probabilities": [], "top10": []}

    probabilities = []
    place_distributions = {}
    game_count = len(games_seen)
    for place in range(1, PLACES + 1):
        place_counts = counts.get(place, {})
        total = sum(place_counts.values())
        prior = PRIOR_MASS * total / game_count if total else 1.0 / game_count
        smoothed = {
            universe_id: place_counts.get(universe_id, 0.0) + prior
            for universe_id in games_seen
        }
        place_distributions[place] = smoothed
        probabilities.extend(
            place_rows(place, list(smoothed.items()))
        )

    used = set()
    top10 = []
    for place in range(1, PLACES + 1):
        best = next(
            (
                universe_id
                for universe_id, _ in sorted(
                    place_distributions[place].items(),
                    key=lambda item: (-item[1], item[0]),
                )
                if universe_id not in used
            ),
            None,
        )
        if best is not None:
            top10.append(best)
            used.add(best)
    return {"probabilities": probabilities, "top10": top10}


def main():
    input_dir = os.path.abspath(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = os.path.abspath(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    task, reads = load_input(input_dir)
    result = forecast(task, reads)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "forecast.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, allow_nan=False, separators=(",", ":"))


if __name__ == "__main__":
    main()
