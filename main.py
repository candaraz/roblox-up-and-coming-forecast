import json
import math
import os
from datetime import datetime, timezone


RECENCY_HOURS = 12.0
MAX_SLOPE = 0.15
PLACES = 10


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


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


def build_games(reads, history_until):
    games = {}
    for read in reads:
        at = parse_time(read["at"])
        for row in read.get("up_and_coming", []):
            universe_id = int(row["universe_id"])
            game = games.setdefault(universe_id, {"appearances": [], "last": row})
            game["appearances"].append((at, int(row["rank"])))
            game["last"] = row
            details = row.get("details")
            if details is None:
                details = read.get("details")
                if isinstance(details, dict):
                    details = details.get(str(universe_id), details.get(universe_id))
                    if details is None and int(read.get("details", {}).get("universe_id", -1)) == universe_id:
                        details = read["details"]
                elif isinstance(details, list):
                    details = next(
                        (
                            item for item in details
                            if int(item.get("universe_id", -1)) == universe_id
                        ),
                        None,
                    )
            if isinstance(details, dict):
                game["details"] = details

    latest = {
        int(row["universe_id"])
        for row in reads[-1].get("up_and_coming", [])
    } if reads else set()
    previous = {
        int(row["universe_id"])
        for row in reads[-2].get("up_and_coming", [])
    } if len(reads) > 1 else set()

    for universe_id, game in games.items():
        appearances = game["appearances"]
        weighted_ranks = []
        for at, rank in appearances:
            dt_hours = (history_until - at).total_seconds() / 3600.0
            weight = math.exp(-dt_hours / RECENCY_HOURS)
            weighted_ranks.append((weight, rank))
        total_weight = sum(weight for weight, _ in weighted_ranks)
        game["wr"] = sum(weight * rank for weight, rank in weighted_ranks) / total_weight

        recent = appearances[-6:]
        points = [
            ((history_until - at).total_seconds() / 3600.0, rank)
            for at, rank in recent
        ]
        mean_x = sum(x for x, _ in points) / len(points)
        mean_y = sum(y for _, y in points) / len(points)
        denominator = sum((x - mean_x) ** 2 for x, _ in points)
        slope = (
            sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
            if denominator
            else 0.0
        )
        game["slope"] = clamp(slope, -MAX_SLOPE, MAX_SLOPE)
        game["absence_penalty"] = (
            0 if universe_id in latest
            else 6 if universe_id not in previous
            else 3
        )
    return games


def fallback_probabilities(reads):
    probabilities = []
    latest_rows = reads[-1].get("up_and_coming", []) if reads else []
    ranked = {}
    for row in latest_rows:
        universe_id = int(row["universe_id"])
        ranked[int(row["rank"])] = universe_id
    for place in range(1, PLACES + 1):
        for rank, probability in (
            (place - 1, 0.15),
            (place, 0.5),
            (place + 1, 0.15),
        ):
            universe_id = ranked.get(rank)
            if universe_id is not None:
                probabilities.append(
                    {"universe_id": universe_id, "place": place, "p": probability}
                )
    return probabilities


def forecast(task, reads):
    if len(reads) < 3:
        probabilities = fallback_probabilities(reads)
        seen = {}
        for read in reads:
            for row in read.get("up_and_coming", []):
                universe_id = int(row["universe_id"])
                seen[universe_id] = int(row["rank"])
        latest = reads[-1].get("up_and_coming", []) if reads else []
        top10 = list(dict.fromkeys(
            int(row["universe_id"])
            for row in sorted(latest, key=lambda row: int(row["rank"]))
        ))[:PLACES]
        if len(top10) < PLACES:
            top10.extend(
                universe_id for universe_id, _ in sorted(seen.items(), key=lambda item: item[1])
                if universe_id not in top10
            )
            top10 = top10[:PLACES]
        return {"probabilities": probabilities, "top10": top10}

    history_until = parse_time(task["history_until"])
    window_from = parse_time(task["window"]["from"])
    window_until = parse_time(task["window"]["until"])
    t_mid = window_from + (window_until - window_from) / 2
    hours_into_window = (t_mid - history_until).total_seconds() / 3600.0
    games = build_games(reads, history_until)

    eligible = []
    for universe_id, game in games.items():
        created_value = game["last"].get("created")
        if created_value is None:
            created_value = game.get("details", {}).get("created")
        if created_value is not None:
            created = parse_time(created_value)
            if (t_mid - created).total_seconds() > 28 * 24 * 3600:
                continue
        prediction = (
            game["wr"]
            + game["slope"] * min(hours_into_window, 48.0)
            + game["absence_penalty"]
        )
        game["pred"] = prediction
        eligible.append((universe_id, game))

    probabilities = []
    for place in range(1, PLACES + 1):
        sigma = 5.0 if place <= 3 else 6.0
        candidates = sorted(
            eligible, key=lambda item: abs(item[1]["pred"] - place)
        )[:30]
        weights = [
            math.exp(-((game["pred"] - place) / sigma) ** 2 / 2.0)
            for _, game in candidates
        ]
        weight_sum = sum(weights)
        if not weight_sum:
            continue
        for (universe_id, _), weight in zip(candidates, weights):
            probability = round(0.8 * weight / weight_sum, 8)
            if probability > 0:
                probabilities.append(
                    {"universe_id": universe_id, "place": place, "p": probability}
                )

    top10 = [
        universe_id
        for universe_id, _ in sorted(eligible, key=lambda item: item[1]["pred"])[:PLACES]
    ]
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
