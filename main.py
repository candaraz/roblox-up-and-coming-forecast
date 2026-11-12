import json
import math
import os
from datetime import datetime, timezone


RECENCY_HOURS = 12.0
PLACES = 10


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
            game = games.setdefault(
                universe_id,
                {"appearances": [], "first_seen": at, "last": row},
            )
            game["appearances"].append((at, row))
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

    for universe_id, game in games.items():
        appearances = game["appearances"]
        weighted_ranks = []
        for at, row in appearances:
            dt_hours = (history_until - at).total_seconds() / 3600.0
            weight = math.exp(-dt_hours / RECENCY_HOURS)
            weighted_ranks.append((weight, int(row["rank"])))
        total_weight = sum(weight for weight, _ in weighted_ranks)
        game["wr"] = sum(weight * rank for weight, rank in weighted_ranks) / total_weight

        recent = appearances[-6:]
        playing_growths = []
        vote_growths = []
        for (previous_at, previous_row), (at, row) in zip(recent, recent[1:]):
            hours = (at - previous_at).total_seconds() / 3600.0
            if hours <= 0:
                continue
            recency_weight = math.exp(
                -max(0.0, (history_until - at).total_seconds() / 3600.0)
                / RECENCY_HOURS
            )
            previous_playing = previous_row.get("playing")
            playing = row.get("playing")
            if previous_playing is not None and playing is not None:
                playing_growths.append(
                    (recency_weight, (float(playing) - float(previous_playing)) / hours)
                )
            previous_votes = previous_row.get("up_votes")
            votes = row.get("up_votes")
            if previous_votes is not None and votes is not None:
                vote_growths.append(
                    (recency_weight, (float(votes) - float(previous_votes)) / hours)
                )

        def weighted_growth(growths):
            weight_sum = sum(weight for weight, _ in growths)
            return (
                sum(weight * growth for weight, growth in growths) / weight_sum
                if weight_sum
                else None
            )

        last_pair = recent[-2:] if len(recent) >= 2 else []
        game["growth_playing_last"] = None
        game["growth_upvotes_last"] = None
        if len(last_pair) == 2:
            previous_at, previous_row = last_pair[0]
            at, row = last_pair[1]
            hours = (at - previous_at).total_seconds() / 3600.0
            if hours > 0:
                for field, growth_key in (
                    ("playing", "growth_playing_last"),
                    ("up_votes", "growth_upvotes_last"),
                ):
                    before = previous_row.get(field)
                    after = row.get(field)
                    if before is not None and after is not None:
                        game[growth_key] = (float(after) - float(before)) / hours
        game["growth_playing"] = weighted_growth(playing_growths)
        game["growth_upvotes"] = weighted_growth(vote_growths)
        game["rank_weighted"] = game["wr"]
        game["present_latest"] = universe_id in latest
    return games


def standardized(values):
    observed = [value for value in values if value is not None]
    if not observed:
        return [0.0] * len(values)
    mean = sum(observed) / len(observed)
    variance = sum((value - mean) ** 2 for value in observed) / len(observed)
    deviation = math.sqrt(variance)
    if deviation == 0:
        return [0.0] * len(values)
    return [0.0 if value is None else (value - mean) / deviation for value in values]


def forecast(task, reads):
    history_until = parse_time(task["history_until"])
    window_from = parse_time(task["window"]["from"])
    window_until = parse_time(task["window"]["until"])
    t_mid = window_from + (window_until - window_from) / 2
    games = build_games(reads, history_until)

    eligible = []
    for universe_id, game in games.items():
        created_value = game["last"].get("created")
        if created_value is None:
            created_value = game.get("details", {}).get("created")
        created = parse_time(created_value) if created_value else game["first_seen"]
        age_days = (t_mid - created).total_seconds() / 86400.0
        if age_days > 28:
            continue
        game["age_days"] = age_days
        eligible.append((universe_id, game))

    if eligible:
        games_only = [game for _, game in eligible]
        growth_playing_z = standardized(
            [game["growth_playing"] for game in games_only]
        )
        growth_upvotes_z = standardized(
            [game["growth_upvotes"] for game in games_only]
        )
        rank_z = standardized([-game["rank_weighted"] for game in games_only])
        for game, playing_z, votes_z, game_rank_z in zip(
            games_only, growth_playing_z, growth_upvotes_z, rank_z
        ):
            game["strength"] = (
                0.9 * playing_z
                + 0.6 * votes_z
                + 0.5 * game_rank_z
                - 0.05 * max(0.0, game["age_days"] - 21.0) ** 1.5
            )

    strength_order = sorted(
        eligible,
        key=lambda item: (-item[1]["strength"], item[0]),
    )
    for base_rank, (_, game) in enumerate(strength_order, start=1):
        game["pred_rank"] = base_rank + (0 if game["present_latest"] else 3)

    probabilities = []
    for place in range(1, PLACES + 1):
        sigma = 2.5 if place <= 3 else 3.5
        candidates = sorted(
            eligible,
            key=lambda item: (
                abs(item[1]["pred_rank"] - place),
                -item[1]["strength"],
                item[0],
            ),
        )[:15]
        weights = [
            math.exp(-((game["pred_rank"] - place) / sigma) ** 2 / 2.0)
            for _, game in candidates
        ]
        weight_sum = sum(weights)
        if not weight_sum:
            continue
        for (universe_id, _), weight in zip(candidates, weights):
            probability = round(0.90 * weight / weight_sum, 8)
            if probability > 0:
                probabilities.append(
                    {"universe_id": universe_id, "place": place, "p": probability}
                )

    top10 = [
        universe_id
        for universe_id, _ in strength_order[:PLACES]
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
