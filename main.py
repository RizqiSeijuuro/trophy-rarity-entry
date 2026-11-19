"""gta6-trophy-rarity-forecast-01 entry: a weighted-trend forecaster, Python 3.13 standard library only.

For every trophy in a case, this fits a recency-weighted least-squares trend of logit(earn-rate) against
log(1 + hours since that trophy's first read in the case's history), shrinks the fitted slope toward zero
when little data backs it, extrapolates to the case's target time with the slope's predicted move clamped to
a sane bound, and builds five quantiles around that prediction from two combined uncertainty terms: a
random-walk term (the house baseline's own spread, so we are never narrower than "nothing changed" would
imply is safe) and a trend-residual term (how well the fitted line actually matched its own history),
widened further when we are asked to extrapolate past the span of data the trend was fit on.

Where a trophy has too little history to fit a trend at all (0 or 1 valid reads), this falls back to the
house baseline's own persistence forecast for that trophy, which is always well-defined.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in python:3.13-slim, standard library
only: no network, no pip packages. -P keeps this file's directory off sys.path, so a module shipped beside
main.py would only import after the sys.path line below puts the directory back first; -s drops user
site-packages. (This entry ships no extra module, but the line is kept so one could be added safely.)

What this program does:
  - reads every <id>.in in the input directory (the grader runs one case at a time, with only that case's
    file there): JSON {"schema": "makefun/gta6-trophies-case/v1", "id", "at", "horizon_minutes", "target",
    "levels", "game", "trophies", "history", "notes"}, where `trophies` lists the case's trophies {id, name,
    description, tier, points, secret} and `history` every read before `at`, oldest first, each {slot, at,
    players_tracked, trophies_earned, rates}, `rates` in the order of `trophies` (null where that read
    lacked the trophy);
  - writes <id>.out to the output directory: {"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]},
    one row for each of the case's trophies, each id once, no other key, each q five finite numbers from 0
    to 100 (percent), non-decreasing; at most 65,536 bytes.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` would find a model.py
# shipped beside this file. This entry has no such module, but the line costs nothing to keep.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# The random-walk spread's scale, in logit units per square-root day: the same constant the house baseline
# uses, so our forecast is never narrower than a plain persistence call would consider safe.
RW_SIGMA = 0.5

# The slope's shrinkage: a trend fit on n points is trusted only n / (n + SHRINK_K) of the way, so a
# two-point "trend" barely moves the forecast while a trend backed by dozens of reads is taken near fully.
SHRINK_K = 4.0

# The extrapolated move (the shrunk trend's predicted change in logit from the last observed read to the
# target) is clamped to this many logit units per square-root day of horizon, so a noisy short trend never
# extrapolates to something wild over a long horizon.
MOVE_CLAMP_PER_SQRT_DAY = 2.5

# When fewer than two valid reads exist for a trophy, there is no trend to fit at all: fall back to a fixed
# residual uncertainty (logit units) standing in for "we don't know how noisy this trophy's rate is yet".
DEFAULT_RESIDUAL = 0.25

# The trend-residual term is scaled up when the horizon reaches past the span of history the trend was
# fit on (extrapolation is less safely guided by the fit the further out it reaches), but the scale-up is
# capped so a tiny span against a long horizon cannot blow the spread up without bound.
MAX_HORIZON_SPAN_SCALE = 6.0

# A floor under the span used in that scale-up, so a trend fit on reads taken seconds apart (or a single
# point) cannot produce a division by (near) zero.
MIN_SPAN_HOURS = 0.5


# Keeps percentages strictly inside (0, 100) so logit() never takes log(0) or divides by zero.
def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100 - p))


def sigmoid(x: float) -> float:
    return 100 / (1 + math.exp(-x))


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_at(iso: str) -> datetime:
    # Every timestamp in a case is UTC ISO 8601, e.g. "2026-11-19T03:00:00Z" or with fractional seconds.
    # Python 3.11+'s fromisoformat accepts the trailing "Z" directly.
    return datetime.fromisoformat(iso)


def hours_between(earlier: datetime, later: datetime) -> float:
    return (later - earlier).total_seconds() / 3600.0


def weighted_least_squares(xs: list[float], ys: list[float], ws: list[float]) -> tuple[float, float]:
    """Weighted least squares intercept and slope for y ~ a + b*x, weights ws. Assumes len(xs) >= 2."""
    sw = sum(ws)
    swx = sum(w * x for w, x in zip(ws, xs))
    swy = sum(w * y for w, y in zip(ws, ys))
    swxx = sum(w * x * x for w, x in zip(ws, xs))
    swxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))
    denom = sw * swxx - swx * swx
    if abs(denom) < 1e-12:
        # Degenerate (e.g. every x identical): no slope is identifiable, fall back to the weighted mean.
        return (swy / sw if sw > 0 else 0.0), 0.0
    b = (sw * swxy - swx * swy) / denom
    a = (swy - b * swx) / sw
    return a, b


def weighted_residual_std(xs: list[float], ys: list[float], ws: list[float], a: float, b: float) -> float:
    sw = sum(ws)
    if sw <= 0:
        return 0.0
    sse = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
    return math.sqrt(max(0.0, sse / sw))


def forecast_trophy(reads: list[tuple[datetime, float]], target: datetime, horizon_days: float) -> list[float]:
    """Build the five logit-scale quantile offsets' base (predicted logit) and spread for one trophy.

    Note on the shrinkage factor n / (n + SHRINK_K): with SHRINK_K == 4, two reads (n=2) trust the fitted
    slope only 2/6 = 1/3 of the way, while fifty reads trust it 50/54 ~= 0.93 of the way, so the clamp and
    the residual-based spread below are doing most of the safety work while data is still thin.

    `reads` is this trophy's own valid (non-null) (at, rate_percent) pairs from the case's history, oldest
    first; there is always at least the case's own last read for every forecast trophy, but this is
    defensive about the (rules say it shouldn't happen) empty case too.
    """
    if not reads:
        # No data at all for this trophy: nothing to anchor on. Treat it as a brand-new, unseen trophy.
        anchor_logit = logit(0.01)
        total_spread = RW_SIGMA * math.sqrt(horizon_days)
        return [round4(sigmoid(anchor_logit + z * total_spread)) for z in Z]

    first_at = reads[0][0]
    xs = [math.log1p(hours_between(first_at, at)) for at, _ in reads]
    ys = [logit(rate) for _, rate in reads]
    n = len(reads)

    last_at, last_rate = reads[-1]
    anchor_logit = logit(last_rate)
    anchor_x = xs[-1]
    target_x = math.log1p(hours_between(first_at, target))

    horizon_hours = horizon_days * 24.0
    rw_term = RW_SIGMA * math.sqrt(horizon_days)

    if n < 2:
        # No trend is fittable from a single read: persist, widened by the default residual uncertainty.
        trend_term = DEFAULT_RESIDUAL
        total_spread = math.sqrt(rw_term * rw_term + trend_term * trend_term)
        return [round4(sigmoid(anchor_logit + z * total_spread)) for z in Z]

    # Recency weights: later reads (closer to the last one) count more. Weight decays by half every
    # half-life, where the half-life scales with the span of history itself, so a trophy with a short
    # history and one with a long history are both weighted on "about a third of the way back is half
    # weight", rather than one fixed clock time suiting every case's cadence.
    span_hours_fit = hours_between(first_at, last_at)
    half_life = max(MIN_SPAN_HOURS, span_hours_fit / 3.0)
    last_hours = hours_between(first_at, last_at)
    read_hours = [hours_between(first_at, at) for at, _ in reads]
    weights = [2.0 ** (-(last_hours - h) / half_life) for h in read_hours]

    a, b = weighted_least_squares(xs, ys, weights)
    shrink = n / (n + SHRINK_K)
    b_shrunk = b * shrink

    raw_move = b_shrunk * (target_x - anchor_x)
    move_bound = MOVE_CLAMP_PER_SQRT_DAY * math.sqrt(horizon_days)
    clamped_move = max(-move_bound, min(move_bound, raw_move))
    predicted_logit = anchor_logit + clamped_move

    residual_std = weighted_residual_std(xs, ys, weights, a, b)
    span_hours = max(MIN_SPAN_HOURS, span_hours_fit)
    horizon_scale = math.sqrt(max(1.0, horizon_hours / span_hours))
    horizon_scale = min(MAX_HORIZON_SPAN_SCALE, horizon_scale)
    trend_term = max(residual_std, 1e-4) * horizon_scale if residual_std > 0 else 0.0
    # Even a perfectly fit trend (residual_std == 0, e.g. identical reads so far) still carries some
    # irreducible uncertainty once we are asked to extrapolate, so floor it at a small fraction of the
    # default when extrapolating beyond the fitted span.
    if residual_std <= 1e-9 and horizon_scale > 1.0:
        trend_term = DEFAULT_RESIDUAL * 0.5 * (horizon_scale - 1.0) / MAX_HORIZON_SPAN_SCALE + 0.0

    total_spread = math.sqrt(rw_term * rw_term + trend_term * trend_term)
    quantiles = [sigmoid(predicted_logit + z * total_spread) for z in Z]
    # Guarantee non-decreasing even after independent rounding below.
    rounded = [round4(q) for q in quantiles]
    rounded.sort()
    return rounded


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")

    target = parse_at(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0

    history = case["history"]
    history_at = [parse_at(h["at"]) for h in history]

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        reads = []
        for at, h in zip(history_at, history):
            rate = h["rates"][i]
            if rate is not None:
                reads.append((at, rate))
        q = forecast_trophy(reads, target, horizon_days)
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
