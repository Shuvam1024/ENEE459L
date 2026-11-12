
from __future__ import annotations

from typing import Any, Iterable, Sequence

from schema import LATENCY_PERCENTILES, REL_FLOOR

import json
from pathlib import Path

########################################################
##########   Helper functions         ##################
########################################################


def deviation_sentence(abs_res: dict[str, Any], rel_res: dict[str, Any]) -> str:
    """The two deviation metrics in one line, with the caveat. Provided complete.

    Read it before writing `max_rel_dev`. The clause after the semicolon is the
    honest disclosure and it appears whenever any denominator was floored --
    which, on a classification head, is almost always.
    """
    line = (f"max |a-b| = {abs_res['max_abs']:.6g} at index {abs_res['index']}; "
            f"max relative = {rel_res['max_rel']:.6g}")
    if rel_res["floored"]:
        line += (f" (denominator floored on {rel_res['floored']} of "
                 f"{rel_res['n']} entries; smallest |b| was "
                 f"{rel_res['denom_min']:.3g})")
    return line


def decision_sentence(dm: dict[str, Any], tol: float) -> str:
    """The claim `decision_margin` licenses, or the one it refuses. Provided."""
    if dm["safe"]:
        return (f"no decision flips: the smallest margin over {dm['n']} inputs "
                f"is {dm['min']:.6g} and the deviation is {tol:.6g}")
    return (f"{dm['below_tol']} of {dm['n']} decisions sit inside the "
            f"deviation: smallest margin {dm['min']:.6g} at row {dm['argmin']} "
            f"against a deviation of {tol:.6g}")


class AgreementError(ValueError):
    """A comparison this module cannot make. `engine.py` exits 3."""


def _finite(v: Any, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise AgreementError(f"{where} is {v!r}, which is not a number.")
    f = float(v)
    if f != f or f in (float("inf"), float("-inf")):
        raise AgreementError(
            f"{where} is {f}. A NaN or an infinity in an output array is the "
            "run failing quietly; it is not a value to take a maximum over."
        )
    return f


def _percentile(xs: Sequence[float], q: float) -> float:
    """Linear interpolation between the two nearest ranks. Provided complete.

    The same implementation Labs 04 through 08 use, here for the same reason:
    a lab whose percentile is nearest-rank and a lab whose percentile
    interpolates produce different p95s from the same samples, and a student
    comparing this week's table against last week's should be comparing the
    board and not the arithmetic.
    """
    s = sorted(float(x) for x in xs)
    if not s:
        raise AgreementError("no samples.")
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (float(q) / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


########################################################
##########    Start from here          #################
########################################################


def _mean(xs: list[float]) -> float:
    # Plain left-to-right addition. Python 3.12+ `sum()` compensates float
    # rounding and returns a different last digit than 3.10 on the same input.
    total = 0.0
    for x in xs:
        total += x
    return total / len(xs)


def latency_report(ms: Iterable[float]) -> dict[str, Any]:
    xs: list[float] = []
    for i, v in enumerate(ms):
        x = _finite(v, f"latency sample {i}")
        if x < 0:
            raise AgreementError(
                f"latency sample {i} is {x} ms. A negative duration means the "
                "start and end times came from different clocks."
            )
        xs.append(x)
    if not xs:
        raise AgreementError("no latency samples; a report needs at least one run.")

    n = len(xs)
    report: dict[str, Any] = {"n": n, "mean": _mean(xs), "worst": max(xs)}
    for q in LATENCY_PERCENTILES:
        report[f"p{q}"] = _percentile(xs, q)
    report["jitter"] = report["p95"] - report["p50"]
    return report


def speedup(base_ms: float, other_ms: float) -> dict[str, Any]:
    base = _finite(base_ms, "base_ms")
    other = _finite(other_ms, "other_ms")
    for name, t in (("base_ms", base), ("other_ms", other)):
        if t <= 0:
            raise AgreementError(
                f"{name} is {t} ms. A stage that took no time was skipped or is "
                "below the clock's resolution."
            )
    factor = base / other
    return {"factor": factor, "faster": factor > 1.0, "saved_ms": base - other}


def _paired(a: Sequence[float], b: Sequence[float]) -> tuple[list[float], list[float]]:
    xa = [_finite(v, f"a[{i}]") for i, v in enumerate(a)]
    xb = [_finite(v, f"b[{i}]") for i, v in enumerate(b)]
    if not xa or not xb:
        raise AgreementError("cannot compare an empty output.")
    if len(xa) != len(xb):
        raise AgreementError(
            f"a has {len(xa)} values and b has {len(xb)}: a batching mismatch "
            "or an output that was not flattened."
        )
    return xa, xb


def max_abs_dev(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    xa, xb = _paired(a, b)
    best_i = 0
    best = abs(xa[0] - xb[0])
    for i in range(1, len(xa)):
        d = abs(xa[i] - xb[i])
        if d > best:
            best, best_i = d, i
    return {"max_abs": best, "index": best_i, "a": xa[best_i], "b": xb[best_i],
            "n": len(xa)}


def max_rel_dev(a: Sequence[float], b: Sequence[float],
                floor: float = REL_FLOOR) -> dict[str, Any]:
    xa, xb = _paired(a, b)
    tau = _finite(floor, "floor")
    if tau <= 0:
        raise AgreementError(
            f"floor is {tau}; a non-positive floor removes the guard against "
            "dividing by a near-zero reference."
        )
    floored = 0
    denom_min = float("inf")
    best = -1.0
    best_i = 0
    for i, (u, v) in enumerate(zip(xa, xb)):
        d = abs(v)
        denom_min = min(denom_min, d)
        if d < tau:
            floored += 1
        r = abs(u - v) / max(d, tau)
        if r > best:
            best, best_i = r, i
    return {"max_rel": best, "index": best_i, "floored": floored,
            "denom_min": denom_min, "n": len(xa)}


def _argmax(row: list[float]) -> int:
    best_c = 0
    for c in range(1, len(row)):
        if row[c] > row[best_c]:
            best_c = c
    return best_c


def top1_agreement(rows_a: Sequence[Sequence[float]],
                   rows_b: Sequence[Sequence[float]]) -> dict[str, Any]:
    ra, rb = list(rows_a), list(rows_b)
    if not ra or not rb:
        raise AgreementError("no rows to compare.")
    if len(ra) != len(rb):
        raise AgreementError(f"{len(ra)} rows against {len(rb)}.")

    agree = 0
    first_disagreement = None
    for k, (u, v) in enumerate(zip(ra, rb)):
        uk = [_finite(x, f"rows_a[{k}][{c}]") for c, x in enumerate(u)]
        vk = [_finite(x, f"rows_b[{k}][{c}]") for c, x in enumerate(v)]
        if not uk or not vk:
            raise AgreementError(f"row {k} is empty.")
        if len(uk) != len(vk):
            raise AgreementError(f"row {k} has {len(uk)} classes against {len(vk)}.")
        if _argmax(uk) == _argmax(vk):
            agree += 1
        elif first_disagreement is None:
            first_disagreement = k

    n = len(ra)
    return {"n": n, "agree": agree, "disagree": n - agree, "rate": agree / n,
            "first_disagreement": first_disagreement}


def decision_margin(rows: Sequence[Sequence[float]],
                    tol: float) -> dict[str, Any]:
    rs = list(rows)
    if not rs:
        raise AgreementError("no rows to take a margin over.")
    t = _finite(tol, "tol")
    if t < 0:
        raise AgreementError(f"tol is {t}; a deviation cannot be negative.")

    margins: list[float] = []
    for k, row in enumerate(rs):
        r = [_finite(x, f"rows[{k}][{c}]") for c, x in enumerate(row)]
        if len(r) < 2:
            raise AgreementError(
                f"row {k} has {len(r)} class(es); a margin needs a runner-up."
            )
        s = sorted(r, reverse=True)
        margins.append(s[0] - s[1])

    lowest = min(margins)
    below = sum(1 for m in margins if m < t)
    return {
        "n": len(margins),
        "min": lowest,
        "median": _percentile(margins, 50),
        "mean": _mean(margins),
        "argmin": margins.index(lowest),
        "below_tol": below,
        "safe": below == 0,
    }

#############################################################
###############  Main function starts here ##################
#############################################################

def main(
    input_path: str | Path = "benchmark_input_data.json",
    output_path: str | Path = "benchmark_results.json",
) -> None:
    in_file = Path(input_path)
    out_file = Path(output_path)

    # 1. Load synthetic data from JSON
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file.resolve()}")

    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    timings_base = data["timings_base"]
    timings_optimized = data["timings_optimized"]
    flat_logits_base = data["flat_logits_base"]
    flat_logits_opt = data["flat_logits_opt"]
    batched_scores_base = data["batched_scores_base"]
    batched_scores_opt = data["batched_scores_opt"]

    # 2. Execute metric and report functions
    report_base = latency_report(timings_base)
    report_opt = latency_report(timings_optimized)
    perf_speedup = speedup(base_ms=report_base["p50"], other_ms=report_opt["p50"])

    abs_dev = max_abs_dev(flat_logits_opt, flat_logits_base)
    rel_dev = max_rel_dev(flat_logits_opt, flat_logits_base, floor=REL_FLOOR)

    agreement = top1_agreement(batched_scores_opt, batched_scores_base)
    margin = decision_margin(batched_scores_base, tol=abs_dev["max_abs"])

    # 3. Structure output results
    results: dict[str, Any] = {
        "latency": {
            "base": report_base,
            "optimized": report_opt,
            "speedup_p50": perf_speedup,
        },
        "deviations": {
            "max_abs_dev": abs_dev,
            "max_rel_dev": rel_dev,
            "summary_sentence": deviation_sentence(abs_dev, rel_dev),
        },
        "agreement": {
            "top1_agreement": agreement,
            "decision_margin": margin,
            "decision_sentence": decision_sentence(margin, tol=abs_dev["max_abs"]),
        },
    }

    # 4. Save results to JSON
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 5. Print a console summary
    print(f"Loaded input from: {in_file.resolve()}")
    print(f"Results saved to:  {out_file.resolve()}\n")
    print("--- Summary ---")
    print(results["deviations"]["summary_sentence"])
    print(results["agreement"]["decision_sentence"])
    print(
        f"Speedup (p50): {perf_speedup['factor']:.2f}x "
        f"({perf_speedup['saved_ms']:.2f} ms saved)"
    )

if __name__ == "__main__":
    main()