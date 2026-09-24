from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from bench import Bench, measured, read_first, read_text, unknown

# A sample is still warm-up while it exceeds the settled rate by this fraction.
WARMUP_TOL = 0.5

# How many samples must sit strictly above a quantile before that quantile is an
# estimate rather than "the biggest number we saw, wearing a hat".
MIN_SAMPLES_ABOVE = 5

# Percentiles the record carries, in the order the schema lists them.
PERCENTILES = (50, 95, 99)

# The widest gap between neighbouring measurements, as a multiple of the typical
# gap, beyond which the sample is treated as coming from two populations.
MULTIMODAL_GAP_RATIO = 20.0

# Neither side of that gap is a mode unless it holds at least this fraction.
MIN_MODE_FRACTION = 0.10

# Below this many retained samples, modality is not a question worth answering.
MIN_SAMPLES_FOR_MODALITY = 20

# How far the last third of a run may drift from the first third, relative to
# the run's own median, before the run is not one population either.
STATIONARITY_TOL = 0.10
MIN_SAMPLES_FOR_STATIONARITY = 12

THERMAL_ZONES = "sys/devices/virtual/thermal"

POWER_RAIL_CANDIDATES = (
    "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
    "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input",
    "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
)

GPU_LOAD_CANDIDATES = (
    "sys/devices/platform/gpu.0/load",
    "sys/devices/gpu.0/load",
)

CPUFREQ_MIN = "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
CPUFREQ_MAX = "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"


# ===========================================================================
# 1. The loop
# ===========================================================================
def run_timed_iterations(bench: Bench, repeats: int = 100) -> list[float]:
    """Time one synchronized pass of the workload, repeats times, in milliseconds."""
    bench.workload.synchronize()
    elapsed_ms: list[float] = []
    for _ in range(repeats):
        start = bench.clock()
        bench.workload.run()
        bench.workload.synchronize()
        end = bench.clock()
        elapsed_ms.append((end - start) / 1_000_000.0)
    return elapsed_ms


def find_warmup_boundary(samples: list[float]) -> dict[str, Any]:
    """Count the leading prefix still above the settled tail's median."""
    src = "leading prefix above (1 + 0.5) x median of the run's second half"
    if len(samples) < 4:
        return unknown(src, "too few samples to separate warm-up from a settled tail")

    settled = statistics.median(samples[len(samples) // 2 :])
    if settled <= 0:
        return unknown(src, "settled median of the second half is not a positive rate")

    threshold = settled * (1 + WARMUP_TOL)
    discarded = 0
    for sample in samples:
        if sample > threshold:
            discarded += 1
        else:
            break

    return measured(
        discarded,
        src,
        settled_rate_ms=round(settled, 4),
        threshold_ms=round(threshold, 4),
        tolerance=WARMUP_TOL,
        retained=len(samples) - discarded,
    )


def _percentile(sorted_samples: list[float], q: float) -> float:
    n = len(sorted_samples)
    h = (n - 1) * q / 100.0
    i = math.floor(h)
    if i + 1 >= n:
        return sorted_samples[i]
    return sorted_samples[i] + (h - i) * (sorted_samples[i + 1] - sorted_samples[i])


def summarize(samples: list[float]) -> dict[str, Any]:
    """Mean, sample stdev, extrema, and linearly interpolated percentiles."""
    keys = ("mean", "std", "min", "max", "p50", "p95", "p99")
    if not samples:
        return {"n": 0, **{k: None for k in keys}}

    ordered = sorted(samples)
    n = len(ordered)
    std = statistics.stdev(ordered) if n > 2 else 0.0
    out: dict[str, Any] = {
        "n": n,
        "mean": round(statistics.fmean(ordered), 4),
        "std": round(std, 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
    }
    for q in PERCENTILES:
        out[f"p{q}"] = round(_percentile(ordered, q), 4)
    return out


def is_multimodal(samples: list[float]) -> dict[str, Any]:
    """True when a trimmed sample splits across a gap much wider than typical."""
    src = (
        f"widest trimmed gap >= {MULTIMODAL_GAP_RATIO}x the median gap, "
        f"with >= {int(MIN_MODE_FRACTION * 100)}% of samples on each side"
    )
    n = len(samples)
    if n < MIN_SAMPLES_FOR_MODALITY:
        return unknown(src, "not enough samples to test for a multimodal split")

    ordered = sorted(samples)
    trim = n * 5 // 100
    trimmed = ordered[trim : n - trim] if trim else ordered
    if len(trimmed) < 2:
        return unknown(src, "not enough samples remain after trimming outliers")

    gaps = [trimmed[i + 1] - trimmed[i] for i in range(len(trimmed) - 1)]
    typical = statistics.median(gaps)
    if typical <= 0:
        return unknown(src, "timer resolution is too coarse to measure gaps between samples")

    widest = max(gaps)
    split_at = gaps.index(widest)
    left = [x for x in ordered if x <= trimmed[split_at]]
    right = [x for x in ordered if x >= trimmed[split_at + 1]]
    ratio = widest / typical
    value = (
        ratio >= MULTIMODAL_GAP_RATIO
        and len(left) >= MIN_MODE_FRACTION * n
        and len(right) >= MIN_MODE_FRACTION * n
    )
    return measured(
        value,
        src,
        gap_ratio=round(ratio, 2),
        widest_gap_ms=round(widest, 3),
        typical_gap_ms=round(typical, 5),
        modes=[
            {
                "n": len(left),
                "share": round(len(left) / n, 2),
                "median_ms": round(statistics.median(left), 4),
            },
            {
                "n": len(right),
                "share": round(len(right) / n, 2),
                "median_ms": round(statistics.median(right), 4),
            },
        ],
    )

# ===========================================================================
# 7. The clock ceiling the run happened under
# ===========================================================================


def probe_power_state(bench: Bench) -> dict[str, Any]:
    """Active nvpmodel mode, and whether cpu0 min/max frequencies are pinned."""
    src = "nvpmodel -q"
    result = bench.runner(["nvpmodel", "-q"])
    if not result.ok:
        return unknown(src, result.error or "nvpmodel -q could not be run")
    if result.returncode != 0:
        return unknown(
            src,
            "nvpmodel -q returned a non-zero exit code — missing sudo permissions, or nvpmodel is absent",
        )

    mode_name = None
    mode_index = None
    lines = result.stdout.splitlines()
    for i, line in enumerate(lines):
        if "NV Power Mode:" in line:
            mode_name = line.split("NV Power Mode:", 1)[1].strip()
            if i + 1 < len(lines):
                try:
                    mode_index = int(lines[i + 1].strip())
                except ValueError:
                    mode_index = None
            break
    if not mode_name or mode_index is None:
        return unknown(src, "nvpmodel -q output did not contain a power mode name and id")

    min_raw = read_text(bench.telemetry, CPUFREQ_MIN)
    max_raw = read_text(bench.telemetry, CPUFREQ_MAX)
    freq_src = f"{CPUFREQ_MIN} vs {CPUFREQ_MAX}"
    if min_raw is None or max_raw is None:
        jetson_clocks = None
        jetson_clocks_source: dict[str, Any] = unknown(
            freq_src, "scaling_min_freq or scaling_max_freq unreadable"
        )
    else:
        min_hz, max_hz = int(min_raw), int(max_raw)
        jetson_clocks = min_hz == max_hz
        jetson_clocks_source = measured(
            f"scaling_min_freq={min_hz}, scaling_max_freq={max_hz}",
            freq_src,
        )

    return measured(
        mode_name,
        src,
        mode_index=mode_index,
        jetson_clocks=jetson_clocks,
        jetson_clocks_source=jetson_clocks_source,
    )


def probe_telemetry(bench: Bench) -> dict[str, Any]:
    """Peak thermal-zone temperature, board power, and GPU load."""
    temp_src = f"{THERMAL_ZONES}/*/temp"
    zones: list[tuple[str, float]] = []
    thermal_root = Path(bench.telemetry) / THERMAL_ZONES
    if thermal_root.exists():
        for zone_dir in sorted(thermal_root.glob("thermal_zone*")):
            if not zone_dir.is_dir():
                continue
            try:
                rel = zone_dir.relative_to(Path(bench.telemetry))
                raw = read_text(bench.telemetry, str(rel / "temp"))
                ztype = read_text(bench.telemetry, str(rel / "type"))
            except (OSError, UnicodeDecodeError, TypeError, ValueError):
                continue
            if raw is None:
                continue
            try:
                milli = int(raw)
            except ValueError:
                continue
            if milli <= -1000:
                continue
            zones.append((ztype or zone_dir.name, milli / 1000.0))
    if not zones:
        temperature: dict[str, Any] = unknown(temp_src, "no readable thermal zones")
    else:
        zone, peak = max(zones, key=lambda z: z[1])
        temperature = measured(
            peak,
            temp_src,
            zone=zone,
            zones_read=len(zones),
        )

    power_src = " | ".join(POWER_RAIL_CANDIDATES)
    power_hit = read_first(bench.telemetry, POWER_RAIL_CANDIDATES)
    if power_hit is None:
        power: dict[str, Any] = unknown(
            power_src, "none of the documented INA3221 rail paths could be read"
        )
    else:
        path, text = power_hit
        try:
            power = measured(int(text), path)
        except ValueError:
            power = unknown(path, "INA3221 rail path was present but not an integer milliwatt reading")

    gpu_src = " | ".join(GPU_LOAD_CANDIDATES)
    gpu_hit = read_first(bench.telemetry, GPU_LOAD_CANDIDATES)
    if gpu_hit is None:
        gpu: dict[str, Any] = unknown(gpu_src, "no GPU load file was found")
    else:
        path, text = gpu_hit
        try:
            gpu = measured(int(text) / 10.0, path, units="per-mille / 10")
        except ValueError:
            gpu = unknown(path, "GPU load file was present but not an integer per-mille reading")

    return {
        "temperature_c": temperature,
        "power_mw": power,
        "gpu_utilization_percent": gpu,
    }

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
    # env = Bench.real()
    # out = find_warmup_boundary(samples)
    # print(out)

# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    env = Bench.real()

    # get your samples
    samples = run_timed_iterations(env, repeats=100)

    # testing measurments and probes
    report = {
        "warmup_boundary": find_warmup_boundary(samples),
        "summarize_setup": summarize(samples),
        "is_multimodal": is_multimodal(samples),
        "probe_power_state": probe_power_state(env),
        "probe_telemetry": probe_telemetry(env),
    }

    # save samples
    path = "samples_analysis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)

    # save report
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)