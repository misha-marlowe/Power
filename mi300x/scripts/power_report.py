#!/usr/bin/env python3
"""Dependency-free text reports for AMD vLLM power benchmark results."""

from __future__ import annotations

import fnmatch
import html
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"
VIEW_ORDER = ["overview", "insights", "efficiency", "gpus", "latency", "timeline", "anomalies"]


@dataclass
class RunResult:
    path: Path
    summary: dict[str, Any]
    samples: list[dict[str, Any]]

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def workload(self) -> dict[str, Any]:
        return self.summary.get("workload", {})

    @property
    def vllm(self) -> dict[str, Any]:
        return self.summary.get("vllm", {})

    @property
    def efficiency(self) -> dict[str, Any]:
        return self.summary.get("efficiency", {})

    @property
    def window(self) -> dict[str, Any]:
        return self.summary.get("benchmark_window", {})


def number(value: Any, decimals: int = 1, suffix: str = "", missing: str = "-") -> str:
    if value is None or isinstance(value, bool):
        return missing
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return missing
    return f"{numeric:,.{decimals}f}{suffix}"


def percent(value: Any, decimals: int = 1) -> str:
    if value is None:
        return "-"
    return number(float(value) * 100, decimals, "%")


def table(headers: list[str], rows: Iterable[Iterable[Any]], right: set[int] | None = None) -> str:
    rendered = [[str(cell) for cell in row] for row in rows]
    all_rows = [headers, *rendered]
    widths = [max(len(row[index]) for row in all_rows) for index in range(len(headers))]
    right = right or set()

    def format_row(row: list[str]) -> str:
        cells = []
        for index, cell in enumerate(row):
            cells.append(cell.rjust(widths[index]) if index in right else cell.ljust(widths[index]))
        return "  ".join(cells)

    separator = ["-" * width for width in widths]
    return "\n".join([format_row(headers), format_row(separator), *(format_row(row) for row in rendered)])


def heading(title: str) -> str:
    return f"\n{title}\n{'=' * len(title)}"


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "monotonic_s" in row and "gpu_w" in row:
                rows.append(row)
    return rows


def result_directories(target: Path) -> list[Path]:
    if target.is_file() and target.name == "summary.json":
        return [target.parent]
    if target.is_dir() and (target / "summary.json").is_file():
        return [target]
    if target.is_dir():
        return sorted(path.parent for path in target.rglob("summary.json"))
    return []


def latest_sweep(results_root: Path = RESULTS_ROOT) -> Path:
    candidates = [
        (path.stat().st_mtime, path.parent)
        for path in results_root.rglob("index.json")
        if path.parent.is_dir() and list(path.parent.glob("*/summary.json"))
    ]
    if not candidates:
        raise RuntimeError(f"no completed sweeps found under {results_root}")
    return max(candidates, key=lambda item: item[0])[1]


def load_runs(path: str | None, run_filter: str | None = None) -> tuple[Path, list[RunResult]]:
    target = Path(path).expanduser().resolve() if path else latest_sweep()
    directories = result_directories(target)
    if run_filter:
        directories = [
            directory
            for directory in directories
            if fnmatch.fnmatch(directory.name, run_filter) or run_filter in directory.name
        ]
    if not directories:
        raise RuntimeError(f"no matching summary.json files under {target}")
    runs = []
    for directory in directories:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        runs.append(RunResult(directory, summary, load_json_lines(directory / "power.jsonl")))
    runs.sort(
        key=lambda run: (
            str(run.workload.get("scenario", "")),
            int(run.workload.get("input_len", 0)),
            int(run.workload.get("output_len", 0)),
            int(run.workload.get("concurrency", 0)),
        )
    )
    return target, runs


def load_run_collection(
    paths: list[str] | None, run_filter: str | None = None
) -> tuple[list[Path], list[RunResult]]:
    requested = paths or [None]
    targets: list[Path] = []
    runs: list[RunResult] = []
    seen: set[Path] = set()
    for path in requested:
        target, loaded = load_runs(path, run_filter)
        targets.append(target)
        for run in loaded:
            resolved = run.path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                runs.append(run)
    runs.sort(
        key=lambda run: (
            str(run.workload.get("scenario", "")),
            int(run.workload.get("input_len", 0)),
            int(run.workload.get("output_len", 0)),
            int(run.workload.get("concurrency", 0)),
        )
    )
    return targets, runs


def benchmark_bounds(run: RunResult) -> tuple[float, float] | None:
    if not run.samples:
        return None
    start = run.window.get("start_monotonic_s")
    end = run.window.get("end_monotonic_s")
    if start is not None and end is not None and float(end) > float(start):
        return float(start), float(end)
    sample_end = float(run.samples[-1]["monotonic_s"])
    duration = float(run.vllm.get("duration", run.window.get("requested_duration_s", 0)) or 0)
    if duration <= 0:
        return None
    return sample_end - duration, sample_end


def benchmark_samples(run: RunResult) -> list[dict[str, Any]]:
    bounds = benchmark_bounds(run)
    if bounds is None:
        return []
    start, end = bounds
    return [sample for sample in run.samples if start <= float(sample["monotonic_s"]) <= end]


def maximum_total_power(run: RunResult) -> float | None:
    stored = run.window.get("maximum_gpu_power_w")
    if stored is not None:
        return float(stored)
    samples = benchmark_samples(run)
    return max((float(sample.get("total_w", 0)) for sample in samples), default=None)


def maximum_gpu_power(run: RunResult, gpu: str) -> float | None:
    stored = run.window.get("per_gpu", {}).get(gpu, {}).get("maximum_power_w")
    if stored is not None:
        return float(stored)
    samples = benchmark_samples(run)
    return max(
        (float(sample.get("gpu_w", {}).get(gpu, 0)) for sample in samples if gpu in sample.get("gpu_w", {})),
        default=None,
    )


def overview(runs: list[RunResult]) -> str:
    rows = []
    for run in runs:
        rows.append(
            [
                run.name,
                f"{run.workload.get('input_len', '-')}/{run.workload.get('output_len', '-')}",
                run.workload.get("concurrency", "-"),
                number(run.vllm.get("duration"), 2, "s"),
                number(run.vllm.get("output_throughput"), 1),
                number(run.window.get("average_gpu_power_w"), 1),
                number(maximum_total_power(run), 1),
                percent(
                    run.efficiency.get("average_power_fraction_of_tdp_proxy")
                    if run.efficiency.get("average_power_fraction_of_tdp_proxy") is not None
                    else run.efficiency.get("average_power_fraction_of_reported_cap")
                ),
            ]
        )
    result = heading("Overview") + "\n" + table(
        ["Run", "ISL/OSL", "Conc", "Time", "Out tok/s", "Avg W", "Max sampled W", "TDP %"],
        rows,
        right={2, 3, 4, 5, 6, 7},
    )
    failures = []
    for run in runs:
        failed = int(run.vllm.get("failed", 0) or 0)
        completed = int(run.vllm.get("completed", 0) or 0)
        expected = int(run.workload.get("num_prompts", completed) or completed)
        if failed or completed != expected:
            failures.append(f"- {run.name}: {failed} failed; completed {completed}/{expected}.")
    if failures:
        result += "\n\nFailures:\n" + "\n".join(failures)
    return result


def efficiency_view(runs: list[RunResult]) -> str:
    ranked = sorted(
        runs,
        key=lambda run: float(run.efficiency.get("gross_j_per_output_token", math.inf) or math.inf),
    )
    rows = []
    for rank, run in enumerate(ranked, 1):
        rows.append(
            [
                rank,
                run.name,
                run.workload.get("concurrency", "-"),
                number(run.efficiency.get("gross_j_per_output_token"), 4),
                number(run.efficiency.get("above_idle_j_per_output_token"), 4),
                number(run.efficiency.get("gross_output_tokens_per_kwh"), 0),
                number(run.efficiency.get("output_tokens_per_s_per_gpu"), 1),
            ]
        )
    return heading("Energy efficiency") + "\n" + table(
        ["#", "Run", "Conc", "Gross J/tok", ">idle J/tok", "tok/kWh", "tok/s/GPU"],
        rows,
        right={0, 2, 3, 4, 5, 6},
    )


def gpu_view(runs: list[RunResult]) -> str:
    sections = [heading("Per-GPU balance")]
    for run in runs:
        per_gpu = run.window.get("per_gpu", {})
        idle = run.summary.get("idle_baseline", {}).get("per_gpu_w", {})
        cap_total = float(run.summary.get("reported_power_cap_w") or 0)
        tdp_total = float(run.summary.get("tdp_proxy_w") or cap_total)
        tdp_each = tdp_total / len(per_gpu) if per_gpu and tdp_total else 0
        average_values = [float(item.get("average_power_w", 0)) for item in per_gpu.values()]
        mean = statistics.fmean(average_values) if average_values else 0
        imbalance = (max(average_values) - min(average_values)) / mean if mean and len(average_values) > 1 else 0
        rows = []
        for gpu, metrics in sorted(per_gpu.items(), key=lambda item: int(item[0])):
            avg = float(metrics.get("average_power_w", 0))
            idle_w = float(idle.get(gpu, 0))
            rows.append(
                [
                    gpu,
                    number(idle_w, 1),
                    number(avg, 1),
                    number(maximum_gpu_power(run, gpu), 1),
                    number(avg - idle_w, 1),
                    number(metrics.get("energy_j"), 1),
                    percent(avg / tdp_each if tdp_each else None),
                ]
            )
        sections.append(f"\n{run.name}  (average-power spread {percent(imbalance)})")
        sections.append(
            table(
                ["GPU", "Idle W", "Avg W", "Max sampled W", "Above idle W", "Energy J", "TDP %"],
                rows,
                right={0, 1, 2, 3, 4, 5, 6},
            )
        )
    return "\n".join(sections)


def latency_view(runs: list[RunResult]) -> str:
    rows = []
    for run in runs:
        rows.append(
            [
                run.name,
                run.workload.get("concurrency", "-"),
                number(run.vllm.get("p50_ttft_ms"), 1),
                number(run.vllm.get("p90_ttft_ms"), 1),
                number(run.vllm.get("p99_ttft_ms"), 1),
                number(run.vllm.get("p50_tpot_ms"), 2),
                number(run.vllm.get("p90_tpot_ms"), 2),
                number(run.vllm.get("p99_e2el_ms"), 1),
            ]
        )
    return heading("Latency") + "\n" + table(
        ["Run", "Conc", "TTFT p50", "TTFT p90", "TTFT p99", "TPOT p50", "TPOT p90", "E2E p99"],
        rows,
        right={1, 2, 3, 4, 5, 6, 7},
    ) + "\nAll latency values are milliseconds."


def binned_values(samples: list[dict[str, Any]], key: str, segments: int) -> list[float]:
    if not samples:
        return []
    segments = max(1, min(segments, len(samples)))
    start = float(samples[0]["monotonic_s"])
    end = float(samples[-1]["monotonic_s"])
    if end <= start:
        return [float(samples[-1].get(key, 0))]
    bins: list[list[float]] = [[] for _ in range(segments)]
    for sample in samples:
        position = (float(sample["monotonic_s"]) - start) / (end - start)
        index = min(segments - 1, int(position * segments))
        if key == "total_w":
            value = float(sample.get("total_w", 0))
        else:
            value = float(sample.get("gpu_w", {}).get(key, 0))
        bins[index].append(value)
    return [statistics.fmean(values) if values else math.nan for values in bins]


def sparkline(values: list[float]) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return "(no samples)"
    low, high = min(finite), max(finite)
    if high == low:
        return blocks[3] * len(values)
    chars = []
    for value in values:
        if not math.isfinite(value):
            chars.append(" ")
        else:
            index = round((value - low) / (high - low) * (len(blocks) - 1))
            chars.append(blocks[index])
    return "".join(chars)


def timeline_view(runs: list[RunResult], segments: int) -> str:
    sections = [heading("Power timelines")]
    for run in runs:
        samples = run.samples
        if not samples:
            sections.append(f"\n{run.name}: no raw power samples")
            continue
        elapsed = float(samples[-1]["monotonic_s"]) - float(samples[0]["monotonic_s"])
        totals = [float(sample["total_w"]) for sample in samples]
        sections.append(
            f"\n{run.name}: {len(samples)} samples over {elapsed:.2f}s; "
            f"raw total {min(totals):.0f}/{statistics.fmean(totals):.0f}/{max(totals):.0f} W min/mean/max"
        )
        total_bins = binned_values(samples, "total_w", segments)
        sections.append(f"  total  {sparkline(total_bins)}  {number(min(total_bins), 0)}-{number(max(total_bins), 0)} W")
        gpu_ids = sorted(samples[0].get("gpu_w", {}), key=int)
        for gpu in gpu_ids:
            values = binned_values(samples, gpu, segments)
            sections.append(f"  GPU {gpu:<3} {sparkline(values)}  {number(min(values), 0)}-{number(max(values), 0)} W")
        sections.append(
            f"  benchmark window: {number(run.vllm.get('duration'), 2, 's')} at "
            f"{number(run.window.get('average_gpu_power_w'), 1, ' W')} average total GPU power"
        )
    return "\n".join(sections)


def time_weighted_power_samples(run: RunResult) -> list[tuple[float, float]]:
    """Return (aggregate watts, represented seconds) over the benchmark window."""
    bounds = benchmark_bounds(run)
    if bounds is None or len(run.samples) < 2:
        return []
    start, end = bounds
    points = sorted(
        (float(sample["monotonic_s"]), float(sample.get("total_w", 0)))
        for sample in run.samples
    )
    weighted: list[tuple[float, float]] = []
    for index, (timestamp, watts) in enumerate(points):
        left = timestamp if index == 0 else 0.5 * (points[index - 1][0] + timestamp)
        right = timestamp if index == len(points) - 1 else 0.5 * (timestamp + points[index + 1][0])
        duration = min(end, right) - max(start, left)
        if duration > 0:
            weighted.append((watts, duration))
    return weighted


def empirical_cdf(weighted: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return (value, cumulative fraction) for a weighted empirical CDF."""
    if not weighted:
        return []
    totals: dict[float, float] = {}
    for value, weight in weighted:
        if weight > 0 and math.isfinite(value):
            totals[value] = totals.get(value, 0.0) + weight
    total_weight = sum(totals.values())
    if not total_weight:
        return []
    cumulative = 0.0
    result = []
    for value in sorted(totals):
        cumulative += totals[value]
        result.append((value, cumulative / total_weight))
    return result


def write_cdf_svg(output: Path, runs: list[RunResult]) -> None:
    scenarios = ["interactive", "document-qa"]
    allowed_concurrency = {1, 4, 16, 64, 128, 256}
    panels: dict[str, list[tuple[int, list[tuple[float, float]]]]] = {}
    for scenario in scenarios:
        candidates = [
            run
            for run in runs
            if run.workload.get("scenario") == scenario
            and int(run.workload.get("concurrency", 0)) in allowed_concurrency
            and not any(level == "ERROR" for level, _ in run_issues(run))
        ]
        by_concurrency: dict[int, RunResult] = {}
        for run in candidates:
            by_concurrency[int(run.workload["concurrency"])] = run
        if 256 in by_concurrency:
            by_concurrency.pop(128, None)
        selected = sorted(by_concurrency.items())
        if len(selected) > 5:
            selected = selected[-5:]
        panels[scenario] = [
            (concurrency, empirical_cdf(time_weighted_power_samples(run)))
            for concurrency, run in selected
        ]
    all_points = [point for curves in panels.values() for _, curve in curves for point in curve]
    if not all_points:
        raise RuntimeError("no successful interactive or document-QA samples available for CDF")

    min_kw = min(point[0] for point in all_points) / 1000
    max_kw = max(point[0] for point in all_points) / 1000
    padding = max(0.1, (max_kw - min_kw) * 0.04)
    x_min = max(0.0, min_kw - padding)
    x_max = max_kw + padding
    width, height = 1180, 560
    panel_width, plot_height = 500, 360
    panel_lefts = [90, 650]
    plot_top, plot_bottom = 70, 430
    colors = {1: "#4c78a8", 4: "#f58518", 16: "#54a24b", 64: "#e45756", 128: "#b279a2", 256: "#b279a2"}

    def sx(value_w: float, left: float) -> float:
        return left + ((value_w / 1000 - x_min) / (x_max - x_min)) * panel_width

    def sy(fraction: float) -> float:
        return plot_bottom - fraction * plot_height

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="560" viewBox="0 0 1180 560">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#222}.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.curve{fill:none;stroke-width:2.25;stroke-linejoin:round}.tick{font-size:12px}.title{font-size:18px;font-weight:600}.panel{font-size:15px;font-weight:600}.legend{font-size:12px}</style>',
        '<text x="590" y="28" text-anchor="middle" class="title">Time-weighted GPU package-power CDF</text>',
    ]
    for panel_index, scenario in enumerate(scenarios):
        left = panel_lefts[panel_index]
        lines.append(f'<text x="{left + panel_width / 2:.1f}" y="55" text-anchor="middle" class="panel">{html.escape(scenario)}</text>')
        for pct in (0, 25, 50, 75, 100):
            y = sy(pct / 100)
            lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + panel_width}" y2="{y:.1f}" class="grid"/>')
            lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{pct}%</text>')
        for tick in range(6):
            fraction = tick / 5
            x = left + fraction * panel_width
            kw = x_min + fraction * (x_max - x_min)
            lines.append(f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" y2="{plot_bottom}" class="grid"/>')
            lines.append(f'<text x="{x:.1f}" y="448" text-anchor="middle" class="tick">{kw:.1f}</text>')
        lines.extend([
            f'<line x1="{left}" y1="{plot_top}" x2="{left}" y2="{plot_bottom}" class="axis"/>',
            f'<line x1="{left}" y1="{plot_bottom}" x2="{left + panel_width}" y2="{plot_bottom}" class="axis"/>',
            f'<text x="{left + panel_width / 2:.1f}" y="476" text-anchor="middle" class="tick">Aggregate power (kW), 8 GPUs</text>',
        ])
        for curve_index, (concurrency, curve) in enumerate(panels[scenario]):
            if not curve:
                continue
            path = [f"M {sx(curve[0][0], left):.2f} {sy(0):.2f}"]
            previous_fraction = 0.0
            for watts, fraction in curve:
                x = sx(watts, left)
                path.append(f"L {x:.2f} {sy(previous_fraction):.2f} L {x:.2f} {sy(fraction):.2f}")
                previous_fraction = fraction
            color = colors[concurrency]
            lines.append(f'<path d="{" ".join(path)}" class="curve" stroke="{color}"/>')
            legend_x = left + 12 + (curve_index % 3) * 150
            legend_y = 500 + (curve_index // 3) * 22
            lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
            lines.append(f'<text x="{legend_x + 31}" y="{legend_y + 4}" class="legend">concurrency {concurrency}</text>')
    lines.append('<text x="22" y="250" transform="rotate(-90 22 250)" text-anchor="middle" class="tick">Time at or below power</text>')
    lines.append('</svg>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_issues(run: RunResult) -> list[tuple[str, str]]:
    issues: list[tuple[str, str]] = []
    if int(run.summary.get("return_code", 0) or 0) != 0:
        issues.append(("ERROR", "benchmark command returned non-zero"))
    failed = int(run.vllm.get("failed", 0) or 0)
    if failed:
        issues.append(("ERROR", f"{failed} requests failed"))
    completed = int(run.vllm.get("completed", 0) or 0)
    expected = int(run.workload.get("num_prompts", completed) or completed)
    if completed != expected:
        issues.append(("ERROR", f"completed {completed}/{expected} prompts"))
    expected_tokens = completed * int(run.workload.get("output_len", 0) or 0)
    actual_tokens = int(run.vllm.get("total_output_tokens", 0) or 0)
    if expected_tokens and actual_tokens != expected_tokens:
        issues.append(("WARN", f"generated {actual_tokens}/{expected_tokens} expected output tokens"))
    errors = run.summary.get("sample_errors", [])
    if errors:
        issues.append(("ERROR", f"{len(errors)} power-sampling errors"))
    requested = float(run.window.get("requested_duration_s", 0) or 0)
    covered = float(run.window.get("covered_s", 0) or 0)
    if requested and covered / requested < 0.95:
        issues.append(("ERROR", f"power samples cover only {covered / requested:.1%} of benchmark"))
    duration = float(run.vllm.get("duration", 0) or 0)
    target_duration = float(run.workload.get("target_duration_s", 0) or 0)
    if target_duration and duration < target_duration * 0.90:
        issues.append(("WARN", f"measured {duration:.2f}s versus {target_duration:.0f}s target"))
    if 0 < duration < 30:
        issues.append(("WARN", f"short {duration:.2f}s run; use longer workloads for stable power"))
    idle_count = int(run.summary.get("idle_baseline", {}).get("sample_count", 0) or 0)
    if idle_count < 5:
        issues.append(("WARN", f"idle baseline has only {idle_count} samples"))
    per_gpu = run.window.get("per_gpu", {})
    powers = [float(item.get("average_power_w", 0)) for item in per_gpu.values()]
    if len(powers) > 1 and statistics.fmean(powers):
        spread = (max(powers) - min(powers)) / statistics.fmean(powers)
        if spread > 0.10:
            issues.append(("WARN", f"per-GPU average-power spread is {spread:.1%}"))
    cap_fraction = run.efficiency.get("average_power_fraction_of_reported_cap")
    if cap_fraction is not None and float(cap_fraction) > 1.02:
        issues.append(("ERROR", f"average power is {float(cap_fraction):.1%} of reported cap"))
    dynamic = run.efficiency.get("above_idle_gpu_energy_j")
    if dynamic is not None and float(dynamic) < 0:
        issues.append(("WARN", "above-idle energy is negative; idle baseline is not representative"))
    if len(run.samples) >= 2:
        gaps = [
            float(right["monotonic_s"]) - float(left["monotonic_s"])
            for left, right in zip(run.samples, run.samples[1:])
        ]
        interval = float(run.summary.get("sample_interval_s", 0.5) or 0.5)
        if max(gaps) > max(2.0, interval * 4):
            issues.append(("WARN", f"largest telemetry gap is {max(gaps):.2f}s"))
    return issues


def anomalies_view(runs: list[RunResult]) -> str:
    rows = []
    for run in runs:
        issues = run_issues(run)
        if not issues:
            rows.append([run.name, "OK", "No automated issues detected"])
        else:
            for index, (level, message) in enumerate(issues):
                rows.append([run.name if index == 0 else "", level, message])
    return heading("Quality and anomaly checks") + "\n" + table(["Run", "Level", "Finding"], rows)


def insights_view(runs: list[RunResult]) -> str:
    lines = [heading("Interpretation")]
    valid = [run for run in runs if run.efficiency.get("gross_j_per_output_token")]
    if not valid:
        lines.append("No successful token/energy results are available.")
        return "\n".join(lines)
    efficient = min(valid, key=lambda run: float(run.efficiency["gross_j_per_output_token"]))
    fastest = max(valid, key=lambda run: float(run.vllm.get("output_throughput", 0)))
    lines.append(
        f"- Best gross energy efficiency: {efficient.name} at "
        f"{number(efficient.efficiency['gross_j_per_output_token'], 3)} J/output-token."
    )
    lines.append(
        f"- Highest output throughput: {fastest.name} at "
        f"{number(fastest.vllm.get('output_throughput'), 1)} output tokens/s."
    )
    groups: dict[tuple[Any, ...], list[RunResult]] = {}
    for run in valid:
        key = (
            run.workload.get("scenario"),
            run.workload.get("input_len"),
            run.workload.get("output_len"),
            tuple(run.summary.get("gpu_ids", [])),
        )
        groups.setdefault(key, []).append(run)
    for group in groups.values():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda run: int(run.workload.get("concurrency", 0)))
        base, last = ordered[0], ordered[-1]
        base_throughput = float(base.vllm.get("output_throughput", 0) or 0)
        base_power = float(base.window.get("average_gpu_power_w", 0) or 0)
        base_energy = float(base.efficiency.get("gross_j_per_output_token", 0) or 0)
        if base_throughput and base_power and base_energy:
            throughput_ratio = float(last.vllm.get("output_throughput", 0)) / base_throughput
            power_ratio = float(last.window.get("average_gpu_power_w", 0)) / base_power
            energy_change = float(last.efficiency.get("gross_j_per_output_token", 0)) / base_energy - 1
            lines.append(
                f"- {base.workload.get('scenario')} concurrency {base.workload.get('concurrency')}→"
                f"{last.workload.get('concurrency')}: throughput {throughput_ratio:.2f}×, "
                f"average power {power_ratio:.2f}×, J/token {energy_change:+.1%}."
            )
    if any(float(run.vllm.get("duration", 0) or 0) < 30 for run in valid):
        lines.append("- At least one run is under 30 seconds; treat its power efficiency as smoke-test evidence, not a stable benchmark.")
    lines.append("- Power values are selected-GPU socket/package measurements, not whole-node wall power.")
    return "\n".join(lines)


def parse_views(value: str) -> list[str]:
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not requested:
        requested = ["overview", "insights", "anomalies"]
    if "all" in requested:
        return VIEW_ORDER
    unknown = set(requested) - set(VIEW_ORDER)
    if unknown:
        raise RuntimeError(f"unknown view(s): {', '.join(sorted(unknown))}; choose {', '.join(VIEW_ORDER)}, or all")
    return requested


def generate_report(target: Path | list[Path], runs: list[RunResult], views: list[str], segments: int) -> str:
    first = runs[0]
    model = first.vllm.get("model_id", "unknown")
    scope = first.summary.get("power_scope", "GPU power")
    sources = target if isinstance(target, list) else [target]
    parts = [
        "AMD vLLM power analysis",
        *(f"Source: {source}" for source in sources),
        f"Model: {model}",
        f"Runs: {len(runs)}",
        f"Scope: {scope}",
        "TDP % source: reported Max Graphics Package Power (proxy; not an independent TDP reading)",
    ]
    renderers = {
        "overview": lambda: overview(runs),
        "insights": lambda: insights_view(runs),
        "efficiency": lambda: efficiency_view(runs),
        "gpus": lambda: gpu_view(runs),
        "latency": lambda: latency_view(runs),
        "timeline": lambda: timeline_view(runs, segments),
        "anomalies": lambda: anomalies_view(runs),
    }
    parts.extend(renderers[view]() for view in views)
    return "\n".join(parts).strip() + "\n"


def run_analysis(args: Any) -> int:
    targets, runs = load_run_collection(args.paths, args.run)
    views = parse_views(args.view)
    report = generate_report(targets, runs, views, args.segments)
    print(report, end="")
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(f"Saved: {output}")
    if args.cdf_output:
        cdf_output = Path(args.cdf_output).expanduser().resolve()
        write_cdf_svg(cdf_output, runs)
        print(f"Saved CDF: {cdf_output}")
    return 0
