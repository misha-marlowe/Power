#!/usr/bin/env python3
"""Portable AMD vLLM power benchmark harness.

The only host dependency is Python 3.11+ plus amd-smi or rocm-smi. vLLM runs
inside a container so the benchmark client and server stay version matched.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "smoke.toml"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    config["_path"] = str(config_path)
    return config


def expand_path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser().resolve()


def model_result_key(model: str) -> str:
    """Return a readable, filesystem-safe directory name for a model ID/path."""
    normalized = model.strip().strip("/").replace("/", "--")
    key = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized).strip("._-")
    return key or "model"


def run_capture(command: list[str], timeout: float = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def parse_json_output(result: subprocess.CompletedProcess[str]) -> Any:
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or f"command exited {result.returncode}")
    return json.loads(result.stdout)


def parse_gpu_ids(args: argparse.Namespace, config: dict[str, Any]) -> list[int]:
    if getattr(args, "gpu_ids", None):
        try:
            ids = [int(item.strip()) for item in args.gpu_ids.split(",") if item.strip()]
        except ValueError as exc:
            raise SystemExit("--gpu-ids must be a comma-separated integer list") from exc
    elif getattr(args, "gpus", None) is not None:
        if args.gpus < 1:
            raise SystemExit("--gpus must be at least 1")
        ids = list(range(args.gpus))
    else:
        configured = config.get("target", {}).get("gpu_ids", [])
        ids = [int(item) for item in configured]
    if not ids:
        raise SystemExit("Specify --gpus N or --gpu-ids 0,1,... explicitly")
    if len(ids) != len(set(ids)) or min(ids) < 0:
        raise SystemExit("GPU IDs must be unique non-negative integers")
    return ids


def target_model(args: argparse.Namespace, config: dict[str, Any]) -> str:
    model = getattr(args, "model", None) or config.get("target", {}).get("model", "")
    if not model:
        raise SystemExit("Specify --model MODEL (run './bench models' to inspect the local cache)")
    return model


def served_name(args: argparse.Namespace, config: dict[str, Any], model: str) -> str:
    return (
        getattr(args, "served_model_name", None)
        or config.get("target", {}).get("served_model_name")
        or model
    )


def detect_runtime() -> list[str]:
    override = os.environ.get("POWER_CONTAINER_RUNTIME")
    candidates = [shlex.split(override)] if override else []
    candidates.extend([["docker"], ["sudo", "-n", "docker"], ["podman"]])
    errors: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            result = run_capture(candidate + ["info"], timeout=15)
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{' '.join(candidate)}: {exc}")
            continue
        if result.returncode == 0:
            return candidate
        errors.append(f"{' '.join(candidate)}: {(result.stderr or result.stdout).strip()}")
    raise RuntimeError(
        "No usable Docker/Podman runtime. Set POWER_CONTAINER_RUNTIME, for example "
        "POWER_CONTAINER_RUNTIME='sudo docker'.\n" + "\n".join(errors)
    )


def amd_smi_gpu_count() -> int:
    result = run_capture(["amd-smi", "list"], timeout=15)
    if result.returncode:
        return 0
    return len(re.findall(r"^GPU:\s+\d+", result.stdout, re.MULTILINE))


def _quantity_value(node: Any) -> float | None:
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return float(node)
    if isinstance(node, dict):
        value = node.get("value")
        unit = str(node.get("unit", "W")).lower()
        if isinstance(value, (int, float)):
            scale = {"uw": 1e-6, "mw": 1e-3, "w": 1.0, "kw": 1e3}.get(unit, 1.0)
            return float(value) * scale
    if isinstance(node, str):
        match = re.search(r"[-+]?[0-9]*\.?[0-9]+", node)
        if match:
            return float(match.group())
    return None


def read_power_amd_smi(gpu_ids: list[int]) -> dict[int, float]:
    command = ["amd-smi", "metric", "--power", "--gpu", *map(str, gpu_ids), "--json"]
    data = parse_json_output(run_capture(command, timeout=15))
    items = data.get("gpu_data", data) if isinstance(data, dict) else data
    if isinstance(items, dict):
        items = list(items.values())
    readings: dict[int, float] = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        gpu = item.get("gpu", item.get("gpu_id"))
        power = item.get("power", {})
        candidates = [
            power.get("socket_power") if isinstance(power, dict) else None,
            power.get("current_socket_power") if isinstance(power, dict) else None,
            item.get("socket_power"),
        ]
        watts = next((value for value in map(_quantity_value, candidates) if value is not None), None)
        if gpu is not None and watts is not None:
            readings[int(gpu)] = watts
    if not readings:
        raise RuntimeError("amd-smi returned no socket-power readings")
    return readings


def read_power_rocm_smi(gpu_ids: list[int]) -> dict[int, float]:
    command = ["rocm-smi", "--showpower", "--json", "-d", *map(str, gpu_ids)]
    data = parse_json_output(run_capture(command, timeout=15))
    readings: dict[int, float] = {}
    for card, fields in data.items():
        match = re.search(r"(\d+)$", card)
        if not match or not isinstance(fields, dict):
            continue
        for key, value in fields.items():
            lowered = key.lower()
            if "power" in lowered and ("current" in lowered or "socket" in lowered):
                watts = _quantity_value(value)
                if watts is not None:
                    readings[int(match.group(1))] = watts
                    break
    if not readings:
        raise RuntimeError("rocm-smi returned no socket-power readings")
    return readings


def read_power(gpu_ids: list[int]) -> tuple[str, dict[int, float]]:
    errors: list[str] = []
    for source, reader in (("amd-smi", read_power_amd_smi), ("rocm-smi", read_power_rocm_smi)):
        try:
            return source, reader(gpu_ids)
        except (FileNotFoundError, RuntimeError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{source}: {exc}")
    raise RuntimeError("; ".join(errors))


class PowerSampler:
    def __init__(self, gpu_ids: list[int], interval: float, output: Path):
        self.gpu_ids = gpu_ids
        self.interval = interval
        self.output = output
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="power-sampler", daemon=True)

    def _sample(self, handle: Any) -> None:
        monotonic_s = time.monotonic()
        try:
            source, readings = read_power(self.gpu_ids)
            missing = sorted(set(self.gpu_ids) - set(readings))
            if missing:
                raise RuntimeError(f"missing GPU readings: {missing}")
            row = {
                "timestamp": utc_now(),
                "monotonic_s": monotonic_s,
                "source": source,
                "gpu_w": {str(gpu): readings[gpu] for gpu in self.gpu_ids},
                "total_w": sum(readings[gpu] for gpu in self.gpu_ids),
            }
            self.samples.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
        except Exception as exc:  # preserve the workload even if one sample fails
            message = f"{utc_now()} {exc}"
            self.errors.append(message)
            handle.write(json.dumps({"timestamp": utc_now(), "error": str(exc)}) + "\n")
            handle.flush()

    def _loop(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("w", encoding="utf-8") as handle:
            self._sample(handle)
            self._ready.set()
            next_sample = time.monotonic() + self.interval
            while not self._stop.wait(max(0.0, next_sample - time.monotonic())):
                self._sample(handle)
                next_sample += self.interval
                if next_sample < time.monotonic():
                    next_sample = time.monotonic() + self.interval
            self._sample(handle)

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(20):
            raise RuntimeError("power sampler did not start")
        if not self.samples:
            raise RuntimeError(self.errors[-1] if self.errors else "power sampler produced no sample")

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=30)


def interpolate(a: tuple[float, float], b: tuple[float, float], at: float) -> float:
    if b[0] == a[0]:
        return b[1]
    fraction = (at - a[0]) / (b[0] - a[0])
    return a[1] + fraction * (b[1] - a[1])


def integrate_power(
    samples: list[dict[str, Any]], gpu_ids: list[int], start: float, end: float
) -> dict[str, Any]:
    if end <= start:
        raise RuntimeError("invalid energy integration interval")
    def maximum_in_window(points: list[tuple[float, float]]) -> float:
        values: list[float] = []
        for left, right in zip(points, points[1:]):
            segment_start = max(start, left[0])
            segment_end = min(end, right[0])
            if segment_end <= segment_start:
                continue
            values.append(interpolate(left, right, segment_start))
            values.append(interpolate(left, right, segment_end))
        return max(values) if values else 0.0

    per_gpu: dict[str, dict[str, float]] = {}
    for gpu in gpu_ids:
        points = [
            (float(row["monotonic_s"]), float(row["gpu_w"][str(gpu)]))
            for row in samples
            if "gpu_w" in row and str(gpu) in row["gpu_w"]
        ]
        points.sort()
        energy = 0.0
        covered = 0.0
        for left, right in zip(points, points[1:]):
            segment_start = max(start, left[0])
            segment_end = min(end, right[0])
            if segment_end <= segment_start:
                continue
            watts_start = interpolate(left, right, segment_start)
            watts_end = interpolate(left, right, segment_end)
            duration = segment_end - segment_start
            energy += 0.5 * (watts_start + watts_end) * duration
            covered += duration
        per_gpu[str(gpu)] = {
            "energy_j": energy,
            "average_power_w": energy / covered if covered else 0.0,
            "maximum_power_w": maximum_in_window(points),
            "covered_s": covered,
        }
    total_energy = sum(item["energy_j"] for item in per_gpu.values())
    covered_values = [item["covered_s"] for item in per_gpu.values()]
    covered = min(covered_values) if covered_values else 0.0
    total_points = [
        (
            float(row["monotonic_s"]),
            sum(float(row["gpu_w"][str(gpu)]) for gpu in gpu_ids),
        )
        for row in samples
        if "gpu_w" in row and all(str(gpu) in row["gpu_w"] for gpu in gpu_ids)
    ]
    total_points.sort()
    return {
        "start_monotonic_s": start,
        "end_monotonic_s": end,
        "requested_duration_s": end - start,
        "covered_s": covered,
        "gpu_energy_j": total_energy,
        "average_gpu_power_w": total_energy / covered if covered else 0.0,
        "maximum_gpu_power_w": maximum_in_window(total_points),
        "per_gpu": per_gpu,
    }


def sample_idle_power(gpu_ids: list[int], seconds: float, interval: float) -> dict[str, Any]:
    samples: list[dict[int, float]] = []
    deadline = time.monotonic() + max(0.0, seconds)
    while True:
        _, readings = read_power(gpu_ids)
        samples.append(readings)
        if time.monotonic() >= deadline:
            break
        time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
    per_gpu = {
        str(gpu): statistics.median(row[gpu] for row in samples if gpu in row)
        for gpu in gpu_ids
    }
    return {
        "sample_count": len(samples),
        "per_gpu_w": per_gpu,
        "total_w": sum(per_gpu.values()),
    }


def tee_process(command: list[str], log_path: Path) -> tuple[int, float, float]:
    print("+ " + shlex.join(command), flush=True)
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return_code = process.wait()
    end = time.monotonic()
    return return_code, start, end


def rocm_inventory(gpu_ids: list[int] | None = None) -> dict[str, Any]:
    selected = ["-d", *map(str, gpu_ids)] if gpu_ids else []
    result = run_capture(
        [
            "rocm-smi", "--showproductname", "--showmeminfo", "vram", "--showuse",
            "--showpower", "--showmaxpower", "--json", *selected,
        ],
        timeout=30,
    )
    if result.returncode:
        return {"error": (result.stderr or result.stdout).strip()}
    return json.loads(result.stdout)


def busy_gpus(
    gpu_ids: list[int], threshold_bytes: int = 2 * 2**30, usage_threshold: float = 5.0
) -> dict[int, tuple[float, float]]:
    """Return selected GPUs with material VRAM allocation or active compute."""
    inventory = rocm_inventory(gpu_ids)
    busy: dict[int, tuple[float, float]] = {}
    for card, fields in inventory.items():
        match = re.search(r"(\d+)$", card)
        if not match or not isinstance(fields, dict):
            continue
        used = _quantity_value(fields.get("VRAM Total Used Memory (B)")) or 0.0
        usage = _quantity_value(fields.get("GPU use (%)")) or 0.0
        if used > threshold_bytes or usage > usage_threshold:
            busy[int(match.group(1))] = (used, usage)
    return busy


def require_idle_gpus(gpu_ids: list[int], allow_busy: bool) -> None:
    busy = busy_gpus(gpu_ids)
    if not busy:
        return
    detail = ", ".join(
        f"GPU {gpu}: {used / 2**30:.1f} GiB used, {usage:.0f}% active"
        for gpu, (used, usage) in sorted(busy.items())
    )
    if allow_busy:
        print(f"WARNING: benchmarking on busy GPUs ({detail})", file=sys.stderr)
        return
    raise RuntimeError(
        f"selected GPUs are already busy ({detail}); choose other --gpu-ids or explicitly pass --allow-busy"
    )


def collect_machine(gpu_ids: list[int]) -> dict[str, Any]:
    data: dict[str, Any] = {
        "collected_at": utc_now(),
        "hostname": os.uname().nodename,
        "kernel": os.uname().release,
        "gpu_ids": gpu_ids,
        "rocm_smi": rocm_inventory(gpu_ids),
    }
    commands = {
        "amd_smi_version": ["amd-smi", "version"],
        "amd_smi_static": ["amd-smi", "static", "--asic", "--vram", "--driver", "--gpu", *map(str, gpu_ids), "--json"],
        "numa_balancing": ["sysctl", "kernel.numa_balancing"],
    }
    for key, command in commands.items():
        try:
            result = run_capture(command, timeout=30)
            text = (result.stdout or result.stderr).strip()
            if key == "amd_smi_static" and result.returncode == 0:
                data[key] = json.loads(text)
            else:
                data[key] = {"returncode": result.returncode, "output": text}
        except Exception as exc:
            data[key] = {"error": str(exc)}
    return data


def max_power_by_gpu(machine: dict[str, Any]) -> dict[int, float]:
    result: dict[int, float] = {}
    for card, fields in machine.get("rocm_smi", {}).items():
        match = re.search(r"(\d+)$", card)
        if not match or not isinstance(fields, dict):
            continue
        for key, value in fields.items():
            if "max" in key.lower() and "power" in key.lower():
                watts = _quantity_value(value)
                if watts is not None:
                    result[int(match.group(1))] = watts
                    break
    return result


def docker_image(config: dict[str, Any]) -> str:
    return str(config.get("server", {}).get("image", "")).strip()


def server_url(config: dict[str, Any]) -> str:
    server = config.get("server", {})
    host = server.get("client_host", "127.0.0.1")
    port = int(server.get("port", 8000))
    return f"http://{host}:{port}"


def server_container(config: dict[str, Any]) -> str:
    return str(config.get("server", {}).get("container_name", "power-vllm"))


def server_command(
    runtime: list[str], config: dict[str, Any], model: str, name: str, gpu_ids: list[int]
) -> list[str]:
    server = config.get("server", {})
    image = docker_image(config)
    if not image:
        raise SystemExit("server.image is missing from the config")
    cache = expand_path(str(server.get("hf_cache", "~/.cache/huggingface")))
    cache.mkdir(parents=True, exist_ok=True)
    ids = ",".join(map(str, gpu_ids))
    command = runtime + [
        "run", "-d",
        "--name", server_container(config),
        "--network", "host",
        "--ipc", "host",
        "--device", "/dev/kfd",
        "--device", "/dev/dri",
        "--group-add", "video",
        "--security-opt", "seccomp=unconfined",
        "--shm-size", str(server.get("shm_size", "64g")),
        "-e", f"HIP_VISIBLE_DEVICES={ids}",
        "-e", f"CUDA_VISIBLE_DEVICES={ids}",
        "-e", "HF_HOME=/root/.cache/huggingface",
        "-v", f"{cache}:/root/.cache/huggingface",
    ]
    for item in server.get("extra_env", []):
        command += ["-e", str(item)]
    for mount in server.get("extra_mounts", []):
        command += ["-v", os.path.expandvars(os.path.expanduser(str(mount)))]
    if "container_entrypoint" in server:
        command += ["--entrypoint", str(server["container_entrypoint"])]
    command += [
        image,
        "vllm", "serve", model,
        "--served-model-name", name,
        "--tensor-parallel-size", str(len(gpu_ids)),
        "--max-model-len", str(server.get("max_model_len", 32768)),
        "--max-num-seqs", str(server.get("max_num_seqs", 256)),
        "--gpu-memory-utilization", str(server.get("gpu_memory_utilization", 0.85)),
        "--kv-cache-dtype", str(server.get("kv_cache_dtype", "auto")),
        "--host", "0.0.0.0",
        "--port", str(server.get("port", 8000)),
    ]
    if server.get("dtype"):
        command += ["--dtype", str(server["dtype"])]
    if server.get("quantization"):
        command += ["--quantization", str(server["quantization"])]
    command += [str(item) for item in server.get("extra_args", [])]
    return command


def start_server(
    runtime: list[str], config: dict[str, Any], model: str, name: str, gpu_ids: list[int], dry_run: bool
) -> None:
    container = server_container(config)
    exists = run_capture(runtime + ["container", "inspect", container], timeout=15)
    if exists.returncode == 0:
        raise RuntimeError(f"container {container!r} already exists; use './bench stop' first")
    command = server_command(runtime, config, model, name, gpu_ids)
    print("+ " + shlex.join(command))
    if dry_run:
        return
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise RuntimeError(f"server container failed to start (exit {result.returncode})")


def stop_server(runtime: list[str], config: dict[str, Any]) -> None:
    container = server_container(config)
    result = subprocess.run(runtime + ["rm", "-f", container], check=False)
    if result.returncode:
        raise RuntimeError(f"could not remove container {container!r}")


def wait_server(runtime: list[str] | None, config: dict[str, Any], timeout: float) -> None:
    url = server_url(config) + "/v1/models"
    started_at = time.monotonic()
    deadline = started_at + timeout
    next_heartbeat = started_at + 30
    last_error = ""
    print(f"Waiting up to {timeout:.0f}s for {url}")
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    print("Server is ready")
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
        container_state = "unknown"
        if runtime:
            state = run_capture(
                runtime + ["container", "inspect", server_container(config), "--format", "{{.State.Status}} {{.State.ExitCode}}"],
                timeout=15,
            )
            if state.returncode == 0:
                container_state = state.stdout.strip()
            if state.returncode == 0 and state.stdout.startswith("exited"):
                logs = run_capture(runtime + ["logs", "--tail", "100", server_container(config)], timeout=30)
                raise RuntimeError("vLLM container exited:\n" + (logs.stdout or logs.stderr))
        now = time.monotonic()
        if now >= next_heartbeat:
            elapsed = now - started_at
            suffix = f"; container={container_state}" if runtime else ""
            print(f"Still waiting for vLLM readiness: {elapsed:.0f}s elapsed{suffix}", flush=True)
            next_heartbeat = now + 30
        time.sleep(3)
    raise RuntimeError(f"server did not become ready: {last_error}")


def benchmark_command(
    runtime: list[str],
    config: dict[str, Any],
    model: str,
    name: str,
    scenario: dict[str, Any],
    concurrency: int,
    prompts: int,
    run_dir: Path,
    gpu_count: int,
    result_filename: str = "vllm.json",
    num_warmups: int | None = None,
) -> list[str]:
    server = config.get("server", {})
    benchmark = config.get("benchmark", {})
    cache = expand_path(str(server.get("hf_cache", "~/.cache/huggingface")))
    request_rate = str(benchmark.get("request_rate", "inf"))
    command = runtime + [
        "run", "--rm",
        "--network", "host",
        "-e", "HF_HOME=/root/.cache/huggingface",
        "-v", f"{cache}:/root/.cache/huggingface",
        "-v", f"{run_dir}:/results",
    ]
    if "container_entrypoint" in server:
        command += ["--entrypoint", str(server["container_entrypoint"])]
    command += [
        docker_image(config),
        "vllm", "bench", "serve",
        "--backend", str(benchmark.get("backend", "openai")),
        "--base-url", server_url(config),
        "--endpoint", str(benchmark.get("endpoint", "/v1/completions")),
        "--model", name,
        "--tokenizer", model,
        "--dataset-name", "random",
        "--random-input-len", str(scenario["input_len"]),
        "--random-output-len", str(scenario["output_len"]),
        "--num-prompts", str(prompts),
        "--num-warmups", str(benchmark.get("num_warmups", 1) if num_warmups is None else num_warmups),
        "--max-concurrency", str(concurrency),
        "--request-rate", request_rate,
        "--ignore-eos",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", str(benchmark.get("metric_percentiles", "50,90,99")),
        "--metadata", f"scenario={scenario['name']}", f"gpu_count={gpu_count}",
        "--save-result",
        "--result-dir", "/results",
        "--result-filename", result_filename,
    ]
    command += [str(item) for item in benchmark.get("extra_args", [])]
    return command


def summarize_run(
    run_dir: Path,
    sampler: PowerSampler,
    idle: dict[str, Any],
    machine: dict[str, Any],
    gpu_ids: list[int],
    command_start: float,
    command_end: float,
    return_code: int,
    scenario: dict[str, Any],
    concurrency: int,
    prompts: int,
    target_duration_s: float | None = None,
    prompt_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    vllm_path = run_dir / "vllm.json"
    vllm_data: dict[str, Any] = {}
    if vllm_path.exists():
        loaded = json.loads(vllm_path.read_text(encoding="utf-8"))
        vllm_data = loaded[-1] if isinstance(loaded, list) else loaded
    command_window = integrate_power(sampler.samples, gpu_ids, command_start, command_end)
    duration = float(vllm_data.get("duration", command_end - command_start))
    benchmark_end = command_end
    benchmark_start = max(command_start, benchmark_end - duration)
    benchmark_window = integrate_power(sampler.samples, gpu_ids, benchmark_start, benchmark_end)
    output_tokens = int(vllm_data.get("total_output_tokens", 0) or 0)
    energy = benchmark_window["gpu_energy_j"]
    idle_energy = idle["total_w"] * benchmark_window["covered_s"]
    dynamic_energy = energy - idle_energy
    max_power = max_power_by_gpu(machine)
    selected_power_cap = sum(max_power.get(gpu, 0.0) for gpu in gpu_ids)
    average_power = benchmark_window["average_gpu_power_w"]
    maximum_power = benchmark_window["maximum_gpu_power_w"]
    efficiency = {
        "gross_j_per_output_token": energy / output_tokens if output_tokens else None,
        "gross_output_tokens_per_kwh": output_tokens * 3_600_000 / energy if energy else None,
        "above_idle_gpu_energy_j": dynamic_energy,
        "above_idle_j_per_output_token": dynamic_energy / output_tokens if output_tokens else None,
        "average_power_fraction_of_reported_cap": (
            average_power / selected_power_cap if selected_power_cap else None
        ),
        "maximum_power_fraction_of_reported_cap": (
            maximum_power / selected_power_cap if selected_power_cap else None
        ),
        "average_power_fraction_of_tdp_proxy": (
            average_power / selected_power_cap if selected_power_cap else None
        ),
        "maximum_power_fraction_of_tdp_proxy": (
            maximum_power / selected_power_cap if selected_power_cap else None
        ),
        "output_tokens_per_s_per_gpu": (
            float(vllm_data.get("output_throughput", 0.0)) / len(gpu_ids)
            if vllm_data.get("output_throughput") is not None
            else None
        ),
    }
    return {
        "created_at": utc_now(),
        "return_code": return_code,
        "gpu_ids": gpu_ids,
        "power_scope": "selected AMD GPU socket/package power; excludes CPU, fans, storage, network, and PSU losses",
        "integration_method": "trapezoidal integration of sampled socket watts over monotonic time",
        "sample_interval_s": sampler.interval,
        "sample_errors": sampler.errors,
        "workload": {
            "scenario": scenario["name"],
            "input_len": int(scenario["input_len"]),
            "output_len": int(scenario["output_len"]),
            "concurrency": concurrency,
            "num_prompts": prompts,
            "target_duration_s": target_duration_s,
            "prompt_planning": prompt_plan,
        },
        "idle_baseline": idle,
        "command_window": command_window,
        "benchmark_window": benchmark_window,
        "reported_power_cap_w": selected_power_cap or None,
        "tdp_proxy_w": selected_power_cap or None,
        "tdp_proxy_source": "reported Max Graphics Package Power",
        "vllm": vllm_data,
        "efficiency": efficiency,
    }


def write_sweep_indexes(sweep_dir: Path, summaries: list[dict[str, Any]]) -> None:
    (sweep_dir / "index.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    columns = [
        "run", "scenario", "input_len", "output_len", "concurrency", "gpu_count",
        "target_duration_s", "duration_s", "completed", "total_output_tokens", "output_throughput",
        "average_gpu_power_w", "maximum_gpu_power_w", "gpu_energy_j", "gross_j_per_output_token",
        "gross_output_tokens_per_kwh", "average_power_fraction_of_reported_cap",
        "maximum_power_fraction_of_reported_cap", "average_power_fraction_of_tdp_proxy",
    ]
    with (sweep_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for summary in summaries:
            workload = summary["workload"]
            vllm = summary["vllm"]
            window = summary["benchmark_window"]
            efficiency = summary["efficiency"]
            writer.writerow({
                "run": summary["run"],
                "scenario": workload["scenario"],
                "input_len": workload["input_len"],
                "output_len": workload["output_len"],
                "concurrency": workload["concurrency"],
                "gpu_count": len(summary["gpu_ids"]),
                "target_duration_s": workload.get("target_duration_s"),
                "duration_s": vllm.get("duration"),
                "completed": vllm.get("completed"),
                "total_output_tokens": vllm.get("total_output_tokens"),
                "output_throughput": vllm.get("output_throughput"),
                "average_gpu_power_w": window["average_gpu_power_w"],
                "maximum_gpu_power_w": window["maximum_gpu_power_w"],
                "gpu_energy_j": window["gpu_energy_j"],
                "gross_j_per_output_token": efficiency["gross_j_per_output_token"],
                "gross_output_tokens_per_kwh": efficiency["gross_output_tokens_per_kwh"],
                "average_power_fraction_of_reported_cap": efficiency["average_power_fraction_of_reported_cap"],
                "maximum_power_fraction_of_reported_cap": efficiency["maximum_power_fraction_of_reported_cap"],
                "average_power_fraction_of_tdp_proxy": efficiency["average_power_fraction_of_tdp_proxy"],
            })


def scenarios_from_config(args: argparse.Namespace, config: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios = list(config.get("scenarios", []))
    requested = set((getattr(args, "scenario", None) or "").split(",")) - {""}
    if requested:
        scenarios = [scenario for scenario in scenarios if scenario.get("name") in requested]
        missing = requested - {str(scenario.get("name")) for scenario in scenarios}
        if missing:
            raise SystemExit(f"unknown scenario(s): {', '.join(sorted(missing))}")
    if not scenarios:
        raise SystemExit("no benchmark scenarios selected")
    override = getattr(args, "concurrency", None)
    if override:
        try:
            levels = [int(item) for item in override.split(",")]
        except ValueError as exc:
            raise SystemExit("--concurrency must be a comma-separated integer list") from exc
        if not levels or min(levels) < 1:
            raise SystemExit("concurrency levels must be positive")
        for scenario in scenarios:
            scenario["concurrency"] = levels
    return scenarios


def calibrated_prompt_plan(
    *,
    base_prompts: int,
    concurrency: int,
    target_duration_s: float,
    calibration_prompts: int,
    calibration_completed: int,
    calibration_duration_s: float,
    minimum_waves: int,
    duration_safety_factor: float,
    max_num_prompts: int,
) -> tuple[int, dict[str, Any]]:
    if calibration_duration_s <= 0 or calibration_completed <= 0:
        raise RuntimeError("calibration returned no successful timed requests")
    scaled_prompts = math.ceil(
        calibration_completed
        * target_duration_s
        / calibration_duration_s
        * duration_safety_factor
    )
    requested_prompts = max(base_prompts, minimum_waves * concurrency, scaled_prompts)
    prompts = min(requested_prompts, max_num_prompts)
    return prompts, {
        "method": "calibrated_target_duration",
        "target_duration_s": target_duration_s,
        "duration_safety_factor": duration_safety_factor,
        "calibration_prompts": calibration_prompts,
        "calibration_completed": calibration_completed,
        "calibration_duration_s": calibration_duration_s,
        "base_prompts": base_prompts,
        "minimum_measured_waves": minimum_waves,
        "uncapped_prompts": requested_prompts,
        "max_num_prompts": max_num_prompts,
        "measured_prompts": prompts,
        "capped": prompts < requested_prompts,
    }


def run_sweep(
    args: argparse.Namespace,
    config: dict[str, Any],
    runtime: list[str],
    model: str,
    name: str,
    gpu_ids: list[int],
) -> None:
    wait_server(None, config, float(config.get("server", {}).get("ready_timeout_s", 1800)))
    machine = collect_machine(gpu_ids)
    machine["model_id"] = model
    machine["served_model_name"] = name
    machine["container_image"] = docker_image(config)
    image_info = run_capture(
        runtime + ["image", "inspect", docker_image(config), "--format", "{{json .Id}} {{json .RepoDigests}}"],
        timeout=30,
    )
    machine["container_image_inspect"] = {
        "returncode": image_info.returncode,
        "output": (image_info.stdout or image_info.stderr).strip(),
    }
    results_root = expand_path(str(config.get("benchmark", {}).get("results_dir", str(ROOT / "results"))))
    model_root = results_root / model_result_key(model)
    model_root.mkdir(parents=True, exist_ok=True)
    sweep_stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    sweep_dir = model_root / sweep_stamp
    sweep_dir.mkdir(parents=True, exist_ok=False)
    (sweep_dir / "machine.json").write_text(json.dumps(machine, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (sweep_dir / "config.toml").write_bytes(Path(config["_path"]).read_bytes())
    benchmark = config.get("benchmark", {})
    interval = float(benchmark.get("sample_interval_s", 0.5))
    idle_seconds = float(benchmark.get("idle_sample_seconds", 3.0))
    multiplier = int(benchmark.get("num_prompts_multiplier", 4))
    minimum = int(benchmark.get("min_num_prompts", 32))
    target_duration_s = float(
        getattr(args, "target_duration", None) or benchmark.get("target_duration_s", 0) or 0
    )
    calibration_waves = int(benchmark.get("calibration_waves", 1))
    minimum_waves = int(benchmark.get("minimum_measured_waves", 2))
    duration_safety_factor = float(benchmark.get("duration_safety_factor", 1.15))
    max_num_prompts = int(benchmark.get("max_num_prompts", 8192))
    summaries: list[dict[str, Any]] = []
    for scenario in scenarios_from_config(args, config):
        max_len = int(config.get("server", {}).get("max_model_len", 32768))
        if int(scenario["input_len"]) + int(scenario["output_len"]) > max_len:
            raise RuntimeError(f"scenario {scenario['name']} exceeds server.max_model_len={max_len}")
        for concurrency in scenario.get("concurrency", []):
            concurrency = int(concurrency)
            base_prompts = int(scenario.get("num_prompts", max(minimum, multiplier * concurrency)))
            prompts = base_prompts
            label = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(scenario["name"]))
            run_dir = sweep_dir / f"{label}-i{scenario['input_len']}-o{scenario['output_len']}-c{concurrency}"
            run_dir.mkdir(parents=True, exist_ok=False)
            prompt_plan: dict[str, Any] | None = None
            if target_duration_s > 0:
                calibration_prompts = max(1, calibration_waves * concurrency)
                calibration_command = benchmark_command(
                    runtime,
                    config,
                    model,
                    name,
                    scenario,
                    concurrency,
                    calibration_prompts,
                    run_dir,
                    len(gpu_ids),
                    result_filename="calibration.json",
                    num_warmups=1,
                )
                (run_dir / "calibration-command.txt").write_text(
                    shlex.join(calibration_command) + "\n", encoding="utf-8"
                )
                print(
                    f"\n--- Calibrating {scenario['name']}: ISL={scenario['input_len']} "
                    f"OSL={scenario['output_len']} c={concurrency} with {calibration_prompts} prompts ---"
                )
                calibration_code, _, _ = tee_process(calibration_command, run_dir / "calibration.log")
                if calibration_code:
                    raise RuntimeError(f"calibration failed; see {run_dir / 'calibration.log'}")
                calibration_data = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
                if isinstance(calibration_data, list):
                    calibration_data = calibration_data[-1]
                calibration_duration = float(calibration_data.get("duration", 0) or 0)
                calibration_completed = int(calibration_data.get("completed", 0) or 0)
                prompts, prompt_plan = calibrated_prompt_plan(
                    base_prompts=base_prompts,
                    concurrency=concurrency,
                    target_duration_s=target_duration_s,
                    calibration_prompts=calibration_prompts,
                    calibration_completed=calibration_completed,
                    calibration_duration_s=calibration_duration,
                    minimum_waves=minimum_waves,
                    duration_safety_factor=duration_safety_factor,
                    max_num_prompts=max_num_prompts,
                )
                (run_dir / "prompt-plan.json").write_text(
                    json.dumps(prompt_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                print(
                    f"Calibration took {calibration_duration:.2f}s for {calibration_completed} prompts; "
                    f"measuring {prompts} prompts for target >= {target_duration_s:.0f}s"
                    + (" (capped)" if prompt_plan["capped"] else "")
                )
            command = benchmark_command(
                runtime, config, model, name, scenario, concurrency, prompts, run_dir, len(gpu_ids)
            )
            (run_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
            print(f"\n=== {scenario['name']}: ISL={scenario['input_len']} OSL={scenario['output_len']} c={concurrency} ===")
            idle = sample_idle_power(gpu_ids, idle_seconds, interval)
            sampler = PowerSampler(gpu_ids, interval, run_dir / "power.jsonl")
            sampler.start()
            try:
                return_code, command_start, command_end = tee_process(command, run_dir / "client.log")
            finally:
                sampler.stop()
            summary = summarize_run(
                run_dir, sampler, idle, machine, gpu_ids, command_start, command_end, return_code,
                scenario, concurrency, prompts, target_duration_s or None, prompt_plan,
            )
            (run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
            )
            summaries.append({"run": run_dir.name, **summary})
            write_sweep_indexes(sweep_dir, summaries)
            metric = summary["efficiency"]["gross_j_per_output_token"]
            print(
                f"Result: {summary['benchmark_window']['gpu_energy_j']:.1f} J GPU energy, "
                + (f"{metric:.6f} J/output-token" if metric is not None else "no token result")
            )
            if return_code:
                raise RuntimeError(f"benchmark failed; see {run_dir / 'client.log'}")
    write_sweep_indexes(sweep_dir, summaries)
    print(f"\nSweep complete: {sweep_dir}")


def cached_models(config: dict[str, Any]) -> list[tuple[str, int, str]]:
    cache = expand_path(str(config.get("server", {}).get("hf_cache", "~/.cache/huggingface"))) / "hub"
    found: list[tuple[str, int, str]] = []
    if not cache.is_dir():
        return found
    for model_dir in sorted(cache.glob("models--*")):
        model_id = model_dir.name[len("models--") :].replace("--", "/")
        snapshots = list((model_dir / "snapshots").glob("*")) if (model_dir / "snapshots").is_dir() else []
        found.append((model_id, len(snapshots), str(model_dir)))
    return found


def command_doctor(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    print("AMD vLLM power benchmark doctor")
    print(f"Config: {config['_path']}")
    count = amd_smi_gpu_count()
    print(f"GPUs visible to amd-smi: {count}")
    ids = list(range(count))
    if count:
        source, readings = read_power(ids)
        print(f"Power telemetry: {source}; {sum(readings.values()):.1f} W across {count} GPUs")
        inventory = rocm_inventory(ids)
        for card, fields in inventory.items():
            if not isinstance(fields, dict):
                continue
            name = fields.get("Card Series", "unknown")
            gfx = fields.get("GFX Version", "unknown")
            used = _quantity_value(fields.get("VRAM Total Used Memory (B)")) or 0
            total = _quantity_value(fields.get("VRAM Total Memory (B)")) or 0
            cap = _quantity_value(fields.get("Max Graphics Package Power (W)"))
            print(f"  {card}: {name}, {gfx}, VRAM {used / 2**30:.1f}/{total / 2**30:.1f} GiB, cap {cap} W")
    numa = run_capture(["sysctl", "-n", "kernel.numa_balancing"])
    print(f"NUMA balancing: {(numa.stdout or numa.stderr).strip()} (0 recommended)")
    try:
        runtime = detect_runtime()
        print(f"Container runtime: {shlex.join(runtime)}")
        image = docker_image(config)
        inspected = run_capture(runtime + ["image", "inspect", image], timeout=30)
        print(f"Configured vLLM image: {image} ({'present' if inspected.returncode == 0 else 'missing'})")
    except RuntimeError as exc:
        print(f"Container runtime: unavailable ({exc})")


def command_models(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    models = cached_models(config)
    if models:
        print("Hugging Face cache entries (a snapshot count does not guarantee every weight is present):")
        for model, snapshots, path in models:
            print(f"  {model}\tsnapshots={snapshots}\t{path}")
    else:
        print("No Hugging Face model cache entries found")
    try:
        with urllib.request.urlopen(server_url(config) + "/v1/models", timeout=2) as response:
            payload = json.load(response)
        print("\nModels served now:")
        for item in payload.get("data", []):
            print(f"  {item.get('id')}")
    except Exception:
        print("\nNo OpenAI-compatible server is responding at " + server_url(config))


def add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="TOML profile")


def add_target(parser: argparse.ArgumentParser) -> None:
    add_config(parser)
    parser.add_argument("--model", help="Hugging Face model ID or mounted model path")
    parser.add_argument("--served-model-name", help="name exposed by the OpenAI API")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gpus", type=int, help="use GPU IDs 0 through N-1")
    group.add_argument("--gpu-ids", help="explicit IDs, for example 4,5,6,7")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="inspect GPUs, telemetry, runtime, and image")
    add_config(doctor)
    models = sub.add_parser("models", help="list local cache entries and currently served models")
    add_config(models)
    analyze = sub.add_parser(
        "analyze",
        aliases=["power-an"],
        help="print text views of power, efficiency, latency, and quality",
    )
    analyze.add_argument(
        "paths",
        nargs="*",
        help="one or more sweep/run directories; defaults to latest completed sweep",
    )
    analyze.add_argument(
        "--view",
        default="overview,insights,anomalies",
        help="comma-separated: overview,insights,efficiency,gpus,latency,timeline,anomalies,all",
    )
    analyze.add_argument("--run", help="run-name substring or shell pattern, e.g. '*-c4'")
    analyze.add_argument("--segments", type=int, default=24, help="timeline sparkline width")
    analyze.add_argument("--output", help="also save the report to this text file")
    analyze.add_argument("--cdf-output", help="also save an interactive/document-QA power CDF as SVG")
    serve = sub.add_parser("serve", help="start the configured vLLM server container")
    add_target(serve)
    serve.add_argument("--dry-run", action="store_true")
    serve.add_argument("--allow-busy", action="store_true", help="permit GPUs with more than 2 GiB VRAM in use")
    stop = sub.add_parser("stop", help="stop and remove this harness's server container")
    add_config(stop)
    logs = sub.add_parser("logs", help="follow the vLLM server log")
    add_config(logs)
    wait = sub.add_parser("wait", help="wait for the OpenAI endpoint")
    add_config(wait)
    wait.add_argument("--timeout", type=float)
    sweep = sub.add_parser("sweep", help="benchmark an already running vLLM server")
    add_target(sweep)
    sweep.add_argument("--scenario", help="comma-separated scenario names")
    sweep.add_argument("--concurrency", help="override concurrency levels, e.g. 1,4,16")
    sweep.add_argument(
        "--target-duration",
        type=float,
        help="calibrate prompt count for at least this many measured seconds",
    )
    all_cmd = sub.add_parser("all", help="start server, wait, sweep, then stop")
    add_target(all_cmd)
    all_cmd.add_argument("--scenario", help="comma-separated scenario names")
    all_cmd.add_argument("--concurrency", help="override concurrency levels, e.g. 1,4,16")
    all_cmd.add_argument(
        "--target-duration",
        type=float,
        help="calibrate prompt count for at least this many measured seconds",
    )
    all_cmd.add_argument("--keep-server", action="store_true")
    all_cmd.add_argument("--dry-run", action="store_true")
    all_cmd.add_argument("--allow-busy", action="store_true", help="permit GPUs with more than 2 GiB VRAM in use")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "doctor":
        command_doctor(args)
        return 0
    if args.command == "models":
        command_models(args)
        return 0
    if args.command in {"analyze", "power-an"}:
        from power_report import run_analysis

        return run_analysis(args)
    config = load_config(args.config)
    runtime = detect_runtime()
    if args.command == "stop":
        stop_server(runtime, config)
        return 0
    if args.command == "logs":
        return subprocess.run(runtime + ["logs", "-f", server_container(config)], check=False).returncode
    if args.command == "wait":
        timeout = args.timeout or float(config.get("server", {}).get("ready_timeout_s", 1800))
        wait_server(runtime, config, timeout)
        return 0
    model = target_model(args, config)
    name = served_name(args, config, model)
    gpu_ids = parse_gpu_ids(args, config)
    visible = amd_smi_gpu_count()
    if visible and max(gpu_ids) >= visible:
        raise SystemExit(f"requested GPU {max(gpu_ids)}, but amd-smi exposes IDs 0..{visible - 1}")
    if args.command == "serve":
        require_idle_gpus(gpu_ids, args.allow_busy)
        start_server(runtime, config, model, name, gpu_ids, args.dry_run)
        return 0
    if args.command == "sweep":
        run_sweep(args, config, runtime, model, name, gpu_ids)
        return 0
    if args.command == "all":
        require_idle_gpus(gpu_ids, args.allow_busy)
        start_server(runtime, config, model, name, gpu_ids, args.dry_run)
        if args.dry_run:
            print("Dry run stops before readiness and workload execution")
            return 0
        started = True
        try:
            wait_server(runtime, config, float(config.get("server", {}).get("ready_timeout_s", 1800)))
            run_sweep(args, config, runtime, model, name, gpu_ids)
        finally:
            if started and not args.keep_server:
                stop_server(runtime, config)
        return 0
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        raise SystemExit(130)
    except (RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
