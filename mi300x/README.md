# AMD vLLM power benchmark

This directory is a portable starting point for measuring the GPU energy of
vLLM inference on AMD accelerators. It was adapted from an MI455X investigation
for this machine's eight MI300X VFs (`gfx942`). Nothing in the measurement code
is tied to MI300X: GPU selection, device identity, power caps, and telemetry are
detected at run time.

The harness measures sampled AMD GPU socket/package watts and integrates them
over monotonic time. This host's `amd-smi` has no cumulative-energy command, so
sampling is the reliable local interface. Results include gross GPU joules,
joules/output-token, tokens/kWh, average and maximum sampled watts, watts as a
fraction of the reported power limit, a clearly labeled TDP proxy, and an
explicitly labeled above-idle estimate. The TDP proxy uses AMD SMI's reported
`Max Graphics Package Power`; it is not an independent manufacturer-TDP reading.

## Quick start

Start Codex in this prepared directory:

```bash
cd ~/power
codex
```

For a direct smoke run:

```bash
cd ~/power
./bench doctor
./bench models
./bench all \
  --config configs/smoke.toml \
  --model MODEL_ID \
  --gpu-ids 4,5,6,7
```

Always choose IDs reported as idle by `./bench doctor`. `--gpus 4` is shorthand
for IDs `0,1,2,3`; use `--gpu-ids` when other jobs occupy those devices. The
harness refuses to launch on a GPU with more than 2 GiB of VRAM already in use
unless `--allow-busy` is explicit.

The managed `all` command starts vLLM, waits for `/v1/models`, runs the sweep,
and removes only its own `power-vllm` container. To keep the server alive, add
`--keep-server`.

## Typical workflow

Run the server and client separately when tuning:

```bash
./bench serve --config configs/smoke.toml --model MODEL_ID --gpu-ids 4,5,6,7
./bench logs --config configs/smoke.toml
./bench wait --config configs/smoke.toml
./bench sweep \
  --config configs/smoke.toml \
  --model MODEL_ID \
  --gpu-ids 4,5,6,7 \
  --concurrency 1,4,16
./bench stop --config configs/smoke.toml
```

The included profiles are:

- `configs/smoke.toml`: 256/128 tokens at concurrency 1 and 4.
- `configs/realistic.toml`: interactive, document-QA, and decode-heavy loads.
- `configs/large-kv.toml`: 32K and 64K input contexts with long decode.
- `configs/full-8gpu.toml`: all six scenarios, all 24 points, eight GPUs, and
  calibrated prompt counts targeting at least 60 measured seconds per point.

Override all profile concurrency levels with `--concurrency 1,4,8,16`, or select
named cases with `--scenario document-qa,decode-heavy`. vLLM continuously
batches requests; client concurrency is the realistic batch-size control.

For GPT-OSS-120B, move from smoke to more detailed profiles like this:

```bash
# Production-shaped interactive and document workloads.
./bench all \
  --config configs/realistic.toml \
  --model openai/gpt-oss-120b \
  --gpu-ids 4,5 \
  --scenario interactive,document-qa \
  --concurrency 1,4,16

# Decode-heavy large-KV work; begin at low concurrency.
./bench all \
  --config configs/large-kv.toml \
  --model openai/gpt-oss-120b \
  --gpu-ids 4,5 \
  --scenario large-kv-32k \
  --concurrency 1,4
```

The large-KV profile restarts the server with a 131,072-token context limit.
Increase concurrency only after checking successful requests and KV capacity.

For the complete accuracy-oriented GPT-OSS suite on eight idle GPUs:

```bash
./run-all-scenarios
```

The runner performs a strict preflight, starts one TP=8 server, calibrates every
scenario/concurrency point, measures all 24 points, then runs every text-analysis
view and saves `analysis.txt` in the new sweep directory. Override its defaults:

```bash
./run-all-scenarios \
  --model openai/gpt-oss-120b \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --target-duration 90
```

MiniMax-M3 uses its official MXFP8 checkpoint and dedicated ROCm image because
the BF16 repository is too large for this host with safe runtime headroom:

```bash
./run-all-scenarios \
  --config configs/minimax-m3-8gpu.toml \
  --model MiniMaxAI/MiniMax-M3-MXFP8 \
  --gpu-ids 0,1,2,3,4,5,6,7
```

Calibration runs one request wave without power accounting, estimates the
prompt count needed for the target duration, adds a 15% duration margin, and
requires at least two measured waves. The measured run begins after a 10-second
idle baseline. Raw power is requested every 0.25 seconds and always integrated
using actual monotonic sample timestamps.

## Results

Each sweep creates `results/<model key>/<UTC timestamp>/` containing. For
example, GPT-OSS results live under `results/openai--gpt-oss-120b/`, while
MiniMax-M3 MXFP8 results live under
`results/MiniMaxAI--MiniMax-M3-MXFP8/`:

- `machine.json`: exact GPU, driver, ROCm, VRAM, power-cap, and NUMA context.
- `config.toml`: the profile used.
- `summary.csv` and `index.json`: comparison tables across all points.
- One directory per point with the exact command, client log, raw `power.jsonl`,
  native `vllm.json`, and derived `summary.json`.

The primary energy window uses vLLM's reported benchmark duration aligned to
the tail of the measured client process. This excludes tokenizer preparation
and vLLM's warm-up requests while retaining raw samples for audit. Power is
integrated with the trapezoidal rule.

`gross_j_per_output_token` is the main comparable result. The above-idle value
subtracts a short server-idle baseline and can be useful diagnostically, but it
is more sensitive to unrelated GPU activity.

## Text power analysis

Analyze the latest completed sweep without installing plotting packages:

```bash
./bench analyze
```

`power-an` is a shorter alias. Select one or more views:

```bash
./bench power-an --view overview,efficiency,gpus
./bench analyze --view latency,timeline --run '*-c4'
./bench analyze ~/power/results/20260729-215615Z --view all
./bench analyze --output ~/power/results/latest-analysis.txt
./bench analyze BASE_SWEEP FOLLOWUP_SWEEP --cdf-output ~/power/power-cdf.svg
```

Point the analyzer at one model directory to combine every sweep for that
model without mixing checkpoints:

```bash
./bench analyze ~/power/results/openai--gpt-oss-120b --view overview,anomalies
```

The CDF output contains separate interactive and document-QA panels. It plots
aggregate selected-GPU package power against the time-weighted percentage of
the benchmark window spent at or below that power, with at most five
concurrency curves per panel.

Available views are `overview`, `insights`, `efficiency`, `gpus`, `latency`,
`timeline`, and `anomalies`. The anomaly view checks failed requests, token
counts, telemetry coverage and gaps, short runs, idle-baseline size, GPU power
imbalance, and reported power-cap consistency.

## Scope and controls

These are GPU socket/package measurements, not wall power. They exclude CPUs,
RAM outside the accelerator package, fans, storage, NICs, cooling, and PSU
losses. Use a metered PDU or BMC feed for whole-node energy and align its samples
to the same benchmark window.

For publishable comparisons:

1. Confirm no unrelated process uses the selected GPUs.
2. Keep image, model revision, quantization, TP size, clocks, power policy, and
   `gpu_memory_utilization` fixed.
3. Run smoke first, then repeat each real point at least three times.
4. Use `request_rate = "inf"` for saturation and a finite rate for service-like
   traffic; do not combine them in one comparison series.
5. Watch for OOM, preemption, cache eviction, failed requests, and thermal or
   power throttling before interpreting joules/token.

Set `POWER_CONTAINER_RUNTIME` if automatic Docker/Podman detection is wrong:

```bash
POWER_CONTAINER_RUNTIME='sudo docker' ./bench doctor
```

Change the image, context limit, cache dtype, extra mounts, or extra vLLM flags
in the TOML profile. The current default image is already present on this host
and targets `gfx94X`; other AMD architectures should use their matching ROCm
vLLM image without changing the measurement code.

Run the dependency-free unit tests with:

```bash
python3 -m unittest discover -s tests -v
```
