# Starting task for Codex

Build and execute a reproducible power characterization of vLLM inference on
the selected AMD GPUs and model. Sweep explicit concurrency levels and ISL/OSL
scenarios, record raw telemetry and native vLLM results, and report throughput,
latency, average GPU power, GPU joules/output-token, tokens/kWh, and power-cap
fraction. Keep the harness portable to other AMD accelerator architectures.

## Recommended next actions

1. Run `./bench doctor` and identify genuinely idle GPUs.
2. Run `./bench models`; verify the chosen checkpoint is complete and supported
   by the configured vLLM image.
3. Dry-run the launch command:

   ```bash
   ./bench serve --config configs/smoke.toml --model MODEL_ID --gpu-ids IDS --dry-run
   ```

4. Run the managed smoke benchmark with the same model and IDs.
5. Inspect `summary.json`, `vllm.json`, `client.log`, and sample coverage. Require
   zero failed requests and no telemetry errors.
6. Move to `configs/realistic.toml`; use `configs/large-kv.toml` only after
   estimating KV capacity and increasing concurrency gradually.
7. Repeat stable points and summarize variance. Treat finite request-rate and
   saturation (`inf`) sweeps as separate experiments.

## Acceptance criteria

- Every result identifies model, container image, TP/GPU IDs, ROCm/driver, GFX
  architecture, ISL, OSL, concurrency, prompt count, and request rate.
- Raw sampled watts can reproduce the reported integrated joules.
- Energy efficiency uses successful generated output tokens.
- Failed or partial runs remain visible and are not averaged into good runs.
- GPU socket energy and whole-node energy are never conflated.
- Architecture-specific vLLM images or flags are configurable without changing
  the telemetry and summary core.
