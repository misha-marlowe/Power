# Codex instructions for this directory

Read `CODEX_TASK.md` and `README.md` before acting. The goal is a reproducible,
architecture-portable AMD vLLM inference power benchmark, with this MI300X host
as the current target.

Begin with read-only discovery:

```bash
./bench doctor
./bench models
```

Ask the user for a model and GPU count/IDs only if they were not supplied. Use
`configs/smoke.toml` first. Do not start a server on devices that `doctor` shows
as occupied, do not stop containers that this harness did not create, and do
not run a realistic or large-KV sweep until the smoke profile completes with no
failed requests and credible telemetry.

For an explicitly requested full eight-GPU accuracy sweep, use
`./run-all-scenarios`. Its preflight must pass without `--allow-busy`; never
bypass that check for comparative power measurements.

The main entry point is `./bench`. Keep raw power samples and exact commands in
each results directory. Preserve the distinction between GPU socket/package
energy and whole-node wall energy. Do not describe `amd-smi` GPU power as
facility, rack, PSU-input, or total-system power.

Portability is a requirement: detect GPU IDs, market name, GFX architecture,
VRAM, power cap, and available telemetry at run time. Add architecture-specific
behavior through config or a clearly isolated adapter, not MI300/MI455X checks
scattered through the benchmark logic.

Current host baseline observed 2026-07-29:

- 8 × AMD Instinct MI300X VF, `gfx942`, about 191.7 GiB VRAM each.
- Reported maximum graphics-package power: 750 W per GPU.
- ROCm 7.2.4; `amd-smi` 26.2.2; amdgpu 6.16.13.
- `kernel.numa_balancing=0`.
- Configured local image: `rocm/vllm:rocm7.13.0_gfx94X-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1`.
- Docker is available to this account through non-interactive `sudo docker`.

That baseline is evidence, not a permanent assumption; rerun `doctor` before a
benchmark. The original MI455X rollout is retained under `context/` only as a
historical reference. Its `gfx1250` compatibility image and old cumulative
energy-counter commands do not apply to this MI300X host.
