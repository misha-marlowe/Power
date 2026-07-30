import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "power_report.py"
SPEC = importlib.util.spec_from_file_location("power_report", MODULE_PATH)
report = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = report
SPEC.loader.exec_module(report)


class PowerReportTests(unittest.TestCase):
    def test_sparkline_preserves_width(self):
        self.assertEqual(len(report.sparkline([100, 200, 300, 400])), 4)

    def test_parse_all_views(self):
        self.assertEqual(report.parse_views("all"), report.VIEW_ORDER)

    def test_load_single_run(self):
        summary = {
            "gpu_ids": [4, 5],
            "power_scope": "GPU package",
            "return_code": 0,
            "sample_errors": [],
            "sample_interval_s": 0.5,
            "reported_power_cap_w": 1500,
            "workload": {"scenario": "smoke", "input_len": 10, "output_len": 2, "concurrency": 1, "num_prompts": 1},
            "vllm": {"model_id": "test/model", "duration": 2, "completed": 1, "failed": 0, "total_output_tokens": 2, "output_throughput": 1},
            "benchmark_window": {"requested_duration_s": 2, "covered_s": 2, "average_gpu_power_w": 500, "gpu_energy_j": 1000, "per_gpu": {"4": {"average_power_w": 255, "energy_j": 510}, "5": {"average_power_w": 245, "energy_j": 490}}},
            "idle_baseline": {"sample_count": 5, "total_w": 300, "per_gpu_w": {"4": 152, "5": 148}},
            "efficiency": {"gross_j_per_output_token": 500, "above_idle_j_per_output_token": 200, "gross_output_tokens_per_kwh": 7200, "output_tokens_per_s_per_gpu": 0.5, "average_power_fraction_of_reported_cap": 1/3, "above_idle_gpu_energy_j": 400},
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "smoke-c1"
            run_dir.mkdir()
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (run_dir / "power.jsonl").write_text(
                json.dumps({"monotonic_s": 1, "total_w": 300, "gpu_w": {"4": 152, "5": 148}}) + "\n"
                + json.dumps({"monotonic_s": 3, "total_w": 500, "gpu_w": {"4": 255, "5": 245}}) + "\n",
                encoding="utf-8",
            )
            target, runs = report.load_runs(str(run_dir))
            text = report.generate_report(target, runs, ["overview", "timeline"], 8)
            self.assertIn("test/model", text)
            self.assertIn("smoke-c1", text)
            self.assertIn("Power timelines", text)

    def test_model_scoped_directory_loads_recursively(self):
        summary = {
            "workload": {"scenario": "smoke", "input_len": 1, "output_len": 1, "concurrency": 1},
            "vllm": {"model_id": "org/model"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary) / "org--model"
            run_dir = model_dir / "20260730-000000Z" / "smoke-c1"
            run_dir.mkdir(parents=True)
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            target, runs = report.load_runs(str(model_dir))
            self.assertEqual(target, model_dir.resolve())
            self.assertEqual([run.name for run in runs], ["smoke-c1"])

    def test_time_weighted_cdf_uses_elapsed_time(self):
        run = report.RunResult(
            Path("example"),
            {
                "benchmark_window": {"start_monotonic_s": 0, "end_monotonic_s": 3},
                "vllm": {"duration": 3},
            },
            [
                {"monotonic_s": 0, "total_w": 100, "gpu_w": {"0": 100}},
                {"monotonic_s": 1, "total_w": 200, "gpu_w": {"0": 200}},
                {"monotonic_s": 3, "total_w": 300, "gpu_w": {"0": 300}},
            ],
        )
        weighted = report.time_weighted_power_samples(run)
        self.assertEqual(weighted, [(100.0, 0.5), (200.0, 1.5), (300.0, 1.0)])
        cdf = report.empirical_cdf(weighted)
        self.assertAlmostEqual(cdf[0][1], 1 / 6)
        self.assertAlmostEqual(cdf[1][1], 2 / 3)
        self.assertEqual(cdf[-1][1], 1.0)

    def test_overview_has_new_power_columns_and_failure_note(self):
        summary = {
            "gpu_ids": [0],
            "reported_power_cap_w": 750,
            "workload": {
                "scenario": "interactive", "input_len": 10, "output_len": 2,
                "concurrency": 1, "num_prompts": 2,
            },
            "vllm": {
                "duration": 2, "completed": 1, "failed": 1,
                "total_output_tokens": 2, "output_throughput": 1,
            },
            "benchmark_window": {"average_gpu_power_w": 300, "maximum_gpu_power_w": 400},
            "efficiency": {
                "gross_j_per_output_token": 300,
                "average_power_fraction_of_reported_cap": 0.4,
                "average_power_fraction_of_tdp_proxy": 0.4,
            },
        }
        text = report.overview([report.RunResult(Path("interactive-c1"), summary, [])])
        header = next(line for line in text.splitlines() if line.startswith("Run"))
        self.assertNotIn("GPUs", header)
        self.assertNotIn("Fail", header)
        self.assertIn("Max sampled W", header)
        self.assertNotIn("Power limit %", header)
        self.assertNotIn("J/out tok", header)
        self.assertIn("TDP %", header)
        self.assertIn("1 failed; completed 1/2", text)

    def test_svg_cdf_is_dependency_free_and_limited_to_five_curves(self):
        runs = []
        for concurrency in (1, 4, 16, 64, 128, 256):
            summary = {
                "return_code": 0,
                "sample_errors": [],
                "sample_interval_s": 1,
                "reported_power_cap_w": 6000,
                "workload": {
                    "scenario": "interactive", "input_len": 10, "output_len": 2,
                    "concurrency": concurrency, "num_prompts": 1,
                },
                "vllm": {
                    "duration": 31, "completed": 1, "failed": 0,
                    "total_output_tokens": 2, "output_throughput": 2,
                },
                "benchmark_window": {
                    "start_monotonic_s": 0, "end_monotonic_s": 31,
                    "requested_duration_s": 31, "covered_s": 31,
                    "average_gpu_power_w": 1000, "per_gpu": {},
                },
                "idle_baseline": {"sample_count": 5, "total_w": 100},
                "efficiency": {
                    "gross_j_per_output_token": 500,
                    "average_power_fraction_of_reported_cap": 1 / 6,
                    "above_idle_gpu_energy_j": 900,
                },
            }
            samples = [
                {"monotonic_s": 0, "total_w": 900, "gpu_w": {"0": 900}},
                {"monotonic_s": 31, "total_w": 1100, "gpu_w": {"0": 1100}},
            ]
            runs.append(report.RunResult(Path(f"interactive-c{concurrency}"), summary, samples))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "cdf.svg"
            report.write_cdf_svg(output, runs)
            svg = output.read_text(encoding="utf-8")
            self.assertIn("<svg", svg)
            self.assertIn("Time at or below power", svg)
            self.assertNotIn("concurrency 128", svg)
            self.assertEqual(svg.count("class=\"curve\""), 5)


if __name__ == "__main__":
    unittest.main()
