import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "powerbench.py"
SPEC = importlib.util.spec_from_file_location("powerbench", MODULE_PATH)
powerbench = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(powerbench)


class EnergyIntegrationTests(unittest.TestCase):
    def test_model_result_key_is_readable_and_safe(self):
        self.assertEqual(
            powerbench.model_result_key("MiniMaxAI/MiniMax-M3-MXFP8"),
            "MiniMaxAI--MiniMax-M3-MXFP8",
        )
        self.assertEqual(powerbench.model_result_key("/models/My Model"), "models--My_Model")

    def test_constant_power_window(self):
        samples = [
            {"monotonic_s": 0.0, "gpu_w": {"0": 100.0, "1": 200.0}},
            {"monotonic_s": 1.0, "gpu_w": {"0": 100.0, "1": 200.0}},
            {"monotonic_s": 2.0, "gpu_w": {"0": 100.0, "1": 200.0}},
        ]
        result = powerbench.integrate_power(samples, [0, 1], 0.25, 1.75)
        self.assertAlmostEqual(result["covered_s"], 1.5)
        self.assertAlmostEqual(result["gpu_energy_j"], 450.0)
        self.assertAlmostEqual(result["average_gpu_power_w"], 300.0)
        self.assertAlmostEqual(result["maximum_gpu_power_w"], 300.0)
        self.assertAlmostEqual(result["per_gpu"]["1"]["maximum_power_w"], 200.0)

    def test_linear_power_uses_trapezoid(self):
        samples = [
            {"monotonic_s": 0.0, "gpu_w": {"0": 0.0}},
            {"monotonic_s": 2.0, "gpu_w": {"0": 200.0}},
        ]
        result = powerbench.integrate_power(samples, [0], 0.5, 1.5)
        self.assertAlmostEqual(result["gpu_energy_j"], 100.0)
        self.assertAlmostEqual(result["maximum_gpu_power_w"], 150.0)
        self.assertEqual(result["start_monotonic_s"], 0.5)
        self.assertEqual(result["end_monotonic_s"], 1.5)

    def test_power_unit_conversion(self):
        self.assertEqual(powerbench._quantity_value({"value": 750, "unit": "W"}), 750.0)
        self.assertEqual(powerbench._quantity_value({"value": 750000, "unit": "mW"}), 750.0)

    def test_calibrated_prompt_count_targets_duration_and_waves(self):
        prompts, plan = powerbench.calibrated_prompt_plan(
            base_prompts=8,
            concurrency=16,
            target_duration_s=60,
            calibration_prompts=16,
            calibration_completed=16,
            calibration_duration_s=4,
            minimum_waves=2,
            duration_safety_factor=1.15,
            max_num_prompts=8192,
        )
        self.assertEqual(prompts, 276)
        self.assertFalse(plan["capped"])

    def test_calibrated_prompt_count_respects_cap(self):
        prompts, plan = powerbench.calibrated_prompt_plan(
            base_prompts=4,
            concurrency=1,
            target_duration_s=60,
            calibration_prompts=1,
            calibration_completed=1,
            calibration_duration_s=0.001,
            minimum_waves=2,
            duration_safety_factor=1.15,
            max_num_prompts=8192,
        )
        self.assertEqual(prompts, 8192)
        self.assertTrue(plan["capped"])


if __name__ == "__main__":
    unittest.main()
