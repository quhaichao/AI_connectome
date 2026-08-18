import unittest

import torch

from fc_pruning.adaptive_allocation import fang_asa_targets, global_layer_targets

from fc_pruning.compute_budget import equal_macs_budget, llama_layer_macs
from fc_pruning.ocp_dynamic import OCPController
from fc_pruning.svd_compression import (
    factorize_from_output_covariance,
    factorize_input_whitened_linear,
)


class ComputeBudgetTests(unittest.TestCase):
    def test_fang_asa_preserves_budget_and_bounds(self):
        complexity = torch.linspace(0.01, 0.32, 32)
        targets, ratios = fang_asa_targets(complexity, 11008, 0.2)
        self.assertEqual(sum(targets), round(32 * 11008 * 0.2))
        self.assertGreaterEqual(min(ratios), 0.1)
        self.assertLessEqual(max(ratios), 0.3)
        self.assertGreater(targets[0], targets[-1])

    def test_global_layer_targets_preserves_budget_and_cap(self):
        scores = [torch.arange(10).float(), torch.arange(10, 20).float()]
        targets = global_layer_targets(scores, total=8, maximum_fraction=0.5)
        self.assertEqual(targets, [5, 3])

    def test_llama2_7b_equal_macs_budget(self):
        budget = equal_macs_budget(4096, 11008, 2048, 0.2)
        self.assertEqual(budget.svd_rank, 2388)
        self.assertEqual(budget.slice_hidden_size, 3144)
        self.assertLess(
            abs(budget.svd_ffn_macs / budget.target_ffn_macs - 1.0),
            1e-3,
        )
        self.assertLess(
            abs(budget.slice_model_macs / budget.target_model_macs - 1.0),
            2e-3,
        )

    def test_layer_macs_increase_with_dimension(self):
        self.assertLess(
            llama_layer_macs(2048, 11008, 2048),
            llama_layer_macs(4096, 11008, 2048),
        )

    def test_full_whitening_equivalent_forms(self):
        torch.manual_seed(0)
        inputs = torch.randn(40, 6)
        linear = torch.nn.Linear(6, 8, bias=False)
        covariance = inputs.t() @ inputs
        rank = 3
        compressed = factorize_input_whitened_linear(
            linear, rank, covariance, torch.device("cpu")
        )
        product = compressed.u_proj.weight @ compressed.v_proj.weight
        weighted = linear.weight @ torch.linalg.cholesky(covariance)
        left = torch.linalg.svd(weighted, full_matrices=False).U[:, :rank]
        expected = left @ left.t() @ linear.weight
        torch.testing.assert_close(product, expected, rtol=1e-5, atol=1e-5)

        outputs = inputs @ linear.weight.t()
        output_covariance = outputs.t() @ outputs
        compressed_output = factorize_from_output_covariance(
            linear, rank, output_covariance, torch.device("cpu")
        )
        output_product = (
            compressed_output.u_proj.weight @ compressed_output.v_proj.weight
        )
        torch.testing.assert_close(
            output_product, expected, rtol=2e-5, atol=2e-5
        )

    def test_ocp_drift_correction_preserves_active_budget(self):
        controller = OCPController(
            number_layers=8,
            first_pruned_layer=3,
            overall_target=0.2,
            beta=0.0,
            gamma=0.1,
            clip_delta=1.0,
        )
        controller.begin_batch()
        densities = [0.01, 0.05, 0.02, 0.08, 0.03]
        ratios = [
            controller.allocate(layer, density)
            for layer, density in zip(range(3, 8), densities)
        ]
        self.assertAlmostEqual(
            sum(ratios), controller.active_layers * controller.active_target,
            places=2,
        )


if __name__ == "__main__":
    unittest.main()
