import unittest

import torch

from fc_pruning.fixed_mask_reconstruction import (
    apply_fixed_mask_reconstruction,
)
from tests.test_pruning import ToyLayer


class FixedMaskReconstructionTests(unittest.TestCase):
    def test_full_ls_preserves_requested_mask_and_width(self):
        torch.manual_seed(0)
        layer = ToyLayer()
        inputs = torch.randn(40, 4)
        hessian = inputs.t() @ inputs
        plan = {
            "pruned": [0],
            "direct": [0],
            "merges": [],
            "bias_compensation": torch.zeros(3),
        }
        audit = apply_fixed_mask_reconstruction(
            layer,
            hessian,
            torch.zeros(4),
            plan,
            "full_ls",
            torch.device("cpu"),
        )
        self.assertEqual(layer.mlp.gate_proj.out_features, 3)
        self.assertEqual(layer.mlp.down_proj.in_features, 3)
        self.assertEqual(audit["target"], 1)

    def test_direct_ls_retains_pair_merge_path(self):
        torch.manual_seed(0)
        layer = ToyLayer()
        inputs = torch.randn(40, 4)
        hessian = inputs.t() @ inputs
        plan = {
            "pruned": [0, 2],
            "direct": [2],
            "merges": [{"prune": 0, "keep": 1, "alpha": 0.5}],
            "bias_compensation": torch.zeros(3),
        }
        audit = apply_fixed_mask_reconstruction(
            layer,
            hessian,
            torch.zeros(4),
            plan,
            "direct_ls",
            torch.device("cpu"),
        )
        self.assertEqual(layer.mlp.gate_proj.out_features, 2)
        self.assertEqual(audit["reconstructed_sources"], 1)

    def test_centered_ls_requires_sample_count(self):
        layer = ToyLayer()
        with self.assertRaises(ValueError):
            apply_fixed_mask_reconstruction(
                layer,
                torch.eye(4),
                torch.zeros(4),
                {"pruned": [0], "direct": [0], "merges": []},
                "full_ls",
                torch.device("cpu"),
                covariance_mode="centered",
            )

    def test_centered_ls_uses_affine_covariance(self):
        torch.manual_seed(1)
        layer = ToyLayer()
        inputs = torch.randn(80, 4) + torch.tensor([2.0, -1.0, 0.5, 3.0])
        plan = {
            "pruned": [0],
            "direct": [0],
            "merges": [],
            "bias_compensation": torch.zeros(3),
        }
        audit = apply_fixed_mask_reconstruction(
            layer,
            inputs.t() @ inputs,
            inputs.mean(dim=0),
            plan,
            "full_ls",
            torch.device("cpu"),
            covariance_mode="centered",
            sample_count=inputs.shape[0],
        )
        self.assertEqual(audit["covariance_mode"], "centered")
        self.assertTrue(torch.isfinite(layer.mlp.down_proj.weight).all())

    def test_direct_ls_can_partially_reconstruct_merge_residual(self):
        torch.manual_seed(2)
        layer = ToyLayer()
        original = layer.mlp.down_proj.weight.detach().clone()
        inputs = torch.randn(80, 4)
        plan = {
            "pruned": [0, 2],
            "direct": [2],
            "merges": [{"prune": 0, "keep": 1, "alpha": 0.4}],
            "bias_compensation": torch.zeros(3),
        }
        audit = apply_fixed_mask_reconstruction(
            layer,
            inputs.t() @ inputs,
            inputs.mean(dim=0),
            plan,
            "direct_ls",
            torch.device("cpu"),
            sample_count=inputs.shape[0],
            merge_residual_weight=0.5,
        )
        self.assertEqual(audit["merge_residual_weight"], 0.5)
        self.assertFalse(torch.allclose(layer.mlp.down_proj.weight[:, 0], original[:, 1]))

if __name__ == "__main__":
    unittest.main()
