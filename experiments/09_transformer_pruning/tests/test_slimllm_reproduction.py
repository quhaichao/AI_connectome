import unittest

import torch
from torch import nn

from fc_pruning.slimllm_reproduction import (
    apply_slimllm_plan,
    fit_slimllm_output_affine,
    select_slimllm_channels,
    slimllm_importance_scores,
)


class ToyMLP(nn.Module):
    def __init__(self, hidden=3, intermediate=5):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.intermediate_size = intermediate


class ToyLayer(nn.Module):
    def __init__(self, hidden=3, intermediate=5):
        super().__init__()
        self.mlp = ToyMLP(hidden, intermediate)


class SlimLLMReproductionTests(unittest.TestCase):
    def test_official_piecewise_keeps_lowest_one_percent(self):
        scores = torch.arange(100, dtype=torch.float32)
        pruned, keep = select_slimllm_channels(scores, 20, "official_piecewise")
        self.assertEqual(pruned.tolist(), list(range(1, 21)))
        self.assertIn(0, keep.tolist())
        self.assertEqual(keep.numel(), 80)

    def test_direct_lowest_prunes_exact_target(self):
        scores = torch.tensor([5.0, 1.0, 4.0, 0.0, 3.0, 2.0])
        pruned, keep = select_slimllm_channels(scores, 2, "direct_lowest")
        self.assertEqual(pruned.tolist(), [1, 3])
        self.assertEqual(keep.numel(), 4)

    def test_importance_matches_explicit_equations(self):
        torch.manual_seed(2)
        layer = ToyLayer(hidden=3, intermediate=5)
        hidden = torch.randn(19, 3)
        intermediate = torch.randn(19, 5)
        output = intermediate @ layer.mlp.down_proj.weight.detach().t()
        scores, _audit = slimllm_importance_scores(
            layer,
            hidden.t() @ hidden,
            intermediate.square().sum(0),
            output.t() @ output,
            intermediate.sum(0),
            hidden.shape[0],
            torch.device("cpu"),
        )

        centered = output - output.mean(0)
        eigenvalues, eigenvectors = torch.linalg.eigh(centered.t() @ centered)
        eigenvalues = eigenvalues.abs()
        weights = torch.sigmoid(eigenvalues / eigenvalues.mean())
        mapped = (
            layer.mlp.down_proj.weight.detach().t() @ eigenvectors
        ).abs() * weights
        expected = torch.linalg.vector_norm(
            layer.mlp.gate_proj.weight.detach() * hidden.square().sum(0).sqrt(), dim=1
        )
        expected += torch.linalg.vector_norm(
            layer.mlp.up_proj.weight.detach() * hidden.square().sum(0).sqrt(), dim=1
        )
        expected += torch.linalg.vector_norm(
            mapped * intermediate.square().sum(0).sqrt().unsqueeze(1), dim=1
        )
        torch.testing.assert_close(scores, expected, rtol=1e-5, atol=1e-5)

    def test_affine_moments_match_direct_fit_and_application(self):
        torch.manual_seed(7)
        layer = ToyLayer(hidden=3, intermediate=5)
        intermediate = torch.randn(80, 5) + torch.tensor([1.0, -0.5, 2.0, 0.0, 0.7])
        weight = layer.mlp.down_proj.weight.detach().float()
        dense = intermediate @ weight.t()
        pruned = torch.tensor([1, 4])
        keep = torch.tensor([0, 2, 3])
        retained = intermediate[:, keep] @ weight[:, keep].t()
        scale, bias, _audit = fit_slimllm_output_affine(
            weight,
            intermediate.t() @ intermediate,
            intermediate.sum(0),
            dense.t() @ dense,
            pruned,
            intermediate.shape[0],
            torch.device("cpu"),
        )
        expected_scale = ((retained - retained.mean(0)) * (dense - dense.mean(0))).sum(0)
        expected_scale /= (retained - retained.mean(0)).square().sum(0)
        expected_bias = dense.mean(0) - expected_scale * retained.mean(0)
        torch.testing.assert_close(scale, expected_scale, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(bias, expected_bias, rtol=2e-5, atol=2e-5)

        plan = {
            "pruned": pruned.tolist(),
            "affine_scale": scale.tolist(),
            "affine_bias": bias.tolist(),
        }
        apply_slimllm_plan(layer, plan, use_regression=True)
        recovered = layer.mlp.down_proj(intermediate[:, keep])
        torch.testing.assert_close(
            recovered, retained * scale + bias, rtol=2e-5, atol=2e-5
        )
        self.assertEqual(layer.mlp.gate_proj.out_features, 3)
        self.assertEqual(layer.mlp.down_proj.in_features, 3)


if __name__ == "__main__":
    unittest.main()
