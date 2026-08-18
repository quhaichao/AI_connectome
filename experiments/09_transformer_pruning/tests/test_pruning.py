import unittest

import torch
from torch import nn

from fc_pruning.gram_similarity import (
    GramPlanConfig,
    plan_from_candidate_pool,
)

from fc_pruning.data import (
    _contains_aligned_token_sequence,
    _wikitext_articles_without_headings,
    _wikitext_heading_level,
)
from fc_pruning.pruning import (
    _fit_no_intercept,
    apply_layer_plan,
    output_importance,
    plan_importance_method,
    plan_similarity_pruning,
)

from scripts.analyze_fc_matrix_stability import matrix_metrics


class ToyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(3, 4, bias=False)
        self.up_proj = nn.Linear(3, 4, bias=False)
        self.down_proj = nn.Linear(4, 3, bias=False)
        self.intermediate_size = 4


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = ToyMLP()


class PruningTests(unittest.TestCase):
    def test_gram_plan_prefers_exact_merge_and_protects_keeper(self):
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.down_proj = nn.Linear(4, 3, bias=False)
        pool = {
            "source": torch.tensor([[0], [0], [2], [2]], dtype=torch.int32),
            "keeper": torch.tensor([[1], [1], [3], [3]], dtype=torch.int32),
            "similarity": torch.tensor([[1.0], [1.0], [0.1], [0.1]]),
            "fit_cross": torch.tensor([[1.0], [1.0], [0.0], [0.0]]),
            "fit_source_second": torch.ones(4, 1),
            "fit_keeper_second": torch.ones(4, 1),
            "holdout_cross": torch.tensor([[1.0], [1.0], [0.0], [0.0]]),
            "holdout_source_second": torch.ones(4, 1),
            "holdout_keeper_second": torch.ones(4, 1),
            "fit_mean": torch.zeros(4),
            "holdout_mean": torch.zeros(4),
            "total_mean": torch.zeros(4),
            "importance": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "output_norm_sq": torch.ones(4),
            "method": "fc",
        }
        config = GramPlanConfig(
            topk=1,
            merge_fraction=1.0,
            keeper_capacity=1,
            minimum_output_gain=0.5,
            protect_fraction=0.0,
            ridge_relative=0.0,
        )
        plan, audit = plan_from_candidate_pool(layer, pool, 0.5, config)
        self.assertEqual(plan["pruned"], [0, 2])
        self.assertEqual(plan["merges"][0]["keep"], 1)
        self.assertNotIn(1, plan["direct"])
        self.assertEqual(audit["merges"], 1)

    def test_token_leakage_search_requires_token_alignment(self):
        sequence = bytes(range(16))
        self.assertTrue(_contains_aligned_token_sequence(b"xxxx" + sequence, sequence))
        self.assertFalse(_contains_aligned_token_sequence(b"x" + sequence, sequence))

    def test_wikitext_headings_and_article_split(self):
        self.assertEqual(_wikitext_heading_level(" = Article = "), 1)
        self.assertEqual(_wikitext_heading_level(" = = History = = "), 2)
        self.assertEqual(_wikitext_heading_level("ordinary prose"), 0)
        articles = _wikitext_articles_without_headings(
            [" = One = ", "first", " = = Plot = = ", "second", " = Two = ", "third"]
        )
        self.assertEqual(articles, [("= One =", ["first", "second"]), ("= Two =", ["third"])])

    def test_no_intercept_fit_recovers_scaled_activation(self):
        receiver = torch.linspace(-2, 2, 100)
        activations = torch.stack([2.5 * receiver, receiver], dim=1)
        alpha, residual, _ = _fit_no_intercept(activations, 0, 1, ridge=0.0)
        self.assertLess(abs(alpha - 2.5), 1e-6)
        self.assertLess(residual, 1e-12)

    def test_output_importance_has_expected_shape(self):
        activations = torch.randn(20, 4)
        down = torch.randn(3, 4)
        importance = output_importance(activations, down)
        self.assertEqual(importance.shape, (4,))
        self.assertTrue(torch.all(importance >= 0))

    def test_simultaneous_merge_preserves_exact_redundancy(self):
        layer = ToyLayer()
        original = layer.mlp.down_proj.weight.detach().clone()
        plan = {
            "pruned": [0],
            "merges": [type("Merge", (), {"prune": 0, "keep": 1, "alpha": 2.0})()],
        }
        apply_layer_plan(layer, plan)
        self.assertEqual(layer.mlp.gate_proj.out_features, 3)
        self.assertEqual(layer.mlp.up_proj.out_features, 3)
        self.assertEqual(layer.mlp.down_proj.in_features, 3)
        self.assertTrue(
            torch.allclose(layer.mlp.down_proj.weight[:, 0], original[:, 1] + 2 * original[:, 0])
        )

    def test_method_specific_merge_cap_overrides_global_cap(self):
        activations = torch.randn(24, 4)
        config = {
            "similarity_fit_fraction": 0.75,
            "ridge_relative": 0.0001,
            "activation_energy_relative_floor": 0.0,
            "protect_top_importance_fraction": 0.0,
            "dead_importance_median_ratio": 0.0,
            "topk_receivers": 2,
            "similarity_block_size": 4,
            "minimum_source_similarity": -1.0,
            "minimum_validation_reconstruction_gain": -1.0,
            "selection_rule": "joint_cost",
            "max_merges_per_keeper": 8,
            "max_merge_fraction": 1.0,
            "max_merge_fraction_by_method": {"fc": 0.0},
            "sampled_positions_per_sequence": 6,
        }
        layer = ToyLayer()
        plan = plan_similarity_pruning(
            "fc",
            layer,
            activations,
            ratio=0.5,
            config=config,
            device=torch.device("cpu"),
        )
        self.assertEqual(plan["merges"], [])

    def test_flap_hybrid_with_zero_merge_cap_matches_flap(self):
        torch.manual_seed(7)
        activations = torch.randn(24, 4) + torch.tensor([0.0, 0.5, -1.0, 2.0])
        config = {
            "similarity_fit_fraction": 0.75,
            "similarity_direct_method": "flap",
            "direct_bias_compensation": True,
            "ridge_relative": 0.0001,
            "activation_energy_relative_floor": 0.0,
            "protect_top_importance_fraction": 0.0,
            "dead_importance_median_ratio": 0.0,
            "topk_receivers": 2,
            "similarity_block_size": 4,
            "minimum_source_similarity": -1.0,
            "minimum_validation_reconstruction_gain": -1.0,
            "selection_rule": "joint_cost",
            "max_merges_per_keeper": 8,
            "max_merge_fraction": 1.0,
            "max_merge_fraction_by_method": {"fc_flap": 0.0},
            "sampled_positions_per_sequence": 6,
        }
        layer = ToyLayer()
        flap = plan_importance_method("flap", layer, activations, ratio=0.5)
        hybrid = plan_similarity_pruning(
            "fc_flap",
            layer,
            activations,
            ratio=0.5,
            config=config,
            device=torch.device("cpu"),
        )
        self.assertEqual(hybrid["merges"], [])
        self.assertEqual(hybrid["pruned"], flap["pruned"])
        self.assertTrue(
            torch.allclose(hybrid["bias_compensation"], flap["bias_compensation"])
        )

        config["max_merge_fraction_by_method"]["fc_flap"] = 1.0
        config["minimum_source_similarity"] = 1.0
        thresholded = plan_similarity_pruning(
            "fc_flap",
            layer,
            activations,
            ratio=0.5,
            config=config,
            device=torch.device("cpu"),
        )
        self.assertEqual(thresholded["merges"], [])
        self.assertEqual(thresholded["pruned"], flap["pruned"])

    def test_fc_matrix_metrics_identical_and_permuted_neighbors(self):
        matrix = torch.tensor(
            [[1.0, 0.9, 0.2], [0.9, 1.0, 0.4], [0.2, 0.4, 1.0]]
        )
        metrics = matrix_metrics(matrix, matrix, topk=1)
        self.assertAlmostEqual(metrics["pearson"], 1.0, places=6)
        self.assertAlmostEqual(metrics["cosine"], 1.0, places=6)
        self.assertAlmostEqual(metrics["mae"], 0.0, places=6)
        self.assertAlmostEqual(metrics["top32_overlap"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
