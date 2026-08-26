import unittest
from types import SimpleNamespace

import torch
from torch import nn

from fc_pruning.fang_reproduction import build_and_apply_fang_flap_plans
from fc_pruning.gram_similarity import (
    GramPlanConfig,
    _center_gram,
    _topk_from_gram,
    build_layer_candidate_pool,
    plan_from_candidate_pool,
)

from fc_pruning.data import (
    _contains_aligned_token_sequence,
    _wikitext_articles_without_headings,
    _wikitext_heading_level,
)
from fc_pruning.pruning import (
    apply_layer_plan,
)
from fc_pruning.sobp_reproduction import apply_sobp_layer

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
    def test_fc_neighbors_use_signed_pearson_correlation(self):
        base = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        activations = torch.stack(
            [
                base,
                3.0 * base + 10.0,
                -2.0 * base + 10.0,
            ],
            dim=1,
        )
        centered_gram = _center_gram(
            activations.t() @ activations,
            activations.sum(dim=0),
            activations.shape[0],
        )
        similarities, neighbors = _topk_from_gram(
            centered_gram, topk=2, block_size=2
        )

        self.assertEqual(int(neighbors[0, 0]), 1)
        self.assertAlmostEqual(float(similarities[0, 0]), 1.0, places=6)
        self.assertEqual(int(neighbors[0, 1]), 2)
        self.assertAlmostEqual(float(similarities[0, 1]), -1.0, places=6)

    def test_candidate_pool_records_signed_pearson_protocol(self):
        base = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        fit = torch.stack(
            [
                base,
                3.0 * base + 10.0,
                -2.0 * base + 10.0,
                torch.tensor([0.0, 1.0, 0.0, 1.0]),
            ],
            dim=1,
        )
        holdout = fit + 0.1
        total = torch.cat([fit, holdout], dim=0)
        pool = build_layer_candidate_pool(
            ToyLayer(),
            fit.t() @ fit,
            holdout.t() @ holdout,
            fit.sum(dim=0),
            holdout.sum(dim=0),
            total.sum(dim=0),
            fit.shape[0],
            holdout.shape[0],
            total.shape[0],
            topk=1,
            block_size=2,
            method="fc",
        )

        self.assertEqual(pool["similarity_metric"], "signed_pearson")
        self.assertEqual(
            {int(pool["source"][0, 0]), int(pool["keeper"][0, 0])},
            {0, 1},
        )
        self.assertAlmostEqual(float(pool["similarity"][0, 0]), 1.0, places=6)

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

    def test_sobp_updates_layer_intermediate_size(self):
        layer = ToyLayer()
        audit = apply_sobp_layer(
            layer,
            torch.eye(4),
            target=1,
            device=torch.device("cpu"),
        )
        self.assertEqual(audit["kept"], 3)
        self.assertEqual(layer.mlp.intermediate_size, 3)
        self.assertEqual(layer.mlp.gate_proj.out_features, 3)
        self.assertEqual(layer.mlp.down_proj.in_features, 3)

    def test_fang_plan_uses_canonical_name_and_shared_adapter(self):
        model = SimpleNamespace(
            config=SimpleNamespace(intermediate_size=4),
            model=SimpleNamespace(layers=[ToyLayer()]),
        )
        statistics = {
            "taylor": torch.tensor([[[4.0, 3.0, 2.0, 1.0]]]),
            "mean": torch.zeros(1, 1, 4),
            "variance": torch.ones(1, 1, 4),
        }
        plans = build_and_apply_fang_flap_plans(
            model,
            statistics,
            ratio=0.25,
            clusters=1,
            apply=False,
        )
        self.assertEqual(plans[0]["method"], "fang_flap_paper_reproduction")
        self.assertEqual(plans[0]["target"], 1)
        self.assertEqual(len(plans[0]["pruned"]), 1)

if __name__ == "__main__":
    unittest.main()
