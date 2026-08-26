import unittest
from types import SimpleNamespace

from torch import nn

from fc_pruning.modeling import MODEL_KEYS, decoder_layers, validate_prunable_model


def _toy_model(model_type: str):
    mlp = SimpleNamespace(
        gate_proj=nn.Linear(3, 5, bias=False),
        up_proj=nn.Linear(3, 5, bias=False),
        down_proj=nn.Linear(5, 3, bias=False),
    )
    return SimpleNamespace(
        config=SimpleNamespace(model_type=model_type),
        model=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)]),
    )


class ModelingTests(unittest.TestCase):
    def test_registry_contains_paper_and_qwen_models(self):
        self.assertEqual(MODEL_KEYS, ("llama32_1b", "qwen25_1_5b"))

    def test_shared_decoder_adapter_accepts_llama_and_qwen2(self):
        for model_type in ("llama", "qwen2"):
            model = _toy_model(model_type)
            validate_prunable_model(model)
            self.assertEqual(len(decoder_layers(model)), 1)

    def test_shared_decoder_adapter_rejects_inconsistent_ffn_width(self):
        model = _toy_model("qwen2")
        model.model.layers[0].mlp.down_proj = nn.Linear(4, 3, bias=False)
        with self.assertRaises(ValueError):
            validate_prunable_model(model)


if __name__ == "__main__":
    unittest.main()
