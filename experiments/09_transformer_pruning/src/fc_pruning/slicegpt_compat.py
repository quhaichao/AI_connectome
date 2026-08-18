from __future__ import annotations

import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


def build_current_llama_adapter():
    """Build a SliceGPT adapter compatible with transformers 4.57.

    SliceGPT pins transformers 4.41. Its rotation and slicing implementation
    remains unchanged; only the compressed decoder forward signature is
    updated for the current Hugging Face cache/position API.
    """
    from slicegpt.adapters.llama_adapter import LlamaModelAdapter

    class CurrentCompressedLlamaDecoderLayer(LlamaDecoderLayer):
        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values=None,
            use_cache: bool | None = False,
            cache_position: torch.LongTensor | None = None,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
            **kwargs,
        ) -> torch.Tensor:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if self.attn_shortcut_Q is not None:
                residual = torch.matmul(residual, self.attn_shortcut_Q)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            if self.mlp_shortcut_Q is not None:
                residual = torch.matmul(residual, self.mlp_shortcut_Q)
            return residual + hidden_states

    class CurrentLlamaModelAdapter(LlamaModelAdapter):
        @property
        def compressed_layer_type(self) -> type:
            return CurrentCompressedLlamaDecoderLayer

        def convert_layer_to_compressed(self, layer, layer_idx):
            compressed = self.compressed_layer_type(self.config, layer_idx).to(
                next(layer.parameters()).dtype
            )
            compressed.load_state_dict(layer.state_dict(), strict=True)
            return compressed

    return CurrentLlamaModelAdapter
