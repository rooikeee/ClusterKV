import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from clusterkv.clusterkv_utils import ClusterKVController, build_cluster, append_kv, prefill_forward, decode_sparse_attn, update_sel_indices
import clusterkv.utils

class ClusterKVAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = config.pretraining_tp
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        # rope_theta is default to 1e4, as set in RoPE kernel API.
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
            self.rope_scale = 1.0
        else:
            rope_scaling = self.config.rope_scaling
            if isinstance(rope_scaling, dict):
                # Compatible with both {"type": ...} and {"rope_type": ...} formats.
                self.rope_scale = float(rope_scaling.get("factor", 1.0))
            else:
                self.rope_scale = float(rope_scaling)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _forward_single(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        q_len: int,
        controller: ClusterKVController,
        shared_sel_token_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.layer_idx >= 2 and not controller.full:
            if q_len > 1:
                # build clusters during prefill
                assert q_len > controller.sink
                if controller.overlap_build:
                    with torch.cuda.stream(controller.build_cluster_stream):
                        build_cluster(
                            controller,
                            self.layer_idx,
                            key_states[controller.sink:],
                            0,
                            controller.nlist,
                            controller.build_cluster_stream,
                        )
                        controller.build_cluster_events[self.layer_idx].record(controller.build_cluster_stream)
                else:
                    build_cluster(
                        controller,
                        self.layer_idx,
                        key_states[controller.sink:],
                        0,
                        controller.nlist,
                        torch.cuda.default_stream(),
                    )

        torch.cuda.nvtx.range_push("append_kv")
        append_kv(
            key_states,
            value_states,
            controller,
            self.layer_idx,
        )
        torch.cuda.nvtx.range_pop()

        if self.layer_idx >= 2 and not controller.full:
            if controller.window > 0 and q_len == 1 and controller.generated_len % controller.window == 0:
                # appending clustering during decoding
                append_key_for_cluster = controller.get_app_k_clustering(self.layer_idx)
                build_cluster(
                    controller,
                    self.layer_idx,
                    append_key_for_cluster,
                    controller.kv_seqlen - controller.sink - controller.window,
                    controller.window_nlist,
                    torch.cuda.default_stream(),
                )
                if controller.should_offload_layer(self.layer_idx):
                    controller.offload_window_kv(self.layer_idx)

        # Prefill/Decode kernels are different.
        if q_len > 1:
            torch.cuda.nvtx.range_push("prefill_attn")
            if controller.offload:
                attn_output = prefill_forward(
                    query_states,
                    controller,
                    self.layer_idx,
                    key_states=key_states,
                    value_states=value_states,
                )
            else:
                attn_output = prefill_forward(
                    query_states,
                    controller,
                    self.layer_idx,
                )
            torch.cuda.nvtx.range_pop()
            if controller.should_offload_layer(self.layer_idx):
                controller.offload_prefill_kv(self.layer_idx, key_states, value_states)
            return attn_output, None

        # Decode stage.
        if not controller.need_estimate():
            torch.cuda.nvtx.range_push("full_attn")
            attn_output = decode_sparse_attn(
                query_states,
                controller,
                self.layer_idx,
                None,
            )
            torch.cuda.nvtx.range_pop()
            return attn_output, None

        torch.cuda.nvtx.range_push("indexing")
        if not controller.build_cluster_finish[self.layer_idx]:
            controller.build_cluster_events[self.layer_idx].wait(controller.build_cluster_stream)
            controller.build_cluster_finish[self.layer_idx] = True
        if shared_sel_token_indices is None:
            update_sel_indices(
                query_states,
                controller,
                self.layer_idx,
            )
            shared_sel_token_indices = controller.sel_token_indices
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("approx_attn")
        attn_output = decode_sparse_attn(
            query_states,
            controller,
            self.layer_idx,
            shared_sel_token_indices,
        )
        torch.cuda.nvtx.range_pop()
        return attn_output, shared_sel_token_indices

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        controller: Optional[ClusterKVController] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        assert hasattr(self, 'layer_idx'), "ClusterKVAttention requires layer_idx to inference."
        if controller is None:
            raise ValueError("ClusterKVAttention requires a valid controller.")

        if isinstance(controller, (list, tuple)):
            controllers = list(controller)
        else:
            controllers = [controller]
        if bsz > 1 and len(controllers) != bsz:
            raise ValueError(
                f"Batch size is {bsz}, but got {len(controllers)} controller(s). "
                "Please call clusterkv_init(batch_size=...) for batched inference."
            )
        if bsz == 1 and len(controllers) != 1:
            raise ValueError(f"Batch size is 1, but got {len(controllers)} controllers.")

        if self.pretraining_tp > 1:
            assert False and "should not happen"
        else:
            torch.cuda.nvtx.range_push("qkv_proj")
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            torch.cuda.nvtx.range_pop()
        
        # Keep NHD layout for Append/KM kernels.
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        attn_output_per_batch = []
        shared_sel_token_indices = None
        for b_idx in range(bsz):
            current_controller = controllers[b_idx]
            single_q = query_states[b_idx]
            single_k = key_states[b_idx]
            single_v = value_states[b_idx]

            torch.cuda.nvtx.range_push("RoPE")
            # -q_len as kv_seqlen has been increased in prepare_metadata
            clusterkv.utils.apply_rope_in_place(
                single_q,
                single_k,
                current_controller.kv_seqlen - q_len,
                rope_scale=self.rope_scale,
                rope_theta=self.config.rope_theta,
            )
            torch.cuda.nvtx.range_pop()

            # Reuse batch-0 selected indices for all batches in decode sparse stage.
            # This follows "same cluster selection across batch" requirement.
            shared_for_this = shared_sel_token_indices if (b_idx > 0 and q_len == 1) else None
            single_output, sel_token_indices = self._forward_single(
                single_q,
                single_k,
                single_v,
                q_len,
                current_controller,
                shared_sel_token_indices=shared_for_this,
            )
            if b_idx == 0 and sel_token_indices is not None:
                shared_sel_token_indices = sel_token_indices

            attn_output_per_batch.append(single_output)

        attn_output = torch.stack(attn_output_per_batch, dim=0)
        # FlashInfer output is naturally NHD
        # Note that we manually control NHD. Should be more general.
        if attn_output.size() != (bsz, q_len, self.num_heads, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, q_len, self.num_heads, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        torch.cuda.nvtx.range_push("o_proj")
        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)
        torch.cuda.nvtx.range_pop()

        attn_weights = None

        return attn_output, attn_weights, past_key_value
