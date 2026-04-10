import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.models.llama.configuration_llama import LlamaConfig

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
        self.decode_cluster_interval = 320
        self.decode_append_nlist = 4
        self._init_rope()

    def _init_rope(self):
        # We use custom in-place RoPE kernel (`apply_rope_in_place`) during forward.
        # Keep only the scale parsing for compatibility across LLaMA/Qwen and
        # different Transformers versions.
        self.rotary_emb = None
        rope_scaling = getattr(self.config, "rope_scaling", None)
        if rope_scaling is None:
            self.rope_scale = 1.0
            return
        if isinstance(rope_scaling, dict):
            # Compatible with both {"type": ...} and {"rope_type": ...} formats.
            self.rope_scale = float(rope_scaling.get("factor", 1.0))
        else:
            self.rope_scale = float(rope_scaling)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _use_strict_cpu_cluster_mode(self, controller: ClusterKVController) -> bool:
        return bool(
            getattr(controller, "offload", False)
            and getattr(controller, "offload_all_layers", False)
            and not getattr(controller, "full", False)
        )

    def _wait_offload_to_cpu(self, controller: ClusterKVController, device: torch.device):
        if controller.offload_events is None:
            return
        event = controller.offload_events[self.layer_idx]
        if event is None:
            return
        stream = torch.cuda.current_stream(device=device)
        stream.wait_event(event)
        stream.synchronize()

    def _maybe_build_prefill_clusters(
        self,
        key_states: torch.Tensor,
        controller: ClusterKVController,
    ):
        if self.layer_idx < 2 or controller.full:
            return
        if key_states.shape[0] <= controller.sink:
            return
        build_cluster(
            controller,
            self.layer_idx,
            key_states[controller.sink:],
            0,
            max(int(controller.nlist), 1),
            torch.cuda.default_stream(),
        )

    def _maybe_build_decode_clusters(
        self,
        controller: ClusterKVController,
        query_device: torch.device,
    ):
        if self.layer_idx < 2 or controller.full:
            return
        generated_len = controller.generated_len
        if generated_len <= 0 or generated_len % self.decode_cluster_interval != 0:
            return
        if controller.kv_cache_cpu[self.layer_idx] is None:
            return

        chunk_end = min(controller.kv_seqlen, controller.max_seq_len)
        chunk_start = max(controller.prompt_len, chunk_end - self.decode_cluster_interval)
        if chunk_end - chunk_start < self.decode_append_nlist:
            return

        self._wait_offload_to_cpu(controller, query_device)
        chunk_keys_cpu = controller.kv_cache_cpu[self.layer_idx][chunk_start:chunk_end, 0, ...]
        chunk_keys = chunk_keys_cpu.to(query_device, non_blocking=True)
        key_offset = max(chunk_start - controller.sink, 0)
        build_cluster(
            controller,
            self.layer_idx,
            chunk_keys,
            key_offset,
            self.decode_append_nlist,
            torch.cuda.default_stream(),
        )

    def _build_head_token_indices(
        self,
        controller: ClusterKVController,
        kv_len: int,
        head_idx: int,
        cluster_indices: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        pieces = []
        sink_len = min(max(int(controller.sink), 0), kv_len)
        if sink_len > 0:
            pieces.append(torch.arange(sink_len, dtype=torch.long, device=device))

        if cluster_indices is not None and cluster_indices.numel() > 0:
            pieces.append(cluster_indices[head_idx].to(device=device, dtype=torch.long))

        if controller.cur_win_size > 0:
            pieces.append(controller.cur_win_indices[head_idx].to(device=device, dtype=torch.long))

        if pieces:
            indices = torch.cat(pieces, dim=0)
        else:
            indices = torch.arange(kv_len, dtype=torch.long, device=device)

        if kv_len > 0:
            indices = indices.clamp(min=0, max=kv_len - 1)
        return indices

    def _decode_attention_from_cpu(
        self,
        query_states: torch.Tensor,  # [1, num_heads, head_dim]
        controller: ClusterKVController,
        cluster_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if controller.kv_cache_cpu[self.layer_idx] is None:
            raise RuntimeError(
                "Strict CPU ClusterKV mode requires per-layer CPU KV cache, but cache is missing."
            )

        device = query_states.device
        self._wait_offload_to_cpu(controller, device)

        kv_len = min(controller.kv_seqlen, controller.max_seq_len)
        if kv_len <= 0:
            return torch.zeros((1, self.num_heads, self.head_dim), dtype=query_states.dtype, device=device)

        kv_cpu = controller.kv_cache_cpu[self.layer_idx][:kv_len]
        cpu_k = kv_cpu[:, 0, ...]  # [kv_len, num_kv_heads, head_dim]
        cpu_v = kv_cpu[:, 1, ...]
        q = query_states[0]        # [num_heads, head_dim]
        scale = 1.0 / math.sqrt(self.head_dim)
        outputs = []

        for h in range(self.num_heads):
            head_indices = self._build_head_token_indices(
                controller=controller,
                kv_len=kv_len,
                head_idx=h,
                cluster_indices=cluster_indices,
                device=device,
            )
            head_indices_cpu = head_indices.to(device="cpu", dtype=torch.long)
            kv_h = h // self.num_key_value_groups

            head_k = cpu_k.index_select(0, head_indices_cpu)[:, kv_h, :].to(device, non_blocking=True)
            head_v = cpu_v.index_select(0, head_indices_cpu)[:, kv_h, :].to(device, non_blocking=True)
            logits = torch.matmul(head_k, q[h]) * scale
            probs = torch.softmax(logits.to(torch.float32), dim=0).to(q.dtype)
            head_out = torch.matmul(probs.unsqueeze(0), head_v).squeeze(0)
            outputs.append(head_out)

        return torch.stack(outputs, dim=0).unsqueeze(0)

    def _forward_single_strict_cpu(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        q_len: int,
        controller: ClusterKVController,
        shared_sel_token_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if q_len > 1:
            self._maybe_build_prefill_clusters(key_states, controller)
            torch.cuda.nvtx.range_push("prefill_attn")
            attn_output = prefill_forward(
                query_states,
                controller,
                self.layer_idx,
                key_states=key_states,
                value_states=value_states,
            )
            torch.cuda.nvtx.range_pop()
            controller.offload_prefill_kv(self.layer_idx, key_states, value_states)
            return attn_output, None

        # Decode: CPU is the source of truth for KV cache.
        controller.offload_decode_kv(self.layer_idx, key_states, value_states)
        self._maybe_build_decode_clusters(controller, query_states.device)

        selected_cluster_indices = None
        if self.layer_idx >= 2 and not controller.full:
            has_cluster_metadata = (
                controller.centroids[self.layer_idx] is not None
                and controller.cluster_size[self.layer_idx] is not None
                and controller.cluster_size_ps[self.layer_idx] is not None
                and controller.cluster_key_indices[self.layer_idx] is not None
            )
            if has_cluster_metadata:
                if shared_sel_token_indices is None:
                    update_sel_indices(
                        query_states,
                        controller,
                        self.layer_idx,
                    )
                    shared_sel_token_indices = controller.sel_token_indices
                selected_cluster_indices = shared_sel_token_indices

        torch.cuda.nvtx.range_push("cpu_decode_attn")
        attn_output = self._decode_attention_from_cpu(
            query_states,
            controller,
            selected_cluster_indices,
        )
        torch.cuda.nvtx.range_pop()
        return attn_output, shared_sel_token_indices

    def _forward_single(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        q_len: int,
        controller: ClusterKVController,
        shared_sel_token_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._use_strict_cpu_cluster_mode(controller):
            return self._forward_single_strict_cpu(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                q_len=q_len,
                controller=controller,
                shared_sel_token_indices=shared_sel_token_indices,
            )

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
        if q_len == 1 and controller.should_offload_layer(self.layer_idx):
            # Keep CPU as source-of-truth cache during decode.
            controller.offload_decode_kv(self.layer_idx, key_states, value_states)

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
        if (not controller.need_estimate()) or (self.layer_idx < 2):
            torch.cuda.nvtx.range_push("full_attn")
            attn_output = decode_sparse_attn(
                query_states,
                controller,
                self.layer_idx,
                None,
            )
            torch.cuda.nvtx.range_pop()
            return attn_output, None

        has_cluster_metadata = (
            controller.centroids[self.layer_idx] is not None
            and controller.cluster_size[self.layer_idx] is not None
            and controller.cluster_size_ps[self.layer_idx] is not None
            and controller.cluster_key_indices[self.layer_idx] is not None
        )
        if not has_cluster_metadata:
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
        if controller.overlap_build and (not controller.build_cluster_finish[self.layer_idx]):
            torch.cuda.current_stream(device=query_states.device).wait_event(
                controller.build_cluster_events[self.layer_idx]
            )
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
