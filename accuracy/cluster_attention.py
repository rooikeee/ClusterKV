from re import L
from tkinter import NO
from typing import Optional, Tuple, List
import math

import torch
from torch import nn, tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.cache_utils import DynamicCache, Cache

from pylibraft.cluster import KMeansParams, fit
from pylibraft.neighbors import ivf_flat
import pylibraft.config
pylibraft.config.set_output_as("torch")
import rmm
# torch.set_printoptions(profile="full")

from clusterkv._clusterkv_knl import search_indices
from .cluster_cache_simulator import CacheSimulator
import os 
from flash_attn import flash_attn_func 

# Use this function as the metadata only has 2-dim
def repeat_metadata(metadata: torch.Tensor, n_rep: int) -> torch.Tensor:
    num_key_value_heads, slen = metadata.shape
    if n_rep == 1:
        return metadata
    metadata = metadata[:, None, :].expand(num_key_value_heads, n_rep, slen)
    return metadata.reshape(num_key_value_heads * n_rep, slen)

def build_cluster(prefill_key, prefill_value, nlist, balance, cluster_params, 
                  num_key_value_groups, gqa_policy, mode):
    _, num_kv_heads, prefill_len, head_dim = prefill_key.shape
    nlist_range = torch.arange(nlist, device=prefill_key.device).reshape(nlist, 1)
    cluster_key_indices = torch.empty((num_kv_heads, prefill_len), dtype=torch.int64,
                                            device=prefill_key.device)
    cluster_key_ptr = torch.empty((num_kv_heads, prefill_len), dtype=torch.int16,
                                        device=prefill_key.device)
    cluster_key_size = torch.empty((num_kv_heads, nlist), dtype=torch.int32,
                                        device=prefill_key.device)
    
    pre_rope = os.getenv("PRE_ROPE")
    if pre_rope:
        all_head_max_indices = torch.empty((num_kv_heads, nlist), dtype=torch.int32,
                                            device=prefill_key.device)
    else:
        all_head_max_indices = None
    device = prefill_key.device
    for h in range(num_kv_heads):
        head_keys = prefill_key[0, h].to(torch.float32)
        seq_len = head_keys.shape[0]
        
        if mode == "max_key_norm" or mode == "max_value_norm":
            if mode == "max_key_norm":
                magnitudes = torch.norm(head_keys.float(), p=2, dim=-1) # [m_len]
            else:
                head_values = prefill_value[0, h].to(torch.float32)
                magnitudes = torch.norm(head_values.float(), p=2, dim=-1)
            chunk_size = (prefill_len + nlist - 1) // nlist
            pad_len = chunk_size * nlist - prefill_len
            
            if pad_len > 0:
                pad_mag = torch.full((pad_len,), -1e9, device=device)
                padded_mag = torch.cat([magnitudes, pad_mag])
            else:
                padded_mag = magnitudes
                
            # [num_buckets, chunk_size]
            reshaped_mag = padded_mag.view(nlist, chunk_size)
            
            local_max_indices = torch.argmax(reshaped_mag, dim=-1)
            
            chunk_offsets = torch.arange(nlist, device=device) * chunk_size
            leader_indices = chunk_offsets + local_max_indices
            leader_indices = torch.clamp(leader_indices, max=prefill_len - 1)
            
            head_centroids = head_keys[leader_indices]
            
            for _ in range(cluster_params.max_iter):
                sim = torch.mm(
                    F.normalize(head_keys, p=2, dim=-1), 
                    F.normalize(head_centroids, p=2, dim=-1).t()
                )
                _, labels = torch.max(sim, dim=-1)
                
                new_centroids = torch.zeros_like(head_centroids)
                new_centroids.index_add_(0, labels, head_keys)
                
                counts = torch.bincount(labels, minlength=nlist).unsqueeze(1).clamp(min=1)
                head_centroids = new_centroids / counts
        else:
            if balance:
                flat_index = ivf_flat.build(cluster_params, head_keys)
                head_centroids = flat_index.centers
            else:
                head_centroids, _, _ = fit(cluster_params, head_keys)
        head_centroids = head_centroids.to(prefill_key.dtype)
        # centoid_indices: (prefill_len,)
        _, centoid_indices = torch.max(torch.mm(F.normalize(prefill_key[0, h], p=2, dim=-1), 
                                                F.normalize(head_centroids, p=2, dim=-1).t().to(prefill_key.device)), 
                                                dim=-1)
        
        if pre_rope:
            token_positions = torch.arange(seq_len, device=head_keys.device)
            max_pos_indices = torch.empty(nlist, dtype=torch.long, device=head_keys.device)
            
            max_pos_indices.scatter_reduce_(
                0, 
                centoid_indices, 
                token_positions, 
                reduce='max', 
                include_self=False
            )
            all_head_max_indices[h] = max_pos_indices
        # if centoid_indices is like [3, 1, 1, 2]
        # cluster_key_ptr is [1, 1, 2, 3], cluster_key_indices is [1, 2, 3, 0]
        cluster_key_ptr[h], cluster_key_indices[h] \
            = torch.where(centoid_indices==nlist_range)
        cluster_key_size[h] = torch.bincount(cluster_key_ptr[h], minlength=nlist)
        # self.cluster_key[0, h] = prefill_key[0, h, self.cluster_key_indices[h], :]
        
        # if self.layer_id == 10 and h == 1:
        #     print(centoid_indices)
        #     print(self.cluster_key_indices[h], self.cluster_key_ptr[h])
        #     print(self.cluster_key_size[h])
        head_centroids = head_centroids.unsqueeze(0)
        if h == 0:
            key_centroids = head_centroids
        else:
            key_centroids = torch.cat([key_centroids, head_centroids], dim=0)
    # (num_kv_heads, nlist)
    cluster_key_size_ps = torch.cumsum(cluster_key_size, dim=-1)
    # if self.layer_id == 10:
    #     print(self.cluster_key_size_ps[1])
    key_centroids = key_centroids.unsqueeze(0)
    # self.key_centroids: (1, num_kv_heads, nlist, head_dim)
    if gqa_policy is None:
        key_centroids = repeat_kv(key_centroids, num_key_value_groups)
        cluster_key_ptr = repeat_metadata(cluster_key_ptr, num_key_value_groups)
        cluster_key_size = repeat_metadata(cluster_key_size, num_key_value_groups)
        cluster_key_size_ps = repeat_metadata(cluster_key_size_ps, num_key_value_groups)

    return key_centroids, cluster_key_indices, cluster_key_ptr, cluster_key_size, cluster_key_size_ps, all_head_max_indices


def init_empty_cluster_metadata(
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    gqa_policy: Optional[str],
):
    meta_heads = num_key_value_heads if gqa_policy else num_heads
    key_centroids = torch.empty((1, meta_heads, 0, head_dim), dtype=dtype, device=device)
    cluster_key_indices = torch.empty((num_key_value_heads, 0), dtype=torch.int64, device=device)
    cluster_key_ptr = torch.empty((meta_heads, 0), dtype=torch.int16, device=device)
    cluster_key_size = torch.empty((meta_heads, 0), dtype=torch.int32, device=device)
    cluster_key_size_ps = torch.empty((meta_heads, 0), dtype=torch.int32, device=device)
    return key_centroids, cluster_key_indices, cluster_key_ptr, cluster_key_size, cluster_key_size_ps


def append_cluster_metadata(
    key_centroids,
    cluster_key_indices,
    cluster_key_ptr,
    cluster_key_size,
    cluster_key_size_ps,
    append_key_centroids,
    append_cluster_key_indices,
    append_cluster_key_ptr,
    append_cluster_key_size,
    append_cluster_key_size_ps,
):
    if key_centroids is None or key_centroids.shape[-2] == 0:
        return (
            append_key_centroids,
            append_cluster_key_indices,
            append_cluster_key_ptr,
            append_cluster_key_size,
            append_cluster_key_size_ps,
        )

    key_centroids = torch.cat([key_centroids, append_key_centroids], dim=-2).contiguous()
    cluster_key_indices = torch.cat([cluster_key_indices, append_cluster_key_indices], dim=-1).contiguous()
    cluster_key_ptr = torch.cat([cluster_key_ptr, append_cluster_key_ptr], dim=-1).contiguous()
    cluster_key_size = torch.cat([cluster_key_size, append_cluster_key_size], dim=-1).contiguous()

    if cluster_key_size_ps.shape[-1] == 0:
        append_cluster_key_size_ps = append_cluster_key_size_ps.to(cluster_key_size_ps.dtype)
    else:
        append_cluster_key_size_ps = append_cluster_key_size_ps.to(cluster_key_size_ps.dtype) + cluster_key_size_ps[:, -1:]
    cluster_key_size_ps = torch.cat([cluster_key_size_ps, append_cluster_key_size_ps], dim=-1).contiguous()

    return key_centroids, cluster_key_indices, cluster_key_ptr, cluster_key_size, cluster_key_size_ps

def build_cluster_global_greedy(prefill_key, prefill_value, nlist, balance, cluster_params, 
                  num_key_value_groups, gqa_policy, mode):
    _, num_kv_heads, prefill_len, head_dim = prefill_key.shape
    device = prefill_key.device
    dtype = prefill_key.dtype
    cluster_key_indices = torch.empty((num_kv_heads, prefill_len), dtype=torch.int64,
                                            device=prefill_key.device)
    cluster_key_ptr = torch.empty((num_kv_heads, prefill_len), dtype=torch.int16,
                                        device=prefill_key.device)
    cluster_key_size = torch.empty((num_kv_heads, nlist), dtype=torch.int32,
                                        device=prefill_key.device)
    nlist_range = torch.arange(nlist, device=prefill_key.device).reshape(nlist, 1)
    for h in range(num_kv_heads):
        post_keys = prefill_key[0, h].to(torch.float32) # [seq_len, dim]
        keys_norm = F.normalize(post_keys.float(), p=2, dim=-1)
        if mode == "max_key_norm" or mode == "max_value_norm":
            if mode == "max_key_norm":
                magnitudes = torch.norm(post_keys.float(), p=2, dim=-1) # [m_len]
            else:
                post_values = prefill_value[0, h].to(torch.float32)
                magnitudes = torch.norm(post_values.float(), p=2, dim=-1) # [m_len]
            chunk_size = (prefill_len + nlist - 1) // nlist
            pad_len = chunk_size * nlist - prefill_len
            
            if pad_len > 0:
                pad_mag = torch.full((pad_len,), -1e9, device=device)
                padded_mag = torch.cat([magnitudes, pad_mag])
            else:
                padded_mag = magnitudes
                
            # [num_buckets, chunk_size]
            reshaped_mag = padded_mag.view(nlist, chunk_size)
            local_max_indices = torch.argmax(reshaped_mag, dim=-1)
        else:
            # find first token of chunk
            local_max_indices = torch.zeros(
                nlist, 
                dtype=torch.long, 
                device=reshaped_mag.device
            )
        
        chunk_offsets = torch.arange(nlist, device=device) * chunk_size
        leader_indices = chunk_offsets + local_max_indices
        leader_indices = torch.clamp(leader_indices, max=prefill_len - 1)
        leader_keys = post_keys[leader_indices].to(prefill_key.dtype)
        leader_keys_norm = keys_norm[leader_indices]
        
        # [m_len, dim] @ [dim, actual_num_buckets] -> [m_len, actual_num_buckets]
        _, centroid_indices = torch.max(torch.mm(keys_norm, leader_keys_norm.t()), dim=-1)
        
        cluster_key_ptr[h], cluster_key_indices[h] \
            = torch.where(centroid_indices==nlist_range)
        cluster_key_size[h] = torch.bincount(cluster_key_ptr[h], minlength=nlist)
        
        leader_keys = leader_keys.unsqueeze(0)
        if h == 0:
            key_centroids = leader_keys
        else:
            key_centroids = torch.cat([key_centroids, leader_keys], dim=0)

    cluster_key_size_ps = torch.cumsum(cluster_key_size, dim=-1)
    # if self.layer_id == 10:
    #     print(self.cluster_key_size_ps[1])
    key_centroids = key_centroids.unsqueeze(0)
        
        # print(f"Head {h}: Generated {len(head_centroids)} clusters from {prefill_len} tokens.")

    if gqa_policy is None:
        key_centroids = repeat_kv(key_centroids, num_key_value_groups)
        cluster_key_ptr = repeat_metadata(cluster_key_ptr, num_key_value_groups)
        cluster_key_size = repeat_metadata(cluster_key_size, num_key_value_groups)
        cluster_key_size_ps = repeat_metadata(cluster_key_size_ps, num_key_value_groups)

    return key_centroids, cluster_key_indices, cluster_key_ptr, cluster_key_size, cluster_key_size_ps

def stat_topk(layer_id, indices, q, prefill_keys, name):
    _, num_heads, k = indices.shape
    attn_weights = torch.matmul(q, prefill_keys.transpose(2, 3))    # [1, num_heads, 1, seq_len]
    _, topk_indices = attn_weights.topk(k, dim=-1)
    topk_indices = topk_indices.squeeze(2)      # [1, 32, k]
    hit_rate = []
    for h in range(num_heads):
        truth = topk_indices[0, h].cpu()
        pred = indices[0, h].cpu()
        hit_rate.append(len( set(truth.numpy()) & set(pred.numpy()) ) / k)
    avg_hit_rate = sum(hit_rate) / len(hit_rate)
    with open(f'topk_stat/top{k}-{name}.csv', 'a') as f:
        f.write(f'layer {layer_id}, {avg_hit_rate}\n')

def cluster_attn_out(query_states, key_states, value_states, attention_mask, prompt_len,
                    key_centroids, cluster_key_indices, cluster_key_size, cluster_key_size_ps,
                    num_key_value_groups, layer_id, token_budget, sink, local_window, head_sel,
                    cluster_cache, topk_stat=False, cluster_params=None, gqa_policy=None, total_sel_cluster=None):
    bsz, num_kv_heads, kv_seq_len, head_dim = key_states.shape
    num_heads = query_states.shape[1]
    _, num_heads, q_len, _ = query_states.shape
    hidden_size = num_heads * head_dim

    include_decode_generated_tokens = prompt_len > token_budget
    sink = min(sink, kv_seq_len)
    local_window = max(local_window, 0)
    if include_decode_generated_tokens:
        # Long-context decode: no local window concept.
        # Keep all generated tokens so selected KV becomes:
        # sink + mid_select + decode_gen_token
        local_start = min(prompt_len, kv_seq_len)
        local_end = kv_seq_len
    else:
        local_end = min(prompt_len, kv_seq_len)
        local_start = max(sink, local_end - local_window) if local_window > 0 else local_end
    cluster_budget = max(token_budget - sink, 0)

    has_cluster_tokens = (
        key_centroids is not None
        and key_centroids.shape[-2] > 0
        and cluster_key_size is not None
        and cluster_key_size.shape[-1] > 0
        and cluster_budget > 0
    )

    c_dist = None
    if has_cluster_tokens:
        if gqa_policy:
            q_grouped = query_states.view(bsz, num_kv_heads, num_key_value_groups, 1, head_dim).mean(dim=2)[0]
            c_dist = torch.matmul(q_grouped.to(key_centroids.device), key_centroids.transpose(2, 3))
        else:
            c_dist = torch.matmul(query_states.to(key_centroids.device), key_centroids.transpose(2, 3))

        # c_dist: (1, num_heads or num_kv_heads, 1, nlist)
        _, c_neighbor = torch.sort(c_dist, dim=-1, descending=True)
        # (num_heads or num_kv_heads, nlist)
        c_neighbor = c_neighbor.squeeze(0).squeeze(-2).to(cluster_key_size.device)
        neighbor_cluster_size = torch.gather(cluster_key_size, -1, c_neighbor)
        neighbor_cluster_key_size_ps = torch.cumsum(neighbor_cluster_size, dim=-1)
        # get the number of needed clusters by mask smaller and get min
        thresholded_ps = neighbor_cluster_key_size_ps.clone()
        thresholded_ps[thresholded_ps < cluster_budget] = 10000000
        # num_need_clusters: (num_heads or num_kv_heads)
        _, num_need_clusters = torch.min(thresholded_ps, dim=-1)
        num_need_clusters += 1
        # now we select same number of clusters for all heads
        max_num_need_clusters = torch.max(num_need_clusters).item()

        # (num_heads or num_kv_heads, max_num_need_clusters)
        sel_cluster_indices = c_neighbor[:, :max_num_need_clusters]
        if os.getenv("GET_CLUSTERS"):
            assert total_sel_cluster is not None
            total_sel_cluster.append(sel_cluster_indices)
        sel_cluster_size = neighbor_cluster_size[:, :max_num_need_clusters]
        # not use thresholded_ps[, :max_num_need_clusters] as it has be modified
        sel_cluster_size_ps = torch.cumsum(sel_cluster_size, dim=-1)
        sel_cluster_key_end = torch.gather(cluster_key_size_ps, -1, sel_cluster_indices)
        sel_cluster_key_start = sel_cluster_key_end - sel_cluster_size

        if cluster_cache is not None:
            cluster_cache.update(sel_cluster_indices)

        # NOTE:
        # `search_indices` custom CUDA kernel may be unsafe in single-process multi-GPU
        # when the extension is built without proper device guard/stream handling.
        # Keep accuracy-first fallback for multi-GPU runs.
        use_search_kernel = torch.cuda.device_count() <= 1
        if os.getenv("FORCE_SEARCH_KERNEL") == "1":
            use_search_kernel = True
        if use_search_kernel:
            max_num_indices = torch.sum(sel_cluster_size, dim=-1).max()
            if gqa_policy:
                sel_key_indices = torch.full((num_kv_heads, max_num_indices), kv_seq_len,
                                            dtype=torch.int64, device=key_states.device)
            else:
                sel_key_indices = torch.full((num_heads, max_num_indices), kv_seq_len,
                                            dtype=torch.int64, device=key_states.device)
            # Important for multi-GPU: launch the custom kernel on the tensor's device context.
            with torch.cuda.device(sel_key_indices.device):
                search_indices(num_need_clusters,
                            sel_cluster_size_ps,
                            sel_cluster_key_start,
                            sel_cluster_key_end,
                            cluster_key_indices,
                            sel_key_indices)
            sel_key_indices = sel_key_indices[:, :cluster_budget]
        else:
            sel_key_indices = []
            meta_heads = num_kv_heads if gqa_policy else num_heads
            for h in range(meta_heads):
                kv_h = h if gqa_policy else h // num_key_value_groups
                head_num_need_clusters = num_need_clusters[h]
                head_sel_key_indices = []
                for i in range(head_num_need_clusters):
                    head_sel_key_indices.append(cluster_key_indices[
                        kv_h, sel_cluster_key_start[h, i]: sel_cluster_key_end[h, i]
                    ])
                head_sel_key_indices = torch.cat(head_sel_key_indices)
                sel_key_indices.append(head_sel_key_indices)

            if head_sel == "pad":
                sel_key_indices = pad_sequence(sel_key_indices, batch_first=True, padding_value=kv_seq_len)
            elif head_sel == "truc":
                sel_key_indices = torch.stack([ind[:cluster_budget] for ind in sel_key_indices])
            else:
                assert False

        sel_key_indices = sel_key_indices.unsqueeze(0)
        sel_key_indices += sink
        sel_key_indices[sel_key_indices > kv_seq_len] = kv_seq_len
    else:
        sel_heads = num_kv_heads if gqa_policy else num_heads
        sel_key_indices = torch.empty((1, sel_heads, 0), dtype=torch.int64, device=key_states.device)

    res_attn_weight = None
    if os.getenv("GET_ATTN") and not os.getenv("NORMAL_ATTN"):
        res_attn_weight = torch.zeros((1, num_heads, 1, prompt_len), dtype=torch.int32, device=key_states.device)
        if sel_key_indices.shape[-1] > 0 and sel_key_indices.shape[1] == num_heads:
            res_attn_weight = res_attn_weight.scatter_(dim=-1, index=sel_key_indices.unsqueeze(2), value=1.0)

    if topk_stat and c_dist is not None and sel_key_indices.shape[-1] > 0 and sel_key_indices.shape[1] == num_heads:
        sink_indices = torch.arange(sink, device=sel_key_indices.device).repeat(1, num_heads, 1)
        full_sel_key_indices = torch.cat([sink_indices, sel_key_indices], dim=-1)
        nlist = c_dist.shape[-1]
        assert cluster_params is not None
        stat_topk(layer_id, full_sel_key_indices, query_states,
                  key_states[:, :, :prompt_len, :],
                  f'nc{nlist}-fi{cluster_params.max_iter}')

    if sel_key_indices.shape[-1] > 0:
        # sel_key_indices: (1, num_heads or num_kv_heads, token_budget, head_dim)
        gather_indices = sel_key_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        if head_sel == "truc":
            device = key_states.device
            sel_key_states = key_states.gather(dim=2, index=gather_indices.to(device))
            sel_value_states = value_states.gather(dim=2, index=gather_indices.to(device))
        elif head_sel == "pad":
            kpad = torch.ones((key_states.shape[0], key_states.shape[1],
                                    1, key_states.shape[3]), dtype=key_states.dtype,
                                    device=key_states.device) * torch.finfo(key_states.dtype).min
            vpad = torch.zeros((value_states.shape[0], value_states.shape[1],
                                    1, value_states.shape[3]), dtype=value_states.dtype,
                                    device=value_states.device)
            sign = (query_states > 0) + (~(query_states > 0)) * -1
            sel_key_states = torch.cat([key_states, kpad*sign], dim=2).gather(dim=2, index=gather_indices)
            sel_value_states = torch.cat([value_states, vpad], dim=2).gather(dim=2, index=gather_indices)
        else:
            assert False
    else:
        sel_key_states = key_states[:, :, :0, :]
        sel_value_states = value_states[:, :, :0, :]

    recent_key_states = key_states[:, :, local_start:local_end, :]
    recent_value_states = value_states[:, :, local_start:local_end, :]
    sel_key_states = torch.cat([key_states[:, :, :sink, :], sel_key_states, recent_key_states], dim=2)
    sel_value_states = torch.cat([value_states[:, :, :sink, :], sel_value_states, recent_value_states], dim=2)

    if gqa_policy:
        query_states = query_states.transpose(1, 2)
        sel_key_states = sel_key_states.transpose(1, 2).contiguous()
        sel_value_states = sel_value_states.transpose(1, 2).contiguous()
        attn_output = flash_attn_func(query_states, sel_key_states, sel_value_states, causal=False)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, hidden_size)
        return attn_output, None

    if os.getenv("NORMAL_ATTN"):
        assert os.getenv("GET_ATTN") or os.getenv("GET_TOPK")
        print("in this")
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        if os.getenv("GET_TOPK"):
            _, res_attn_weight = attn_weights.topk(128, dim=-1)
            print(res_attn_weight.shape)
    else:
        attn_weights = torch.matmul(query_states.contiguous(), sel_key_states.transpose(2, 3)) / math.sqrt(head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : sel_key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    if os.getenv("NORMAL_ATTN"):
        attn_output = torch.matmul(attn_weights, value_states)
    else:
        attn_output = torch.matmul(attn_weights, sel_value_states)

    if attn_output.size() != (bsz, num_heads, q_len, head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is"
            f" {attn_output.size()}"
        )

    if not os.getenv("GET_ATTN") and not os.getenv("GET_TOPK"):
        attn_weights = None
    else:
        if res_attn_weight is not None:
            attn_weights = res_attn_weight
        print(attn_weights.shape)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)
    return attn_output, attn_weights

def streaming_attn_out(query_states, key_states, value_states, attention_mask, prompt_len, sink, budget):
    bsz, _, _, head_dim = key_states.shape
    _, num_heads, q_len, _ = query_states.shape
    hidden_size = num_heads * head_dim
    select_len = budget - sink
    include_decode_generated_tokens = prompt_len > budget
    prompt_tail_start = max(0, prompt_len - select_len)
    pieces_k = [key_states[:, :, :sink, :], key_states[:, :, prompt_tail_start:prompt_len, :]]
    pieces_v = [value_states[:, :, :sink, :], value_states[:, :, prompt_tail_start:prompt_len, :]]
    if include_decode_generated_tokens:
        pieces_k.append(key_states[:, :, prompt_len:, :])
        pieces_v.append(value_states[:, :, prompt_len:, :])
    sel_key_states = torch.cat(pieces_k, dim=2).contiguous()
    sel_value_states = torch.cat(pieces_v, dim=2)
    
    attn_weights = torch.matmul(query_states, sel_key_states.transpose(2, 3)) / math.sqrt(head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : sel_key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
   
    attn_output = torch.matmul(attn_weights, sel_value_states)

    if attn_output.size() != (bsz, num_heads, q_len, head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)
    return attn_output


def build_decode_cluster_params(cluster_params, balance, nlist):
    if balance:
        fit_iter = getattr(cluster_params, "kmeans_n_iters", 20)
        return ivf_flat.IndexParams(
            n_lists=nlist,
            metric='inner_product',
            kmeans_n_iters=fit_iter,
            kmeans_trainset_fraction=1,
            add_data_on_build=False,
        )
    max_iter = getattr(cluster_params, "max_iter", 20)
    metric = getattr(cluster_params, "metric", "cosine")
    return KMeansParams(n_clusters=nlist, max_iter=max_iter, metric=metric)


def maybe_append_decode_clusters(self, key_states, value_states, sink):
    update_interval = getattr(self, "cluster_update_interval", 320)
    append_nlist = getattr(self, "cluster_update_nlist", 4)
    if update_interval <= 0 or append_nlist <= 0:
        return
    # Only keep decode-generated tokens when prompt is longer than token budget.
    if self.prompt_len <= self.token_budget:
        return

    generated_len = key_states.shape[-2] - self.prompt_len
    if generated_len <= 0 or generated_len % update_interval != 0:
        return

    clustered_decode_tokens = getattr(self, "clustered_decode_tokens", 0)
    if generated_len <= clustered_decode_tokens:
        return

    append_start = max(sink, key_states.shape[-2] - update_interval)
    append_end = key_states.shape[-2]
    if append_end - append_start < append_nlist:
        return

    append_key = key_states[..., append_start:append_end, :].contiguous()
    append_value = value_states[..., append_start:append_end, :].contiguous()
    append_cluster_params = build_decode_cluster_params(self.cluster_params, self.balance, append_nlist)

    if os.getenv("GREEDY"):
        (
            append_key_centroids,
            append_cluster_key_indices,
            append_cluster_key_ptr,
            append_cluster_key_size,
            append_cluster_key_size_ps,
        ) = build_cluster_global_greedy(
            append_key,
            append_value,
            append_nlist,
            self.balance,
            append_cluster_params,
            self.num_key_value_groups,
            self.gqa_policy,
            self.mode,
        )
    else:
        (
            append_key_centroids,
            append_cluster_key_indices,
            append_cluster_key_ptr,
            append_cluster_key_size,
            append_cluster_key_size_ps,
            append_all_head_max_indices,
        ) = build_cluster(
            append_key,
            append_value,
            append_nlist,
            self.balance,
            append_cluster_params,
            self.num_key_value_groups,
            self.gqa_policy,
            self.mode,
        )
        if append_all_head_max_indices is not None:
            self.all_head_max_indices = append_all_head_max_indices

    # cluster indices are stored in the "without-sink" coordinate system.
    append_cluster_key_indices = append_cluster_key_indices + (append_start - sink)
    (
        self.key_centroids,
        self.cluster_key_indices,
        self.cluster_key_ptr,
        self.cluster_key_size,
        self.cluster_key_size_ps,
    ) = append_cluster_metadata(
        self.key_centroids,
        self.cluster_key_indices,
        self.cluster_key_ptr,
        self.cluster_key_size,
        self.cluster_key_size_ps,
        append_key_centroids,
        append_cluster_key_indices,
        append_cluster_key_ptr,
        append_cluster_key_size,
        append_cluster_key_size_ps,
    )
    self.clustered_decode_tokens = generated_len

def forward_cluster(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    assert bsz == 1

    if not hasattr(self, "num_heads"):
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

    if not hasattr(self, "long_decode_sink"):
        self.long_decode_sink = getattr(self, "sink", 0)
    if not hasattr(self, "local_window"):
        if hasattr(self, "window") and self.window is not None:
            self.local_window = self.window
        else:
            self.local_window = 0
    if not hasattr(self, "cluster_update_interval"):
        self.cluster_update_interval = 320
    if not hasattr(self, "cluster_update_nlist"):
        self.cluster_update_nlist = 4
    if not hasattr(self, "clustered_decode_tokens"):
        self.clustered_decode_tokens = 0

    sink = self.long_decode_sink
    local_window = self.local_window

    cached_kv_len = 0
    if past_key_value is not None:
        layer_cache = past_key_value[self.layer_id]
        if layer_cache is not None and layer_cache[0] is not None:
            cached_kv_len = layer_cache[0].shape[-2]
    current_kv_len = cached_kv_len + q_len

    current_layer = self.layer_idx
    num_layers = self.config.num_hidden_layers
    is_heavy_layer = (current_layer == 0) or (current_layer == num_layers - 1)
    if is_heavy_layer or q_len > 1 \
        or current_kv_len < self.token_budget:
        if q_len > 1:
            self.prompt_len = q_len
            self.clustered_decode_tokens = 0
            # reset cache for each request
            if self.cache_steps > 0 and self.layer_id >= 2:
                self.cluster_cache = CacheSimulator(self.layer_id, self.cache_steps+1)

        return self.flash_forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_value,
            cache_position,
            **kwargs,
        )

    prefill_key = past_key_value[self.layer_id][0]
    prefill_value = past_key_value[self.layer_id][1]
    if prefill_key.shape[-2] <= sink:
        return self.flash_forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_value,
            cache_position,
            **kwargs,
        )

    # clustering for prefilled keys: only the middle region [sink, prompt_len-local_window)
    if self.key_centroids is None:
        prompt_middle_end = max(sink, self.prompt_len - local_window)
        prefill_middle_key = prefill_key[..., sink:prompt_middle_end, :]
        prefill_middle_value = prefill_value[..., sink:prompt_middle_end, :]

        if prefill_middle_key.shape[-2] == 0:
            (
                self.key_centroids,
                self.cluster_key_indices,
                self.cluster_key_ptr,
                self.cluster_key_size,
                self.cluster_key_size_ps,
            ) = init_empty_cluster_metadata(
                self.num_heads,
                self.num_key_value_heads,
                self.head_dim,
                prefill_key.dtype,
                prefill_key.device,
                self.gqa_policy,
            )
            self.all_head_max_indices = None
            self.cluster_key = prefill_middle_key
        else:
            if self.nlist == 0:
                self.nlist = max(1, prefill_middle_key.shape[-2] // 80)
                self.cluster_params = build_decode_cluster_params(self.cluster_params, self.balance, self.nlist)
            if os.getenv("GREEDY"):
                self.key_centroids, self.cluster_key_indices, \
                self.cluster_key_ptr, self.cluster_key_size, self.cluster_key_size_ps = \
                build_cluster_global_greedy(prefill_middle_key, prefill_middle_value, self.nlist, self.balance, self.cluster_params,
                            self.num_key_value_groups, self.gqa_policy, self.mode)
            else:
                self.key_centroids, self.cluster_key_indices, \
                self.cluster_key_ptr, self.cluster_key_size, self.cluster_key_size_ps, self.all_head_max_indices = \
                build_cluster(prefill_middle_key, prefill_middle_value, self.nlist, self.balance, self.cluster_params,
                            self.num_key_value_groups, self.gqa_policy, self.mode)
            if self.cluster_key is None:
                self.cluster_key = prefill_middle_key
        self.clustered_decode_tokens = 0

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )

    if hasattr(self, "q_norm") and self.q_norm is not None:
        query_states = self.q_norm(query_states)
    if hasattr(self, "k_norm") and self.k_norm is not None:
        key_states = self.k_norm(key_states)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_id)
        # every 320 generated tokens append 4 new clusters
        maybe_append_decode_clusters(self, key_states, value_states, sink)

    if self.gqa_policy is None:
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

    token_budget = min(self.prompt_len, self.token_budget)
    attn_weights = None
    if os.getenv("STREAMING"):
        attn_output = streaming_attn_out(query_states, key_states, value_states, attention_mask,
                                         self.prompt_len, sink, token_budget)
    else:
        attn_output, attn_weights = cluster_attn_out(
            query_states, key_states, value_states, attention_mask,
            self.prompt_len, self.key_centroids, self.cluster_key_indices,
            self.cluster_key_size, self.cluster_key_size_ps,
            self.num_key_value_groups, self.layer_id, token_budget,
            sink, local_window, self.head_sel, self.cluster_cache, self.topk_stat, self.cluster_params, self.gqa_policy, self.total_sel_cluster
        )

    attn_output = self.o_proj(attn_output)
    if os.getenv("GET_ATTN") or os.getenv("GET_TOPK"):
        assert attn_weights is not None

        attn_weights = attn_weights.squeeze(0)
        if self.attn_weight is None:
            self.attn_weight = attn_weights
        else:
            self.attn_weight = torch.cat([self.attn_weight, attn_weights], dim=1)

    attn_weights = None

    return attn_output, attn_weights


def split_tensor_along_last_dim(
        tensor: torch.Tensor,
        num_partitions: int,
        contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list

@torch.jit.script
def glm_apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [b, np, sq, hn]
    b, np, sq, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:, :sq]
    xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
    rope_cache = rope_cache.view(-1, 1, sq, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)

def forward_cluster_glm(
    self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
):
    bsz, q_len, _ = hidden_states.size()
    assert bsz == 1

    cached_kv_len = 0
    if kv_cache is not None:
        cached_kv_len = kv_cache[0].shape[-2]
    current_kv_len = cached_kv_len + q_len

    if q_len > 1 or self.layer_number < 3 \
        or current_kv_len < self.token_budget:
        if q_len > 1:
            self.prompt_len = q_len
            if self.cache_steps > 0 and self.layer_number>= 2:
                self.cluster_cache = CacheSimulator(self.layer_number, self.cache_steps+1)
        return self.flash_forward(
            hidden_states,
            attention_mask,
            rotary_pos_emb,
            kv_cache,
            use_cache
        )

    sink = self.sink
    prefill_key = kv_cache[0]
    prefill_key = prefill_key[..., sink:, :]
    num_key_value_group = self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition
    # clustering for prefilled keys
    if self.key_centroids is None:
        self.key_centroids, self.cluster_key_indices, \
        self.cluster_key_ptr, self.cluster_key_size, self.cluster_key_size_ps = \
		build_cluster(prefill_key, self.nlist, self.balance, self.cluster_params, 
                    num_key_value_group, self.gqa_policy)
    # hidden_states: [b, sq, h]

    # =================================================
    # Pre-allocate memory for key-values for inference.
    # =================================================
    # =====================
    # Query, Key, and Value
    # =====================

    # Attention heads [b, sq, h] --> [b, sq, (np * 3 * hn)]
    mixed_x_layer = self.query_key_value(hidden_states)

    if self.multi_query_attention:
        (query_layer, key_layer, value_layer) = mixed_x_layer.split(
            [
                self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
            ],
            dim=-1,
        )
        query_layer = query_layer.view(
            query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        )
        key_layer = key_layer.view(
            key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
        )
        value_layer = value_layer.view(
            value_layer.size()[:-1]
            + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
        )
    else:
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
                            (self.num_attention_heads_per_partition,
                            3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [b, sq, np, 3 * hn] --> 3 [b, sq, np, hn]
        (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

    # [b, sq, np, hn] -> [b, np, sq, hn]
    query_layer, key_layer, value_layer = [k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]]

    # apply relative positional encoding (rotary embedding)
    if rotary_pos_emb is not None:
        query_layer = glm_apply_rotary_pos_emb(query_layer, rotary_pos_emb)
        key_layer = glm_apply_rotary_pos_emb(key_layer, rotary_pos_emb)

    # adjust key and value for inference
    if kv_cache is not None:
        cache_k, cache_v = kv_cache
        key_layer = torch.cat((cache_k, key_layer), dim=2)
        value_layer = torch.cat((cache_v, value_layer), dim=2)
    if use_cache:
        if kv_cache is None:
            kv_cache = torch.cat((key_layer.unsqueeze(0).unsqueeze(0), value_layer.unsqueeze(0).unsqueeze(0)),
                                    dim=1)
        else:
            kv_cache = (key_layer, value_layer)
    else:
        kv_cache = None

    if self.multi_query_attention:
        key_layer = key_layer.unsqueeze(2)
        key_layer = key_layer.expand(
            -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
        )
        key_layer = key_layer.contiguous().view(
            key_layer.size()[:1] + (self.num_attention_heads_per_partition,) + key_layer.size()[3:]
        )
        value_layer = value_layer.unsqueeze(2)
        value_layer = value_layer.expand(
            -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
        )
        value_layer = value_layer.contiguous().view(
            value_layer.size()[:1] + (self.num_attention_heads_per_partition,) + value_layer.size()[3:]
        )

    # ==================================
    # core attention computation
    # ==================================

    # context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
    token_budget = min(self.prompt_len, self.token_budget)
    context_layer, _ = cluster_attn_out(
        query_layer, key_layer, value_layer, attention_mask, 
        self.prompt_len, self.key_centroids, self.cluster_key_indices, 
        self.cluster_key_size, self.cluster_key_size_ps,
        num_key_value_group, self.layer_number, token_budget, 
        sink, getattr(self, "local_window", getattr(self, "window", 0) or 0), self.head_sel, self.cluster_cache, self.topk_stat, self.cluster_params
    )

    # =================
    # Output. [sq, b, h]
    # =================

    output = self.dense(context_layer)

    return output, kv_cache

MAX_POOL_SIZE = 10*1024**3
def cluster_reset(model):
    if isinstance(model, PreTrainedModel):
        rmm.reinitialize(pool_allocator=True, initial_pool_size=MAX_POOL_SIZE, maximum_pool_size=MAX_POOL_SIZE)
        torch.cuda.empty_cache()
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            cluster_reset(module)
        module.key_centroids = None
        module.cluster_key_indices = None
        module.cluster_key_ptr = None
        module.cluster_key_size = None
        module.cluster_key_size_ps = None
        module.cluster_key = None
        module.all_head_max_indices = None
        module.total_sel_cluster = []
        module.clustered_decode_tokens = 0
        if os.getenv("GET_ATTN") or os.getenv("GET_TOPK"):
            module.attn_weight = None
             
def apply_cluster_config(module, args):
    nlist = args.nlist
    module.nlist = nlist
    module.head_sel = args.head_sel
    module.balance = True if args.balance else False
    module.sink = args.sink
    module.long_decode_sink = args.sink
    module.local_window = args.window if args.window is not None else 0
    module.cluster_update_interval = 320
    module.cluster_update_nlist = 4
    module.clustered_decode_tokens = 0
    module.gqa_policy = args.gqa_policy
    module.mode = args.mode
    if args.window is not None:
        module.window = args.window
    if args.balance:
        module.cluster_params = ivf_flat.IndexParams(
        n_lists=nlist, metric='inner_product', kmeans_n_iters=args.fit_iter,
        kmeans_trainset_fraction=1, add_data_on_build=False)
    else:
        module.cluster_params = KMeansParams(
            n_clusters=nlist, max_iter=args.fit_iter, metric=args.dist_t)


