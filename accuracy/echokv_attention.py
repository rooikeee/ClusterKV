import math
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
# 必须安装 flash-attn: pip install flash-attn
from flash_attn import flash_attn_func 
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import os
import types

@triton.jit
def _echokv_1d_nms_kernel(
    qk_probs_ptr, 
    anchors_ptr,
    seq_len, 
    num_anchors, 
    window_size,
    stride_probs_h,    # 允许输入张量在显存中不连续的步长保护
    stride_anchors_h,
    BLOCK_SIZE: tl.constexpr
):
    head_idx = tl.program_id(0)
    
    probs_offset = head_idx * stride_probs_h
    anchor_offset = head_idx * stride_anchors_h
    
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < seq_len
    
    scores = tl.load(qk_probs_ptr + probs_offset + offs, mask=mask, other=-float('inf'))
    scores = scores.to(tl.float32) 

    for i in range(num_anchors):
        # 找出当前存活的最高分绝对索引
        best_idx = tl.argmax(scores, axis=0)
        
        # 写入外部显存
        tl.store(anchors_ptr + anchor_offset + i, best_idx)
        
        # 构造掩码：精准复刻 [idx - window_size, idx + window_size) 的切片范围
        is_in_window = (offs >= (best_idx - window_size)) & (offs < (best_idx + window_size))
        
        scores = tl.where(is_in_window, -float('inf'), scores)

def echokv_get_initial_anchors_kv(qk_probs, num_anchors=32, window_size=32):
    """
    全并行 Triton 加速版 1D-NMS (EchoKV 核心寻峰算子)
    qk_probs 形状: [kv_heads, seq_len]
    """
    kv_heads, seq_len = qk_probs.shape
    device = qk_probs.device
    
    # 准备输出容器
    anchors = torch.empty((kv_heads, num_anchors), dtype=torch.long, device=device)
    
    # 获取能够容纳 seq_len 的最小 2 的幂次作为 Block 尺寸
    BLOCK_SIZE = triton.next_power_of_2(seq_len)
    
    grid = (kv_heads,)
    _echokv_1d_nms_kernel[grid](
        qk_probs, 
        anchors, 
        seq_len, 
        num_anchors, 
        window_size,
        qk_probs.stride(0),
        anchors.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # 维持单调递增排序逻辑，为后续的 Gather 和显存拼装做准备
    anchors, _ = torch.sort(anchors, dim=-1)
    
    return anchors

@triton.jit
def _exact_greedy_nms_kernel(
    scores_ptr, indices_ptr, out_anchors_ptr, old_anchors_ptr,
    num_anchors, suppression_radius, N,
    BLOCK_SIZE: tl.constexpr
):
    # 1. 映射：每个 KV Head 分配一个独立的 GPU 线程块 (Block)
    head_idx = tl.program_id(0)
    head_offset = head_idx * N
    
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # 🌟 修复关键点：载入后，立刻将其强制转换为 FP32！
    # 这样进入循环的是 FP32，和后面的 -float('inf') 完美匹配，不再报错！
    scores = tl.load(scores_ptr + head_offset + offs, mask=mask, other=-float('inf'))
    scores = scores.to(tl.float32) 
    
    indices = tl.load(indices_ptr + head_offset + offs, mask=mask, other=-1)
    
    # 4. 纯 SRAM 内的极速贪心循环
    for k in range(num_anchors):
        # 寻找当前存活的最大分数 (Triton 中 1D 张量的 max 返回标量)
        best_score = tl.max(scores, axis=0)
        best_idx = tl.argmax(scores, axis=0)
        
        # 提取绝对坐标
        is_best = offs == best_idx
        best_pos = tl.max(tl.where(is_best, indices, -1), axis=0)
        
        # --- 饥荒兜底机制 ---
        old_anchor = tl.load(old_anchors_ptr + head_idx * num_anchors + k)
        final_pos = tl.where(best_score == -float('inf'), old_anchor, best_pos)
        
        # 将找到的锚点写入外部全局显存
        tl.store(out_anchors_ptr + head_idx * num_anchors + k, final_pos)
        
        # --- SRAM 内部掩码惩罚 ---
        distances = tl.abs(indices - final_pos)
        penalty_mask = distances < suppression_radius
        
        # 此时 scores 是 FP32，-float('inf') 也是 FP32，类型完美一致！
        scores = tl.where(penalty_mask, -float('inf'), scores)

def echokv_triton_exact_nms(flat_scores, flat_indices, num_anchors, suppression_radius, old_anchors):
    kv_heads, N = flat_scores.shape
    device = flat_scores.device
    out_anchors = torch.empty((kv_heads, num_anchors), dtype=torch.long, device=device)
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    grid = (kv_heads,)
    _exact_greedy_nms_kernel[grid](
        flat_scores, flat_indices, out_anchors, old_anchors,
        num_anchors, suppression_radius, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    out_anchors, _ = torch.sort(out_anchors, dim=-1)
    return out_anchors

def echokv_bidirectional_boundaries_kv(anchors, page_size, middle_start, middle_end):
    """双向倒逼，榨干预算，绝不越界"""
    num_anchors = anchors.shape[-1]
    ideal_starts = torch.clamp(anchors - page_size // 2, min=middle_start)
    
    penalty = torch.arange(num_anchors, device=anchors.device) * page_size
    y = ideal_starts - penalty
    max_y, _ = torch.cummax(y, dim=-1)
    starts_lr = max_y + penalty
    
    max_valid_start = middle_end - page_size 
    upper_bounds = max_valid_start - torch.arange(num_anchors - 1, -1, -1, device=anchors.device) * page_size
    
    return torch.min(starts_lr, upper_bounds)

def update_page_digest(self,
                       k,):
    sink_size = self.sink
    _, kv_seq_len, num_kv_heads, head_dim = k.shape
    page_size = self.page_size
    if self.num_pages == 0:
        num_init_pages = (kv_seq_len-sink_size) // page_size
        assert num_init_pages < self.max_k.shape[1]
        if num_init_pages > 0:
            paged_k = k[:, sink_size:sink_size+num_init_pages*page_size, :, :].reshape(
                k.shape[0], num_init_pages, page_size, num_kv_heads, head_dim
            )
            mins = paged_k.min(dim=2).values
            maxs = paged_k.max(dim=2).values
            self.min_k[:, :num_init_pages, ...] = mins
            self.max_k[:, :num_init_pages, ...] = maxs
            self.num_pages += num_init_pages
    elif (kv_seq_len-sink_size) // page_size > self.num_pages:
        new_paged_k = k[:, sink_size + self.num_pages*page_size:, ...].reshape(
            k.shape[0], page_size, num_kv_heads, head_dim 
        )
        mins = new_paged_k.amin(dim=1)
        maxs = new_paged_k.amax(dim=1)
        self.min_k[:, self.num_pages, ...] = mins
        self.max_k[:, self.num_pages, ...] = maxs
        self.num_pages += 1

def repeat_kv_BLH(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None,:].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

def quest_sel(q, min_k_for_sel, max_k_for_sel, GQA_policy, num_heads, num_kv_heads):
    if GQA_policy == "avgQ":
        # [bsz, 1, num_kv_heads, head_dim]
        grouped_q = q.reshape(
            q.shape[0], q.shape[1], num_kv_heads, num_heads // num_kv_heads, q.shape[-1]
        ).mean(dim=-2)
        # [bsz, num_used_pages, num_kv_heads, head_dim]
        q_min_k = grouped_q * min_k_for_sel
        q_max_k = grouped_q * max_k_for_sel
        # [bsz, num_kv_heads, num_used_pages]
        max_qk = torch.maximum(q_min_k, q_max_k).sum(dim=-1).transpose(1, 2)
    elif GQA_policy in ["maxS", "avgS", "avgSM", "avgSdM"]:
        # [bsz, num_used_pages, num_heads, head_dim]
        q_min_k = q * repeat_kv_BLH(min_k_for_sel, num_heads // num_kv_heads)
        q_max_k = q * repeat_kv_BLH(max_k_for_sel, num_heads // num_kv_heads)
        # [bsz, num_heads, num_used_pages]
        max_qk = torch.maximum(q_min_k, q_max_k).sum(dim=-1).transpose(1, 2)
        # [bsz, num_kv_heads, num_used_pages]
        if GQA_policy == "maxS":
            max_qk = max_qk.reshape(
                max_qk.shape[0], num_kv_heads, num_heads//num_kv_heads, max_qk.shape[-1]
            ).max(dim=-2).values
        elif GQA_policy == "avgS":
            max_qk = max_qk.reshape(
                max_qk.shape[0], num_kv_heads, num_heads//num_kv_heads, max_qk.shape[-1]
            ).mean(dim=-2)
        elif GQA_policy == "avgSM":
            max_qk = F.softmax(max_qk, dim=-1)
            max_qk = max_qk.reshape(
                max_qk.shape[0], num_kv_heads, num_heads//num_kv_heads, max_qk.shape[-1]
            ).mean(dim=-2)
        elif GQA_policy == "avgSdM":
            max_qk = F.softmax(max_qk / math.sqrt(q.shape[-1]), dim=-1)
            max_qk = max_qk.reshape(
                max_qk.shape[0], num_kv_heads, num_heads//num_kv_heads, max_qk.shape[-1]
            ).mean(dim=-2)
    else:
        assert False
    
    return max_qk  

def echokv_llama_forward(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None, past_key_value=None, **kwargs):
    # ==========================================
    # 0. 初始化状态参数 (无纠错设定)
    # ==========================================
    bsz, q_len, _ = hidden_states.size()
    
    device = hidden_states.device
    # if not hasattr(self, "echo_anchors"):
    #     self.page_size = self.page_size   
    #     self.sink = self.sink         
    #     self.window = self.window     
    #     self.echo_num_anchors = (self.token_budget-self.sink-self.window) // self.page_size   
    #     self.echo_anchors = None # [kv_heads, num_anchors]
    #     self.corr_threshold = 0.9
    #     self.corr_count = 0

    #     self.num_heads = self.config.num_attention_heads
    #     self.num_key_value_heads = self.config.num_key_value_heads
    #     self.hidden_size = self.config.hidden_size
    #     self.head_dim = self.hidden_size // self.num_heads

    current_layer = self.layer_idx
    num_layers = self.config.num_hidden_layers
   
    heavy_layers = [0, 1, num_layers-1, num_layers-2]
    is_heavy_layer = current_layer in heavy_layers 
    
    # is_heavy_layer = (current_layer == 0) or (current_layer == num_layers-1) or (current_layer == 13)

    if self.num_pages == 0 and not is_heavy_layer:
        max_page_size = 128 * 1024
        max_page_num = max_page_size // self.page_size
        self.min_k = torch.empty((bsz, max_page_num, self.num_key_value_heads, self.head_dim), 
                                       device=device,
                                       dtype=self.q_proj.weight.dtype,)
        self.max_k = torch.empty((bsz, max_page_num, self.num_key_value_heads, self.head_dim), 
                                       device=device,
                                       dtype=self.q_proj.weight.dtype,)
        self.num_window_pages = self.window // self.page_size
        self.page_budget = (self.token_budget-self.sink-self.window) // self.page_size 
        self.update_page_digest = types.MethodType(update_page_digest, self)

    # 获取当前层信息，判断是否为 U-Shaped 的“重装层”
    # is_heavy_layer = False

    # QKV 投影
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # 重塑为 [bsz, seq_len, heads, head_dim]
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    
    if hasattr(self, "q_norm") and self.q_norm is not None:
        query_states = self.q_norm(query_states)
    if hasattr(self, "k_norm") and self.k_norm is not None:
        key_states = self.k_norm(key_states)

    # RoPE 旋转位置编码 (Hugging Face 原生算子通常需要 transpose)
    # 算完后再转回 [bsz, seq_len, heads, head_dim]
    q_trans = query_states.transpose(1, 2)
    k_trans = key_states.transpose(1, 2)
    cos, sin = position_embeddings
    q_trans, k_trans = apply_rotary_pos_emb(q_trans, k_trans, cos, sin)
    
    query_states = q_trans.transpose(1, 2)
    key_states = k_trans.transpose(1, 2)

    # KV Cache 更新
    if past_key_value is not None:
        # HF update API 通常需要 [bsz, heads, seq_len, head_dim]
        k_up, v_up = past_key_value.update(k_trans, value_states.transpose(1, 2), self.layer_idx, {"sin": sin, "cos": cos})
        key_states = k_up.transpose(1, 2)   # 转回 [bsz, seq_len, kv_heads, head_dim]
        value_states = v_up.transpose(1, 2)

    total_seq_len = key_states.shape[1]
    num_q_per_kv = self.num_heads // self.num_key_value_heads
    
    scale = 1.0 / math.sqrt(self.head_dim)
    if os.getenv("GET_TOPK"):
        # 提取转置后的张量用于矩阵乘法 [bsz, heads, seq_len, head_dim]
        q_t = query_states.transpose(1, 2)
        k_t = key_states.transpose(1, 2)
        v_t = value_states.transpose(1, 2)
        
        # GQA 广播，将 KV 头数复制对齐到 Q 头数
        k_rep = repeat_kv(k_t, num_q_per_kv)
        v_rep = repeat_kv(v_t, num_q_per_kv)
        
        # 1. 计算原生 Attention Scores
        attn_scores = torch.matmul(q_t, k_rep.transpose(-1, -2)) * scale
        
        # 2. 加上 causal mask (处理 prefill 阶段)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
            
        # 3. Softmax 获取真实权重
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # 4. 乘上 V 得到输出
        attn_output = torch.matmul(attn_weights, v_rep)
        
        # ------------------------------------------
        # 🌟 核心探针：只在 Decoding 阶段 (q_len == 1) 收集 Top-K
        # ------------------------------------------
        q_len = query_states.shape[1]
        if q_len == 1:
            top_k_num = min(64, attn_weights.shape[-1]) # 防止初始序列长度不足 64
            # 拿到 top 64 的 indices, 形状: [bsz, q_heads, 1, 64]
            _, topk_indices = torch.topk(attn_weights, k=top_k_num, dim=-1)
            
            topk_indices = topk_indices.squeeze(0)
            # 使用 cat 进行时间步(dim=2)上的拼接
            if getattr(self, "attn_weight", None) is None:
                self.attn_weight = topk_indices
            else:
                self.attn_weight = torch.cat([self.attn_weight, topk_indices], dim=1)
                print(self.attn_weight.shape)
        
        # 返回原生结果
        self.decode_step += 1
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output), None
    
    # 动态上下文判断
    is_prefill = q_len > 1
    middle_start = self.sink
    middle_end = total_seq_len - self.window
    middle_len = middle_end - middle_start
    is_short_context = middle_len < (self.echo_num_anchors * self.page_size)

    self.past_query = query_states.clone()
    scale = 1.0 / math.sqrt(self.head_dim)
    
    if not is_heavy_layer:
        self.update_page_digest(key_states)
    if is_prefill or is_short_context or is_heavy_layer:
        attn_output = flash_attn_func(query_states, key_states, value_states, causal=is_prefill)

    else:
        assert q_len == 1
        k_t = key_states.transpose(1, 2)
        v_t = value_states.transpose(1, 2)
        if self.echo_anchors is None:
            # init page digest
            attn_output = flash_attn_func(query_states, key_states, value_states, causal=False)
            key_states = key_states.transpose(1, 2)
            query_states = query_states.transpose(1, 2)
            last_query = query_states[:, :, -1:, :]
            key_states_rep = repeat_kv(key_states, num_q_per_kv)
            scale = 1.0 / math.sqrt(self.head_dim)
              
            # 仅仅对这一行做 matmul，拿到我们梦寐以求的概率波
            last_scores = torch.matmul(last_query, key_states_rep.transpose(2, 3)) * scale
            last_weights = torch.softmax(last_scores, dim=-1)
            
            # 切出中间那段广袤的区域
            last_weights_middle = last_weights[:, :, :, middle_start:middle_end] # [1, num_heads, 1, middle_len]
            
            # GQA 分组求平均
            kv_grouped_weights = last_weights_middle.view(1, self.num_key_value_heads, num_q_per_kv, -1).mean(dim=2).squeeze(0) # [kv_heads, middle_len]
            
            # 独立 NMS 空降！
            rel_anchors = echokv_get_initial_anchors_kv(kv_grouped_weights, self.echo_num_anchors, self.page_size // 2)
            
            self.echo_anchors = rel_anchors + middle_start

        # ==========================================
        # 🌟 路由 B：EchoKV 极速解码 (中间层 + q_len == 1)
        # ==========================================
        else:
            # 1. 边界计算
            device = query_states.device
            q_per_kv = self.num_heads // self.num_key_value_heads
            q_grouped = query_states.transpose(1, 2).view(bsz, self.num_key_value_heads, q_per_kv, 1, self.head_dim).mean(dim=2)

                # current_pages_num = self.num_pages
            offsets = torch.arange(self.page_size, device=query_states.device)

            full = False
            if self.decode_step % self.page_size == 0:
                # update anchors
                min_k_for_sel = self.min_k[:, :self.num_pages - self.num_window_pages, ...]
                max_k_for_sel = self.max_k[:, :self.num_pages - self.num_window_pages, ...]
                max_qk = quest_sel(query_states, min_k_for_sel, max_k_for_sel, \
                                self.gqa_policy, self.num_heads, self.num_key_value_heads)
                _, sel_page_indices = torch.topk(max_qk, self.page_budget)
                starts = self.sink + sel_page_indices.unsqueeze(0) * self.page_size
            
                q_t = query_states.transpose(1, 2)
                key_states_rep = repeat_kv(k_t, num_q_per_kv)
                scale = 1.0 / math.sqrt(self.head_dim)
                
                last_scores = torch.matmul(q_t, key_states_rep.transpose(2, 3)) * scale
                last_weights = torch.softmax(last_scores, dim=-1)
                # 切出中间那段广袤的区域
                last_weights_middle = last_weights[:, :, :, middle_start:middle_end] # [1, num_heads, 1, middle_len]
                
                # GQA 分组求平均
                kv_grouped_weights = last_weights_middle.view(1, self.num_key_value_heads, num_q_per_kv, -1).mean(dim=2).squeeze(0) # [kv_heads, middle_len]
                # kv_grouped_weights_sum = kv_grouped_weights.view(self.num_key_value_heads, middle_len // self.page_size, -1).sum(dim=-1) # [kv_heads, middle_page_len]
                _, gather_indices = torch.topk(kv_grouped_weights, self.token_budget-self.sink-self.window, dim=-1)

                gather_indices = gather_indices.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            
                # # 独立 NMS 空降！
                # rel_anchors = echokv_get_initial_anchors_kv(kv_grouped_weights, self.echo_num_anchors, self.page_size // 2)
                
                # self.echo_anchors = rel_anchors + middle_start
                
            else:

                starts = echokv_bidirectional_boundaries_kv(self.echo_anchors, self.page_size, middle_start, middle_end)
                starts = starts.to(offsets.device)
                indices = starts.unsqueeze(-1) + offsets
                indices_flat = indices.view(1, self.num_key_value_heads, -1)
                gather_indices = indices_flat.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            
            assert gather_indices is not None
            echo_k = torch.gather(k_t, dim=2, index=gather_indices)
            echo_v = torch.gather(v_t, dim=2, index=gather_indices)
            
            sink_k, sink_v = k_t[:, :, :self.sink, :], v_t[:, :, :self.sink, :]
            window_k, window_v = k_t[:, :, -self.window:, :], v_t[:, :, -self.window:, :]
            
            local_k_trans = torch.cat([sink_k, echo_k, window_k], dim=2)
            local_v_trans = torch.cat([sink_v, echo_v, window_v], dim=2)
        
            # # [bsz, kv_heads, 1, total_budget] 
            # # local_k_trans_rep = repeat_kv(local_k_trans, num_q_per_kv)
            # # track_scores = torch.matmul(query_states, local_k_trans_rep.transpose(-1, -2)) * scale
            
            # track_scores = torch.matmul(q_grouped, local_k_trans.transpose(2, 3)) * scale
            # if os.getenv("GET_CORR"):
            #     local_k_trans_sp = repeat_kv(local_k_trans, self.num_key_value_groups)
            #     tmp_scores = torch.matmul(query_states, local_k_trans_sp.transpose(-1, -2)) * scale
            #     local_attn_weights_kv = torch.softmax(tmp_scores, dim=-1)
            #     gini_impurity = 1.0 - torch.sum(local_attn_weights_kv ** 2, dim=-1).mean()
                
            #     if gini_impurity > self.corr_threshold:
            #         self.corr_count += 1
                
            # absolute_indices = starts.unsqueeze(-1) + offsets # [kv_heads, num_anchors, page_size]
            # flat_scores = track_scores[0, :, :, self.sink : self.sink + (self.echo_num_anchors * self.page_size)].squeeze(-2).clone()                 # [kv_heads, num_anchors * page_size]
            # flat_indices = absolute_indices.view(self.num_key_value_heads, -1)
            
            # # 使用极速雷达池化引擎进行洗牌！惩罚半径拉满至 32 防扎堆！
            # self.echo_anchors = echokv_triton_exact_nms(
            #     flat_scores=flat_scores, 
            #     flat_indices=flat_indices, 
            #     num_anchors=self.echo_num_anchors, 
            #     suppression_radius=self.page_size // 2, 
            #     old_anchors=self.echo_anchors
            # )     

            local_k_trans_rep = repeat_kv(local_k_trans, self.num_key_value_groups)
            
            # query_states: [bsz, q_heads, 1, head_dim]
            # local_k_trans_rep: [bsz, q_heads, total_budget, head_dim]
            track_scores = torch.matmul(query_states.transpose(1, 2), local_k_trans_rep.transpose(-1, -2)) * scale
            
            track_probs = torch.softmax(track_scores, dim=-1) # [bsz, q_heads, 1, total_budget]
            if os.getenv("GET_CORR"):
                # 如果需要计算基尼不纯度，可以直接复用算好的 track_probs
                gini_impurity = 1.0 - torch.sum(track_probs ** 2, dim=-1).mean()
                if gini_impurity > self.corr_threshold:
                    self.corr_count += 1
                    
            bsz, q_heads, _, total_budget = track_probs.shape
            # Reshape 切分维度: [bsz, kv_heads, num_q_per_kv, total_budget]
            track_probs_grouped = track_probs.view(bsz, self.num_key_value_heads, self.num_key_value_groups, total_budget)
            
            # 对 group 维度求平均，得到 KV Head 收到的综合概率投票！
            kv_track_probs = track_probs_grouped.mean(dim=2) # [bsz, kv_heads, total_budget]
            
            echo_start = self.sink
            echo_end = self.sink + (self.echo_num_anchors * self.page_size)
            
            absolute_indices = starts.unsqueeze(-1) + offsets # [kv_heads, num_anchors, page_size]
            
            # 注意：现在提取的是经过 Softmax 和 Mean 之后的概率！
            flat_probs = kv_track_probs[:, :, echo_start:echo_end].squeeze(0).clone().to(device) # [kv_heads, num_anchors * page_size]
            flat_indices = absolute_indices.view(self.num_key_value_heads, -1).to(device)
            flat_probs = flat_probs.contiguous()
            flat_indices = flat_indices.contiguous()
    
            self.echo_anchors = echokv_triton_exact_nms(
                flat_scores=flat_probs,  # 喂进去的是 [0,1] 的民主选票
                flat_indices=flat_indices, 
                num_anchors=self.echo_num_anchors, 
                suppression_radius=self.page_size // 2, 
                old_anchors=self.echo_anchors.to(device)
            )

            # 转回 FlashAttention 要求的格式 [bsz, seq_len, kv_heads, head_dim]，连续化物理显存

            local_k_fa = local_k_trans.transpose(1, 2).contiguous()
            local_v_fa = local_v_trans.transpose(1, 2).contiguous()
            
            # 原生 FlashAttention 自动处理 GQA 广播，极速完成计算！
            attn_output = flash_attn_func(query_states, local_k_fa, local_v_fa, causal=False).to(device)

    # attn_output 本来就是 [bsz, q_len, heads, head_dim]，只需 reshape
    self.decode_step += 1
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    return self.o_proj(attn_output), None

def apply_echo_config(module, args):
    module.page_size = args.chunk_size
    module.sink = args.sink
    module.window = args.window
    module.echo_anchors = None
    module.echo_num_anchors = (args.token_budget-args.sink-args.window) // args.chunk_size
    module.corr_threshold = 0.9
    module.corr_count = 0
    module.num_heads = module.config.num_attention_heads
    module.num_key_value_heads = module.config.num_key_value_heads
    module.hidden_size = module.config.hidden_size
    module.head_dim = module.hidden_size // module.num_heads
    module.gqa_policy = args.gqa_policy
    module.decode_step = 0
    module.num_pages = 0

def echo_reset(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            echo_reset(module)
        module.corr_count = 0
        module.echo_anchors = None
        module.num_pages = 0
        module.decode_step = 0
