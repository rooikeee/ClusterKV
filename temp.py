import math
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

def echokv_llama_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, **kwargs):
    if not hasattr(self, "echo_anchors"):
        self.echo_num_anchors = 32      
        self.echo_chunk_size = 64       
        self.echo_gini_threshold = 0.95 
        self.echo_num_sink = 64         
        self.echo_num_window = 256      
        self.echo_fallback_count = 0    
        self.echo_anchors = None # [kv_heads, num_anchors]

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, {"sin": sin, "cos": cos})

    total_seq_len = key_states.shape[-2]
    scale = 1.0 / math.sqrt(self.head_dim)
    
    # GQA 映射关系：多少个 Q 头共享一个 KV 头？
    num_q_per_kv = self.num_heads // self.num_key_value_heads

    is_prefill = q_len > 1 or self.echo_anchors is None
    middle_start = self.echo_num_sink
    middle_end = total_seq_len - self.echo_num_window
    middle_len = middle_end - middle_start
    
    # 只要中间部分能塞下我们预算，就启启动态追踪
    is_short_context = middle_len < (self.echo_num_anchors * self.echo_chunk_size)

    if is_prefill or is_short_context:
        # --- 阶段 A：Prefill / 短文本 ---
        key_states_rep = repeat_kv(key_states, num_q_per_kv)
        value_states_rep = repeat_kv(value_states, num_q_per_kv)
        
        attn_weights = torch.matmul(query_states, key_states_rep.transpose(2, 3)) * scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        if is_prefill and not is_short_context:
            # 🌟 以 KV Head 为粒度聚合 Q 的关注度！
            last_step_weights = attn_weights[:, :, -1, middle_start:middle_end] # [1, num_heads, middle_len]
            # 把 Q 头的分数按照 KV 头的分组求平均
            kv_grouped_weights = last_step_weights.view(1, self.num_key_value_heads, num_q_per_kv, -1).mean(dim=2).squeeze(0) # [kv_heads, middle_len]
            
            # 各自为战：每个 KV Head 拥有独立的追踪雷达
            rel_anchors = echokv_get_initial_anchors_kv(kv_grouped_weights, self.echo_num_anchors, self.echo_chunk_size // 2)
            self.echo_anchors = rel_anchors + middle_start
            
        attn_output = torch.matmul(attn_weights, value_states_rep)

    else:
        # --- 阶段 B：极其硬核的 Gather 碎片提取 ---
        # 1. 用双向倒逼算法，算出绝对不越界、绝对填满预算的真实起跑线！
        starts = echokv_bidirectional_boundaries_kv(
            self.echo_anchors, self.echo_chunk_size, middle_start, middle_end
        ) # [kv_heads, num_anchors]
        
        # 2. 生成 Gather 索引 (利用广播机制神仙操作)
        offsets = torch.arange(self.echo_chunk_size, device=starts.device)
        indices = starts.unsqueeze(-1) + offsets # [kv_heads, num_anchors, chunk_size]
        indices_flat = indices.view(1, self.num_key_value_heads, -1) # [1, kv_heads, 总预算 Token 数]
        gather_indices = indices_flat.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        
        # 3. CUDA 极限操作：用极其底层的 Gather 直接从显存深处抓取所需特征！
        echo_k = torch.gather(key_states, dim=2, index=gather_indices)
        echo_v = torch.gather(value_states, dim=2, index=gather_indices)
        
        # 提取 Sink 和 Window
        sink_k = key_states[:, :, :self.echo_num_sink, :]
        sink_v = value_states[:, :, :self.echo_num_sink, :]
        window_k = key_states[:, :, -self.echo_num_window:, :]
        window_v = value_states[:, :, -self.echo_num_window:, :]
        
        # 拼接物理块 (以 KV Head 为单位)
        local_k_kv = torch.cat([sink_k, echo_k, window_k], dim=2)
        local_v_kv = torch.cat([sink_v, echo_v, window_v], dim=2)
        
        # 展开为 GQA 要求的 Q Head 数量
        local_k = repeat_kv(local_k_kv, num_q_per_kv)
        local_v = repeat_kv(local_v_kv, num_q_per_kv)
        
        # --- 阶段 C：计算与基尼警报 ---
        local_scores = torch.matmul(query_states, local_k.transpose(2, 3)) * scale
        local_attn_weights = torch.softmax(local_scores, dim=-1)
        
        gini_impurity = 1.0 - torch.sum(local_attn_weights ** 2, dim=-1).mean()
        
        if gini_impurity > self.echo_gini_threshold:
            self.echo_fallback_count += 1
            key_states_rep = repeat_kv(key_states, num_q_per_kv)
            value_states_rep = repeat_kv(value_states, num_q_per_kv)
            
            full_scores = torch.matmul(query_states, key_states_rep.transpose(2, 3)) * scale
            if attention_mask is not None:
                full_scores = full_scores + attention_mask
            full_attn_weights = torch.softmax(full_scores, dim=-1)
            
            # 全量兜底后，同样以 KV Head 为单位重新空降！
            last_weights = full_attn_weights[:, :, -1, middle_start:middle_end]
            kv_grouped_weights = last_weights.view(1, self.num_key_value_heads, num_q_per_kv, -1).mean(dim=2).squeeze(0)
            self.echo_anchors = echokv_get_initial_anchors_kv(kv_grouped_weights, self.echo_num_anchors, self.echo_chunk_size // 2) + middle_start
            
            attn_output = torch.matmul(full_attn_weights, value_states_rep)
        else:
            # --- 阶段 D：安全更新锚点 ---
            echo_attn_start = self.echo_num_sink
            echo_attn_len = self.echo_num_anchors * self.echo_chunk_size
            echo_attn = local_attn_weights[..., echo_attn_start : echo_attn_start + echo_attn_len] # [1, num_heads, 1, 预算长度]
            
            # 把拿到的 Attention 分数重新映射回对应的 KV Head
            echo_attn_kv = echo_attn.view(1, self.num_key_value_heads, num_q_per_kv, -1).mean(dim=2).squeeze(0) # [kv_heads, 预算长度]
            
            # 在各个 Chunk 内部寻找下一个高地
            chunk_attns = echo_attn_kv.view(self.num_key_value_heads, self.echo_num_anchors, self.echo_chunk_size)
            local_max_idx = torch.argmax(chunk_attns, dim=-1) # [kv_heads, num_anchors]
            
            # 精确加上该 Chunk 实际起跑线 (starts 刚才在边界算法里已经算好了！)
            self.echo_anchors = starts + local_max_idx
            
            attn_output = torch.matmul(local_attn_weights, local_v)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    return self.o_proj(attn_output), None, past_key_value