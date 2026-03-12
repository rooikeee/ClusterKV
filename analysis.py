from re import L
from statistics import variance
from tkinter import N
from unittest import skip
from unittest.util import _count_diff_all_purpose
import torch
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from requests.exceptions import ProxyError, SSLError
import json

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3.1-8b-chat-8k")
    parser.add_argument("--task", type=str, default="hotpotqa")
    parser.add_argument("--type", type=str, default=None)
    parser.add_argument("--rope_type", type=str, default=None)
    parser.add_argument("--draw_type", type=str, default="overlap")
    parser.add_argument("--analysis_type", type=str, default="normal")

    return parser.parse_args(args)

def analysis(cluster_key_indices, cluster_key_ptrs, layer_key_states, key_centroids, sel_clusters, draw_type):
    picture_save_dir = os.path.join("pictures", model_name, task, draw_type)
    sel_data = get_select_cluster(sel_clusters)
    os.makedirs(picture_save_dir, exist_ok=True)
    for layer_idx in range(2, len(cluster_key_indices)):
        cluster_key_index = cluster_key_indices[layer_idx]
        cluster_key_ptr = cluster_key_ptrs[layer_idx]
        k_centroids = key_centroids[layer_idx].squeeze(0)
        num_kv_head = cluster_key_index.shape[0]
        layer_sel_data = sel_data[layer_idx]
        num_head = cluster_key_ptr.shape[0]
        num_group_head = num_head // num_kv_head
        fig, axes = plt.subplots(8, 1, figsize=(10, 16), constrained_layout=True)
        sel_count = count_sel_clusters(layer_sel_data)

        plt.show()
        for kv_head in range(num_kv_head):
            k_idx = cluster_key_index[kv_head].tolist()
            head = kv_head * num_group_head
            k_ptr = cluster_key_ptr[head].tolist()
            ax = axes[kv_head]
            if draw_type == "overlap":
                overlap_result, ratio, cluster_size = get_overlap_result(k_idx, k_ptr)
                draw_ratio(ax, ratio, cluster_size, layer_idx, kv_head, model_name)
            else:
                assert draw_type == "variance"
                k_state = layer_key_states[layer_idx][0, kv_head, ...].to('cpu')
                k_centroid = k_centroids[kv_head * num_group_head].to('cpu')
                variance, min_variance, cluster_size = get_variance(k_state, k_idx, k_ptr, k_centroid)
                min_k_cos_sim, mean_k_cos_sim = get_min_cosine_similarity(k_state)
                min_c_cos_sim, mean_c_cos_sim = get_min_cosine_similarity(k_centroid)
                
                cluster_labels = get_data(k_idx, k_ptr)
                kv_head_sel_count = sel_count[kv_head]
                ssb, msb = calculate_inter_cluster_variance(k_state, k_centroid, cluster_labels)
                draw_variance(ax, variance, min_variance, kv_head_sel_count, cluster_size, ssb, msb, layer_idx, kv_head, min_k_cos_sim, min_c_cos_sim, mean_k_cos_sim, mean_c_cos_sim)
                
        plt.savefig(os.path.join(picture_save_dir, f"layer_{layer_idx}.png"))

def count_sel_clusters(sel_data):
    count = {}
    num_kv_head = 8
    num_head = 32
    for head_idx in range(num_head):
        kv_head_idx = head_idx // 4
        if kv_head_idx not in count:
            count[kv_head_idx] = {}
        head_sel_data = sel_data[head_idx]
        decode_step = len(head_sel_data)
        for decode_idx in range(decode_step):
            step_sel_data = head_sel_data[decode_idx]
            for cluster_idx in step_sel_data:
                if cluster_idx not in count[kv_head_idx]:
                    count[kv_head_idx][cluster_idx] = 0
                count[kv_head_idx][cluster_idx] += 1
    return count

def draw_ratio(ax, ratio, cluster_size, layer_idx, kv_head, model_name):
    color1, color2 = 'tab:red', 'tab:blue'
    ax.plot(ratio, color=color1, linewidth=2)
    ax.set_xlabel('Cluster Index')
    ax.set_ylabel('Increase Ratio', color=color1)
    
    ax.set_title(f"{model_name}, layer {layer_idx}, kv_head {kv_head}, avg {round(sum([x for x in ratio if x >= 0]) / len([x for x in ratio if x >= 0]), 2)}")

    ax2 = ax.twinx()
    ax2.set_ylabel('Cluster Size', color=color2)
    ax2.plot(cluster_size, color=color2, linewidth=2)
    ax.grid(True, alpha=0.3)

def draw_variance(ax, 
                  variance, 
                  min_variance,
                  kv_head_sel_count,
                  cluster_size, 
                  ssb, 
                  msb, 
                  layer_idx, 
                  kv_head, 
                  min_k_cos_sim,
                  min_c_cos_sim,
                  mean_k_cos_sim,
                  mean_c_cos_sim):
    color1, color2 = 'tab:red', 'tab:blue'
    ax.plot(variance, color=color1, linewidth=2)
    ax.set_xlabel('Cluster Index')
    ax.set_ylabel('Consine Similarity', color=color1)
    
    ax.set_title(f"layer {layer_idx}," 
                 f"kv_head {kv_head},"
                 f"min_k_cos: {round(min_k_cos_sim, 3)},"
                 f"min_c_cos: {round(min_c_cos_sim, 3)},"
                 f"mean_k_cos: {round(mean_k_cos_sim, 3)},"
                 f"mean_c_cos: {round(mean_c_cos_sim, 3)}") 

    x = range(len(min_variance))
    ax.scatter(x, min_variance, color=color1, s=5, marker='o')
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    x_2 = kv_head_sel_count.keys()
    y_2 = kv_head_sel_count.values()
    ax2.set_ylabel('Select Count', color=color2)
    ax2.scatter(x_2, y_2, color=color2, s=5, marker='^')

def get_overlap_result(k_idx, k_ptr):
    
    def find_longest_ordered_subarray(lst):
        """
        找到列表中连续升序数字的最长子数组
        
        参数:
            lst: 输入的数字列表
            
        返回:
            一个元组 (最大长度, 子数组)
        """
        if not lst:
            return 0, []
        
        max_length = 1
        current_length = 1
        max_start = 0
        current_start = 0
        
        asc_subarray_len = 0
        for i in range(1, len(lst)):
            # 检查当前元素是否大于前一个元素（保持升序）
            if lst[i] == lst[i-1] + 1:
                current_length += 1
            else:
                if current_length > 1:
                    asc_subarray_len += current_length
                # 如果当前升序序列结束，检查是否需要更新最大值
                if current_length > max_length:
                    max_length = current_length
                    max_start = current_start
                
                # 重置当前序列
                current_length = 1
                current_start = i
        
        if current_length > 1:
            asc_subarray_len += current_length

        # 检查最后一个序列
        if current_length > max_length:
            max_length = current_length
            max_start = current_start
        
        # 提取最长有序子数组
        longest_subarray = lst[max_start:max_start + max_length]
        
        return max_length, longest_subarray, asc_subarray_len
    
    data = {}
    for idx in range(len(k_idx)):
        if k_ptr[idx] not in data:
            data[k_ptr[idx]] = []
        else:
            data[k_ptr[idx]].append(k_idx[idx])
    
    result = {}
    ratio = []
    cluster_size_list = []
    for cluster_idx in data:
        cluster_size = len(data[cluster_idx])
        cluster_size_list.append(cluster_size)
        if cluster_size == 0:
            ratio.append(-1)
            continue
        max_length, longest_subarray, asc_subarray_len = find_longest_ordered_subarray(data[cluster_idx])
        result[cluster_idx] = {
            'size': cluster_size,
            'longest_increasing_subarray_length': max_length,
            'longest_increasing_subarray': longest_subarray,
            "ratio": round(asc_subarray_len / cluster_size, 2)
        }
        ratio.append(round(asc_subarray_len / cluster_size, 2))

    return result, ratio, cluster_size_list

def calculate_intra_cluster_variance(data, centroids):
    """
    Args:
        data: shape [N, 128]
        centroids: shape [C, 128]
    Returns:
        total_variance: 所有数据的平均簇内方差
        inertia: 簇内距离平方和 (Sum of squares)
    """
    # 1. 计算两两距离矩阵
    # torch.cdist 计算的是欧氏距离 (L2 norm)
    # output shape: [N, C]
    dists_matrix = torch.cdist(data, centroids, p=2)
    
    # 2. 找到每个样本距离最近的簇中心
    # min_dists: 每个样本到最近中心的距离 [N]
    # labels: 每个样本归属的簇索引 [N]
    min_dists, labels = torch.min(dists_matrix, dim=1)
    
    # 3. 计算平方距离 (因为 cdist 开了根号，方差需要平方)
    squared_dists = min_dists ** 2
    
    # 4. 计算指标
    inertia = torch.sum(squared_dists)       # 簇内平方和 (Inertia)
    total_variance = torch.mean(squared_dists) # 总体簇内方差 (MSE)
    
    return total_variance, inertia, labels

def calculate_inter_cluster_variance(data, centroids, labels, use_cos=True):
    """
    计算簇间方差 (Inter-Cluster Variance)
    :param data: [N, 128] 原始数据
    :param centroids: [C, 128] 聚类中心
    :param labels: dict: {c, cluster_size}
    :return: ssb (组间平方和), msb (组间方差/均方)
    """
    
    # 2. 计算全局均值 (Global Mean) [128]
    global_mean = torch.mean(data, dim=0)
    # 3. 计算 SSB (Sum of Squares Between)
    ssb = 0.0
    num_clusters = centroids.shape[0]
    for k in range(num_clusters):
        n_k = len(labels[k])
        if n_k == 0:
            continue
            
        if use_cos:
            # 余弦相似度范围是[-1, 1]，通常我们希望差异越大值越大
            # 所以用 1 - cos_sim 或 1 - |cos_sim|，取决于是否需要考虑方向
            cos_sim = F.cosine_similarity(
                centroids[k].unsqueeze(0), 
                global_mean.unsqueeze(0)
            ).item()
            dist_sq = 1 - cos_sim  # 值越大表示差异越大（0-2范围）
        else:
            # 计算该簇中心与全局中心的欧氏距离平方
            # || mu_k - mu ||^2
            diff = centroids[k] - global_mean
            dist_sq = torch.sum(diff ** 2).item()
        
        # 加权累加
        ssb += n_k * dist_sq
        
    # 4. 计算 MSB (Mean Square Between) -> 即簇间方差
    # 自由度通常为 C - 1
    if num_clusters > 1:
        msb = ssb / (num_clusters - 1)
    else:
        msb = 0.0 # 只有一个簇，方差为0

    return ssb, msb

def get_variance(k_states, k_idx, k_ptr, key_centroids):
    
    def calculate_per_cluster_variance(k_states, centroid, use_cos=True):
        if use_cos:
            # 使用余弦相似度计算簇内方差
            # 计算所有样本与中心的余弦相似度
            # shape: [N]
            cos_sims = F.cosine_similarity(k_states, centroid, dim=1)
            
            # 计算余弦距离的平均值作为方差
            var = cos_sims.mean().item()
            min_var = cos_sims.min().item()
            return var, min_var
        else:
            # 原始欧氏距离方法
            diff = k_states - centroid
            sq_dist = (diff ** 2).sum(dim=1)
            var = sq_dist.mean()
            return var.item()

    data = get_data(k_idx, k_ptr)

    variance = []
    min_variance = []
    cluster_size_list = []
    for cluster_idx in data:
        cluster_size = len(data[cluster_idx])
        cluster_size_list.append(cluster_size)
        if cluster_size == 0:
            variance.append(None)
            min_variance.append(None)
            continue
        indices = torch.tensor(data[cluster_idx])
        # k_states shape: [seq, head_dim]
        k_state = torch.gather(k_states, 0, indices.unsqueeze(1).expand(-1, k_states.shape[1])) # [cluster_size, head_dim]
        var, min_var = calculate_per_cluster_variance(k_state, key_centroids[cluster_idx])
        variance.append(var)
        min_variance.append(min_var)
    
    return variance, min_variance, cluster_size_list

def get_min_cosine_similarity(keys):
    """
    计算key states之间的最小余弦相似度（排除自身比较）
    
    Args:
        keys: Tensor of shape [N, D]
    
    Returns:
        min_sim: 最小余弦相似度
        min_pair: 最小相似度对应的向量索引 (i, j)
    """
    # 计算余弦相似度矩阵
    sim_matrix = compute_cosine_similarity_fast(keys)
    
    # 创建掩码排除对角线（自身比较）
    mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=keys.device)
    
    # 获取最小值及其位置
    masked_sim = sim_matrix[mask]
    min_value = torch.min(masked_sim)
    mean_sim = torch.mean(masked_sim)
    
    return min_value.item(), mean_sim.item()

# 辅助函数：计算余弦相似度矩阵
def compute_cosine_similarity_fast(keys):
    """高效的余弦相似度计算"""
    import torch.nn.functional as F
    norm_keys = F.normalize(keys, p=2, dim=1)
    return torch.mm(norm_keys, norm_keys.T)


def get_select_cluster(sel_clusters):
    data = {}
    for layer_idx in range(2, len(sel_clusters)):
        layer_sel_clusters = sel_clusters[layer_idx]
        decode_step = len(layer_sel_clusters)
        head_sel_clusters = {}
        for step_idx in range(decode_step):
            step_sel_clustetrs = layer_sel_clusters[step_idx]
            num_head = step_sel_clustetrs.shape[0]
            for head_idx in range(num_head):
                if head_idx not in head_sel_clusters:
                    head_sel_clusters[head_idx] = []
                head_sel_clusters[head_idx].append(step_sel_clustetrs[head_idx].tolist())
        data[layer_idx] = head_sel_clusters
    return data

def get_data(k_idx, k_ptr):
    data = {}
    for idx in range(len(k_idx)):
        if k_ptr[idx] not in data:
            data[k_ptr[idx]] = []
        else:
            data[k_ptr[idx]].append(k_idx[idx])
    
    return data

def analysis_select(sel_clusters):
    picture_save_dir = os.path.join("pictures", model_name, task, analysis_type)
    os.makedirs(picture_save_dir, exist_ok=True)
    for layer_idx in range(2, len(sel_clusters)):
        layer_sel_clusters = sel_clusters[layer_idx]
        decode_step = len(layer_sel_clusters)
        head_sel_clusters = {}
        for step_idx in range(decode_step):
            step_sel_clustetrs = layer_sel_clusters[step_idx]
            num_head = step_sel_clustetrs.shape[0]
            for head_idx in range(num_head):
                if head_idx not in head_sel_clusters:
                    head_sel_clusters[head_idx] = []
                head_sel_clusters[head_idx].append(step_sel_clustetrs[head_idx].tolist())
            
        
        fig, axes = plt.subplots(4, 8, figsize=(18, 9), constrained_layout=True)
        for idx, ax in enumerate(axes.T.flat):
            data = head_sel_clusters[idx]
            x_coords = []
            y_coords = []
            for i, row in enumerate(data):
                for value in row:
                    x_coords.append(i+1)  # x坐标从1开始
                    y_coords.append(value)
            
            # 绘制散点图
            if idx < 4:
                ax.set_ylabel('Cluster Index')
            if (idx+1) % 4 == 0:
                ax.set_xlabel('Decoding Step')
            ax.set_title(f"layer {layer_idx}, head {idx}")
            ax.scatter(x_coords, y_coords, s=10, alpha=0.6)
            
            plt.savefig(os.path.join(picture_save_dir, f"layer_{layer_idx}.png"))

def analysis_text(cluster_key_indices, cluster_key_ptrs, input, tokenizer):
    picture_save_dir = os.path.join("pictures", model_name, task, draw_type)
    text = [tokenizer.decode(x, skip_special_tokens=True) \
            for x in input]

    os.makedirs(picture_save_dir, exist_ok=True)
    for layer_idx in range(2, len(cluster_key_indices)):
        cluster_key_index = cluster_key_indices[layer_idx]
        cluster_key_ptr = cluster_key_ptrs[layer_idx]
        num_kv_head = cluster_key_index.shape[0]
        num_head = cluster_key_ptr.shape[0]
        num_group_head = num_head // num_kv_head
        fig, axes = plt.subplots(8, 1, figsize=(10, 16), constrained_layout=True)

        plt.show()
        for kv_head in range(num_kv_head):
            k_idx = cluster_key_index[kv_head].tolist()
            head = kv_head * num_group_head
            k_ptr = cluster_key_ptr[head].tolist()
            ax = axes[kv_head]
            data = get_text_data(k_idx, k_ptr)
            color_print(text, data)
        # plt.savefig(os.path.join(picture_save_dir, f"layer_{layer_idx}.png"))

def get_text_data(k_idx, k_ptr):
    data = [0 for _ in range(len(k_idx))]
    for i in range(len(k_idx)):
        data[k_idx[i]] = k_ptr[i]
    return data

def color_print(text, cluster_data):
    class Colors:
        """ANSI颜色代码"""
        RED = '\033[91m'
        GREEN = '\033[92m'
        BLUE = '\033[94m'
        YELLOW = '\033[93m'
        PURPLE = '\033[95m'
        CYAN = '\033[96m'
        WHITE = '\033[97m'
        RESET = '\033[0m'
        BOLD = '\033[1m'
    
    text_group = [(text[i], cluster_data[i]) for i in range(len(text))]
    group_colors = {
        0: Colors.RED,
        1: Colors.GREEN,
        2: Colors.BLUE,
        3: Colors.YELLOW
    }

    result = []
    for word, group in text_group:
        color = group_colors.get(group, Colors.WHITE)
        result.append(f"{color}{word}{Colors.RESET}")
    
    # 组合并打印
    print(" ".join(result))

def load_model_and_tokenizer(path, model_name, device):
    if "intern" in model_name or "qwen" in model_name or "glm4" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        # model = AutoModelForCausalLM.from_pretrained(
        #     path, trust_remote_code=True, torch_dtype=torch.float16,
        #     device_map="auto", low_cpu_mem_usage=True,
        #     attn_implementation="flash_attention_2", use_cache=True
        # ).to(device)
    elif "llama" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path)
        # model = LlamaForCausalLM.from_pretrained(
        #     path, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True,
        #     attn_implementation="flash_attention_2", use_cache=True
        # )
    else:
        assert False
    model = None

    return model, tokenizer

def load_model_with_retry(model_path, model_name, device, retries=3, delay=1):
    import time
    for attempt in range(retries):
        try:
            model, tokenizer = load_model_and_tokenizer(model_path, model_name, device)
            return model, tokenizer
        except (ProxyError, SSLError) as e:
            print(f"Attempt {attempt + 1} failed due to network error: {e}")
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                raise  # Re-raise the last exception if all retries fail

def analysis_clusters(k_states):
    
    def evaluate_k(keys, k_list):
    # keys: [N, D] Pre-RoPE keys (flattened)
    # 1. 必须归一化！
    # 这样点积 (Dot Product) 就等于 Cosine Similarity
        keys_normalized = F.normalize(keys, p=2, dim=-1).cpu().numpy()
        
        results = []
        for k in k_list:
            
            # 训练 KMeans
            # 虽然它的损失函数是欧式的，但在单位球面上这等价于优化 Cosine
            kmeans = MiniBatchKMeans(n_clusters=k, batch_size=2048, random_state=42).fit(keys_normalized)
            
            # --- 手动计算 Cosine SSE ---
            
            # 1. 获取聚类标签和质心
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            
            # 2. 极其重要：Sklearn 更新质心后，质心可能不再是单位向量
            # 所以必须再次对质心进行归一化，才能保证计算的是 Cosine 相似度
            centroids_norm = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
            
            # 3. 计算每个点到其对应质心的 Cosine Similarity
            # 也就是点积: x . centroid
            # keys_normalized[i] dot centroids_norm[labels[i]]
            
            # 利用 numpy 的高效索引，取出每个样本对应的质心
            assigned_centroids = centroids_norm[labels]
            
            # 计算点积 (batch dot product)
            # sum(A * B, axis=1)
            similarities = np.sum(keys_normalized * assigned_centroids, axis=1)
            
            # 5. 求和得到 SSE (Sum of Cosine Errors)
            total_cosine_sse = np.sum(1-similarities)
            results.append(total_cosine_sse)
            
        return results
    
    num_layers = len(k_states)
    picture_save_dir = os.path.join("pictures", model_name, task, "sse")
    os.makedirs(picture_save_dir, exist_ok=True)
    for layer_id in range(2, num_layers):
        k_state = k_states[layer_id].squeeze(0)
        num_kv_head = k_state.shape[0]
        fig, axes = plt.subplots(num_kv_head, 1, figsize=(10, 16), constrained_layout=True)
        k_list = [10, 50, 100, 150, 200, 250, 300, 350, 400]
        for head_id in range(num_kv_head):
            head_k_state = k_state[head_id]
            ax = axes[head_id]
            sse = evaluate_k(head_k_state, k_list)    
            ax.plot(sse, color='tab:red', linewidth=2)
            ax.set_xlabel('Cluster Number')
            ax.set_ylabel('sse')
            ax.set_title(f"{model_name}, layer {layer_id}, kv_head {head_id}")
            ax.set_xticks([i for i in range(len(k_list))])
            ax.set_xticklabels(k_list)
            
        plt.savefig(os.path.join(picture_save_dir, f"layer_{layer_id}"))

def analysis_cos_sim(k_states):
    num_layers = len(k_states)
    picture_save_dir = os.path.join("pictures", model_name, task, "cos_sim")

    def get_cos_sim(head_k_states):
        keys_norm = F.normalize(head_k_states.float(), p=2, dim=-1)
        adjacent_sims = torch.sum(keys_norm[:-1] * keys_norm[1:], dim=-1)
        is_cut = adjacent_sims < 0.65
            
        # 补齐第一个位置 (Index 0 永远不是切分点，它是起点)
        concat_cuts = torch.cat([torch.tensor([False], device=device), is_cut])
        token_bucket_ids = torch.cumsum(concat_cuts.long(), dim=0)
        
        # 获取当前 Head 切分出了多少个 Bucket
        num_buckets = token_bucket_ids[-1].item() + 1

        return adjacent_sims.tolist(), num_buckets

    os.makedirs(picture_save_dir, exist_ok=True)
    for layer_id in range(2, num_layers):
        k_state = k_states[layer_id].squeeze(0)
        num_kv_head = k_state.shape[0]
        fig, axes = plt.subplots(num_kv_head, 1, figsize=(10, 20), constrained_layout=True)
        for head_id in range(num_kv_head):
            head_k_state = k_state[head_id][16:]
            adjacent_sims, num_buckets = get_cos_sim(head_k_state)
            ax = axes[head_id]    
            ax.plot(adjacent_sims, color='tab:red', linewidth=0.5)
            ax.set_xlabel('Token Idx')
            ax.set_ylabel('Cosine Similarity')
            ax.set_title(f"{model_name}, layer {layer_id}, kv_head {head_id} "
                         f"avg {round(sum(adjacent_sims) / len(adjacent_sims), 2)}"
                         f"num buckets {num_buckets}")
            
        plt.savefig(os.path.join(picture_save_dir, f"layer_{layer_id}"))

def analysis_attn_weight(data_dir, key_states):
    layers = len(os.listdir(data_dir))
    print(data_dir)
    topk = os.getenv("topk")
    save_dir = os.path.join("pictures", f"{model_name}", f"topk-{topk}")
    os.makedirs(save_dir, exist_ok=True)
    rows, cols = 4, 8
    for layer in range(2, layers):
        attn_maps = torch.load(os.path.join(data_dir, f"layer-{layer}.pt")).to('cpu')
        layer_k_states = key_states[layer].squeeze(0)
        if topk:
            _, indices = torch.topk(attn_maps, int(topk), dim=-1)
            # 3. 创建一个全 0 的张量，形状与原张量一致
            mask = torch.zeros_like(attn_maps)

            attn_maps = mask.scatter_(dim=-1, index=indices, value=0.5)
        print(f"Working layer-{layer}")
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
        axes = axes.T.flatten()
        
        for idx in range(rows * cols):
            attn_map = attn_maps[idx]
            seq_len, _ = attn_map.shape
            head_k_states = layer_k_states[idx // rows]
            norms = torch.norm(head_k_states, p=2, dim=1)
            _, topk_indices = torch.topk(norms, k=100)
            
            attn_map[..., topk_indices] = 1.0
            attn_map = attn_map.float()
            # attn_map = attn_map * 10000

            attn_rows, attn_cols = attn_map.shape
            if attn_map.ndimension() != 2:
                raise ValueError("attn_map should be 2D")
            ax = axes[idx]
            cax = ax.imshow(attn_map.numpy(), aspect='auto', vmin=0, vmax=1.0)
            
            ax.set_title(f"Head {idx}")
            ax.axis('off')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(save_dir, f"layer-{layer}.png"), )
        plt.close()

def analysis_topk(data_dir):
    layers = len(os.listdir(data_dir))
    print(data_dir)
    topk = os.getenv("topk")
    save_dir = os.path.join("pictures", f"{model_name}", f"topk-{topk}")
    os.makedirs(save_dir, exist_ok=True)
    if "qwen" in model_name:
        rows, cols = 7, 4
    else:
        rows, cols = 8, 4

    for layer in range(2, layers):
        topk_indices = torch.load(os.path.join(data_dir, f"layer-{layer}.pt")).to('cpu')
        print(topk_indices.shape)

        print(f"Working layer-{layer}")
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
        axes = axes.T.flatten()
        
        for idx in range(rows * cols):
            top_index = topk_indices[idx]
            ax = axes[idx]
            data_np = top_index.detach().cpu().numpy()

            # 准备 Y 轴的坐标 (0 到 254)
            steps = np.arange(255)

            # Y坐标：把每个 step 复制 64 次，对应这64个数据点
            y_coords = np.repeat(steps, 64)

            # X坐标：把 numpy 数组展平为一维
            x_coords = data_np.flatten()

            # 画散点图
            ax.scatter(x_coords, y_coords, s=2, alpha=0.5, color='blue')

        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(save_dir, f"layer-{layer}.png"), )
        plt.close()

if __name__ == '__main__':
    args = parse_args()
    model_name = args.model 
    task = args.task
    rope_type = args.rope_type
    analysis_type = args.analysis_type
    draw_type = args.draw_type
    sink = 16
    model2path = json.load(open("accuracy/config/model2path.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, tokenizer = load_model_with_retry(
        model2path[model_name], model_name, device
    )

    if rope_type is not None:
        model_name = f"{model_name}-{rope_type}"
    else:
        model_name = f"{model_name}"

    data_dir = os.path.join("/state", "partition", "cwli", model_name, task)
    if task != "topk":
        cluster_dir = os.path.join("/state", "partition", "cwli", f"{model_name}post_rope", "gov_report")
        with open(os.path.join(cluster_dir, 'cluster_key_indices.pkl'), 'rb') as f:
            loaded_data = pickle.load(f)
        cluster_key_indices = loaded_data['cluster_key_indices']
        cluster_key_ptrs = loaded_data['cluster_key_ptr']
        layer_key_states = loaded_data['key_states']
        key_centroids = loaded_data['key_centroids']
        sel_clusters = loaded_data['sel_clusters']

    if analysis_type == "normal":
        analysis(cluster_key_indices, cluster_key_ptrs, layer_key_states, key_centroids, sel_clusters, draw_type)
    elif analysis_type == "select":
        analysis_select(sel_clusters)
    elif analysis_type == "text":
        input = loaded_data['input'].squeeze(0).tolist()[sink:]
        analysis_text(cluster_key_indices, cluster_key_ptrs, input, tokenizer)
    elif analysis_type == "cluster":
        analysis_clusters(layer_key_states)
    elif analysis_type == "cos_sim":
        analysis_cos_sim(layer_key_states)
    elif analysis_type == "attn_weights":
        analysis_attn_weight(data_dir, layer_key_states)
    else:
        analysis_topk(data_dir)