from email.policy import default
from sys import modules
from tkinter import N
import types
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from accuracy.quest_attention import forward_quest, forward_quest_glm, forward_greedy
from accuracy.cluster_attention import forward_cluster, forward_cluster_glm, apply_cluster_config
from accuracy.echokv_attention import echokv_llama_forward, apply_echo_config
import torch 
import torch.nn.functional as F
import os
import numpy as np

def parse_common_args(parser):
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "llama3-8b-chat-4k",
            "llama3-8b-chat-8k",
            "llama3.1-8b-chat-4k",
            "llama3.1-8b-chat-8k",
            "llama3.1-8b-chat-32k",
            "glm4-9b-chat-4k",
            "glm4-9b-chat-8k",
            "glm4-9b-chat-32k",
            "ds-r1-llama-8b",
            "qwen-2.5-7b",
            "qwen-2.5-14b",
            "qwen-3-8b"
        ],
    )
    parser.add_argument("--token_budget", type=int, default=128)

    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--quest", action="store_true", help="Enable Quest Attention")

    parser.add_argument("--sink", type=int, default=128)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--cluster", action="store_true", help="Enable ClusterKV Attention")
    parser.add_argument("--greedy", action="store_true", help="Enable greedy Attention")
    parser.add_argument("--echo", action="store_true", help="Enable echo Attention")
    parser.add_argument("--head_sel", type=str, choices=["truc", "pad"], default="truc",
                        help="truncate or pad for different heads to make same selection budget")
    parser.add_argument("--balance", action="store_true", help="Use Balanced KMeans")
    parser.add_argument("--nlist", type=int, default=0, help="Number of clusters")
    parser.add_argument("--fit_iter", type=int, default=20, help="Max steps for clustering")
    parser.add_argument("--gqa_policy", 
                        type=str, 
                        choices=[
                            "avgQ",
                            "maxS",
                            "avgS",
                            "avgSM",
                            "avgSdM"
                        ], default=None)
    parser.add_argument("--dist_t", type=str, 
                        choices=["cosine", "inner_product", "l2", "l1", "euclidean", 
                                 "chebyshev", "canberra"], 
                        default="cosine", help="Distance for clustering")
    parser.add_argument("--cache_steps", type=int, default=0, 
                        help="Stat cache hit rate of recent steps")
    parser.add_argument("--topk_stat", action="store_true", help="Stat hit rate of TopK tokens")
    return parser


@torch.no_grad()
def reorder_linear_weights(
    linear_module: torch.nn.Linear,
    full_attention_heads: torch.Tensor,
    repeat_num,
    reorder_channel,
    dyn_attention_heads: torch.Tensor = None,
):
    assert reorder_channel in ["in", "out"]
    full_attention_heads = torch.repeat_interleave(
        full_attention_heads, repeats=repeat_num
    ).to(linear_module.weight.device)
    full_attn_mask = full_attention_heads > 0.5
    # if dyn_attention_heads is not None:
    #     dyn_attention_heads = torch.repeat_interleave(
    #         dyn_attention_heads, repeats=repeat_num
    #     ).to(linear_module.weight.device)
    #     dyn_attn_mask = dyn_attention_heads > 0.5

    if reorder_channel == "in":
        weight1 = linear_module.weight.data[:, full_attn_mask]
        weight2 = linear_module.weight.data[:, ~full_attn_mask]
        reordered_weight = torch.cat([weight1, weight2], dim=1)
    else:
        weight1 = linear_module.weight.data[full_attn_mask, :]
        weight2 = linear_module.weight.data[~full_attn_mask, :]
        reordered_weight = torch.cat([weight1, weight2], dim=0)

    linear_module.weight.data = reordered_weight
    # for linear modules with bias
    if linear_module.bias is not None:
        bias1 = linear_module.bias.data[full_attn_mask]
        bias2 = linear_module.bias.data[~full_attn_mask]
        reordered_bias = torch.cat([bias1, bias2], dim=0)
        linear_module.bias.data = reordered_bias

    return linear_module

def enable_attention_eval(model_name, model, args):
    layer_id = model.config.num_hidden_layers
    print(layer_id)
   
    for layer in reversed(model.model.layers):
        module = layer.self_attn
        if isinstance(module, LlamaAttention) or \
           isinstance(module, Qwen2Attention) or \
           isinstance(module, Qwen3Attention):
            layer_id -= 1
            module.layer_id = layer_id
            module.flash_forward = module.forward
            module.cache_steps = args.cache_steps
            module.token_budget = args.token_budget
            module.chunk_size = args.chunk_size
            module.page_size = args.chunk_size
            module.cluster_cache = None
            if args.quest:
                module.forward = types.MethodType(forward_quest, module)
                module.gen = False
            elif args.cluster:
                module.forward = types.MethodType(forward_cluster, module)
                apply_cluster_config(module, args)
            elif args.greedy:
                module.forward = types.MethodType(forward_greedy, module)
                apply_cluster_config(module, args)
            elif args.echo:
                module.forward = types.MethodType(echokv_llama_forward, module)
                apply_echo_config(module, args)

            module.topk_stat = True if args.topk_stat else False

        elif "glm4" in model_name and module.__class__.__name__ == "SelfAttention":
            module.flash_forward = module.forward
            module.cache_steps = args.cache_steps
            module.token_budget = args.token_budget
            module.chunk_size = args.chunk_size
            module.cluster_cache = None
            if args.quest:
                module.forward = types.MethodType(forward_quest_glm, module)
                module.gen = False
            elif args.cluster:
                module.forward = types.MethodType(forward_cluster_glm, module)
                apply_cluster_config(module, args)
            module.topk_stat = True if args.topk_stat else False


def get_config_output_affix(args):
    config_affix = ''
    if args.quest:
        if args.chunk_size == 16:
            config_affix = f"-{args.token_budget}"
        else:
            config_affix = f"-{args.token_budget}c{args.chunk_size}"
    elif args.cluster:
        km_id = ("h" + args.head_sel + ("b" if args.balance else "") 
                 + str(args.nlist) + f"fi{args.fit_iter}")
        if args.sink > 0:
            km_id += f"sink{args.sink}"
        if args.gqa_policy:
            km_id += f"gqa"
        config_affix = f"-{km_id}_{args.token_budget}"
    elif args.echo:
        config_affix = f"-echokv_sink{args.sink}_{args.token_budget}"
        if args.window:
            config_affix = f"-echokv_s{args.sink}w{args.window}t{args.token_budget}"    
        if args.re:
            config_affix = f"-sink{args.sink}_{args.token_budget}_echokv_re"
        
        if args.chunk_size != 32:
            config_affix = f"-echokv_s{args.sink}w{args.window}t{args.token_budget}c{args.chunk_size}"
        
        if args.gqa_policy:
            config_affix += f"gqa{args.gqa_policy}"

    return config_affix

def sample_token(output, temperature, top_p):
    logits = output.logits[:, -1, :]
    if  temperature <= 0:
        pred_token_idx = logits.argmax(dim=-1).unsqueeze(1)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            probs[indices_to_remove] = 0.0
            renormalized_probs = probs / (torch.sum(probs, dim=-1, keepdim=True) + 1e-9)
            final_probs = renormalized_probs
        else:
            final_probs = probs

        pred_token_idx = torch.multinomial(final_probs, num_samples=1)
    return pred_token_idx

def rotate_half(x):
    """
    将输入 x 的最后维度切分为两半，并执行旋转操作:
    [x1, x2] -> [-x2, x1]
    ]"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def build_chat(tokenizer, prompt, model_name, enalbe_thinking=False):
    if "ds-r1" in model_name or "skywork-or1" in model_name:
        return prompt + "<think>\n"
    elif "llama" in model_name:
        messages = [
            {"role": "user", "content": f"{prompt}"}
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
    elif "qwen-2.5" in model_name:
        messages = [
            {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba."},
            {"role": "user", "content": prompt}
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif "qwen-3" in model_name:
        messages = [
            {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba."},
            {"role": "user", "content": prompt}
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking = enalbe_thinking
        )
    elif "QwQ" in model_name:
        messages = [
            {"role": "user", "content": prompt}
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return chat_prompt