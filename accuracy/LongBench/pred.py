import os, time
from requests.exceptions import ProxyError, SSLError
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
from transformers.cache_utils import DynamicCache
from tqdm import tqdm
import numpy as np
import random
import argparse
from accuracy.patch import parse_common_args, enable_attention_eval, get_config_output_affix, build_chat
from accuracy.cluster_attention import cluster_reset
from accuracy.echokv_attention import echo_reset
import pickle

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser = parse_common_args(parser)
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--task", type=str, help="task name", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data_idx", type=int, default=None)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--re", action="store_true")
    return parser.parse_args(args)

def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    model_name,
):
    preds = []
    pbar = tqdm(data)
    if os.getenv("GET_TOPK"):
        max_gen = 512
        
    for idx, json_obj in enumerate(pbar):
        pbar.set_description(
            f"Generating for dataset {dataset}, q_idx {idx+1}"
        )
        
        if args.cluster:
            cluster_reset(model)
        
        if args.echo:
            echo_reset(model)

        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer.encode(prompt)

        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            
        if dataset not in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "lcc",
            "repobench_p",
        ]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
    
        if isinstance(prompt, str):
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(
                model.device
            ).input_ids
        else:
            input = prompt
            
        context_length = input.shape[-1]
        output = model.generate(
            input,
            max_new_tokens=max_gen,
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )[0]

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)

        if len(data) == 1:
            print(pred)
            print(len(output[context_length:]))
        # pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        if args.echo:
            for idx, layer in enumerate(model.model.layers):
                corr_count = layer.self_attn.corr_count
                if corr_count > 0:
                    print(f"layer {idx}, corr_count {corr_count}, gen_len: {len(output[context_length:])}")
          
        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )

        if os.getenv("GET_CLUSTERS"):
            get_clusters(model)
        
        if os.getenv("GET_ATTN"):
            get_attention(model)
        
        if os.getenv("GET_TOPK"):
            get_topk(model)

    return preds

def get_clusters(model: LlamaForCausalLM):
    cluster_key_indices, cluster_key_ptr, \
        layer_key_states, key_centroids, sel_clusters = [], [], [], [], []
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "cluster_key_indices"):
            cluster_key_indices.append(layer.self_attn.cluster_key_indices)
            cluster_key_ptr.append(layer.self_attn.cluster_key_ptr)
            layer_key_states.append(layer.self_attn.cluster_key)
            key_centroids.append(layer.self_attn.key_centroids)
            sel_clusters.append(layer.self_attn.total_sel_cluster)
    suffix = "pre_rope" if os.getenv("PRE_ROPE") else "post_rope"
    save_dir = os.path.join("/state", "partition", "cwli", f"{model_name}-{suffix}", args.task)
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "cluster_key_indices.pkl"), 'wb') as f:
        pickle.dump({'cluster_key_indices': cluster_key_indices, 
                    'cluster_key_ptr': cluster_key_ptr,
                    'key_states': layer_key_states,
                    'key_centroids': key_centroids,
                    'sel_clusters': sel_clusters}, f)

def get_attention(model: LlamaForCausalLM):
    if os.getenv("PRE_ROPE"):
        suffix = "pre_rope"
    elif os.getenv("NORMAL_ATTN"):
        suffix = ""
    else:
        suffix = "post_rope"
    save_dir = os.path.join("/state", "partition", "cwli", f"{model_name}-{suffix}", "attn_weight")
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer.self_attn, "attn_weight"):
            attn_weigths = layer.self_attn.attn_weight
            torch.save(attn_weigths, os.path.join(save_dir, f"layer-{layer_idx}.pt"))

def get_topk(model: LlamaForCausalLM):
    save_dir = os.path.join("/state", "partition", "cwli", f"{model_name}", "topk")
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer.self_attn, "attn_weight"):
            attn_weigths = layer.self_attn.attn_weight
            torch.save(attn_weigths, os.path.join(save_dir, f"layer-{layer_idx}.pt"))

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device):
    if "intern" in model_name or "qwen" in model_name or "glm4" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.bfloat16,
            device_map="auto", low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2", use_cache=True
        )
    elif "llama" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2", use_cache=True
        )
    else:
        assert False
    model = model.eval()

    if args.quest or args.cluster or args.echo:
        enable_attention_eval(model_name, model, args)

    return model, tokenizer

def load_model_with_retry(model_path, model_name, device, retries=3, delay=1):
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

def load_data(data_dir, method, task, qid=None):
    data_path = os.path.join(os.getcwd(), data_dir)
    data = []
    file_name = f"{task}.jsonl"
    with open(os.path.join(data_path, file_name), 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            data.append(example)
    if qid is not None:
        data = data[qid:qid+1]
    return data

if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    assert not (args.quest and args.cluster)     # cannot be enabled at same time
    if args.dist_t != "cosine":
        assert args.debug
    mode = args.mode
    model2path = json.load(open("accuracy/config/model2path.json", "r"))
    model2maxlen = json.load(open("accuracy/config/model2maxlen.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    # define your model
    model, tokenizer = load_model_with_retry(
        model2path[model_name], model_name, device
    )
    max_length = model2maxlen[model_name]
    if args.task is not None:
        datasets = [args.task]
    elif args.re:
        datasets = [
            "gov_report",
            "musique",
            "dureader",
            "lcc",
            "2wikimqa",
            "samsum"
        ]
    else:
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench_p",
            "narrativeqa",
            "multifieldqa_zh",
            "dureader",
            "vcsum",
            "passage_retrieval_zh",
            "lsht",
            "musique",
            "qmsum",
        ]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("accuracy/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("accuracy/config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    if not os.path.exists("debug"):
        os.makedirs("debug")
    data_dir = os.path.join("accuracy/LongBench/datasets")
    for dataset in datasets:
        if args.e:
            data = load_data(data_dir, "longbench", dataset, args.data_idx)
            res_dir = "debug" if args.debug or args.data_idx is not None else "pred_e"
            if not os.path.exists(f"{res_dir}/{model_name}"):
                os.makedirs(f"{res_dir}/{model_name}")
            out_path = f"{res_dir}/{model_name}/{dataset}.jsonl"
            if args.quest:
                out_path = f"{res_dir}/{model_name}/{dataset}-{args.token_budget}.jsonl"
            else:
                out_path = f"{res_dir}/{model_name}/{dataset}.jsonl"
        else:
            data = load_data(data_dir, "longbench", dataset, args.data_idx)
            res_dir = "debug" if args.debug or args.data_idx is not None else "pred"
            if not os.path.exists(f"{res_dir}/{model_name}"):
                os.makedirs(f"{res_dir}/{model_name}")
            config_affix = get_config_output_affix(args)
            if mode:
                out_path = f"{res_dir}/{model_name}/{dataset}{config_affix}_{mode}.jsonl"
            else:
                out_path = f"{res_dir}/{model_name}/{dataset}{config_affix}.jsonl"
        print(f"result save in {out_path}")
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        
        preds = get_pred(
            model,
            tokenizer,
            data,
            max_length,
            max_gen,
            prompt_format,
            dataset,
            model_name,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")
