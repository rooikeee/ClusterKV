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
from accuracy.patch import parse_common_args, enable_attention_eval, get_config_output_affix, sample_token, build_chat
from accuracy.cluster_attention import cluster_reset
from accuracy.echokv_attention import echo_reset


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser = parse_common_args(parser)
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--task", type=str, help="task name", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data_idx", type=int, default=None)
    parser.add_argument("--max_gen", type=int, default=1024 * 16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--re", action="store_true")
    return parser.parse_args(args)


# This is the customized building prompt for chat models
def load_data(data_dir, method, task, qid=None):
    data_path = os.path.join(os.getcwd(), data_dir)
    if method == "o1":
        data = []
        file_name = f"{task}.jsonl"
        with open(os.path.join(data_path, file_name), 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                data.append(example)
    elif method == "longbench":
        file_name = f"{task}.json"
        with open(os.path.join(data_path, file_name), 'r', encoding='utf-8') as f:
            data = json.load(f)
    if qid is not None:
        data = data[qid:qid+1]
    return data

def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    model_name,
    seed,
):
    preds = []
    pbar = tqdm(data)
    for idx, json_obj in enumerate(pbar):
        if args.cluster:
            cluster_reset(model)
        
        if args.echo:
            echo_reset(model)
        
        assert model.model.layers[10].self_attn.echo_anchors is None
        pbar.set_description(
            f"Generating for dataset {dataset}, seed {seed}, q_idx {idx+1}"
        )
        prompt = prompt_format.format(**json_obj)
        tokenizer_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if len(tokenizer_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenizer_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenizer_prompt[-half:], skip_special_tokens=True)
        prompt = build_chat(tokenizer, prompt, model_name, enalbe_thinking=True)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        
        output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )[0]
        
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        if len(data) == 1:
            print("len context length: ", len(output[context_length:]))
            print(pred)

            if args.echo:
                for idx, layer in enumerate(model.model.layers):
                    corr_count = layer.self_attn.corr_count
                    if corr_count > 0:
                        print(f"layer_idx: {idx}, corr_count {corr_count}")
                    
        preds.append(
            {
                "input": prompt,
                "pred": pred,
                "answer": json_obj["answer"],
                "gen_len":len(output[context_length:]),
            }
        )   
        
    return preds

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
            path, trust_remote_code=True, torch_dtype=torch.float16,
            device_map="auto", low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2", use_cache=True
        ).to(device)
    elif "llama" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True,
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

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    max_gen = args.max_gen
    dataset = args.task
    assert not (args.quest and args.cluster)     # cannot be enabled at same time
    if args.dist_t != "cosine":
        assert args.debug
    
    model2path = json.load(open("accuracy/config/model2path.json", "r"))
    model2maxlen = json.load(open("accuracy/config/model2maxlen.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    # define your model
    model, tokenizer = load_model_with_retry(
        model2path[model_name], model_name, device
    )
    max_length = model2maxlen[model_name]
    ds_dir = "accuracy/o1/datasets"

    data = load_data(ds_dir, "o1", dataset, args.data_idx)

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("accuracy/config/dataset2prompt.json", "r"))

    prompt_format = dataset2prompt[dataset.upper()]
    if "cot" in model_name:
        prompt_format += "<Thought> {thought} </Thought>\n"
    
    res_dir = os.path.join("accuracy", "o1", "results")
    os.makedirs(res_dir, exist_ok=True)
    config_affix = get_config_output_affix(args)
    os.makedirs(os.path.join(res_dir, model_name), exist_ok=True)
    out_path = f"{res_dir}/{model_name}/{dataset}{config_affix}-seed{args.seed}.jsonl"
    print(f"out path: {out_path}")

    preds = get_pred(
            model,
            tokenizer,
            data,
            max_length,
            max_gen,
            prompt_format,
            dataset,
            model_name,
            seed=args.seed,
        )
    
    with open(out_path, "w", encoding="utf-8") as f:
        for pred in preds:
            json.dump(pred, f, ensure_ascii=False)
            f.write("\n")