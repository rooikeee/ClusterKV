# Based on Punica Project
# Check: https://github.com/efeslab/Atom/blob/main/e2e/punica-atom/benchmarks/bench_textgen.py

import argparse
import dataclasses
import json
import os
import time
from typing import List

import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from clusterkv.quest_models.llama import LlamaForCausalLM as QuestLlamaForCausalLM
from clusterkv.clusterkv_models.llama import LlamaForCausalLM as ClusterKVLlamaForCausalLM

c = torch.cuda.get_device_capability()
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{c[0]}.{c[1]}"


@dataclasses.dataclass
class ModelConfig:
    model_path: str
    dtype: str = dataclasses.field(default="float16")
    device: str = dataclasses.field(default="cuda:0")


MODEL_CFGS = {
    "llama2-7b": ModelConfig(model_path="meta-llama/Llama-2-7b-chat-hf"),
    "llama3-8b": ModelConfig(model_path="meta-llama/Meta-Llama-3-8B-Instruct"),
}


def ensure_tokenizer_padding(tokenizer):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError(
                "Tokenizer has no pad/eos/unk token for batching. "
                "Please set pad token explicitly."
            )
    # For decoder-only models, left padding keeps the last token position aligned.
    tokenizer.padding_side = "left"


def load_model(model_cfg: ModelConfig, method: str):
    device = torch.device(model_cfg.device)
    dtype = getattr(torch, model_cfg.dtype)
    torch.set_default_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_path, trust_remote_code=True)
    ensure_tokenizer_padding(tokenizer)

    with device:
        if method == "quest":
            model = QuestLlamaForCausalLM.from_pretrained(
                model_cfg.model_path,
                device_map=device,
                torch_dtype=dtype,
            )
        elif method in ["clusterkv", "full"]:
            model = ClusterKVLlamaForCausalLM.from_pretrained(
                model_cfg.model_path,
                device_map=device,
                torch_dtype=dtype,
            )
        else:
            raise ValueError(f"Unsupported method: {method}")
    model.eval()
    return model, tokenizer


def truncate_in_middle(tokenizer, prompt: str, max_tokens: int) -> str:
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized_prompt) <= max_tokens:
        return prompt
    half = max_tokens // 2
    return tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
        tokenized_prompt[-half:], skip_special_tokens=True
    )


def load_longgenbench_records(path: str) -> List[dict]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"LongGenBench path not found: {path}")

    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return payload["data"]
        if "examples" in payload and isinstance(payload["examples"], list):
            return payload["examples"]
    raise ValueError(f"Unsupported LongGenBench file format: {path}")


def load_json_records(path: str) -> List[dict]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Dataset path not found: {path}")
    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return payload["data"]
        if "examples" in payload and isinstance(payload["examples"], list):
            return payload["examples"]
    raise ValueError(f"Unsupported dataset format: {path}")


def build_longgenbench_prompt(example: dict, prompt_key: str = "prompt") -> str:
    if prompt_key in example and isinstance(example[prompt_key], str):
        return example[prompt_key]

    preferred_keys = [
        "instruction",
        "context",
        "input",
        "question",
        "query",
        "document",
        "passage",
    ]
    chunks = []
    for key in preferred_keys:
        value = example.get(key, None)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            chunks.append(value.strip())
    if chunks:
        return "\n\n".join(chunks)
    return json.dumps(example, ensure_ascii=False)


def load_prompts(tokenizer, args) -> List[str]:
    prompts: List[str] = []
    if args.bench_dataset == "longbench":
        data = load_dataset("THUDM/LongBench", args.longbench_task, split="test")
        prompt_format = (
            "Answer the question based on the given passage. "
            "Only give me the answer and do not output any other words. "
            "The following are some examples.\n\n{context}\n\n{input}"
        )
        for example in data:
            prompt = prompt_format.format(**example)
            prompts.append(truncate_in_middle(tokenizer, prompt, args.context_len))
            if len(prompts) >= args.num_prompts:
                break
    elif args.bench_dataset == "gov_report":
        records = load_json_records(args.gov_report_path)
        prompt_format = (
            "You are given a report by a government agency. "
            "Write a one-page summary of the report.\n\n"
            "Report:\n{context}\n\n"
            "Now, write a one-page summary of the report.\n\nSummary:"
        )
        for example in records:
            prompt = prompt_format.format(**example)
            prompts.append(truncate_in_middle(tokenizer, prompt, args.context_len))
            if len(prompts) >= args.num_prompts:
                break
    else:
        records = load_json_records(args.longgenbench_path)
        for example in records:
            prompt = build_longgenbench_prompt(example, prompt_key=args.longgen_prompt_key)
            prompts.append(truncate_in_middle(tokenizer, prompt, args.context_len))
            if len(prompts) >= args.num_prompts:
                break

    if not prompts:
        raise RuntimeError("No valid prompts were loaded.")
    return prompts


def build_batch_prompts(prompts: List[str], batch_size: int, batch_mode: str) -> List[str]:
    if batch_mode == "same":
        return [prompts[0]] * batch_size
    if len(prompts) < batch_size:
        repeats = (batch_size + len(prompts) - 1) // len(prompts)
        prompts = (prompts * repeats)[:batch_size]
        return prompts
    return prompts[:batch_size]


def _sec_to_ms(sec: float) -> float:
    return sec * 1000.0


@torch.inference_mode()
def benchmark_clusterkv():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CFGS.keys(), default="llama3-8b")
    parser.add_argument("--context_len", type=int, default=4 * 1024, help="Prefill length")
    parser.add_argument("--decode_len", type=int, default=256, help="Generation length")
    parser.add_argument("--page_size", type=int, default=16, help="Page size for Quest")
    parser.add_argument("--token_budget", type=int, default=512, help="Token budget for ClusterKV and Quest")
    parser.add_argument("--iteration", type=int, default=3, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup iterations")
    parser.add_argument("--method", type=str, choices=["quest", "clusterkv", "full"], required=True)
    parser.add_argument("--nlist", type=int, default=200, help="Number of clusters")
    parser.add_argument("--niter", type=int, default=20, help="Number of max cluster iterations")
    parser.add_argument("--sink", type=int, default=16, help="Sink size")
    parser.add_argument("--window", type=int, default=320, help="Window size")
    parser.add_argument("--window_nlist", type=int, default=8, help="Number of clusters in a window")
    parser.add_argument("--offload", action="store_true", help="Offloading cache to CPU")
    parser.add_argument(
        "--cpu_kv_all",
        action="store_true",
        help="Force all-layer KV offload to CPU (enables --offload automatically).",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for batched inference")
    parser.add_argument(
        "--batch_mode",
        choices=["same", "distinct"],
        default="same",
        help="same: duplicate one prompt across batch; distinct: use different prompts.",
    )
    parser.add_argument(
        "--bench_dataset",
        choices=["longbench", "longgenbench", "lgbench", "gov_report"],
        default="longbench",
        help="Benchmark prompt source.",
    )
    parser.add_argument("--longbench_task", type=str, default="triviaqa", help="LongBench subset name.")
    parser.add_argument(
        "--longgenbench_path",
        type=str,
        default=os.path.join("efficiency", "datasets", "longgenbench.json"),
        help="Path to LongGenBench json/jsonl file.",
    )
    parser.add_argument(
        "--gov_report_path",
        type=str,
        default=os.path.join("efficiency", "datasets", "gov_report.jsonl"),
        help="Path to gov_report json/jsonl file.",
    )
    parser.add_argument("--longgen_prompt_key", type=str, default="prompt", help="Prompt field in LongGenBench.")
    parser.add_argument("--num_prompts", type=int, default=256, help="How many samples to scan from dataset.")
    args = parser.parse_args()
    assert args.warmup < args.iteration, "Warmup iterations must be less than total iterations"

    # Alias kept for compatibility with FreeKV naming.
    if args.bench_dataset == "lgbench":
        args.bench_dataset = "longgenbench"

    assert args.model in MODEL_CFGS, f"Model {args.model} not found in MODEL_CFGS"
    model_cfg = MODEL_CFGS[args.model]

    if args.offload:
        assert args.method == "clusterkv", "Offloading is only supported for clusterkv"
    if args.cpu_kv_all:
        if args.method != "clusterkv":
            raise ValueError("--cpu_kv_all is only supported for clusterkv")
        args.offload = True
    if args.method == "quest" and args.batch_size > 1:
        raise ValueError("Quest path currently supports batch_size=1 only.")

    max_seq_len = args.context_len + args.decode_len + 512
    method = args.method
    token_budget = 102400 if method == "full" else args.token_budget

    model, tokenizer = load_model(model_cfg, method)

    dtype = getattr(torch, model_cfg.dtype)
    device = torch.device(model_cfg.device)
    if method == "quest":
        model.quest_init(
            page_size=args.page_size,
            max_seq_len=max_seq_len,
            token_budget=token_budget,
            dtype=dtype,
            device=device,
        )
    elif method in ["clusterkv", "full"]:
        model.clusterkv_init(
            nlist=args.nlist,
            niter=args.niter,
            max_seq_len=max_seq_len,
            token_budget=token_budget,
            dtype=dtype,
            device=device,
            full=(method == "full"),
            sink=args.sink,
            window=args.window,
            window_nlist=args.window_nlist,
            offload=True if args.offload else False,
            offload_all_layers=True if args.cpu_kv_all else False,
            batch_size=args.batch_size,
        )

    prompts = load_prompts(tokenizer, args)
    batch_prompts = build_batch_prompts(prompts, args.batch_size, args.batch_mode)
    print("=" * 100)
    print(f"method={method}, dataset={args.bench_dataset}, batch_size={args.batch_size}, batch_mode={args.batch_mode}")
    print(f"context_len={args.context_len}, decode_len={args.decode_len}, token_budget={token_budget}")
    print(f"offload={args.offload}, cpu_kv_all={args.cpu_kv_all}")
    print(f"example prompt chars={len(batch_prompts[0])}")

    prefill_latency = []
    decode_latency = []
    decode_total_latency_per_iter = []
    decode_steps_per_iter = []

    for _ in tqdm(range(args.iteration)):
        torch.cuda.empty_cache()

        batch_input = tokenizer(batch_prompts, truncation=False, padding=True, return_tensors="pt").to(device)
        input_ids = batch_input.input_ids
        attention_mask = batch_input.attention_mask

        generated_ids: List[List[int]] = [[] for _ in range(args.batch_size)]

        # Prefill stage.
        ts = time.perf_counter()
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        te = time.perf_counter()
        prefill_latency.append(te - ts)

        pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
        for b_idx in range(args.batch_size):
            generated_ids[b_idx].append(pred_token_idx[b_idx].item())

        # Decode stage.
        step_count = 0
        decode_total_this_iter = 0.0
        for _ in range(args.decode_len):
            ts = time.perf_counter()
            output = model(input_ids=pred_token_idx)
            te = time.perf_counter()
            step_latency = te - ts
            decode_latency.append(step_latency)
            decode_total_this_iter += step_latency
            step_count += 1
            pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
            for b_idx in range(args.batch_size):
                generated_ids[b_idx].append(pred_token_idx[b_idx].item())
        decode_total_latency_per_iter.append(decode_total_this_iter)
        decode_steps_per_iter.append(step_count)

        # print first sample decode as sanity-check
        sample_pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(sample_pred)

        if method == "quest":
            model.quest_clear()
        elif method in ["clusterkv", "full"]:
            model.clusterkv_clear()

    warmup = args.warmup
    prefill_post = prefill_latency[warmup:]
    decode_total_post = decode_total_latency_per_iter[warmup:]
    decode_steps_post = decode_steps_per_iter[warmup:]

    avg_prefill_latency = float(np.mean(prefill_post)) if prefill_post else 0.0
    avg_decode_total_latency = float(np.mean(decode_total_post)) if decode_total_post else 0.0
    total_decode_time = float(np.sum(decode_total_post)) if decode_total_post else 0.0
    total_decode_steps = int(np.sum(decode_steps_post)) if decode_steps_post else 0
    avg_decode_latency = (total_decode_time / total_decode_steps) if total_decode_steps > 0 else 0.0

    report_decode_tokens = 512
    decode_512_latency = avg_decode_latency * report_decode_tokens
    total_512_latency = avg_prefill_latency + decode_512_latency

    print("=" * 100)
    print("Timing Summary (post-warmup)")
    print(
        f"Config: batch={args.batch_size}, token_budget={token_budget}, "
        f"context_len={args.context_len}, decode_len={args.decode_len}, warmup={warmup}"
    )
    print(f"Prefill Time: {_sec_to_ms(avg_prefill_latency):.3f} ms")
    print(f"Decode Time (avg/token): {_sec_to_ms(avg_decode_latency):.3f} ms")
    print(f"Decode Time ({report_decode_tokens} tokens): {_sec_to_ms(decode_512_latency):.3f} ms")
    print(f"Total Time = Prefill + Decode{report_decode_tokens}: {_sec_to_ms(total_512_latency):.3f} ms")
    print("=" * 100)


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    seed_everything(42)
    benchmark_clusterkv()
