CUDA_VISIBLE_DEVICES=0 python accuracy/LongBench/pred.py --model qwen-2.5-7b --token_budget 2048 --sink 128 --echo

CUDA_VISIBLE_DEVICES=0 python accuracy/LongBench/pred.py --model qwen-2.5-7b --token_budget 2048 --sink 128 --quest