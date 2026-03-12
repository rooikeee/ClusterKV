import os
import json
import argparse
import re
import io
# from grader import check_is_correct

def parse_agrs():
    parse = argparse.ArgumentParser(description='Visualize the result of the experiment.')
    parse.add_argument("--model_name", type=str, default=None, help="The name of the model.")
    parse.add_argument("--dataset", type=str, default=None)
    parse.add_argument("--data_dir", type=str, required=True, help="The directory of the data.")
    parse.add_argument("--max_length", type=int, default=8192, help="The max length of the data.")
    parse.add_argument("--loose", action="store_true")
    
    return parse.parse_args()

def get_trimmed_average(data_list):
    n = len(data_list)
    if n < 3:
        return None

    sorted_list = sorted(data_list)
    trimmed_list = sorted_list[1:-1]
    trimmed_sum = sum(trimmed_list)
    trimmed_len = n - 2
    average = trimmed_sum / trimmed_len
    return round(average, 2)
    
def get_result(path):
    data_list = []
    with open(path, "r", encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)

    correct_nums = 0
    corrects = []
    output_lens = []
    output_lens_fin = []
    
    num_split_out_win = []
    num_split_out_win_fin = []
    
    # Count the correct numbers
    for pid, example in enumerate(data_list):
        output_len = example["gen_len"]
        pred = example["pred"]
        answer = str(example["answer"])
        # Avoid dismatch pred
        start_index = pred.find("**Final Answer**")
        output_lens.append(output_len)
        if "num_split_out_win" in example:
            num_split_out_win.append(example["num_split_out_win"])
        if output_len < max_length:
            output_lens_fin.append(output_len)
            if "num_split_out_win" in example:
                num_split_out_win_fin.append(example["num_split_out_win"])
        if args.loose:
            start_index = 0     # do not check **Final Answer**
        if start_index != -1:
            pred = pred[start_index:]
            pattern = r"\\boxed{((?:[^{}]|\{[^{}]*\})*)}"
            # pred = pred.replace("\\", "").replace("\n", "").strip()
            match = re.search(pattern, pred)
            if match:
                pred = match.group(1)
                # print(f"==========={pid}==============")
                # print(pred, answer)
                pred = pred.replace(" ", "")
                answer = answer.replace(" ", "")
                if pred == answer:
                # if check_is_correct(pred, answer):
                    correct_nums += 1
                    corrects.append(pid+1)  # start from 1

    accuracy = round((correct_nums/len(data_list))*100, 2) if len(data_list) > 0 else 0
    avg_output_len = round(sum(output_lens) / len(output_lens), 1) if len(output_lens) > 0 else 0

    avg_num_split_out_win = -1
    avg_num_split_out_win_fin = -1
    if len(num_split_out_win) > 0:
        avg_num_split_out_win = round(sum(num_split_out_win)/len(num_split_out_win), 1)
    if len(num_split_out_win_fin) > 0:

        avg_num_split_out_win_fin = round(sum(num_split_out_win_fin)/len(num_split_out_win_fin), 1)
    if len(output_lens_fin) > 0:
        avg_output_len_fin = round(sum(output_lens_fin) / len(output_lens_fin), 1)
    else:
        avg_output_len_fin = 99999
    return accuracy, avg_output_len, avg_output_len_fin, corrects, len(data_list), \
            avg_num_split_out_win, avg_num_split_out_win_fin

if __name__ == "__main__":
    args = parse_agrs()
    # load the result
    data_dir = args.data_dir
    max_length = args.max_length
    model_name = args.model_name
    
    print(data_dir)
    all_results = dict()
    all_corrects = dict()
    for i, method_dir in enumerate(sorted(os.listdir(data_dir))):
        # method_name = method_dir.split("-")[-1]
        if not os.path.isdir(os.path.join(data_dir, method_dir)):
            continue
        if model_name is not None and not model_name in method_dir:
            continue
        for data_file in sorted(os.listdir(os.path.join(data_dir, method_dir))):
            # dataset should be at first
            dataset = data_file.replace(".jsonl", "").split("-")[0]
            if args.dataset is not None and dataset != args.dataset:
                continue
            print("Eval on", data_file)
            id = data_file.replace(dataset, method_dir).replace(".jsonl", "")
            # seed should be at last (or not exist)
            id_no_seed = id.split("-seed")[0]
            acc, avg_len, avg_len_fin, corrects, data_list_len, avg_num_split_out_win, avg_num_split_out_win_fin = \
                get_result(os.path.join(data_dir, method_dir, data_file))

            if not dataset in all_results:
                all_results[dataset] = {}
                all_corrects[dataset] = {}
            if not id_no_seed in all_results[dataset]:
                all_results[dataset][id_no_seed] = {
                    "accuracy": [acc],
                    "avg_len": [avg_len],
                    "avg_len_fin": [avg_len_fin],
                    "correct": set(corrects),
                    "n_problem": [data_list_len],
                }
                if avg_num_split_out_win > 0:
                    all_results[dataset][id_no_seed].update({
                        "avg_num_split_out_win": [avg_num_split_out_win],
                        "avg_num_split_out_win_fin": [avg_num_split_out_win_fin],
                    })
            else:
                all_results[dataset][id_no_seed]["accuracy"].append(acc)
                all_results[dataset][id_no_seed]["avg_len"].append(avg_len)
                all_results[dataset][id_no_seed]["avg_len_fin"].append(avg_len_fin)
                all_results[dataset][id_no_seed]["correct"] = \
                    all_results[dataset][id_no_seed]["correct"].union(corrects)
                all_results[dataset][id_no_seed]["n_problem"].append(data_list_len)
                if avg_num_split_out_win > 0:
                    all_results[dataset][id_no_seed]["avg_num_split_out_win"].append(avg_num_split_out_win)
                    all_results[dataset][id_no_seed]["avg_num_split_out_win_fin"].append(avg_num_split_out_win_fin)
            all_corrects[dataset][id] = corrects
    
    for dataset in all_results:
        for id in all_results[dataset]:
            acc = all_results[dataset][id]["accuracy"]
            all_results[dataset][id]["avg@k"] = round(sum(acc) / len(acc), 2)
            if (trim_avg:=get_trimmed_average(acc)) is not None:
                all_results[dataset][id]["avg@k_trim"] = trim_avg
            all_results[dataset][id]["pass@k"] = round(
                len(all_results[dataset][id]["correct"]) / all_results[dataset][id]["n_problem"][0] *100
            , 2)
            all_results[dataset][id]["correct"] = len(all_results[dataset][id]["correct"])

    with open(os.path.join(data_dir, "results.json"), 'w', encoding='utf-8') as f:
        buffer = io.StringIO()
        json.dump(all_results, buffer, ensure_ascii=False, indent=4)
        json_str = buffer.getvalue()
        # json_str = re.sub(r'\[\n\s+([^\]\n]+)\n\s+\]', r'[\1]', json_str)
        list_split = ",\n"
        json_str = re.sub(
            r'\[\n(\s+)(.*?)\n(\s+)\]',
            lambda m: f'[{", ".join([x.strip() for x in m.group(2).split(list_split)])}]',
            json_str,
            flags=re.DOTALL
        )
        f.write(json_str)
    with open(os.path.join(data_dir, "corrects.json"), 'w', encoding='utf-8') as f:
        json.dump(all_corrects, f)