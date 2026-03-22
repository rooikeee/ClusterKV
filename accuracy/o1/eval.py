import os
import json
import argparse
import re
import io
from decimal import Decimal, InvalidOperation

try:
    import sympy as sp
except Exception:
    sp = None


def parse_agrs():
    parse = argparse.ArgumentParser(description='Visualize the result of the experiment.')
    parse.add_argument("--model_name", type=str, default=None, help="The name of the model.")
    parse.add_argument("--dataset", type=str, default=None)
    parse.add_argument("--data_dir", type=str, required=True, help="The directory of the data.")
    parse.add_argument("--max_length", type=int, default=8192, help="The max length of the data.")
    parse.add_argument("--loose", action="store_true")
    parse.add_argument("--no_symbolic", action="store_true", help="Disable symbolic equivalence via sympy.")

    return parse.parse_args()


BOXED_START_PATTERN = re.compile(r'\\boxed\s*\{')
TEXT_WRAPPER_PATTERN = re.compile(r'^\\(?:text|mathrm|operatorname)\{(.*)\}$', re.DOTALL)


def extract_all_boxed(text):
    """
    提取文本中所有 \boxed{...} 的内容。
    不能只用简单正则，因为 boxed 内部可能嵌套大括号，
    例如 \boxed{\frac{14}{3}}。
    """
    results = []
    matches = list(BOXED_START_PATTERN.finditer(text))
    if not matches:
        return results

    for match in matches:
        start = match.end()
        depth = 1
        i = start
        while i < len(text):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    results.append(text[start:i].strip())
                    break
            i += 1

    return results



def extract_last_boxed(text):
    all_boxed = extract_all_boxed(text)
    if not all_boxed:
        return None
    return all_boxed[-1]



def is_balanced_braces(s):
    depth = 0
    for ch in s:
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth < 0:
                return False
    return depth == 0



def strip_outer_braces(s):
    """仅当整个字符串被一层完整外括号包裹时，去掉这一层。"""
    s = s.strip()
    while len(s) >= 2 and s[0] == '{' and s[-1] == '}':
        depth = 0
        ok = True
        for i, ch in enumerate(s):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    ok = False
                    break
        if ok:
            s = s[1:-1].strip()
        else:
            break
    return s



def strip_outer_parens(s):
    s = s.strip()
    pairs = {'(': ')', '[': ']', '{': '}'}
    while len(s) >= 2 and s[0] in pairs and s[-1] == pairs[s[0]]:
        left = s[0]
        right = pairs[left]
        depth = 0
        ok = True
        for i, ch in enumerate(s):
            if ch == left:
                depth += 1
            elif ch == right:
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    ok = False
                    break
        if ok:
            s = s[1:-1].strip()
        else:
            break
    return s



def unwrap_text_like(s):
    s = s.strip()
    while True:
        m = TEXT_WRAPPER_PATTERN.fullmatch(s)
        if not m:
            return s
        inner = m.group(1).strip()
        if not is_balanced_braces(inner):
            return s
        s = inner



def normalize_latex_answer(s):
    """
    保守型文本归一化：尽量减少纯格式差异导致的误判。
    不在这里做过强推断，数学等价放到后面的 stronger matcher。
    """
    if s is None:
        return ""

    s = str(s).strip()
    s = s.replace('$', '')
    s = re.sub(r'\\left\s*', '', s)
    s = re.sub(r'\\right\s*', '', s)
    s = s.replace(r'\dfrac', r'\frac')
    s = s.replace(r'\tfrac', r'\frac')
    s = unwrap_text_like(s)
    s = s.replace('−', '-')
    s = s.replace('–', '-')
    s = s.replace('—', '-')
    s = s.replace('∶', ':')
    s = s.replace('，', ',')
    s = s.replace('；', ';')
    s = s.replace('（', '(').replace('）', ')')
    s = s.replace('【', '[').replace('】', ']')
    s = s.replace('°', '')
    s = re.sub(r'\^\{?\\circ\}?', '', s)
    s = re.sub(r'\\!', '', s)
    s = re.sub(r'\s+', '', s)
    s = strip_outer_braces(s)
    return s



def split_top_level(s, delimiters=',;'):
    """只在顶层分割，避免把 \frac{1,2} 之类内部内容切开。"""
    parts = []
    start = 0
    depth_round = depth_square = depth_curly = 0
    for i, ch in enumerate(s):
        if ch == '(':
            depth_round += 1
        elif ch == ')':
            depth_round -= 1
        elif ch == '[':
            depth_square += 1
        elif ch == ']':
            depth_square -= 1
        elif ch == '{':
            depth_curly += 1
        elif ch == '}':
            depth_curly -= 1
        elif ch in delimiters and depth_round == depth_square == depth_curly == 0:
            parts.append(s[start:i].strip())
            start = i + 1
    parts.append(s[start:].strip())
    return [p for p in parts if p != '']



def maybe_strip_assignment(s):
    """
    对 x=2 / y=\frac{1}{2} 这类形式，尽量提取右侧答案。
    仅在左边非常像单变量时触发，避免误伤一般方程。
    """
    s = s.strip()
    if s.count('=') != 1:
        return s
    left, right = s.split('=', 1)
    if re.fullmatch(r'[A-Za-zα-ωΑ-Ω]+(?:_\{?[A-Za-z0-9]+\}?)?', left):
        return right.strip()
    return s



def latex_frac_to_plain(s):
    pattern = re.compile(r'\\frac\{([^{}]+)\}\{([^{}]+)\}')
    prev = None
    while prev != s:
        prev = s
        s = pattern.sub(r'((\1)/(\2))', s)
    return s



def latex_sqrt_to_plain(s):
    s = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', s)
    return s



def latex_misc_to_plain(s):
    replacements = {
        r'\cdot': '*',
        r'\times': '*',
        r'\pi': 'pi',
        r'\pm': '+-',
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    s = re.sub(r'\\(?:mathrm|text|operatorname)\{([^{}]+)\}', r'\1', s)
    s = s.replace('{', '(').replace('}', ')')
    s = s.replace('^', '**')
    return s



def to_sympy_expr_str(s):
    s = normalize_latex_answer(s)
    s = maybe_strip_assignment(s)
    s = strip_outer_parens(strip_outer_braces(s))
    s = latex_frac_to_plain(s)
    s = latex_sqrt_to_plain(s)
    s = latex_misc_to_plain(s)
    return s



def parse_decimal(s):
    s = normalize_latex_answer(s)
    s = maybe_strip_assignment(s)
    s = strip_outer_parens(s)
    if re.fullmatch(r'[+-]?\d+(?:\.\d+)?', s):
        try:
            return Decimal(s)
        except InvalidOperation:
            return None
    return None



def parse_simple_fraction(s):
    s = normalize_latex_answer(s)
    s = maybe_strip_assignment(s)
    s = strip_outer_parens(s)

    m = re.fullmatch(r'([+-]?\d+)\s*/\s*([+-]?\d+)', s)
    if m:
        den = int(m.group(2))
        if den == 0:
            return None
        return sp.Rational(int(m.group(1)), den) if sp is not None else Decimal(m.group(1)) / Decimal(m.group(2))

    m = re.fullmatch(r'\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}', s)
    if m:
        den = int(m.group(2))
        if den == 0:
            return None
        return sp.Rational(int(m.group(1)), den) if sp is not None else Decimal(m.group(1)) / Decimal(m.group(2))

    return None



def numeric_equivalent(a, b):
    da = parse_decimal(a)
    db = parse_decimal(b)
    if da is not None and db is not None:
        return da == db

    fa = parse_simple_fraction(a)
    fb = parse_simple_fraction(b)
    if fa is not None and fb is not None:
        return fa == fb

    # fraction vs decimal
    if fa is not None and db is not None:
        return Decimal(str(float(fa))) == db
    if fb is not None and da is not None:
        return Decimal(str(float(fb))) == da

    return False



def safe_sympy_parse(expr_str):
    if sp is None or not expr_str:
        return None
    try:
        transformations = sp.parsing.sympy_parser.standard_transformations + (
            sp.parsing.sympy_parser.implicit_multiplication_application,
        )
        return sp.parsing.sympy_parser.parse_expr(expr_str, transformations=transformations, evaluate=True)
    except Exception:
        return None



def symbolic_equivalent(a, b):
    if sp is None:
        return False

    sa = to_sympy_expr_str(a)
    sb = to_sympy_expr_str(b)
    if not sa or not sb:
        return False
    if '+-' in sa or '+-' in sb:
        return False

    ea = safe_sympy_parse(sa)
    eb = safe_sympy_parse(sb)
    if ea is None or eb is None:
        return False

    try:
        diff = sp.simplify(ea - eb)
        if diff == 0:
            return True
    except Exception:
        pass

    try:
        return bool(ea.equals(eb))
    except Exception:
        return False



def compare_ratio(a, b):
    """2:1 与 2/1 这类可选支持，但只在双方都明显是 ratio/fraction 时启用。"""
    aa = normalize_latex_answer(a)
    bb = normalize_latex_answer(b)

    def ratio_to_pair(x):
        m = re.fullmatch(r'([+-]?\d+)\:([+-]?\d+)', x)
        if not m:
            return None
        x1, x2 = int(m.group(1)), int(m.group(2))
        if x2 == 0:
            return None
        g = abs(sp.gcd(x1, x2)) if sp is not None else 1
        if g:
            x1 //= g
            x2 //= g
        return (x1, x2)

    def frac_to_pair(x):
        m = re.fullmatch(r'([+-]?\d+)\s*/\s*([+-]?\d+)', x)
        if not m:
            m = re.fullmatch(r'\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}', x)
        if not m:
            return None
        x1, x2 = int(m.group(1)), int(m.group(2))
        if x2 == 0:
            return None
        g = abs(sp.gcd(x1, x2)) if sp is not None else 1
        if g:
            x1 //= g
            x2 //= g
        return (x1, x2)

    pa = ratio_to_pair(aa) or frac_to_pair(aa)
    pb = ratio_to_pair(bb) or frac_to_pair(bb)
    return pa is not None and pb is not None and pa == pb



def compare_sequence_like(a, b, symbolic=True):
    """比较 a,b 是否为顶层列表/元组形式，并逐项比较。"""
    aa = maybe_strip_assignment(normalize_latex_answer(a))
    bb = maybe_strip_assignment(normalize_latex_answer(b))
    aa = strip_outer_parens(aa)
    bb = strip_outer_parens(bb)

    parts_a = split_top_level(aa, ',;')
    parts_b = split_top_level(bb, ',;')

    if len(parts_a) <= 1 or len(parts_b) <= 1 or len(parts_a) != len(parts_b):
        return False

    return all(math_equal(x, y, symbolic=symbolic) for x, y in zip(parts_a, parts_b))



def compare_set_like(a, b, symbolic=True):
    aa = maybe_strip_assignment(normalize_latex_answer(a))
    bb = maybe_strip_assignment(normalize_latex_answer(b))

    if not ((aa.startswith('{') and aa.endswith('}')) and (bb.startswith('{') and bb.endswith('}'))):
        return False

    inner_a = strip_outer_braces(aa)
    inner_b = strip_outer_braces(bb)
    parts_a = split_top_level(inner_a, ',;')
    parts_b = split_top_level(inner_b, ',;')
    if len(parts_a) != len(parts_b):
        return False

    used = [False] * len(parts_b)
    for x in parts_a:
        matched = False
        for i, y in enumerate(parts_b):
            if not used[i] and math_equal(x, y, symbolic=symbolic):
                used[i] = True
                matched = True
                break
        if not matched:
            return False
    return True



def math_equal(pred_ans, gold_ans, symbolic=True):
    """
    强判定入口，按从保守到激进的顺序比较：
    1) 文本归一化后完全相同
    2) 比例/分数/小数等价
    3) 顶层序列逐项等价
    4) 顶层集合无序等价
    5) sympy 符号等价
    """
    pred_norm = maybe_strip_assignment(normalize_latex_answer(pred_ans))
    gold_norm = maybe_strip_assignment(normalize_latex_answer(gold_ans))

    if pred_norm == gold_norm:
        return True

    if compare_ratio(pred_norm, gold_norm):
        return True

    if numeric_equivalent(pred_norm, gold_norm):
        return True

    if compare_sequence_like(pred_norm, gold_norm, symbolic=symbolic):
        return True

    if compare_set_like(pred_norm, gold_norm, symbolic=symbolic):
        return True

    if symbolic and symbolic_equivalent(pred_norm, gold_norm):
        return True

    return False



def verify_answer(text, correct_answer, symbolic=True):
    cleaned_answer = str(correct_answer)
    extracted_matches = extract_all_boxed(text)

    for match in extracted_matches:
        if math_equal(match, cleaned_answer, symbolic=symbolic):
            return True
    return False



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

    for pid, example in enumerate(data_list):
        output_len = example["gen_len"]
        pred = example["pred"]
        answer = str(example["answer"])

        start_index = pred.find("**Final Answer**")

        output_lens.append(output_len)
        if "num_split_out_win" in example:
            num_split_out_win.append(example["num_split_out_win"])
        if output_len < max_length:
            output_lens_fin.append(output_len)
            if "num_split_out_win" in example:
                num_split_out_win_fin.append(example["num_split_out_win"])

        if args.loose:
            start_index = 0

        if start_index != -1:
            pred_segment = pred[start_index:]

            if args.loose:
                if verify_answer(pred_segment, answer, symbolic=not args.no_symbolic):
                    correct_nums += 1
                    corrects.append(pid + 1)
            else:
                pred_final = extract_last_boxed(pred_segment)
                if math_equal(pred_final, answer, symbolic=not args.no_symbolic):
                    correct_nums += 1
                    corrects.append(pid + 1)

    accuracy = round((correct_nums / len(data_list)) * 100, 2) if len(data_list) > 0 else 0
    avg_output_len = round(sum(output_lens) / len(output_lens), 1) if len(output_lens) > 0 else 0

    avg_num_split_out_win = -1
    avg_num_split_out_win_fin = -1
    if len(num_split_out_win) > 0:
        avg_num_split_out_win = round(sum(num_split_out_win) / len(num_split_out_win), 1)
    if len(num_split_out_win_fin) > 0:
        avg_num_split_out_win_fin = round(sum(num_split_out_win_fin) / len(num_split_out_win_fin), 1)
    if len(output_lens_fin) > 0:
        avg_output_len_fin = round(sum(output_lens_fin) / len(output_lens_fin), 1)
    else:
        avg_output_len_fin = 99999
    return accuracy, avg_output_len, avg_output_len_fin, corrects, len(data_list), \
        avg_num_split_out_win, avg_num_split_out_win_fin


if __name__ == "__main__":
    args = parse_agrs()
    data_dir = args.data_dir
    max_length = args.max_length
    model_name = args.model_name

    print(data_dir)
    all_results = dict()
    all_corrects = dict()
    for i, method_dir in enumerate(sorted(os.listdir(data_dir))):
        if not os.path.isdir(os.path.join(data_dir, method_dir)):
            continue
        if model_name is not None and model_name not in method_dir:
            continue
        for data_file in sorted(os.listdir(os.path.join(data_dir, method_dir))):
            dataset = data_file.replace(".jsonl", "").split("-")[0]
            if args.dataset is not None and dataset != args.dataset:
                continue
            print("Eval on", data_file)
            id = data_file.replace(dataset, method_dir).replace(".jsonl", "")
            id_no_seed = id.split("-seed")[0]
            acc, avg_len, avg_len_fin, corrects, data_list_len, avg_num_split_out_win, avg_num_split_out_win_fin = \
                get_result(os.path.join(data_dir, method_dir, data_file))

            if dataset not in all_results:
                all_results[dataset] = {}
                all_corrects[dataset] = {}
            if id_no_seed not in all_results[dataset]:
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
            if (trim_avg := get_trimmed_average(acc)) is not None:
                all_results[dataset][id]["avg@k_trim"] = trim_avg
            all_results[dataset][id]["pass@k"] = round(
                len(all_results[dataset][id]["correct"]) / all_results[dataset][id]["n_problem"][0] * 100,
                2
            )
            all_results[dataset][id]["correct"] = len(all_results[dataset][id]["correct"])

    with open(os.path.join(data_dir, "results.json"), 'w', encoding='utf-8') as f:
        buffer = io.StringIO()
        json.dump(all_results, buffer, ensure_ascii=False, indent=4)
        json_str = buffer.getvalue()
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
