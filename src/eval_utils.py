"""
Lightweight evaluation utilities.
Used for quick pass@1 checks between training stages without running full lm-eval.
"""

import re
import json
import glob
from typing import List, Dict, Optional


def extract_boxed(text: str) -> Optional[str]:
    m = re.search(r'\\boxed\{([^}]*)\}', text)
    return m.group(1).strip() if m else None


def normalize_answer(ans: str) -> str:
    if ans is None:
        return ''
    return ans.strip().lower().replace(' ', '').replace(',', '')


def score_responses(responses: List[str], gold_answers: List[str]) -> Dict:
    """
    Given a list of generated responses and gold answers, returns:
      - pass@1 accuracy
      - format compliance rate (% with \\boxed{})
      - avg response length (words)
    """
    assert len(responses) == len(gold_answers)
    correct, has_format, total_len = 0, 0, 0

    for resp, gold in zip(responses, gold_answers):
        pred = extract_boxed(resp)
        if normalize_answer(pred) == normalize_answer(gold):
            correct += 1
        if pred is not None:
            has_format += 1
        total_len += len(resp.split())

    n = len(responses)
    return {
        'pass_at_1': correct / n,
        'format_rate': has_format / n,
        'avg_len_words': total_len / n,
        'n': n,
    }


def load_lm_eval_result(result_dir: str, task: str) -> Optional[float]:
    """
    Parse lm-eval output directory and return the primary accuracy metric for a task.
    """
    files = glob.glob(f'{result_dir}/**/*.json', recursive=True)
    if not files:
        return None
    with open(files[0]) as f:
        data = json.load(f)
    task_results = data.get('results', {}).get(task, {})
    # Try common metric keys in order
    for key in ['exact_match,flexible-extract', 'exact_match,strict-match', 'acc,none']:
        if key in task_results:
            return task_results[key]
    return None


def build_results_table(result_dirs: Dict[str, str], tasks: List[str]) -> List[Dict]:
    """
    result_dirs: {'baseline': '/path/...', 'sft': '/path/...', ...}
    tasks: ['gsm8k_cot', 'minerva_math']
    Returns a list of dicts ready for pandas DataFrame or printing.
    """
    rows = []
    for model_tag, result_dir in result_dirs.items():
        row = {'model': model_tag}
        for task in tasks:
            row[task] = load_lm_eval_result(result_dir, task)
        rows.append(row)
    return rows
