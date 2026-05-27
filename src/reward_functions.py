"""
Verifiable reward functions for GRPO math training.
All functions follow the TRL GRPOTrainer signature:
    fn(completions: list[str], **kwargs) -> list[float]
The 'solution' column from the dataset is passed via **kwargs.
"""

import re
from typing import List


def extract_boxed(text: str):
    m = re.search(r'\\boxed\{([^}]*)\}', text)
    return m.group(1).strip() if m else None


def normalize_answer(ans: str) -> str:
    if ans is None:
        return ''
    return ans.strip().lower().replace(' ', '').replace(',', '')


def reward_correctness(completions: List[str], solution, **kwargs) -> List[float]:
    """
    +1.0 if the extracted \\boxed{} answer matches the gold solution (after normalization).
    -1.0 otherwise (includes no \\boxed{} present).
    This is the primary reward signal.
    """
    rewards = []
    for comp in completions:
        pred = extract_boxed(comp)
        if normalize_answer(pred) == normalize_answer(str(solution)):
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards


def reward_format(completions: List[str], **kwargs) -> List[float]:
    """
    +0.2 if the completion contains a non-empty \\boxed{} expression.
    0.0 otherwise.
    Encourages the model to use the expected output format.
    Note: does NOT check correctness — that is reward_correctness's job.
    """
    rewards = []
    for comp in completions:
        if re.search(r'\\boxed\{[^}]+\}', comp):
            rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards


def reward_length_penalty(completions: List[str], **kwargs) -> List[float]:
    """
    Penalizes excessively long completions to prevent verbosity reward hacking.
    -0.1 per 512 extra words beyond 2048. Capped at -0.5.
    Uses word count as a fast proxy for token count.
    """
    rewards = []
    for comp in completions:
        n_words = len(comp.split())
        excess = max(0, n_words - 2048)
        penalty = -0.1 * (excess // 512)
        rewards.append(max(penalty, -0.5))
    return rewards


# Reward function sets for the three ablation experiments
REWARD_SET_A = [reward_correctness]
REWARD_SET_B = [reward_correctness, reward_format]
REWARD_SET_C = [reward_correctness, reward_format, reward_length_penalty]
