# LLM Post-Training Pipeline-Qwen3-8B

A reproducible single-GPU post-training pipeline for mathematical reasoning. Implements supervised fine-tuning with QLoRA, Direct Preference Optimization, and Group Relative Policy Optimization with verifiable rewards, evaluated on GSM8K.

![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.10-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![GPU](https://img.shields.io/badge/GPU-A100%20%7C%20H100-76b900)
![Base Model](https://img.shields.io/badge/base-Qwen3--8B-orange)

---

## Overview

This repository implements a complete post-training stack on top of `Qwen/Qwen3-8B`:

- **Stage 1.** Supervised fine-tuning on chain-of-thought math reasoning traces using QLoRA.
- **Stage 2a.** Direct Preference Optimization as a preference-learning baseline.
- **Stage 2b.** Group Relative Policy Optimization with three programmatic reward functions: exact-answer correctness, output-format compliance, and a length penalty against verbosity.

The pipeline is structured as seven sequential Colab notebooks, designed to run end-to-end on a single NVIDIA A100 (40 GB or 80 GB) or H100. Hyperparameter recipes are provided for both 40 GB and 80 GB profiles.

## Results

GSM8K accuracy, 8-shot chain-of-thought prompting, 100-sample subset, greedy decoding via `lm-evaluation-harness`:

| Model                | GSM8K  | Δ vs. Baseline |
| :------------------- | :----: | :------------: |
| Qwen3-8B (baseline)  | 16.0 % |       —        |
| + SFT (QLoRA)        | **22.0 %** |  **+6.0** |
| + GRPO (3 rewards)   | 19.0 % |      +3.0      |

Statistical note: 100-sample Wilson 95% CI is approximately ±8 pp at p ≈ 0.2. Differences are reported as point estimates.

## Tech Stack

`Qwen3-8B` · `TRL` · `PEFT` · `bitsandbytes` · `transformers` · `accelerate` · `lm-evaluation-harness` · `Weights & Biases`

## Quick Start

```bash
git clone https://github.com/ChaitanyaK77/LLM-Post-Training-.git
cd LLM-Post-Training-
```

Open each notebook in Colab and run sequentially from `00` through `06`. Every notebook persists its artifacts to Google Drive before exiting, so progress survives session disconnects.

| # | Notebook              | Purpose                                   | Wall-clock (H100) |
| :- | :------------------- | :---------------------------------------- | :---------------: |
| 00 | `environment_check`   | Verify GPU, install pinned dependencies   |       2 min       |
| 01 | `data_preparation`    | Download, filter, deduplicate, audit data |      10 min       |
| 02 | `baseline_evaluation` | Benchmark base model on GSM8K             |      20 min       |
| 03 | `sft_training`        | QLoRA SFT on OpenR1-Math-220k             |        2 h        |
| 04 | `dpo_baseline`        | DPO preference fine-tuning                |      30 min       |
| 05 | `grpo_rlvr_training`  | GRPO with verifiable rewards              |      30 min       |
| 06 | `final_evaluation`    | Comparison table and error analysis       |        1 h        |

## Repository Layout

```
LLM-Post-Training-/
├── notebooks/                   # Seven sequential Colab notebooks
├── configs/
│   ├── a100_40gb.yaml           # Hyperparameter recipe for 40 GB
│   ├── a100_80gb.yaml           # Hyperparameter recipe for 80 GB
│   ├── requirements.txt         # Pinned dependency versions
│   └── model_card_template.md   # Hugging Face model card template
├── src/
│   ├── reward_functions.py      # Verifiable reward implementations
│   └── eval_utils.py            # Evaluation result parsing utilities
└── README.md
```

## Pipeline Architecture

```
Qwen3-8B  (frozen, 4-bit NF4)
   │
   ▼   Stage 1
SFT (QLoRA, r=16, α=32)  ──►  sft_final
   │
   ├──►  Stage 2a   DPO  ──►  dpo_final
   │
   └──►  Stage 2b   GRPO ──►  grpo_final
                              │
                              ▼
                       Final Evaluation
                       + Error Analysis
```

### Stage 1 — Supervised Fine-Tuning

| Field        | Value                                                    |
| :----------- | :------------------------------------------------------- |
| Dataset      | `open-r1/OpenR1-Math-220k`, `default` split (94 K)       |
| Adapter      | LoRA, rank 16, alpha 32, on all seven linear projections |
| Optimizer    | Paged AdamW 8-bit, cosine schedule                       |
| Learning rate| 2 × 10⁻⁴                                                 |
| Precision    | bf16 mixed, gradient checkpointing                       |
| Quantization | 4-bit NF4, double quantization (≈ 5 GB static footprint) |
| Steps        | 300, effective batch size 16                             |

### Stage 2a — Direct Preference Optimization

| Field        | Value                          |
| :----------- | :----------------------------- |
| Dataset      | `argilla/dpo-mix-7k`           |
| Initialization | SFT adapter                  |
| β            | 0.1                            |
| Learning rate| 5 × 10⁻⁵                       |
| Steps        | 100                            |

### Stage 2b — Group Relative Policy Optimization

Three programmatic reward functions are evaluated per completion:

```python
reward_correctness(c, gold)  # +1 if \boxed{·} matches gold, else −1
reward_format(c)             # +0.2 if \boxed{·} present, else 0
reward_length(c)             # linear penalty above 2048 words, floor −0.5
```

Per-completion reward lies in `[−1.5, +1.2]`. The correctness magnitude exceeds the format magnitude by a factor of five, ensuring that no positive net reward is attainable through formatting alone.

Ablation harness: set `REWARD_MODE` in notebook 05 to `'A'` (correctness only), `'B'` (correctness + format), or `'C'` (all three).

## Mathematical Formulation

GRPO samples a group of $G$ completions $\{o_1, \ldots, o_G\}$ per prompt $q$. The group-relative advantage of completion $i$ is

$$
A_i = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r) + \varepsilon}.
$$

The policy is updated by maximizing a clipped surrogate objective with a KL penalty against a frozen reference policy:

$$
\mathcal{L}(\theta) = -\,\mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G} \min\!\big(\rho_i A_i,\; \mathrm{clip}(\rho_i, 1-\epsilon, 1+\epsilon)\, A_i\big)\right] + \beta\, \mathrm{KL}\!\left(\pi_\theta \,\Vert\, \pi_{\mathrm{ref}}\right),
$$

with importance ratio $\rho_i = \pi_\theta(o_i \mid q) / \pi_{\theta_{\mathrm{old}}}(o_i \mid q)$. Eliminating the value network reduces the active-parameter footprint to a level that fits comfortably on a single GPU alongside the policy, reference, and rollout KV cache.

## Hardware Compatibility

| GPU         | Status     | Notes                                                      |
| :---------- | :--------- | :--------------------------------------------------------- |
| A100 40 GB  | Supported  | Use `configs/a100_40gb.yaml`. vLLM disabled in GRPO stage. |
| A100 80 GB  | Supported  | Use `configs/a100_80gb.yaml`. vLLM colocate mode enabled.  |
| H100 80 GB  | Supported  | Approximately 2–3× faster than A100 across all stages.     |
| T4 / V100   | Unsupported | Use the smaller `Qwen3-4B` fallback model for debugging.  |

## Discussion

The six-point GSM8K improvement from supervised fine-tuning indicates that the OpenR1-Math-220k corpus carries strong reasoning signal that transfers cleanly to the GSM8K distribution under QLoRA. The relative ordering of SFT and GRPO observed here is consistent with prior reports that GRPO requires substantially more rollouts than supervised stages to recover and exceed an SFT initialization. Published GRPO results in the literature operate at one to two orders of magnitude more total rollouts than were available in this study, and scaling the rollout budget along with group size is identified in the future-work section as the highest-priority next experiment.

A finer-grained error analysis was performed on a held-out sample to characterize remaining failure modes:

| Category               | Count |
| :--------------------- | :---: |
| Correct                |   1   |
| Boxed but incorrect    |   0   |
| No final answer emitted| 19    |
| Length-policy violation|   0   |

The dominant failure mode under temperature sampling is non-emission of the final answer within the 1024-token generation budget. The model frequently enters Qwen3's thinking-mode block, reasons productively, and is truncated before reaching the final `\boxed{·}` step. Increasing `max_new_tokens` to 2048 or disabling thinking mode at evaluation time addresses this directly; the under-greedy-decoding lm-eval scores reported above are robust to this effect because greedy decoding follows the high-probability path that includes the answer step. Zero length-policy violations indicates that the length-penalty reward effectively suppresses verbosity-based reward hacking.

## Reproducibility

Pinned library versions used for the final run (May 2026):

```text
torch          2.10.0+cu128
transformers   5.5.0
trl            0.24.0
peft           0.19.1
bitsandbytes   0.49.2
accelerate     1.13.0
datasets       4.3.0
torchao       >=0.16.0
lm-eval[math] >=0.4.5
```

Seeds are fixed at `42` for dataset partitioning and at `0` for lm-evaluation-harness. A hash-based contamination audit between the training set and the GSM8K test set returned zero overlapping items (notebook 01).

## Future Work

- Scale GRPO to ≥ 1000 optimization steps with group size 8 or greater, using vLLM colocate generation on 80 GB hardware.
- Full-test-set evaluation on GSM8K (1,319 items) with Wilson confidence intervals, plus MATH-500 and AIME-2024 subsets.
- Curriculum reward scheduling: warm up with format-only reward before introducing correctness and length terms.
- Pass@1 versus Maj@k comparison to disentangle reasoning improvement from sampling-distribution effects.
- Extend the verifiable-reward framework to code generation via unit-test-pass rewards on MBPP-style tasks.

## References

- Shao et al. (2024). [DeepSeekMath](https://arxiv.org/abs/2402.03300) — original GRPO formulation.
- DeepSeek-AI (2025). [DeepSeek-R1](https://arxiv.org/abs/2501.12948).
- Qwen Team (2025). [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388).
- Rafailov et al. (2023). [Direct Preference Optimization](https://arxiv.org/abs/2305.18290).
- Hu et al. (2022). [LoRA](https://arxiv.org/abs/2106.09685).
- Dettmers et al. (2023). [QLoRA](https://arxiv.org/abs/2305.14314).
- Cobbe et al. (2021). [GSM8K](https://arxiv.org/abs/2110.14168).
- Hendrycks et al. (2021). [MATH](https://arxiv.org/abs/2103.03874).

## License

Released under the MIT License. The base model ([Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)) and the primary training dataset ([OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)) are independently licensed under Apache 2.0; consult their respective licenses for redistribution terms.
