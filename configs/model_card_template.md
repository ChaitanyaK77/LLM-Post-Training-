---
language: en
license: apache-2.0
base_model: Qwen/Qwen3-8B
tags:
  - math
  - reasoning
  - rlvr
  - grpo
  - qlora
datasets:
  - open-r1/OpenR1-Math-220k
---

# Qwen3-8B Math Reasoning (SFT + GRPO)

Fine-tuned from [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) using:
1. QLoRA SFT on [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
2. GRPO with three verifiable reward functions (correctness, format, length)

## Results

| Benchmark | Base | + SFT | + GRPO |
|-----------|------|-------|--------|
| GSM8K     | -    | -     | -      |
| MATH-500  | -    | -     | -      |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "YOUR_HF_USERNAME/qwen3-8b-math-grpo-reward-C")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

messages = [
    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
    {"role": "user",   "content": "If x^2 - 5x + 6 = 0, what are the values of x?"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=1024, temperature=0.6, top_p=0.95, do_sample=True)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## Training Details

- Hardware: 1× NVIDIA A100 (40GB or 80GB)
- SFT: QLoRA r=16, α=32, 1 epoch, lr=2e-4
- GRPO: 500 steps, lr=5e-6, β=0.04, 4 generations/prompt
- Reward functions: correctness (+1/-1), format (+0.2), length penalty
