---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
---

# Qwen2.5-0.5B-Instruct

[Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) as the Q4_K_M file of [Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF), run through llama.cpp on one Windows laptop on the CPU and the NVIDIA RTX GPU (with CUDA and with Vulkan). Each cell is judged on the qa20 line: 20 short questions answered with greedy decoding, compared with the same model in fp32 (Transformers, CPU), and one prompt that has to end within 16 tokens.

<!-- zoo:cells begin -->
## Lines

- Line `qa20` (the task line): the 20 questions of suites/llm/qa20.jsonl, each one user turn in the model's chat template rendered by the reference step (apply_chat_template, add_generation_prompt=True, enable_thinking=False) and passed as it is, with no template of the runtime and no special token added (PLAN.md section Z6 gives the form per runtime: the rendered string's token ids, a leading <bos> dropped where the runtime adds the start token itself, Foundry Local through a local registration of the same package files whose chat template returns the message content as it is), plain greedy (repetition penalty 1.0), at most 48 new tokens: the first line of the answer, normalized (trimmed, lower case, quotes, backticks and asterisks around it and a final . ! ? dropped, spaces collapsed), equals the reference's on at least 18 of 20; and the gate prompt (Reply with only the number: 1+1=?) ends on an end token within 16 new tokens; reference = transformers fp32 on this PC's CPU. Written 2026-11-13T09:50:05+09:00 (PLAN.md section 1b). Answers right by the answer key (`suites/llm/qa20.key.jsonl`; a count outside the line, added after the runs): the reference 17 of 20; each `task-*` cell's count ends its `line.verdict` (PLAN.md, How to read the qa20 line).

## Cells

One JSON per cell (this model on one runtime, device and package) in [`results/`](results/): 3 files. Each one's `line.verdict` gives its verdict, its numbers and its line in one string; how the cell ran is in `recipe.toml` and in the run record the JSON names (`source.record`).
<!-- zoo:cells end -->
