---
license: apache-2.0
base_model: Qwen/Qwen3-4B
base_model_relation: quantized
library_name: onnxruntime-genai
pipeline_tag: text-generation
---

# Qwen3-4B

[Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) as published packages of several runtimes (LiteRT-LM, llama.cpp, ONNX Runtime GenAI, Foundry Local, OpenVINO GenAI), run on one Windows laptop on the Intel NPU, the Intel GPU, the RTX and the CPU. Its cells were copied from the LLM zoo [`john-rocky/windows-llm-zoo`](https://github.com/john-rocky/windows-llm-zoo) with that zoo's line; the cells `task-*`, added on 2026-11-10, run the same packages on the task line (below). Each cell's package and quantization is in its result JSON.

Licence: apache-2.0 (the base model's; each package's own repo is in its cell's JSON: `form.package` in the copied cells, `form.from` in the `task-*` cells).

<!-- zoo:cells begin -->
## Lines

- Line `text-check` (text integrity): edge-llm-bench's textCheck on every counted run's cold and warm text: at least 32 alphanumeric characters and 10 distinct words, a word of the task (on.device|phone|local|cloud|offline|privacy|ai), and no line repeated over half of 4 or more lines; a failing run is not counted (the answer's correctness is not judged). Written 2026-10-06T19:20:09+09:00 (edge-llm-bench fairness-rules.md section 12; windows-llm-zoo PLAN.md section 1).
- Line `qa20` (the task line): the 20 questions of suites/llm/qa20.jsonl, each one user turn in the model's chat template rendered by the reference step (apply_chat_template, add_generation_prompt=True, enable_thinking=False) and passed as it is, with no template of the runtime and no special token added (PLAN.md section Z6 gives the form per runtime: the rendered string's token ids, a leading <bos> dropped where the runtime adds the start token itself, Foundry Local through a local registration of the same package files whose chat template returns the message content as it is), plain greedy (repetition penalty 1.0), at most 48 new tokens: the first line of the answer, normalized (trimmed, lower case, quotes, backticks and asterisks around it and a final . ! ? dropped, spaces collapsed), equals the reference's on at least 18 of 20; and the gate prompt (Reply with only the number: 1+1=?) ends on an end token within 16 new tokens; reference = transformers fp32 on this PC's CPU. Written 2026-11-10T07:58:00+09:00 (PLAN.md section Z6 and section 1b). Right by the answer key (`suites/llm/qa20.key.jsonl`; outside the line, added after the runs): the reference 19 of 20; each cell's count ends its `line.verdict` (PLAN.md, How to read the qa20 line).

## Cells

One JSON per cell (this model on one runtime, device and package) in [`results/`](results/): 42 files. Each one's `line.verdict` gives its verdict, its numbers and its line in one string; how the cell ran is in `recipe.toml` and in the run record the JSON names (`source.record`).
<!-- zoo:cells end -->
