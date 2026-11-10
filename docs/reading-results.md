# Reading the results

## Words used in the files

The result files, recipes and tools use short names for a few ideas.

| Word | Meaning |
|---|---|
| cell | one result: a model on one runtime, device and package |
| line | the pass/fail rule a result is judged on |
| `qa20` | the LLM check: 20 short questions |
| gate | the extra LLM prompt that must end within 16 tokens |
| sitting | one timed measuring session on the PC |
| lane | this repo's name on the shared PC's lock and notes |
| walk | one run of [reproduce.md](reproduce.md) from an empty folder |

## What the results cover

28 models (5 LLMs, and 23 models of the vision zoo,
[`john-rocky/windows-zoo`](https://github.com/john-rocky/windows-zoo): 20 image or video models and 3 audio models) run
on one Windows laptop's runtimes and devices (Intel NPU, Intel GPU, NVIDIA RTX, CPU): the LLMs as published packages,
the other 23 as the vision zoo's converted files. One JSON per cell (a model on one runtime, device and package); each
cell of this laptop is judged on a line, a pass / fail rule in its model's recipe (the `task-*` cells' lines were
written before any of them ran). What the task lines (the `task-*` cells) look at: for an LLM, whether the first line of
each of 20 short answers, normalized (trimmed and lower case; quotes, backticks or asterisks around it and a final `.`
`!` `?` dropped; spaces collapsed), equals the first line of the same model's answer in transformers on the CPU (fp32;
bf16 for gemma-4-E4B-it), the reference, on at least 18 of 20, and a gate prompt ends within 16 tokens. That is
agreement with the reference, not answers a person graded right or wrong; a second count outside the line, at the end
of each LLM `task-*` cell's verdict, is the number of answers that contain the right answer of an answer key
(`suites/llm/qa20.key.jsonl`). For the other 23, the model's task output
after a copy of its sample app's post-processing (`john-rocky/LiteRT-Models`: boxes, points, masks, depth, classes and
the like; for the audio models a keyword or the decoded sound) against the same model's fp32 file on the CPU. Each
`task-*` cell is one run that was not a timing: the speed values its JSON keeps from that run (`speed.timed` false) are
not timing results. A FAIL means this package's form did not reach the line on that runtime and device: it is recorded
against that package and is not a measure of the model or of the runtime in general (the cells do not say what caused
it).

The other cells were copied from the vision zoo ([`john-rocky/windows-zoo`](https://github.com/john-rocky/windows-zoo))
and the LLM zoo ([`john-rocky/windows-llm-zoo`](https://github.com/john-rocky/windows-llm-zoo)) with those zoos' lines
and numbers; the `task-*` cells, added on 2026-11-10, judge the same files and packages again on task lines
([`PLAN.md`](../PLAN.md), sections Z5 and Z6): the vision zoo's 47 cells that fail its internal line and the LLM zoo's 86
cells that pass its text-integrity line. The other copied cells were not judged again. 60 of the copied JSONs are the
vision zoo's Qualcomm AI Hub records of two other devices (Samsung Galaxy S26, Snapdragon X2 Elite CRD;
`role: other-device`): they keep those numbers and have no line. The cards list no cells: each card carries its
model's lines and a link to its `results/`. No file here puts runtimes side by side; `tools/table.py --local` writes
those tables on your own disk only.

## Reading a result file

`line.verdict` gives the verdict, its numbers and the line in one string, for example
`FAIL on qa20: 9 of 20 answers' first lines equal the reference's (pass mark 18); gate PASS; outside the line: 12 of
20 right by the answer key` or
`FAIL on task-keypoints: 53 of 60 points shown by either side agree (pass mark 95 %)`; `no verdict on <line>: <state>`
when the run did not end OK, and `reference of <line>` for a reference cell. `line.pass` and `line.detail` hold the same
as values (the count by the answer key, the last part of an LLM `task-*` verdict, is outside the line and is in
`llm.task.key`), `run` how the cell ran, `form` the file or package (the copied LLM cells name it in `form.package`, the LLM
`task-*` cells in `form.from`, the vision `task-*` cells the vision zoo's file in `form.path` and `form.detail`).
`source.by` says who ran the cell: the vision zoo (`windows-zoo (Z1)`), the LLM zoo (`windows-llm-zoo (Z2)`),
edge-llm-bench, Qualcomm AI Hub (the 60 `other-device` records), or `windows-zoo-template`, the private working repo
this one was cut from, where the `task-*` cells ran (its PC folder, `C:\src\zoo-template`, is in their records'
paths). A copied cell names its zoo's file in `source.migrated_from`. In the recipes' rules, the cards' generated
blocks and `source.by`, Z1 is the vision zoo, Z2 the LLM zoo and Z3 the audio zoo
([`john-rocky/windows-audio-zoo`](https://github.com/john-rocky/windows-audio-zoo)); Z5 and Z6 are this repo's two
passes of 2026-11-10 that ran the `task-*` cells (PLAN.md), and Z7 is this repo's `lane` in `zoo.toml` (its name on
the shared PC's lock and notes).

The three Qwen3 `foundry_auto_webgpu` cells (copied from the LLM zoo) and their `task-*` cells ran Foundry Local's
generic-gpu package with the main decoder on the CUDA EP, not WebGPU (`raw/z6/foundry_ep_lines.txt`; PLAN.md, "Z6
changes" item 8): `webgpu` in their names and in the copied cells' `runtime.ep` is the package's, not the EP that ran.

## Pass/fail rules

A cell is judged on a line written before it runs. Its kind, in each cell's `line.kind`:

| kind | what it compares | here |
|---|---|---|
| `task` | the model's job against the same model in fp32 on the CPU (the reference), on a fixed input set | the `task-*` cells: PLAN.md sections 1a, 1b, Z5, Z6 |
| `internal` | outputs against the reference's outputs (cosine, max \|d\|) | the vision zoo's line, on its copied cells |
| `text-integrity` | the text is well formed (length, words, no loop), not its correctness | the LLM zoo's line (edge-llm-bench's textCheck), on its copied cells |
| `reference` | none: the cell is the reference of its line | the vision zoo's CPU cells |

[PLAN.md, How to read the qa20 line](../PLAN.md#how-to-read-the-qa20-line) says what the LLM line counts and what it does
not. `schema/lines.md` has the forms per kind and task type.

## Tables on your disk

```bash
python3 tools/table.py --local          # RESULTS_VISION.md, RESULTS_LLM.md, models/index.json
python3 tools/task_pairs_vision.py      # TASK_PAIRS_VISION.md: each task cell next to the copied cell it judges again
python3 tools/task_pairs_llm.py         # TASK_PAIRS_LLM.md: the same for the LLM cells
```

These files put runtimes side by side, so they are in `.gitignore`. `python3 tools/table.py` (without `--local`)
writes what is committed: the cards' blocks and `LICENSES.md`. Never edit a generated file; `tools/lint.py` fails when
one differs from what the JSONs give (a local file only when it is on disk).
