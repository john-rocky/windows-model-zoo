# The README's walk, 2026-11-10

The README's "Walk from an empty folder" commands, run once from a fresh clone on a Mac against a PC folder that did
not exist, `C:\src\model-zoo`, on the shared Windows laptop the cells ran on (Intel Core Ultra 9 386H with its NPU and
GPU, RTX 5080 Laptop, 31.5 GiB of RAM). The clone held an earlier version of this tree, pushed privately at 12:29
(commit `5ecaf06`, not in this repository's history). Finding 2 below was fixed in the clone before the first PC
command; for Finding 1, step 0 ran under bash, where the old lines work, and the README was changed afterwards. How
this commit differs from the walked version is in the next section. Times are JST. The run records, the references and
the PC's logs of the walk are in `raw/walk/` (without the six npz output files: the four vision cells' and the two
vision references').

## Result

All nine cells give the same verdict and the same numbers as this repo's cells of the same name, and the three
references give the same rows (`tools/walk_compare.py`, below). Seven of the nine verdicts are FAIL and two PASS, in
the walk as in the repo; each verdict string is in its cell's JSON (`line.verdict`). `pc\walk_check.ps1`: 0 missing,
0 failed.

## This commit and the walked version

The walk ran the code of `5ecaf06`. This commit differs from it in:

- the docs: the README (its first paragraph, which cells the `task-*` cells judge again, `source.by`, the Z names, the
  Foundry Local EP note, "Not walked", the walk's model names and two pull lines), PLAN.md (the note on
  gemma-4-E4B-it's reference and "Z6 changes" item 9), one sentence of gemma-4-E4B-it's card, the vision images' row
  of `LICENSES.md` (and its text in `tools/table.py`), this file and `raw/walk/`;
- the mode of `tools/pc.sh` and `tools/sitting.sh` (100755, Finding 2);
- names of other jobs on the shared PC, their decision numbers and private paths, taken out of: comments and
  docstrings in 10 code files, the description in `suites/audio/S150.json`, a header line of
  `raw/z6/foundry_ep_lines.txt`, and the descriptive strings `source.by` (60 copied cells) and `regime.qos_basis`
  (31 copied cells);
- four lookups of other jobs' presence notes (`pc/steps.ps1`, `pc/chain.ps1`, `pc/run_llm_foundry.py`,
  `pc/z6_setup.py`): the note patterns now come from `pcup\zoo_yield.json`, a file of the PC that this repo does not
  ship (`Get-YieldNotes` in `pc/zlib.ps1`, `yield_notes` in `pc/zoolib.py` and `pc/z6_setup.py`; no file = no note
  to yield to).

The checks of the code, on 2026-11-10 at 15:53: PowerShell's tokenizer on the PC gives the walked and this commit's
`pc/zlib.ps1`, `pc/steps.ps1` and `pc/chain.ps1` the same tokens apart from comments, except the lookups and
`Get-YieldNotes`; Python's parser gives the changed `.py` files the same tree apart from docstrings, except the
lookups and `yield_notes`. The lookups ran on the PC against a test folder (`ZOO_PCUP`), in PowerShell (through
`Test-LockBarred` and `Wait-FoundryFree`) and in Python 3.13 (`yield_notes` and `foreign_notes`): no file, a file with
matching notes, a file with other notes only, and a file that does not parse gave none, the notes, none, and an error.

## Before the walk

- Not on the PC at 13:05: `C:\src\model-zoo`, `C:\src\mz-litert-lm`, `C:\bzl-mz`.
- New with the folder: uv's managed Python 3.12 and uv's package cache. `pc/steps.ps1` keeps both inside the zoo's
  folder (`python\`, `uvcache\`), and the setup logs show each download (`raw/walk/logs/setup/python.txt.err`,
  `pip-*.txt.err`).
- Shared with the PC's other jobs and not emptied:
  - Bazel's repository cache (the build's log does not show which external repositories came from it);
  - the Foundry Local model cache. The three Qwen3-0.6B variants were already in it. `pc/z6_setup.py` did not
    download them; it checked each one's 8 files against the record (size and sha256), and all matched.

## Steps

| step | ran | outcome |
|---|---|---|
| 0. `tools/plan.py manifest`, `steps`, `first` (Mac) | 12:33 | 9 sources (8 Hub repos and the vision input images by URL), 4.62 GiB; the lists setup, fetch, engines, engines-litert and one per model; the plans walk-v (4 units) and walk-l (5 units). Finding 1 |
| 1. `tools/pc.sh push`, the note | 13:13 | after Finding 2: the 7 `.ps1` files parse on the PC with the Mac's sha256 |
| 2. `setup` | 13:13 to 13:17:53 | OK: uv's Python 3.12 and seven venvs |
| 2. `fetch` | 13:13 to 13:17:44 | OK in 3.8 min: 77 files, 4,962,342,274 bytes, each matching its hash on the first try (`raw/walk/fetch_report.json`) |
| 2. `engines` | 13:18:15 to 13:20:34 | OK: the CUDA 12 wheels (17 DLLs); the four llama.cpp b11721 zips (each sha256 matched on the first try); LiteRT-LM cloned at v0.19.0 = 83ce6067, with its prebuilt Windows DLLs from git LFS |
| 3. `engines-litert` | 13:21 to 13:37:42 | OK: this zoo's own lock taken at 13:22:04 after 60 s free; the step took 15.5 min (`raw/walk/logs/engines-litert.log`; Bazel's own time 913 s, `raw/walk/logs/litert_build.log`), then the lock was released |
| (pause) | 13:38 to 14:15 | two locks of other jobs (13:39 to 13:49, 13:50 to 14:09), then the separate job below under this zoo's own lock (14:11:43 to 14:15:06); nothing of the walk ran |
| 4. the three model lists | 14:15 to 14:16:33 | OK: the references of rtmpose_s_fp16 and rtmw_fp16, and Qwen3-0.6B's setup and reference (details below) |
| 5. chain `walk-v` | 14:17:12 to 14:18:02 | 4 units OK |
| 5. chain `walk-l` | 14:18:15 to 14:21:00 | 5 units OK |
| 6. pulls, `tools/results.py`, `tools/walk_compare.py` | 14:21 to 14:42 | the runs and references pulled at 14:21; 9 cells written at 14:36, all the same as this repo's (below), after the Mac had waited 14.5 min for another job's timing; the logs and `fetch_report.json` pulled at 14:42 |
| 7. `pc\walk_check.ps1` | 14:21:31 | "walk check: 0 missing, 0 failed" |

Qwen3-0.6B's setup (step 4, `raw/walk/logs/z6_setup_Qwen3-0.6B.json`):

- The engines were checked in place against the record:
  - LiteRT-LM: 10 files; its three compiled files recorded, not compared;
  - llama.cpp: CPU 50, CUDA 54 and Vulkan 51 files, all matching.
- The model files were linked from the walk's Hub folder.
- The Foundry Local packages were found in the Foundry cache, as above.
- The reference was transformers fp32 on the CPU (0.3 min).

At 14:09 to 14:15 a separate job used this folder's `.venv-ref`, `pc\ref_llm_hf.py` and `pc\steps.ps1` under this
zoo's own lock: it put six of the 20 questions through the reference script again for four model cards on the Hub. It
was not a README step, and its logs (`logs\refcheck*`) are left out of `raw/walk/logs`.

## `tools/walk_compare.py walk-out` (the first line of each cell and reference)

```
Qwen3-0.6B reference: rows the same
Qwen3-0.6B/task-foundry_cpu_cpu: verdict and numbers the same
Qwen3-0.6B/task-litert-lm_cpu_cpu: verdict and numbers the same
Qwen3-0.6B/task-llama.cpp_cpu_cpu: verdict and numbers the same
Qwen3-0.6B/task-openvino-genai_cpu_cpu: verdict and numbers the same
Qwen3-0.6B/task-ort-genai_cpu_openvino: verdict and numbers the same
rtmpose_s_fp16 reference: rows, task_values the same
rtmpose_s_fp16/task-cuda32: verdict and numbers the same
rtmpose_s_fp16/task-ovnpu: verdict and numbers the same
rtmpose_s_fp16/task-ovnpu_pre: verdict and numbers the same
rtmw_fp16 reference: rows, task_values the same
rtmw_fp16/task-ovgpu: verdict and numbers the same
```

## What the walk found, and what changed

1. **The README's `$M`.** The README set `M="rtmpose_s_fp16 rtmw_fp16 Qwen3-0.6B"` and passed `$M`. bash splits it
   into three names; zsh (the Mac's default shell) passes one word: `no model directory: rtmpose_s_fp16 rtmw_fp16
   Qwen3-0.6B`. The README now writes the three names out.
2. **`tools/pc.sh` was not executable.** The earlier version (`5ecaf06`) had `tools/pc.sh` and `tools/sitting.sh` as
   mode 100644 (the copy from the working repo wrote the bytes, not the mode), so `tools/pc.sh` gave "Permission
   denied" in the fresh clone (on `tools/pc.sh status` at 13:05, before the README's first PC command). Both are
   100755 now.

## `pc\walk_check.ps1` (its last line)

```
walk check: 0 missing, 0 failed
```
