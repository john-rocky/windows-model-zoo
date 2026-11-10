# The README's walk, 2026-11-10

The README's "Walk from an empty folder" commands, run once from a fresh clone on a Mac against a PC folder that did
not exist, `C:\src\model-zoo`, on the shared Windows laptop the cells ran on (Intel Core Ultra 9 386H with its NPU and
GPU, RTX 5080 Laptop, 31.5 GiB of RAM). The clone held an earlier version of this tree, pushed privately at 12:29
(commit `5ecaf06`, not in this repository's history). Finding 2 below was fixed in the clone before the first PC
command; for Finding 1, step 0 ran under bash, where the old lines work, and the README was changed afterwards. How
`8baf723`, the commit that added this log, differs from the walked version is in the next section. Times are JST. The run records, the references and
the PC's logs of the walk are in `raw/walk/` (without the six npz output files: the four vision cells' and the two
vision references').

## Result

All nine cells give the same verdict and the same numbers as this repo's cells of the same name, and the three
references give the same rows (`tools/walk_compare.py`, below). Seven of the nine verdicts are FAIL and two PASS, in
the walk as in the repo; each verdict string is in its cell's JSON (`line.verdict`). `pc\walk_check.ps1`: 0 missing,
0 failed.

## `8baf723` and the walked version

The walk ran the code of `5ecaf06`. `8baf723` differs from it in:

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

The checks of the code, on 2026-11-10 at 15:53: PowerShell's tokenizer on the PC gives the walked and `8baf723`'s
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

## A second walk: the files on the Hub branches and the keyword clips, 2026-11-10

The README's commands, run for the 15 models whose cells or references read the files on the Hub branches
`windows-model-zoo` or the keyword clips ("Where the files come from"), from a fresh clone and a PC folder that did
not exist, `C:\src\z9-walk`, in three starts:

| start (sitting) | models | cells | code | ran (JST) |
|---|---|---|---|---|
| `z9-walk-min` | dfine_graphB_fp16, rtmpose_s_fp16 | 5 | `6fcf626` | 17:28 to 17:42 |
| `z9-walk-rest` | the other 11 that read the branch files | 10 | `6fcf626` | 17:42 to 17:53 |
| `z9-walk-kws` | w2v2_head_fp16, w2v2_frontend_fp16, dac_16khz_deconly_zs_fp16 | 4 | `e3fc1f5` | 17:53 to 17:58 |

All 19 cells give the same verdict and numbers as this repo's cells, and the 15 references give the same rows
(below). Ten of the 19 verdicts are FAIL and nine PASS, in the walk as in the repo. `pc\walk_check.ps1`: 0 missing,
0 failed after each start. The first cell, dfine_graphB_fp16/task-cuda16, was stopped by another job's lock and ran
again from its start once the lock had been free 60 s (17:30:56 to 17:41:02; `raw/walk-z9/runs/z9-walk-min/chain.log`).
Records: `raw/walk-z9/` (no npz files; `fetch_report.json` and the step outputs in `logs/setup/` and `logs/fetch/` are
the third start's; each `ref.json` and `run1.json` names the files it read with their sha256).

What it found: in the second start, w2v2_head_fp16's reference stopped on a missing WAV, because no step fetched the
keyword clips that `suites/audio/kws120.json` names. The start was launched again without that model: its ten cells
ran, and its check is over those ten models. `e3fc1f5` adds that fetch, and the third start ran with it.

```
dac_16khz_deconly_zs_fp16 reference: rows, task_values the same
dac_16khz_deconly_zs_fp16/task-ovgpu: verdict and numbers the same
dfine_graphB_fp16 reference: rows, task_values the same
dfine_graphB_fp16/task-cuda16: verdict and numbers the same
dfine_graphB_fp16/task-ovgpu: verdict and numbers the same
dfine_graphB_fp16/task-ovnpu: verdict and numbers the same
dfine_graphB_fp16/task-ovnpu_pre: verdict and numbers the same
dfine_graphB_fp16_fixed reference: rows, task_values the same
dfine_graphB_fp16_fixed/task-cuda16: verdict and numbers the same
metric3d_fp16 reference: rows, task_values the same
metric3d_fp16/task-cuda16: verdict and numbers the same
ppocr_det_fp16 reference: rows, task_values the same
ppocr_det_fp16/task-cuda16: verdict and numbers the same
rtdetr_graphB_fp16 reference: rows, task_values the same
rtdetr_graphB_fp16/task-ovgpu: verdict and numbers the same
rtm_animal_fp16 reference: rows, task_values the same
rtm_animal_fp16/task-cuda16: verdict and numbers the same
rtm_face_fp16 reference: rows, task_values the same
rtm_face_fp16/task-cuda16: verdict and numbers the same
rtmhand_fp16 reference: rows, task_values the same
rtmhand_fp16/task-cuda16: verdict and numbers the same
rtmpose_s_fp16 reference: rows, task_values the same
rtmpose_s_fp16/task-cuda16: verdict and numbers the same
rtmw_fp16 reference: rows, task_values the same
rtmw_fp16/task-cuda16: verdict and numbers the same
sam2_decoder reference: rows, task_values the same
sam2_decoder/task-ovnpu_pre: verdict and numbers the same
ssdlite_mobilenetv3_320_fp16 reference: rows, task_values the same
ssdlite_mobilenetv3_320_fp16/task-cuda16: verdict and numbers the same
w2v2_frontend_fp16 reference: rows, task_values the same
w2v2_frontend_fp16/task-ovnpu: verdict and numbers the same
w2v2_frontend_fp16/task-ovnpu_pre: verdict and numbers the same
w2v2_head_fp16 reference: rows, task_values the same
w2v2_head_fp16/task-cuda16: verdict and numbers the same
```

## The timed sitting (step 8), 2026-11-10

Run from a new worktree of this repository at `8baf723` on the Mac, against another PC folder that did not exist, on
the same laptop (MSI Center's Extreme mode, on AC power, the desktop locked). Steps 0 to 5 and step 6's pulls ran first
for the two vision models and their four cells (the two models have no engines list and no step 3; the first pass was
pulled and not judged); then the sitting. Its run records are not in this repository: this repo's `task-*` cells
stay untimed. Times are JST.

| step | ran | outcome |
|---|---|---|
| 0 to 5 and step 6's pulls for `rtmpose_s_fp16` and `rtmw_fp16` | 17:03:56 to 17:09:23, the pulls at 17:20 | OK: setup (3.1 min), fetch (0.4 min), the two references, the four cells' first pass |
| a test folder: the chain with a stand-in runner | 17:09:54 to 17:16:44 | Findings 3 and 5 below; a control head 13 % off the known value stopped the sitting before any cell; a tail 37 % off gave "TAIL OFF 37.0 %: the rows of this sitting are not used" |
| a test folder: `engines_gate` | 17:18:09 to 17:19:51 | the GPU engines' counter read 0 % before the load and after it, and 88 to 98 % while a test process ran rtmw on the Intel GPU with OpenVINO; the per-run gate waited ("gpu engines 95 %") and opened after the load ended |
| 8. the first sitting | lock 17:21:41 to 17:27:02 | the control head's first run was 5.8 % off the known value, so the sitting stopped there with no cell run (Finding 6); the other two head runs were 1.6 % and 1.8 % off |
| 8. the second sitting | lock 17:30:55 to 17:40:01 | control head 3.1 % off, tail 2.5 % off: PASS. The three cells' nine runs ended OK; every run's gate read a quiet PC, and the eight runs with `engines_gate` read 0 % |
| `tools/results.py`, `tools/walk_compare.py` | 17:40 | the three cells: "verdict and numbers the same"; the two references: "rows, task_values the same" |
| `pc\walk_check.ps1` | 17:40 | "walk check: 0 missing, 0 failed" for the lists, the first pass and the second sitting (the first sitting alone: "20 missing, 1 failed") |
| a test folder: the changed `pc/chain.ps1`, two of the same plans | 17:53:55 to 18:01:00 | the same outcomes: "TAIL OFF 37.0 %: the rows of this sitting are not used"; the two NPU cells left, the tail PASS |

The sittings ran `8baf723` with Finding 3's change to `tools/plan.py`. Two changes came after them: the chain's run-time
key for the control rows (Finding 3) and `tools/results.py`'s check of `control.json` (Finding 7); the test folder and
copies of the second sitting's records checked them (below).

## What the timed sitting found, and what changed

3. **The control rows shared one run folder.** `tools/plan.py timed` gave the warm-up, the head and the tail the
   control cell's own name, so all three had one folder and one `run1.*` (`runs\<sitting>\<model>\<cell>\run1.*`).
   `pc/chain.ps1` skips a run whose records are on disk: the head's first run and the tail read the warm-up's record,
   and the tail passed without running. A stand-in runner with a different value per row showed it in the test folder:
   the sitting ended OK with "PASS (tail off 1.0 %)", the warm-up's value, though the tail's own value was 37 % off;
   only the chain log's two "run 1 on disk, skipped" lines told. Each control row now has its own folder, `runs\<sitting>\control-<row>\<model>\<cell>`, and
   `pc/chain.ps1` keys the run times of the three rows as one cell (`Get-RunKey`), so `--end-by` still counts the
   tail at the head's run time. Taken from `pc/chain.ps1` by PowerShell's parser on the PC and called with the control
   rows' names, its functions give the tail the head's run time + 60 s, as before the folders were split (0 s without
   `Get-RunKey`).
4. **The push between the plan and the sitting.** `tools/plan.py timed` writes `pc/plans/<name>.json` on the Mac and
   `tools/sitting.sh` runs the PC's copy. The README now has `tools/pc.sh push` between them.
5. **`--end-by` and `est_sec`.** The chain counts each run of a cell with no `est_sec` at its `timeout_min`, and no
   recipe here sets `est_sec`. In the test folder, with the end time 85 min after the plan, the cell with 20 min ran and
   the two NPU cells with 30 min (about 95 min for three runs) were left, though each run takes a few seconds. The
   README says so; no recipe changed.
6. **One control run against ±5 %.** Within 75 s, `task-cuda32`'s three head runs spread 7.7 % and 7.5 %
   ((max − min) / median) in the two sittings, so the first run can be more than 5 % off a known value from the middle
   of its values. The first sitting's known value was the mean of the cell's two untimed runs in this repository
   (`raw/z5`, `raw/walk`); the second took the median of seven runs of the cell on this PC in the same mode (those
   two, the walk's own first pass and the first sitting's four control runs). Without `--known`, the head's own runs
   (spread over 5 %) would have stopped both sittings.
7. **A failed control tail still gave timed values.** `tools/results.py` took a timed sitting whose runs ended OK
   without reading `control.json`, so a sitting whose tail was off ("the rows of this sitting are not used") gave its
   cells timed values. Now such a sitting is not a timing (`speed.timed` false and a note in the cell), and a first
   pass of the cell is taken before it. Checked on copies of the second sitting's records with the verdict set to
   "TAIL OFF 6.0 %" and with `control.json` removed; on the second sitting's own records it writes the same five files
   as at 17:40. On this repository's first-pass records (`raw/walk/runs`, `raw/z5/runs` and `raw/walk-z9/runs`, each
   judged for all 133 of this zoo's own cells), the old and the new `tools/results.py` write the same 440 files.
