# Windows model zoo

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
([`PLAN.md`](PLAN.md), sections Z5 and Z6): the vision zoo's 47 cells that fail its internal line and the LLM zoo's 86
cells that pass its text-integrity line. The other copied cells were not judged again. 60 of the copied JSONs are the
vision zoo's Qualcomm AI Hub records of two other devices (Samsung Galaxy S26, Snapdragon X2 Elite CRD;
`role: other-device`): they keep those numbers and have no line. The cards list no cells: each card carries its
model's lines and a link to its `results/`. No file here puts runtimes side by side; `tools/table.py --local` writes
those tables on your own disk only.

```
zoo.toml                    the PC (ssh name, folder), the lane (this zoo's name on the shared PC's lock and notes),
                            device and runtime names
PLAN.md                     the lines and their dated changes
models/<model>/README.md    the card: Hub front matter (license, base_model, base_model_relation, library_name,
                            pipeline_tag), text, and the generated block (its lines and where its cells are)
models/<model>/recipe.toml  how each form is made and how each cell runs; the lines, written before any run
models/<model>/results/     one JSON per cell (schema/cell.schema.json)
models/<model>/reference/   the compact record of the line's reference
raw/z5/, raw/z6/            the run records and references the task cells name (`source.record`)
suites/                     the input sets: vision images (pinned URLs + sha256), LLM questions and their answer key,
                            audio clip lists
schema/                     cell.schema.json, recipe.md (the recipe's keys), lines.md (the lines per kind and task)
tools/                      the Mac side: table.py, lint.py, results.py, vision_tasks.py, plan.py, pc.sh, sitting.sh,
                            walk_compare.py, task_pairs_vision.py, task_pairs_llm.py, scrub.py, qa20_key.py
pc/                         the PC side: chain.ps1, steps.ps1, lock.ps1, launch.ps1, wait_done.ps1, zlib.ps1,
                            fetch.py, engines.py, the runners and their library zoolib.py, req/ (one pinned venv each)
tests/                      synthetic outputs and records through the vision task lines, the LLM qa20 records and the
                            LLM runners with fake engines, then results → table → lint (Mac, no model runs):
                            python3 tests/test_vision_tasks.py, test_records.py, test_llm_records.py,
                            test_llm_runners.py
LICENSES.md                 generated: each model's licence, the other repos its cells use, the input sets' terms
```

Every Mac tool runs with `python3` (3.11 or later: `tomllib`; `tools/results.py` also needs numpy, and scipy for the
vision task cells). No model runs on the Mac: references, exports and cells all run on the PC.

## Reading a cell

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

## Lines

A cell is judged on a line written before it runs. Its kind, in each cell's `line.kind`:

| kind | what it compares | here |
|---|---|---|
| `task` | the model's job against the same model in fp32 on the CPU (the reference), on a fixed input set | the `task-*` cells: PLAN.md sections 1a, 1b, Z5, Z6 |
| `internal` | outputs against the reference's outputs (cosine, max \|d\|) | the vision zoo's line, on its copied cells |
| `text-integrity` | the text is well formed (length, words, no loop), not its correctness | the LLM zoo's line (edge-llm-bench's textCheck), on its copied cells |
| `reference` | none: the cell is the reference of its line | the vision zoo's CPU cells |

[PLAN.md, How to read the qa20 line](PLAN.md#how-to-read-the-qa20-line) says what the LLM line counts and what it does
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

## Add a model

1. **Folder.** `models/<model>/` with a name of letters, digits, `.`, `_`, `-`, starting with a letter or a digit.
2. **Recipe** (`recipe.toml`, keys in `schema/recipe.md`):
   - `[model]`: `kind` (vision, llm, asr, tts), `title`, `source` (the Hub id of the original) and
     `source_revision`, `license`, `pipeline_tag`, `inputs` (a file of `suites/`), `order` (its place in a table).
   - `[[fetch]]`: each Hub repo the PC downloads, at a pinned revision, with `include` / `exclude` globs; or one plain
     file by URL (`url`, `dest`, `path`, `size`, `sha256`).
   - `[[form]]`: each form a cell runs (`id`, a short label `short`, a longer `label`; `from` = a published repo, or
     `made_by` = the step that writes it).
   - `[[step]]`: the untimed steps before the cells, in order: the reference (`reference` = the record it writes) and
     any export or setup. A step with `phase = "after"` and a `cell` runs after that cell.
   - `[[line]]`: the line the cells are judged on, with `written` = the time it was written, **before any run of the
     cells it judges** (the lint checks it against each cell's start time).
   - `[[cell]]`: `id`, `runtime`, `device` (cpu, igpu, rtx, npu), `form`, `line`, `venv` (a `pc/req/<venv>.txt`),
     `script` (a runner in `pc/`), `args` (with the placeholders below), `first_args` (the subset of a first pass),
     `timeout_min`, `ram_need_gb`. A GPU cell's runner records the adapter the run used (the runtime's log line, or
     the GPU memory per adapter): a flag is not proof.
3. **Card** (`README.md`): the front matter with the five keys above, a paragraph, and the two marker lines
   `<!-- zoo:cells begin -->` and `<!-- zoo:cells end -->` (`tools/table.py` fills the space between them).
4. **What the PC runs**, written from the recipes into `pc/` (not committed):
   ```bash
   python3 tools/plan.py manifest <model> ...    # pc/hub_manifest.json: each [[fetch]] file with its size and hash
   python3 tools/plan.py steps <model> ...       # pc/steps/setup.json, fetch.json, <model>.json (+ engines lists)
   python3 tools/plan.py first --sitting <name> --cells <model>/<cell>,...   # pc/plans/<name>.json (one untimed run each)
   ```
5. **Run them on the PC** (the walk below has the commands): push, setup and fetch, the model's steps, the first
   pass, the pull.
6. **Results, cards, lint:**
   ```bash
   python3 tools/results.py --runs raw/<runs> --refs raw/<refs>   # one JSON per cell, the line judged
   python3 tools/table.py                                         # the cards' blocks, LICENSES.md
   python3 tools/lint.py                                          # PASS / FAIL per model folder, with the reasons
   ```

Placeholders in `args` (pc/chain.ps1, pc/steps.ps1): `{ROOT}` the zoo folder on the PC, `{HUB}` its `hub\` (Hub files
as `<owner>__<repo>\<path>`), `{FIX}` its `fixtures\`, `{HOME}` the user folder, `{RUN}` the cell's run folder.

## Add a cell to a model

Add a `[[cell]]` (and its `[[form]]`, `[[fetch]]` and steps when new) to the model's recipe, then steps 4 to 6 above
with that cell alone in `--cells`. To judge the cells of one walk without touching `models/`, write them elsewhere:
`python3 tools/results.py --runs ... --refs ... --out <folder> --cells <model>/<cell>,...`.

## The shared PC

The PC tools run on a PC that other jobs share. The shared files on the PC are in `%USERPROFILE%\pcup\`: `busy.txt`
(the lock, one holder at a time), `active\<lane>*.txt` (what a job runs, since when) and `yield\<lane>.txt` (the job
waits for the lock). Set `[zoo] lane` in `zoo.toml` to the zoo's own name there.

`tools/pc.sh` reads the shared lock feed on the Mac (`~/code/standup/tools/winpc/busy_now.sh`, no ssh) before every
connection but `wait`, and refuses one (exit 3) during another job's timing or when the feed is older than 3 min; a
Mac without that feed runs it with `ZOO_FEED=none`. Untimed steps and first passes run without the lock: each is
killed when another job's lock appears and runs again from its start once the lock has been free 60 s (a suspended
process would keep its memory). A step over 8 GB (`lock = true`) takes the lock as "(not a timing)", at most 30 min.
Each list and chain keeps its note in `active\` while it runs.

A timed sitting (`python3 tools/plan.py timed ...`, `tools/pc.sh push`, then `tools/sitting.sh <name>`) waits on the
Mac until that feed shows the lock free for 60 s, takes it, passes the lane's quiet gate and starts the chain, which
settles 180 s before its first row. `tools/sitting.sh` has no `ZOO_FEED=none` path: on a Mac without the feed it waits
and never takes the lock. Before each run the chain waits for a quiet PC: CPU, NVIDIA memory, free RAM and, for a cell
with `engines_gate` (by default the Intel GPU and RTX cells), the GPU engines' load; each run's `.meta.json` keeps
these reads (`gate`). Control rows bracket the cells, each in its own folder (`runs\<name>\control-<row>\`): an
uncounted warm-up, a control head that stops the sitting when its first run is off the known value by more than 5 %,
the cells, and a control tail. `control.json` holds the verdict, and `tools/results.py` counts the sitting as a timing
only when it is a PASS. Give the known value (`--known`) from several of the control cell's runs on the same PC and
power mode: without it, the chain takes the median of the head's own runs, and stops the sitting when they spread more
than 5 % ((max − min) / median) or are fewer than three. With `--end-by`, the chain starts no cell that would not end,
with the control tail, before that time; it counts a run at the cell's `est_sec`, else at its
`timeout_min` (no recipe here sets `est_sec`: a cell with the default 20 min and three runs needs over 65 min). The
records come to `raw/<name>/` before the lock goes back. No cell of this repo's `task-*` lines is timed.

## Walk from an empty folder

The walk runs nine `task-*` cells (at least one per runtime the `task-*` cells use) and the references they need from
a folder that does not exist yet, with every file from a public source, and compares each result with this repo's cell of the same name:

| model | cell | runtime, device |
|---|---|---|
| rtmpose_s_fp16 | `task-ovnpu` | ONNX Runtime + OpenVINO EP, Intel NPU |
| rtmpose_s_fp16 | `task-ovnpu_pre` | OpenVINO (the compiled NPU blob), Intel NPU |
| rtmpose_s_fp16 | `task-cuda32` | ONNX Runtime CUDA EP, RTX |
| rtmw_fp16 | `task-ovgpu` | ONNX Runtime + OpenVINO EP, Intel GPU |
| Qwen3-0.6B | `task-openvino-genai_cpu_cpu` | OpenVINO GenAI, CPU |
| Qwen3-0.6B | `task-ort-genai_cpu_openvino` | ONNX Runtime GenAI + OpenVINO EP, CPU |
| Qwen3-0.6B | `task-llama.cpp_cpu_cpu` | llama.cpp, CPU |
| Qwen3-0.6B | `task-litert-lm_cpu_cpu` | LiteRT-LM, CPU |
| Qwen3-0.6B | `task-foundry_cpu_cpu` | Foundry Local, CPU |

Where the files come from:

- **Hub files**: each recipe's `[[fetch]]` at a pinned revision; `pc/fetch.py` checks every file against the Hub's
  sha256. The vision forms are the vision zoo's own files, with the same sha256 as the files measured: 28 of the 43
  forms the `task-*` cells use are on the main branch of `mlboydaisuke/<repo>-OpenVINO`; the other 15 (all 11 fp16
  forms, two fp32 files and two NPU blobs) and five files the references read are on the branch `windows-model-zoo`
  of the same repo (since 2026-11-10; each repo's card says what the branch adds).
- **The vision inputs and the host steps' files**: the input images and three parameter files from the app repo
  (`john-rocky/LiteRT-Models` at 96e0bc1f) and SAM2's prompt file from the Hub, by URL, pinned by sha256; the
  keyword clips of the audio models out of the Speech Commands v0.01 test archive (download.tensorflow.org), pinned
  by the archive's sha256 and each clip's.
- **llama.cpp b11721**: the release zips (CPU, Vulkan, CUDA 13.4 with its runtime zip) by URL, pinned by sha256
  (edge-llm-bench's pins), unpacked by `pc/engines.py --make llama`.
- **LiteRT-LM v0.19.0**: LiteRT-LM publishes no Windows build of its CLI, so `pc/engines.py` builds the tag with
  edge-llm-bench's `windows/build_litert_lm.ps1` (downloaded at a pinned commit, checked by sha256), with the DirectX
  shader compiler DLLs of the PyPI wheel `litert-lm-api` 0.19.0. The tag's prebuilt DLLs and the wheel's two DLLs have
  the record's sha256s; the three files the build compiles (`litert_lm_advanced_main_windows.exe`,
  `litert_lm_main.exe`, `LiteRtDispatch.dll`) get other bytes on every build, so `pc/z6_setup.py` records them and
  does not compare them. The build is a step over 8 GB: it runs under the lane's lock (`pc\steps\engines-litert.json`).
- **Foundry Local 0.13.0 packages**: the Foundry catalog, `foundry model download <variant>` into the user's Foundry
  cache when the package is not there; every file checked against the record (size and sha256).
- **CUDA 12 for Foundry Local's daemon**: the NVIDIA wheels of `pc/req/cuda12.txt` (`pc/engines.py --make cuda12`).

Needs on the PC: Windows 11, [uv](https://docs.astral.sh/uv/) (on the PATH or in winget's `Links` folder; it installs
Python 3.12 and the venvs), Windows' own `curl.exe`, Foundry Local 0.13.0 (winget), OpenSSH server, the NVIDIA, Intel
GPU and Intel NPU drivers. For the LiteRT-LM build, what the build script's header lists: Visual Studio 2022 or its
Build Tools with the C++ workload, Git for Windows with LFS, Python 3.13 for all users, Bazelisk, a JDK with
`JAVA_HOME`, LongPathsEnabled, and an elevated shell or Developer Mode for symlinks; its source and Bazel output go to
the short paths `pc/z6_spec.json` names (`C:\src\mz-litert-lm`, `C:\bzl-mz`). When the build dies with "Server
terminated abruptly", read edge-llm-bench's `windows/README.md` (Microsoft Defender). Free disk: the walk's Hub
files (4.6 GiB), the venvs, the engines and the build. On the Mac: `ssh <pc_ssh>` reaches the PC (its default shell is
PowerShell) and python3.

Set `[zoo] pc_ssh` and `pc_root` (a folder that does not exist yet) in `zoo.toml`. Each `launch` prints
`STARTED <pid>` (or `FAILED <why>`); each `wait` streams the logs and ends with one `DONE OK <time> (<done file>)` per
list or chain (`DONE FAIL <step>` when a step failed; the last number is the most hours it waits).

```bash
R='C:\src\model-zoo'      # zoo.toml's pc_root
# 0. the recipes -> pc/ (Hub API for the pins)
python3 tools/plan.py manifest rtmpose_s_fp16 rtmw_fp16 Qwen3-0.6B
python3 tools/plan.py steps rtmpose_s_fp16 rtmw_fp16 Qwen3-0.6B
python3 tools/plan.py first --sitting walk-v --cells rtmpose_s_fp16/task-ovnpu,rtmpose_s_fp16/task-ovnpu_pre,rtmpose_s_fp16/task-cuda32,rtmw_fp16/task-ovgpu
python3 tools/plan.py first --sitting walk-l --cells Qwen3-0.6B/task-openvino-genai_cpu_cpu,Qwen3-0.6B/task-ort-genai_cpu_openvino,Qwen3-0.6B/task-llama.cpp_cpu_cpu,Qwen3-0.6B/task-litert-lm_cpu_cpu,Qwen3-0.6B/task-foundry_cpu_cpu
# 1. the scripts and the suites into the empty folder, and the lane's note in %USERPROFILE%\pcup\active\
tools/pc.sh push
tools/pc.sh lock note "walk of the model zoo README (untimed; killed and run again around the locks of other jobs) since $(date +%H:%M)"
# 2. uv's Python 3.12 and the venvs; the Hub files and the vision inputs; the engines that need no lock
tools/pc.sh launch 'pc\steps.ps1' '-List setup'
tools/pc.sh launch 'pc\steps.ps1' '-List fetch'
tools/pc.sh wait "$R\logs\setup.done,$R\logs\fetch.done" "$R\logs\setup.log,$R\logs\fetch.log" 3
tools/pc.sh launch 'pc\steps.ps1' '-List engines'
tools/pc.sh wait "$R\logs\engines.done" "$R\logs\engines.log" 2
# 3. the LiteRT-LM build, under the lane's own lock (a step over 8 GB: in the slot the shared PC's order gives)
tools/pc.sh launch 'pc\steps.ps1' '-List engines-litert'
tools/pc.sh wait "$R\logs\engines-litert.done" "$R\logs\engines-litert.log" 2
# 4. each model's steps: the references (and the LLM setup: engines checked, files linked, Foundry packages)
tools/pc.sh launch 'pc\steps.ps1' '-List rtmpose_s_fp16'
tools/pc.sh launch 'pc\steps.ps1' '-List rtmw_fp16'
tools/pc.sh launch 'pc\steps.ps1' '-List Qwen3-0.6B'
tools/pc.sh wait "$R\logs\rtmpose_s_fp16.done,$R\logs\rtmw_fp16.done,$R\logs\Qwen3-0.6B.done" "$R\logs\rtmpose_s_fp16.log,$R\logs\rtmw_fp16.log,$R\logs\Qwen3-0.6B.log" 2
# 5. the first passes
tools/pc.sh launch 'pc\chain.ps1' "-Plan $R\pc\plans\walk-v.json -Mode first"
tools/pc.sh wait "$R\runs\walk-v\chain.done" "$R\runs\walk-v\chain.log" 2
tools/pc.sh launch 'pc\chain.ps1' "-Plan $R\pc\plans\walk-l.json -Mode first"
tools/pc.sh wait "$R\runs\walk-l\chain.done" "$R\runs\walk-l\chain.log" 2
# 6. the records to the Mac; the cells judged into walk-out/ (models/ is left as it is) and compared with this repo's
tools/pc.sh pull runs/walk-v raw/walk/runs
tools/pc.sh pull runs/walk-l raw/walk/runs
tools/pc.sh pull refs raw/walk
tools/pc.sh pull logs raw/walk                    # the lists' logs, the engines' and the LLM setup's reports
tools/pc.sh pull hub/fetch_report.json raw/walk   # every Hub file and input against its hash
python3 tools/results.py --runs raw/walk/runs --refs raw/walk/refs --out walk-out --cells rtmpose_s_fp16/task-ovnpu,rtmpose_s_fp16/task-ovnpu_pre,rtmpose_s_fp16/task-cuda32,rtmw_fp16/task-ovgpu,Qwen3-0.6B/task-openvino-genai_cpu_cpu,Qwen3-0.6B/task-ort-genai_cpu_openvino,Qwen3-0.6B/task-llama.cpp_cpu_cpu,Qwen3-0.6B/task-litert-lm_cpu_cpu,Qwen3-0.6B/task-foundry_cpu_cpu
python3 tools/walk_compare.py walk-out    # one line per cell and per reference: the same or not
# 7. the walk's check on the PC: every file a step leaves, each done line, each unit's records; it ends with
#    "walk check: N missing, M failed" and exits 1 unless both are 0
printf '%s\n' "& '$R\pc\walk_check.ps1' -Lists setup,fetch,engines,engines-litert,rtmpose_s_fp16,rtmw_fp16,Qwen3-0.6B -Sittings walk-v,walk-l; exit \$LASTEXITCODE" | tools/pc.sh ps
tools/pc.sh lock note ""
```

Then one timed sitting of three of these cells between control rows of `rtmpose_s_fp16/task-cuda32`, in a slot the
shared PC's order gives (the walk's second sitting held the lock for 9 min; its first stopped at the control head after
5 min). `--known` is the median of several values of `task-cuda32` on the PC of the sitting, in the same power mode:
`summary.value` of its `run<k>.json` files (step 5's first pass gives one; a sitting that stops at its control head
gives four more, under `raw/<name>/control-warm/` and `raw/<name>/control-head/`). `--end-by` is when the lock has to
be back, as an ISO time with its UTC offset (the walk's was `2026-11-10T19:17:00+09:00`). The chain counts every run at
its cell's `timeout_min`, 30 min for the two NPU cells, so the last cell needs about 96 min left when the chain reaches
it, or it is left unrun: the walk gave 106 min after taking the lock, and the sitting took 9.

```bash
# 8. a timed sitting: the plan, the plan to the PC, the sitting (its log: raw/walk-t.sitting.log; the records:
#    raw/walk-t/), the cells judged into walk-out-t/ and compared with this repo's, the walk's check of the sitting
python3 tools/plan.py timed --sitting walk-t --cells rtmw_fp16/task-ovgpu,rtmpose_s_fp16/task-ovnpu_pre,rtmpose_s_fp16/task-ovnpu --control rtmpose_s_fp16/task-cuda32 --known <value> --end-by <ISO time> --no-windows
tools/pc.sh push
tools/sitting.sh walk-t
python3 tools/results.py --runs raw --refs raw/walk/refs --out walk-out-t --cells rtmw_fp16/task-ovgpu,rtmpose_s_fp16/task-ovnpu_pre,rtmpose_s_fp16/task-ovnpu
python3 tools/walk_compare.py walk-out-t
printf '%s\n' "& '$R\pc\walk_check.ps1' -Sittings walk-t; exit \$LASTEXITCODE" | tools/pc.sh ps
```

`R` is the PC folder set in the first block. Each cell's timed value is `speed.value` (the median of its three runs,
`speed.timed` true) in `walk-out-t/<model>/<cell>.json`. A sitting that stops at its control head ends with `DONE FAIL
control head off the known value by more than 5 %: no other rows` and gives the lock back; plan the next one under a
new `--sitting` name (the chain skips every run whose records are on disk).

`WALK.md` is the log of this walk on 2026-11-10: each step's outcome and each cell's line from `tools/walk_compare.py`.
Its later sections are a second walk (the 15 models whose cells or references read the files on the Hub branches or
the keyword clips: 19 cells, 15 references) and the timed sitting of step 8.

### Not walked

The walk's timed sitting ran in one block (`--no-windows`). Not run from this repo yet: the control rows' run-time key
in `pc/chain.ps1` (`Get-RunKey`) under `tools/sitting.sh` (it came after the walk's sittings and ran only in a test
folder with a stand-in runner, `WALK.md`), the lock given back while others wait (a plan without `--no-windows`),
fixed slots (`--slots`), a cell left by `--end-by` with the real runners (that path ran only in the same test
folder), `pc/z6_setup.py`'s `foundry model download` (the walk found its three packages already in the Foundry Local
cache), `tools/pc.sh` with `ZOO_FEED=none`, and the runners of the copied cells (they stay in their zoos).

## Licences

`LICENSES.md` is generated with the cards: each model's licence (its recipe's `[model] license`, the same as its
card's front matter), the other repos its cells use, and the input sets' terms. This repo's own code and records carry
no separate licence file yet.
