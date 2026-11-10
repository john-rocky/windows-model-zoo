# Run everything from an empty folder

The words used here are explained in [reading-results.md](reading-results.md#words-used-in-the-files).

The commands take turns with other jobs on the PC: [shared-pc.md](shared-pc.md) explains how.

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
gives four more, under `raw/<name>/control-warm/` and `raw/<name>/control-head/`; runs inside a lock can sit below
first passes, `WALK.md` Finding 12). `--end-by` is when the lock has to
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
new `--sitting` name (the chain skips every run whose records are on disk). A cell the chain left by `--end-by`
(`left_cells` in the sitting's `control.json`) is listed as `LEFT` by `pc\walk_check.ps1` and does not count as missing.

[`WALK.md`](../WALK.md) is the log of this walk on 2026-11-10: each step's outcome and each cell's line from `tools/walk_compare.py`.
Its later sections are a second walk (the 15 models whose cells or references read the files on the Hub branches or
the keyword clips: 19 cells, 15 references), the timed sitting of step 8, and the sittings that walked step 8's other
paths on 2026-11-10 and 11-11: a fixed slot (`--slots`), cells left by `--end-by`, `ZOO_FEED=none`, and the lock given
back in a window while another job waits (in a test lock folder).

### Not walked

Not run from this repo yet: a window that gives the shared lock itself to another job (the walk's window gave back the
lock of a test lock folder, `ZOO_PCUP`, and a stand-in took it there, while the walk kept the shared lock); a sitting
with a window that ends with its control tail (the walk's was stopped on purpose near the end of its slot, before the
tail, and is not a timing); a fixed slot in which another job takes the lock (in the walk's none came, and the chain
took the lock back at the slot's give-up time); the night stop on real steps and on a `lock = true` step (it ran in a
test lock folder with stand-in steps, under `ZOO_NIGHT`); `pc/z6_setup.py`'s `foundry model download` (the walk found
its three packages already in the Foundry Local cache); and the runners of the copied cells (they stay in their zoos).
`WALK.md` has each case.
