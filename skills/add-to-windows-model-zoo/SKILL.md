---
name: add-to-windows-model-zoo
description: Add a model, or one more result (a cell) of a model, to windows-model-zoo - write the recipe and the card, run the untimed steps and one first pass on the Windows PC over SSH from a Mac, judge each cell on its line, and pass tools/lint.py. Use when asked to add a converted LLM, vision or audio package to this repo, or to run a model of this repo on one more runtime, device or package. Cold start: this file and a checkout of the repo are the only inputs.
---

# Add a model or a result to windows-model-zoo

You add one model folder, or one cell to a model folder. You are done when:

1. `python3 tools/lint.py` exits 0; its last line starts `lint: <n> model directories, <n> PASS, 0 FAIL; generated files fresh`.
2. Each new cell has `models/<model>/results/<cell>.json` whose `run.state` is `ok` and whose `line.verdict` gives
   PASS or FAIL on the line with its numbers. A FAIL is a result too. A cell whose runner failed has `run.state`
   `error` and `run.error_line`, and its `line.verdict` reads `no verdict on <line>: error (...)`.
3. `git status` shows only your model's files, `LICENSES.md`, the run records under `raw/<name>/`, and the lines of
   `.gitignore`, `docs/add-a-model.md`, `README.md` and `docs/reading-results.md` that name them or count the results
   (section 7).

Words (more in [`docs/reading-results.md`](../../docs/reading-results.md#words-used-in-the-files)): a **cell** is one
model on one runtime, device and package, one JSON in `results/`; a **form** is the package a cell runs; a **line** is
the pass / fail rule a cell is judged on, written in the recipe before any run of its cells; the **reference** is the
original model in fp32 on the PC's CPU, the record a line compares with; a **first pass** is one untimed run of a cell;
a **list** is a set of untimed steps on the PC; the **lock** is the shared PC's `%USERPROFILE%\pcup\busy.txt`.
`<name>` below is a short name you pick for this job (letters, digits, `-`); `<model>` is the model's folder name.

This skill does not cover: timed sittings (`docs/reproduce.md` step 8), a new runner or runtime, a new kind of line,
cells copied from another zoo. The LLM path below (OpenVINO GenAI on the CPU and the Intel NPU) is the one walked
on the PC; for the other runtimes and for vision and audio models, section 2 says what to copy.

## 0. Before you start

1. On the Mac, at the repo root:

   ```bash
   python3 -c 'import sys, tomllib, numpy, scipy; print(sys.version.split()[0], numpy.__version__, scipy.__version__)'
   for t in tests/test_*.py; do python3 "$t" >/dev/null 2>&1 && echo "ok   $t" || echo "FAIL $t"; done
   python3 tools/lint.py | tail -n 1
   ```

   Python 3.11 or later with numpy and scipy (the tests and the vision task cells use scipy). Every test prints `ok`,
   and lint's last line says `0 FAIL; generated files fresh`. If not, the checkout is broken before your change: stop
   and report it.
2. Work on the current `main`. When others use the same clone, take a new worktree:
   `git fetch origin && git worktree add -b <name> ../windows-model-zoo-<name> origin/main`, then work there.
3. In `zoo.toml`, set two keys of `[zoo]` for this job (and do not commit them; `git checkout -- zoo.toml` before
   the commit): `pc_root` = a folder on the PC that does not exist yet, such as `'C:\src\zoo-<name>'` (no spaces), and
   `lane` = this job's name on the shared PC (letters and digits; it names your lock and your notes there).
4. Before each connection except `wait`'s, `tools/pc.sh` reads a lock feed on the Mac
   (`~/code/standup/tools/winpc/busy_now.sh`). On a Mac without that file those calls stop with `REFUSED (lock feed
   stale or missing)`: put `export ZOO_FEED=none;` at the start of every shell call that runs `tools/pc.sh` (when each
   command runs in a new shell, a variable set in one call is gone in the next).
5. Check the PC (`[zoo] pc_ssh` is its ssh host; its default shell is PowerShell 5.1):

   ```bash
   tools/pc.sh status
   printf '%s\n' "Test-Path '<pc_root>'" | tools/pc.sh ps
   ```

   `status` prints the lock, the other jobs' notes and the free disk; the second prints `False`. The PC needs uv (on
   the PATH or in winget's `Links` folder), Windows' own `curl.exe` and the driver of each device your cells use; the
   Foundry Local and LiteRT-LM cells need more (`docs/reproduce.md`, "Needs on the PC"). Sections 1 to 4 need no
   PC: when `tools/pc.sh` is refused (section 8), do them first and check the PC again before section 5.

## 1. A model or a result

- **A model**: a new folder `models/<model>/` with a recipe (section 2) and a card (section 3), then sections 4 to 7.
- **A result** of a model already here: one more `[[cell]]` in `models/<model>/recipe.toml` (and its `[[form]]` and
  `[[fetch]]` when they are new), then sections 4 to 7 with that cell alone in `--cells`. When its device is not in the
  card's paragraph yet (section 3), add it there. It is judged on the model's line against the model's reference
  record, which `pulled_from` in `models/<model>/reference/ref.json` names (`raw/<folder>/refs/<model>/ref.json`):
  section 5 puts that record on the PC instead of running the reference again. This path is walked for an LLM model
  only. A vision or audio model's committed reference has no outputs file (`.gitignore`), so `results.py` cannot judge
  a new cell against it: stop and report it.

## 2. The recipe

`models/<model>/recipe.toml`; the keys of the template below are in `schema/recipe.md`. The folder name: letters,
digits, `.`, `_`, `-`, starting with a letter or a digit.

Read each Hub repo first (`pc/fetch.py` downloads without a Hugging Face token, so a gated repo does not download):

```bash
curl -s https://huggingface.co/api/models/<owner>/<repo> | python3 -c 'import json, sys; d = json.load(sys.stdin); print(d["sha"], "gated:", d.get("gated"), "license:", (d.get("cardData") or {}).get("license"))'
curl -s "https://huggingface.co/api/models/<owner>/<repo>/tree/<sha>?recursive=true" | python3 -c 'import json, sys; [print(f["size"], f["path"]) for f in json.load(sys.stdin) if f["type"] == "file"]'
```

When `gated` is not `False`, stop and report it.

An LLM on the `qa20` line, run by OpenVINO GenAI (the walked form; values in `<>` are yours):

```toml
# <model>: <package> on OpenVINO GenAI, judged on the qa20 line

[model]
kind = "llm"
title = "<model>"
order = <3000 or the next free number (below)>
source = "<owner>/<repo of the original model>"
source_revision = "<its 40-hex sha>"
license = "<its licence id>"
pipeline_tag = "text-generation"
inputs = "suites/llm/qa20.jsonl"

[[fetch]]
repo = "<owner>/<repo of the original model>"
revision = "<its 40-hex sha>"
include = ["*.json", "*.safetensors", "*.txt"]

[[fetch]]
repo = "<owner>/<repo of the package>"
revision = "<its 40-hex sha>"

[[form]]
id = "ov:<owner>/<repo of the package>"
short = "<the quantization in a few words, from its card>"
label = "<short>; <owner>/<repo of the package>@<its 40-hex sha>"
from = "<owner>/<repo of the package>"

[[step]]
name = "reference"
venv = "ref"
script = "pc/ref_llm_hf.py"
args = ["--model-dir", "{HUB}\\<owner>__<repo of the original model>", "--suite", "{ROOT}\\suites\\llm\\qa20.jsonl", "--out", "{ROOT}\\refs\\<model>\\ref.json"]
reference = "refs/<model>/ref.json"
ram_gb = <the fp32 weights in GB + 1>
timeout_min = 30

[[line]]
id = "qa20"
kind = "task"
rule = "<the rule of the qa20 line of models/Qwen3-0.6B/recipe.toml, as it is>"
written = "<the output of date -Iseconds when you write this line>"
plan = "PLAN.md section 1b"

[[cell]]
id = "task-openvino-genai_cpu_cpu"
runtime = "openvino-genai"
device = "cpu"
line = "qa20"
venv = "ovg"
script = "pc/run_llm_ovg.py"
form = "ov:<owner>/<repo of the package>"
timeout_min = 30
args = ["--model-dir", "{HUB}\\<owner>__<repo of the package>", "--device", "CPU", "--ref", "{ROOT}\\refs\\<model>\\ref.json"]
ram_need_gb = 5
```

- **The order** of a new model: 3000, or the next number after the highest one from 3000 on
  (`grep -h '^order' models/*/recipe.toml | sort -n -k3 | tail -n 1`).
- **The line's time.** `written` is when you wrote the line, `date -Iseconds` (with its UTC offset), before any run of
  the cells it judges: `tools/lint.py` checks it against each cell's run start. After a cell has run, do not change
  `written` (section 8, "is not before the run").
- **The rule.** Print the qa20 rule to copy:
  `python3 -c 'import tomllib; r = tomllib.load(open("models/Qwen3-0.6B/recipe.toml", "rb")); print(next(l["rule"] for l in r["line"] if l["id"] == "qa20"))'`.
- **Revisions** are commit shas (40 hex), never a branch. `include` globs must each match a file of that revision
  (`plan.py manifest` stops otherwise); no `include` = every file but `README.md` and `.gitattributes`. The
  original's `include` covers its config, weights, tokenizer and chat template files in the tree you printed (add
  `"*.jinja"` when the template is a file, as in `models/gemma-4-E2B-it/recipe.toml`).
- **Memory.** `ram_gb` of the reference is an estimate of its peak, at least the original's safetensors bytes × 2 for
  bf16 or fp16 files (× 1 for fp32) plus 1 GB (the Qwen3 recipes give more: 6, 10 and 18 for 1.5, 4.1 and 8.0 GB of
  bf16 files). Over 8 GB, add `lock = true`: the step then runs under your own lock, at most 30 min. Its slot comes from
  the shared PC's order, which is not in this repo: before you launch the model's list, stop and ask your user when it
  may run.
- **Cell ids** start with `task-` (the README and `tools/qa20_key.py` count the `task-*.json` files): `task-<runtime>_<device>_<backend>`,
  as in the other models (`task-openvino-genai_npu_npu`, `task-ort-genai_igpu_openvino`, `task-llama.cpp_rtx_cuda`).
  A second package on a runtime and device the model already has takes a suffix that names it
  (`task-openvino-genai_cpu_cpu-int8-sym`).
- **Another device.** The same cell with another `id` (`task-openvino-genai_npu_npu`; the Intel GPU:
  `task-openvino-genai_igpu_gpu`), `"--device", "NPU"` (or `"GPU.0"`), `"--cache-dir",
  "{ROOT}\\ovcache\\<model>_<device>"` and `device = "npu"` (or `"igpu"`). An Intel NPU runs only some quantizations
  (this repo's OpenVINO GenAI `task-*` cells on the NPU run symmetric int4 or int8 IRs). A GPU cell's proof of its
  adapter is the GPU memory per adapter that `pc/chain.ps1` records in each run's `.meta.json`.
- **Other runtimes and kinds** (not walked here): copy a `task-*` `[[cell]]` of the same runtime and device from the
  closest model, not one with `migrated = true`, and the `[[form]]`, `[[fetch]]` and `[[step]]` it names
  (`grep -n 'runtime = "<id>"' models/*/recipe.toml`). Leave out its `z1_cell` or `z2_cell` key: it marks a cell that
  judges a copied cell again. The LLM cells of llama.cpp, LiteRT-LM and Foundry Local, and an ONNX Runtime GenAI cell
  that runs a WebGPU overlay, also need the model's `z6-setup` step and its entry in `pc/z6_spec.json` (engines,
  links, Foundry packages, overlays): see `models/Qwen3-0.6B/recipe.toml` and the header of `pc/z6_setup.py`. A vision
  or audio model needs its task's line and reference step: see `PLAN.md` and a model of the same task.

## 3. The card

`models/<model>/README.md`. Its front matter has five keys, then a paragraph, then the two marker lines
(`tools/table.py` fills the space between them):

```markdown
---
license: <the original's licence id>
base_model: <owner>/<repo of the original model>
base_model_relation: quantized
library_name: openvino
pipeline_tag: text-generation
---

# <model>

[<owner>/<repo of the original model>](https://huggingface.co/<owner>/<repo of the original model>) as <the package, linked to its Hub repo>, run on one Windows laptop on <the devices of the cells>. Each cell is judged on the qa20 line: 20 short questions answered with greedy decoding, compared with the same model in fp32 (Transformers, CPU), and one prompt that has to end within 16 tokens.

<!-- zoo:cells begin -->
<!-- zoo:cells end -->
```

`library_name` is the package's library (`openvino`, `onnxruntime-genai`, `gguf`, ...).

## 4. What the PC runs

Written from the recipes into `pc/` (not committed; `tools/pc.sh push` sends it):

```bash
python3 tools/plan.py manifest <model>
python3 tools/plan.py steps <model>
python3 tools/plan.py first --sitting <name>-1 --cells <model>/<cell>,<model>/<cell>
```

- `manifest` reads the Hub API: one line per repo (`<repo> <sha10> <GiB> GiB <n> files`) and
  `pc/hub_manifest.json: <k> sources, <x> GiB`.
- `steps` prints `pc/steps/setup.json: <n> steps`, `pc/steps/fetch.json: 2 steps` and `pc/steps/<model>.json: <n> steps`
  (and engines lists when the model has a `z6-setup` step).
- `first` prints `pc/plans/<name>-1.json: <n> units (first pass)`. A later pass takes a new sitting name
  (`<name>-2`, ...): the chain skips every run whose record is already on the PC.
- For a result of a model already here, `manifest <model>` and `steps <model>` take every `[[fetch]]` of the model
  and every venv its cells name, not only your cell's (`models/Qwen3-0.6B/recipe.toml` has 7 `[[fetch]]`).

## 5. Run it on the PC

```bash
R='<pc_root>'
tools/pc.sh push
tools/pc.sh lock note "adding <model> to the model zoo (untimed; killed and run again around the locks of other jobs) since $(date +%H:%M)"
tools/pc.sh launch 'pc\steps.ps1' '-List setup'
tools/pc.sh launch 'pc\steps.ps1' '-List fetch'
tools/pc.sh wait "$R\logs\setup.done,$R\logs\fetch.done" "$R\logs\setup.log,$R\logs\fetch.log" 3
tools/pc.sh launch 'pc\steps.ps1' '-List <model>'
tools/pc.sh wait "$R\logs\<model>.done" "$R\logs\<model>.log" 2
tools/pc.sh launch 'pc\chain.ps1' "-Plan $R\pc\plans\<name>-1.json -Mode first"
tools/pc.sh wait "$R\runs\<name>-1\chain.done" "$R\runs\<name>-1\chain.log" 2
tools/pc.sh pull runs/<name>-1 raw/<name>/runs
tools/pc.sh pull refs raw/<name>
tools/pc.sh pull hub/fetch_report.json raw/<name>
tools/pc.sh pull logs raw/<name>
tools/pc.sh lock note ""
```

- `R` is a shell variable. When each command runs in a new shell, write `R='<pc_root>';` at the start of every call
  that uses `$R`. With `$R` empty, `wait` finds no done file and prints nothing until its hours are over.
- Whenever you stop before the last line, run `tools/pc.sh lock note ""` first: a note under 2 hours old counts as a
  job that waits for the PC, and this repo's timed sittings give their lock back for it.
- `push` ends with `pushed pc/, suites/, zoo.toml -> <pc_root>: <n> .ps1 files parse on the PC with the Mac's sha;
  zlib loaded, lane <lane>, others waiting <k>`. Push again after every `tools/plan.py` call.
- Each `launch` prints `STARTED <pid> (...)`. Each `wait` prints the logs from their start (a list launched again
  appends to its log, so an earlier try's lines come first) and ends with one `DONE OK <time> (<done file>)` per list
  or chain, or `DONE FAIL <steps> <time> (<done file>)`; its last number is the most hours it waits.
- `lock note` prints `NOTE written <time>` (`lock note ""` prints `NOTE removed <time>`). No `'` in the note's text
  (it breaks the PowerShell quoting). The lists and the chain write their own notes too, in `%USERPROFILE%\pcup\active\`.
- Untimed work runs without the lock. When another job's lock appears, the step or run in flight is killed and runs
  again from its start once the lock has been free 60 s (section 8). Do not kill, suspend or start it again yourself;
  keep the `wait` running, or start the same `wait` again if its connection ended.
- In the PC's night (`[zoo] night` in `zoo.toml`) nothing starts and what runs is killed: the list ends
  `FAIL <step> (night <night>)`, the chain `FAIL the night (<night>): ...`. Launch the same list or plan after the night.
- **A result** of a model already here is judged against the model's reference record (`pulled_from` in
  `models/<model>/reference/ref.json`: `raw/<folder>/refs/<model>/ref.json`). Instead of the model's list
  (`-List <model>`, its reference step), put that record where the cell reads it. Launch `setup` and `fetch` and wait
  for them first, as above: they add only what the PC folder lacks (seconds when it has it all). Leave out
  `tools/pc.sh pull refs raw/<name>`: the record is already in `raw/<folder>/`. Put it and compare the two sha256:

  ```bash
  tools/pc.sh put raw/<folder>/refs/<model> refs
  printf '%s\n' "(Get-FileHash '$R\refs\<model>\ref.json' -Algorithm SHA256).Hash" | tools/pc.sh ps
  shasum -a 256 raw/<folder>/refs/<model>/ref.json
  ```

  The two sha256 are the same (the PC prints upper case). When your cell needs a step of the model's list other than
  the reference (the LLM engines' `z6-setup`), run the model's list before the `put`: its reference step writes
  `refs\<model>\ref.json` again, and the `put` then sets the committed record back.

## 6. Judge the cells, write the cards, lint

```bash
python3 tools/results.py --runs raw/<name>/runs --refs raw/<name>/refs --model <model>            # a new model
python3 tools/results.py --runs raw/<name>/runs --refs raw/<folder>/refs --cells <model>/<cell>   # a result of a model already here
python3 tools/table.py
python3 tools/lint.py
```

- Run one of the two `results.py` lines. Each prints `<model>/<cell>: ok, first, <verdict>` per cell and
  `results: <n> cells written`.
- `results.py` writes again every own cell of the models it is given, and of every model when it is given none. A
  cell with no record under `--runs` becomes `not_run`. So `--model` is for a new model only, and a result takes
  `--cells`. `<folder>` is the folder of section 5.
- To judge without touching `models/`: `python3 tools/results.py --runs ... --refs ... --out walk-out --cells <model>/<cell>`.
- `table.py` prints `wrote <n> of <m> generated files ...`: the card's block and `LICENSES.md`.
- `lint.py` prints `PASS <model> (task: ...)` for your model (the count of its cells' verdicts) and the last line of
  done item 1. Its `WARN ... no start time in the record` lines belong to cells copied from other zoos and were there
  before your change.

## 7. Before the commit

1. The PC's account name can be in the logs and records (paths under the user folder): replace it with `USER` and
   check, without printing it:

   ```bash
   u=$(printf '%s\n' "Split-Path -Leaf ([Environment]::GetFolderPath('UserProfile'))" | tools/pc.sh ps | tr -d '\r')
   [ -n "$u" ] && python3 tools/scrub.py --names "$u" raw/<name> models/<model> && python3 tools/scrub.py --names "$u" --check raw/<name> models/<model>; echo "exit $?"
   ```

   Run the two lines in one shell call. The first `scrub.py` prints `scrubbed <file> (<hits>)` per file and
   `<n> file(s) changed`; the check prints nothing, and the last line is `exit 0`. Never write the name in a file or a
   commit message.
2. On a shared PC the logs quote the other jobs' lock texts (`another lane holds the lock (<text>)`, `killed for
   another lane's lock (<text>)`). Cut them before the commit:

   ```bash
   python3 -c 'import pathlib, re, sys
   pat = re.compile(r"(another lane holds the lock|killed for another lane.s lock) \((?:[^()]|\([^()]*\))*\)")
   for p in pathlib.Path(sys.argv[1]).rglob("*.log"):
       t = p.read_bytes().decode("utf-8")
       u = pat.sub(r"\1", t)
       if u != t:
           p.write_bytes(u.encode("utf-8")); print("cut", p)' raw/<name>
   ```

3. What goes in: `models/<model>/` (recipe, card, `results/`, `reference/`), `LICENSES.md`, and from `raw/<name>/` the
   `runs/` and `fetch_report.json`, and for a new model its `refs/`. `raw/*/` is ignored: add `!raw/<name>/` and
   `raw/<name>/logs/` (the step logs stay on your disk) to `.gitignore`, next to the lines of the other folders, and
   add `raw/<name>/` to the list in `docs/add-a-model.md` ("Where things are").
4. `README.md` and `docs/reading-results.md` count and list the results (the `task-*` files). Change each place your
   cells change.
   - In `README.md`: the number of results in four places (`grep -n -E '[0-9]+ (test )?results' README.md`); the
     sentence that starts `The other <n> run packages`, with its count and what your cells run; the last sentence of
     the paragraph that starts `The <n> results ran on`, with the day your cells ran; and for a new model, its link in
     its job's row of "What's covered".
   - In `docs/reading-results.md`: for a new model, the counts of models and LLMs at the start of "What the results
     cover"; the paragraph on the `task-*` cells added after 2026-11-10; for an NPU cell, the count of `task-*` cells
     on the Intel NPU and a sentence with its driver and MSI Center mode (`npu` and `power.msiUserScenarioMode` in
     `raw/<name>/runs/<name>-<n>/env.*.json`: 2 is Balanced, 1 is Extreme); and under `source.by`, the day and the PC
     folder of your cells and what `raw/<name>/` holds.
   - The README's figure counts only cells that judge a copied cell again: `python3 tools/readme_figure.py` leaves
     `docs/images/` as it is.
5. `git checkout -- zoo.toml`, then `git status`: only the files of done item 3. Run section 0's tests and lint once
   more.

## 8. When something fails

| what you see | what it means | what to do |
|---|---|---|
| `REFUSED (another lane's timing lock): ...` (exit 3) | another job times on the PC; no connection is opened | wait (minutes), then the same command |
| `REFUSED (lock feed stale or missing): ...` (exit 3) | the Mac's lock feed is old or missing | a Mac without the feed: `export ZOO_FEED=none;` at the start of the call (section 0); else wait for the feed |
| `STOP: <repo>@<sha10> has no file for ['<glob>']` (plan.py manifest) | an `include` glob matches no file of that revision | fix the glob from the repo's tree (section 2) |
| `FAILED steps.ps1 ended within 20 s with exit 1 (pid <n>)` (launch) and `logs\<x>.done` = `FAIL no list file pc\steps\<x>.json` | no such list | the list names are the files in `pc/steps/` (`setup`, `fetch`, the model's folder name) |
| `DONE FAIL fetch`; in the pulled `raw/<name>/logs/fetch/fetch.txt`, a row with `"ok": false` and the last line `fetch done: <n> files, <k> not matching` | a file's bytes are not the manifest's hash after 6 tries | `python3 tools/plan.py manifest <model>`, `tools/pc.sh push`, then launch the `fetch` list and wait for it again (the files that match are only hashed); a URL `[[fetch]]`: check its `size` and `sha256` |
| `<step>: killed for another lane's lock (...) after <s> s; runs again when it is gone`, then `<step> start (try 2)` | another job took the lock | nothing: it runs again by itself |
| `<model>/<cell> run 1 stopped (another lane's lock, a slot or the night); its files removed, run again after it` | the same for a first-pass run | nothing (the cell's `run.tries` counts the starts) |
| `DONE FAIL <step>` (a list) | a step failed | its output: `tools/pc.sh pull logs raw/<name>`, then `raw/<name>/logs/<list>/<step>.txt` and `.txt.err` |
| results.py: `<model>/<cell>: error, first, ...` | the runner failed | `run.error_line` in the cell JSON and the run's `run1.stderr.txt`; fix it, then run the cell again as in the row "is not before the run" below |
| other cells' `results/*.json` in `git status` after results.py | results.py ran without `--model` / `--cells`, or with `--model` for a result of a model already here | `git checkout -- 'models/*/results/*.json' 'models/*/reference/*.json'` (with the quotes: your recipe, your card and your new files stay), then section 6 with `--model` (a new model) or `--cells` (a result) |
| lint: `card: the cells block is stale (python3 tools/table.py)` or `generated file LICENSES.md is stale` | the generated files are older than the JSONs | `python3 tools/table.py`, then lint |
| lint: `results/<cell>.json: line qa20 written <t> is not before the run (<started>)` | the line was written (or dated) after that run | leave `written` as it is. Run the cell again after the line's time: section 4's `first` with a sitting name not used yet (`--sitting <name>-<n>`), then from section 5 only `push`, the note, the chain's `launch` and `wait`, the pulls and `lock note ""`, with that name. Then section 6 (it takes the newest first pass) |
| lint: `recipe: ...` | a key is missing or names something that is not there | the reason names the key; `schema/recipe.md` |

## Not walked

Walked on the PC with this text, from a new worktree and an empty PC folder: a new LLM model with an OpenVINO GenAI
cell on the CPU, then one more result of it on the Intel NPU (in the same PC folder and `raw/` folder). These rows of
section 8 were walked: the missing lock feed, a glob with no file, a list name with no file, a fetch hash mismatch, a
step killed by a lock and run again (a stand-in lock in a lock folder of its own), results.py without `--model`, stale
generated files, and a line dated after a run. Not walked: `REFUSED (another lane's timing lock)`, a first-pass run
stopped by a lock, a step or a runner that fails in another way, the PC's night stop, a `lock = true` step, a GPU cell,
the other runtimes' cells, a vision or audio model, a result of a model that an earlier job added, and a timed sitting.
