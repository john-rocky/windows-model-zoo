# recipe.toml

One per model folder (`models/<model>/recipe.toml`). Read by `tools/zoo.py` (`tomllib`), checked by `tools/lint.py`,
turned into the PC's step lists and plans by `tools/plan.py`. Required keys are marked **R**.

## `[model]`

| key | | |
|---|---|---|
| `kind` | **R** | `vision`, `llm`, `asr`, `tts` (the table the cells go to) |
| `source` | **R** | the Hub id of the original model |
| `license` | **R** | the model's licence (the card's front matter says the same) |
| `title` | | the name in the tables (default: the folder's name) |
| `source_revision` | | the revision read when the model was added |
| `pipeline_tag` | | the Hub task |
| `order` | | the model's place in its table (copied zoos: their order; new models from 3000) |
| `group` | | vision: `main` or `fixed` (the vision zoo's second table of fixed conversions) |
| `task` | | vision: the task type of the line (`classification`, ...; PLAN.md section 1a) |
| `inputs` | | the input set in `suites/` (`suites/vision/images.json`, `suites/llm/qa20.jsonl`, `suites/audio/S150.json`) |
| `first_args` | | the arguments a first pass adds to every cell (a cell's own `first_args` wins) |
| `migrated_from` | | a copied model: the zoo's repo and commit |
| `fixed_of`, `notes`, `source_file` | | the vision zoo's own fields, kept by its copy |

## `[[fetch]]` — Hub files the PC downloads

| key | | |
|---|---|---|
| `repo` | **R** | the Hub repo |
| `revision` | **R** | a commit sha (pinned: `tools/plan.py manifest` writes each file's size and hash) |
| `include` | | globs; empty = every file except README.md and .gitattributes |
| `exclude` | | globs left out |

## `[[form]]` — what a cell runs

| key | | |
|---|---|---|
| `id` | **R** | the name cells use in `form` |
| `short` | | the label in the tables (`fp32`, `int8`, `q8`) |
| `label` | | the longer description (the cell JSON's `form.detail`) |
| `from` | | a published repo, used as it is |
| `made_by` | | the `[[step]]` that writes it, and `path` = where (with the placeholders) |

## `[[step]]` — untimed steps on the PC

| key | | |
|---|---|---|
| `name` | **R** | the step's name in `logs\<list>\` |
| `venv` | **R** | a `pc/req/<venv>.txt` |
| `script` | **R** | a script in `pc/` (`pc/<name>.py`) |
| `args` | | its arguments (placeholders `{ROOT}` `{HUB}` `{FIX}` `{HOME}`; an after-step also `{RUN}`) |
| `phase` | | `before` (default: the model's step list, before its cells) or `after` (after the cell named by `cell`) |
| `cell` | | an after-step's cell |
| `reference` | | the record this step writes, relative to the PC folder (`refs/<model>/ref.json`): the line's reference |
| `timeout_min`, `ram_gb` | | the step's limit and memory; `lock = true` for a step over 8 GB (it runs under the lane's lock) |

## `[[line]]` — what the cells are judged on

| key | | |
|---|---|---|
| `id` | **R** | the name cells use in `line` |
| `kind` | **R** | `task`, `internal`, `text-integrity` (`reference` is a cell's role, not a line's kind) |
| `rule` | **R** | the line in words: what is compared, against what, the bar |
| `written` | **R** | ISO time with a zone, before the first run of any cell it judges (`tools/lint.py` checks) |
| `plan`, `written_note` | | where it was written and how its time is known |

## `[[cell]]` — one model × runtime × device × form

| key | | |
|---|---|---|
| `id` | **R** | the result's file name (`results/<id>.json`) |
| `runtime`, `device`, `form` | **R** | ids: `zoo.toml` `[runtimes]` / `[devices]` give the labels; `form` = a `[[form]]` |
| `line` | | a `[[line]]` id |
| `venv`, `script`, `args` | **R** for a new cell | how it runs (`script` in `pc/`, a list `args`); the chain adds `--out <run>.json` |
| `first_args` | | what a first pass adds (`["--first", "10"]`: the first 10 clips per language) |
| `reference` | | the line's reference as a file of this repo (instead of a step's record) |
| `timeout_min`, `ram_need_gb`, `est_sec`, `engines_gate` | | the chain's limits and its per-run gate |
| `migrated` | | `true` = a copied cell: its runner is in its own zoo (`script` names it) |
