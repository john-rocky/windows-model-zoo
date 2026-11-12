# Add a model

The words used here are explained in [reading-results.md](reading-results.md#words-used-in-the-files).

## Where things are

```
zoo.toml                    the PC (ssh name, folder), the lane (this zoo's name on the shared PC's lock and notes),
                            the PC's night (no untimed work in it), device and runtime names
PLAN.md                     the lines and their dated changes
models/<model>/README.md    the card: Hub front matter (license, base_model, base_model_relation, library_name,
                            pipeline_tag), text, and the generated block (its lines and where its cells are)
models/<model>/recipe.toml  how each form is made and how each cell runs; the lines, written before any run
models/<model>/results/     one JSON per cell (schema/cell.schema.json)
models/<model>/reference/   the compact record of the line's reference
raw/z5/, raw/z6/, raw/z14/, the run records and references the task cells name (`source.record`)
raw/z13/
suites/                     the input sets: vision images (pinned URLs + sha256), LLM questions and their answer key,
                            audio clip lists
schema/                     cell.schema.json, recipe.md (the recipe's keys), lines.md (the lines per kind and task)
tools/                      the Mac side: table.py, lint.py, results.py, vision_tasks.py, plan.py, pc.sh, sitting.sh,
                            walk_compare.py, task_pairs_vision.py, task_pairs_llm.py, scrub.py, qa20_key.py, readme_figure.py
pc/                         the PC side: chain.ps1, steps.ps1, lock.ps1, launch.ps1, wait_done.ps1, zlib.ps1,
                            fetch.py, engines.py, the runners and their library zoolib.py, req/ (one pinned venv each)
tests/                      synthetic outputs and records through the vision task lines, the LLM qa20 records and the
                            LLM runners with fake engines, then results → table → lint; tools/sitting.sh with a
                            missing, stale or skipped lock feed (Mac, no model runs, no PC):
                            python3 tests/test_vision_tasks.py, test_records.py, test_llm_records.py,
                            test_llm_runners.py, test_sitting_feed.py
docs/                       how to read a result, add a model and run everything again; images/ is the README's figure
LICENSES.md                 generated: each model's licence, the other repos its cells use, the input sets' terms
```

Every Mac tool runs with `python3` (3.11 or later: `tomllib`; `tools/results.py` also needs numpy, and scipy for the
vision task cells). No model runs on the Mac: references, exports and cells all run on the PC.

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
5. **Run them on the PC** ([reproduce.md](reproduce.md) has the commands): push, setup and fetch, the model's steps, the first
   pass, the pull.
6. **Results, cards, lint:**
   ```bash
   python3 tools/results.py --runs raw/<runs> --refs raw/<refs>   # one JSON per cell, the line judged
   python3 tools/table.py                                         # the cards' blocks, LICENSES.md
   python3 tools/lint.py                                          # PASS / FAIL per model folder, with the reasons
   ```

Placeholders in `args` (pc/chain.ps1, pc/steps.ps1): `{ROOT}` the zoo folder on the PC, `{HUB}` its `hub\` (Hub files
as `<owner>__<repo>\<path>`), `{FIX}` its `fixtures\`, `{HOME}` the user folder, `{RUN}` the cell's run folder.

## Add a result to a model

Add a `[[cell]]` (and its `[[form]]`, `[[fetch]]` and steps when new) to the model's recipe, then steps 4 to 6 above
with that cell alone in `--cells`. To judge the cells of one walk without touching `models/`, write them elsewhere:
`python3 tools/results.py --runs ... --refs ... --out <folder> --cells <model>/<cell>,...`.
