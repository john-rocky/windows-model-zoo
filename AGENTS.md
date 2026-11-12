# AGENTS.md — for coding agents in windows-model-zoo

Read this first if someone asked you to add a converted model to this repo, to run one of its models on one more
runtime, device or package, or to say what a result here means. This file routes; the skill and the docs have the
commands.

**What this is.** Test results showing whether converted LLM, vision and audio models still do their job on one
Windows laptop (Intel NPU, Intel GPU, NVIDIA RTX GPU, CPU). One folder per model in `models/`: the card
(`README.md`), the recipe (`recipe.toml`), `results/` and `reference/`. A result (a cell) is one `task-*.json` in
`results/`: one model on one runtime, device and package. A line is the pass / fail rule in the recipe that a cell is
judged on. The other JSONs in `results/` are records copied from the earlier repos. The tools run on a Mac and drive
the Windows PC over SSH; no model runs on the Mac.

## Route

| Task | Do | Where |
|---|---|---|
| Add a model, or one more result (a cell) of a model | follow the skill from its section 0 | [skills/add-to-windows-model-zoo/SKILL.md](skills/add-to-windows-model-zoo/SKILL.md) |
| Say what a result means | read its `line.verdict` and `run.state` | [docs/reading-results.md](docs/reading-results.md) |
| See every result in one table | `python3 tools/table.py --local`. Run it again after a change under `models/`, or delete the three files: `tools/lint.py` checks them while they are on your disk. | `RESULTS_LLM.md`, `RESULTS_VISION.md` and `models/index.json` on your disk |
| Check the checkout before and after a change | the test loop and `python3 tools/lint.py` (below) | stdout |
| The recipe's keys, the lines per kind and task | read | [schema/recipe.md](schema/recipe.md), [schema/lines.md](schema/lines.md), [PLAN.md](PLAN.md) |
| Rebuild results from an empty PC folder, or run a timed sitting | read | [docs/reproduce.md](docs/reproduce.md) |
| Run on a PC that other jobs share | read | [docs/shared-pc.md](docs/shared-pc.md) |

The check, on the Mac at the repo root (Python 3.11 or later with numpy and scipy):

```bash
for t in tests/test_*.py; do python3 "$t" >/dev/null 2>&1 && echo "ok   $t" || echo "FAIL $t"; done
python3 tools/lint.py | tail -n 1
```

Every test prints `ok`, and lint's last line starts `lint: <n> model directories, <n> PASS, 0 FAIL; generated files
fresh`.

## Rules

1. **A line is written before its cells run.** Each `[[line]]` of a recipe has a `written` time, and
   `tools/lint.py` checks it against the start of every cell the line judges. After a cell has run, do not change
   `written` (the skill's section 8 says what to do instead).
2. **A FAIL is a result.** It is committed with its numbers, like a PASS.
3. **A result speaks for one package on one runtime and device.** Do not turn it into a rating of a model or a
   runtime. Its speed values come from one untimed run (`speed.timed` is false): do not give them as timings. A FAIL
   does not say what caused it.
4. **Tables that put runtimes side by side stay on your disk.** `tools/table.py --local`,
   `tools/task_pairs_vision.py` and `tools/task_pairs_llm.py` write them, and `.gitignore` keeps them out. Do not
   paste them into the README, a card, a commit, a pull request or an issue.
5. **The PC is shared.** Before each connection except `wait`'s, `tools/pc.sh` reads a lock feed on the Mac with the
   command in `PC_FEED_CMD` (its output is described in `tools/pc.sh`'s header; this repo has no such command). It
   refuses the call (exit 3) while another job times on the PC: wait, then run the same command. It also refuses
   (exit 3) when `PC_FEED_CMD` is not set (`REFUSED (PC_FEED_CMD is not set ...`) or the feed is stale or missing
   (`REFUSED (lock feed stale or missing)`). Put `export PC_FEED_CMD='<that command>';` at the start of each shell
   call that runs `tools/pc.sh` or `tools/sitting.sh`; only on a Mac that has no lock feed, put
   `export ZOO_FEED=none;` there instead (the skill's section 0). The PC kills an untimed step when another job takes the
   lock and runs it again once the lock has been free 60 s: do not kill, suspend or start it again yourself. In the
   PC's night (`[zoo] night` in `zoo.toml`) nothing untimed starts and what runs is killed: launch the same list or
   plan after the night.
6. **No PC account name and no other job's lock text in a commit.** Before the commit, run `tools/scrub.py` on your
   records and your model's folder, and cut the lock texts from the logs (the skill's section 7).
7. **Set `pc_root` and `lane` in `zoo.toml` for your job, and never commit them.** `pc_root` is a folder on the PC
   that does not exist yet. `lane` is your job's name on the shared PC (the skill's section 0). Run
   `git checkout -- zoo.toml` before the commit.
8. **Ask before you push.** Commit on your branch (the skill's section 0 makes one). Push it or open a pull request
   only when the person who gave you the task says so.

Maintainer: john-rocky (GitHub). Issues: https://github.com/john-rocky/windows-model-zoo/issues
