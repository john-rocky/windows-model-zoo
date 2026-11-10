#!/usr/bin/env python3
"""The qa20 record path (Mac, no model runs): synthetic PC records for the Z6 lane's LLM task cells (the 86
task-<Z2 cell> cells of migrate/z6.py) through the repo's own pipeline. Copies the repo's files (tracked and new) to a
scratch folder, writes a reference per model (the rows pc/ref_llm_hf.py writes) and one first-pass run per cell in the
row shape of its runner (ids: pc/run_llm_ovg.py LLMPipeline, run_llm_genai.py, run_llm_llamacpp.py; token counts:
run_llm_ovg.py VLMPipeline, run_llm_litert.py, run_llm_foundry.py), five broken on purpose (17 of 20 answers, a gate
without an end token, an error record, a prompt count one over the reference's, no run at all), then runs
tools/results.py, tools/table.py and tools/lint.py there and checks each cell's verdict, the prompt instrument and that
a first pass shows no speed. The repo itself is not written. Exit 1 on any unexpected result.

    python3 tests/test_llm_records.py [--keep]
"""
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "gemma-4-E2B-it", "gemma-4-E4B-it"]
REF_STARTED = "2026-11-10T09:00:00.0000000+09:00"   # after the qa20 lines' written (07:58)
RUN_STARTED = "2026-11-10T09:30:00.1234567+09:00"
BROKEN = {("Qwen3-0.6B", "task-openvino-genai_cpu_cpu"): "fail-17",
          ("Qwen3-1.7B", "task-litert-lm_cpu_cpu"): "gate-no-eos",
          ("Qwen3-4B", "task-foundry_cpu_cpu"): "error",
          ("gemma-4-E2B-it", "task-litert-lm_npu_npu"): "count-off",
          ("gemma-4-E4B-it", "task-llama.cpp_rtx_cuda"): "not-run"}


def ref_rows(m):
    bos = m.startswith("gemma")
    rows = []
    for i in range(1, 21):
        ids = ([2] if bos else []) + list(range(100, 112 + i))
        rows.append({"id": f"q{i:02d}", "role": "question", "question": f"question {i}",
                     "prompt": ("<bos>" if bos else "") + f"<turn>user question {i}", "prompt_ids": ids,
                     "new_ids": [7, 8, 1], "new_tokens": 3, "text": f"Answer {i}.", "eos": True, "max_new": 48, "wall_s": 1.0})
    rows.append({"id": "gate", "role": "gate", "question": "Reply with only the number: 1+1=?",
                 "prompt": ("<bos>" if bos else "") + "<turn>user 1+1", "prompt_ids": ([2] if bos else []) + list(range(100, 118)),
                 "new_ids": [9, 1], "new_tokens": 2, "text": "2", "eos": True, "max_new": 16, "wall_s": 0.5})
    return rows


def run_rows(m, c, how, ref):
    """the cell's rows in its runner's shape"""
    script, args = c["script"], c.get("args") or []
    counts = (script.endswith(("run_llm_litert.py", "run_llm_foundry.py"))
              or (script.endswith("run_llm_ovg.py") and m.startswith("gemma")))
    out = []
    for k, r in enumerate(ref):
        text = r["text"]
        if how == "fail-17" and r["role"] == "question" and k < 3:
            text = "something else"
        if r["role"] == "question" and k == 5:
            text = text.lower().rstrip(".") + "\nmore"   # agrees after the normalization of the first line
        eos = not (how == "gate-no-eos" and r["role"] == "gate")
        row = {"id": r["id"], "role": r["role"], "new_tokens": r["new_tokens"] if eos else r["max_new"], "text": text,
               "eos": eos, "max_new": r["max_new"], "wall_s": 0.2, "first_token_ms": 50.0}
        if counts:
            row["prompt_tokens"] = len(r["prompt_ids"]) + (1 if how == "count-off" and k % 2 == 0 else 0)
        else:
            row["prompt_ids"] = list(r["prompt_ids"])
            row["new_ids"] = list(r["new_ids"])
        out.append(row)
    return out


def main():
    keep = "--keep" in sys.argv[1:]
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="zoo-llm-test-")).resolve()
    top = tmp / "repo"
    files = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "-co", "--exclude-standard"], capture_output=True,
                           check=True).stdout
    for f in files.decode().split("\0"):
        if f and (ROOT / f).is_file():
            (top / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, top / f)
    cells, bad = {}, []
    for m in MODELS:
        recipe = tomllib.loads((top / "models" / m / "recipe.toml").read_text())
        ref = ref_rows(m)
        rd = top / "raw" / "t" / "refs" / m
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "ref.json").write_text(json.dumps({"runner": "ref_llm_hf.py", "started": REF_STARTED,
                                                 "ended": "2026-11-10T09:05:00.0000000+09:00", "status": "OK",
                                                 "versions": {"transformers": "5.22.2"}, "workload": {"prompts": 21},
                                                 "rows": ref}))
        for c in recipe["cell"]:
            if c.get("migrated") or c.get("line") != "qa20":
                continue
            how = BROKEN.get((m, c["id"]), "good")
            cells[(m, c["id"])] = how
            if how == "not-run":
                continue
            d = top / "raw" / "t" / "runs" / "t-z6" / m / c["id"]
            d.mkdir(parents=True, exist_ok=True)
            run = {"runner": c["script"].split("/")[-1], "started": RUN_STARTED,
                   "ended": "2026-11-10T09:31:00.0000000+09:00", "argv": c["args"], "status": "OK",
                   "power_throttling": {"set_ok": True, "execution_speed_throttling": "off (High QoS)"},
                   "rows": run_rows(m, c, how, ref), "versions": {"x": "1"}, "load_s": 2.0,
                   "summary": {"prompts": 21, "new_tokens": 60, "wall_s": 4.2, "tok_per_s": 14.3, "eos": 21, "value": 14.3}}
            if how == "error":
                run.update(status="ERROR", error="Traceback (most recent call last):\nRuntimeError: synthetic failure")
            (d / "run1.json").write_text(json.dumps(run))
            (d / "run1.meta.json").write_text(json.dumps({"mode": "first", "start": "2026-11-10T09:30:00+09:00", "tries": 1,
                                                         "power": {"msiUserScenarioMode": 2}, "desktopLocked": False,
                                                         "engine": {"peakWorkingSetBytes": 123456789}, "command": "python ..."}))
    r = subprocess.run([sys.executable, "tools/results.py", "--runs", str(top / "raw/t/runs"), "--refs", str(top / "raw/t/refs"),
                        "--model", *MODELS], cwd=top, capture_output=True, text=True)
    if r.returncode:
        bad.append(f"results.py exit {r.returncode}: {(r.stdout + r.stderr)[-800:]}")
    want = {"good": ("ok", True), "fail-17": ("ok", False), "gate-no-eos": ("ok", False), "error": ("error", None),
            "count-off": ("ok", True), "not-run": ("not_run", None)}
    for (m, cid), how in sorted(cells.items()):
        doc = json.loads((top / "models" / m / "results" / f"{cid}.json").read_text())
        state, got = doc["run"]["state"], doc["line"]["pass"]
        task = (doc.get("llm") or {}).get("task") or {}
        sp = doc.get("speed") or {}
        if (state, got) != want[how]:
            bad.append(f"{m}/{cid} ({how}): state {state}, line {got}, want {want[how]}")
        if state == "ok":
            n_same = task.get("prompt_ids_same")
            if how == "count-off" and n_same != 10:
                bad.append(f"{m}/{cid}: prompt instrument {n_same} of 21, want 10 (every other row one over)")
            if how != "count-off" and n_same != 21:
                bad.append(f"{m}/{cid}: prompt instrument {n_same} of 21 ({task.get('prompt_basis')})")
            if task.get("agree") != (17 if how == "fail-17" else 20):
                bad.append(f"{m}/{cid}: agree {task.get('agree')}")
            if any(sp.get(k) is not None for k in ("value", "first_value", "ttft_ms")):
                bad.append(f"{m}/{cid}: a first pass shows a speed: {sp}")
        if how != "good":
            print(f"{m}/{cid} ({how}): state {state}, line {got}, {doc['line'].get('text')}, prompt basis "
                  f"{task.get('prompt_basis')} same {task.get('prompt_ids_same')}")
    for step in (("tools/table.py", "--local"), ("tools/lint.py", *MODELS)):   # --local: RESULTS_LLM.md is read below
        r = subprocess.run([sys.executable, *step], cwd=top, capture_output=True, text=True)
        print(f"{step[0]}: exit {r.returncode}; {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''}")
        if r.returncode:
            bad.append(f"{step[0]} exit {r.returncode}: {(r.stdout + r.stderr)[-1200:]}")
    tab = (top / "RESULTS_LLM.md").read_text()
    n_rows = sum(1 for ln in tab.splitlines() if ln.startswith("| ") and "qa20" not in ln and " | PASS | " in ln)
    print(f"cells {len(cells)} (broken on purpose {sum(h != 'good' for h in cells.values())}), RESULTS_LLM.md PASS rows "
          f"{n_rows}, unexpected {len(bad)}")
    for b in bad:
        print("UNEXPECTED", b)
    if keep:
        print("kept", top)
    else:
        shutil.rmtree(tmp)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
