#!/usr/bin/env python3
"""Fixture records through the repo's own pipeline (Mac, no model runs). Copies the tracked files to a scratch folder,
writes synthetic PC records there for the references and task cells of pc/z5_spec.json (in the shapes
pc/ref_vision_task.py, pc/run_vision_task.py and pc/chain.ps1 write; four cells broken on purpose: a failing output,
an error record, no outputs file, other inputs), runs tools/results.py (absolute --runs / --refs inside the copy),
tools/table.py and tools/lint.py there, and checks each cell's verdict, the compact references and the generated
table. The repo itself is not written. Exit 1 on any unexpected result.

    python3 tests/test_records.py [--keep]
"""
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import tomllib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tests")]
import synth  # noqa: E402

TASK_SCRIPT = "pc/run_vision_task.py"
BROKEN = {("dmcount", "task-ovgpu"): "fail", ("sinet", "task-ovnpu"): "error",
          ("mlsd_fp16", "task-ovnpu_pre"): "no-outputs", ("dinov2_s_fp16", "task-ovnpu_pre"): "other-inputs"}
REF_STARTED = "2026-11-10T09:00:00.0000000+09:00"   # after every line's `written`
RUN_STARTED = "2026-11-10T09:10:00.1234567+09:00"


def save(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "arrays": len(arrays)}


def rows_for(ms):
    frames = (ms.get("stream") or {}).get("frames")
    if frames:
        return [{"id": f"{i}#f{t}", "input": i, "frame": t, "wall_s": 0.004, "nonfinite": 0}
                for i in ms["ids"] for t in range(1, frames + 1)]
    return [{"id": i, "wall_s": 0.01 + 0.001 * k, "nonfinite": [0]} for k, i in enumerate(ms["ids"])]


def write_records(top, spec):
    """-> {(model, cell): how}"""
    rng = np.random.default_rng(11)
    cells = {}
    for m, ms in sorted(spec.items()):
        recipe = tomllib.loads((top / "models" / m / "recipe.toml").read_text())
        inputs_sha = hashlib.sha256(m.encode()).hexdigest()
        ref, good, bad = synth.gen(m, ms, rng)
        rd = top / "raw" / "t" / "refs" / m
        rj = {"runner": "ref_vision_task.py", "started": REF_STARTED, "ended": "2026-11-10T09:01:00.0000000+09:00",
              "status": "OK", "rows": rows_for(ms), "versions": {"onnxruntime": "1.31.1"},
              "model": {"path": ms["forms"]["fp32"]["path"], "sha256": ms["forms"]["fp32"]["sha256"]}, "files": [],
              "inputs": {"path": f"C:\\src\\zoo-template\\work\\{m}\\inputs.npz", "n": len(ms["ids"]),
                         "sha256": inputs_sha, "bytes": 1, "feeds": {}, "set": ms["set"]},
              "workload": {"inputs": len(ms["ids"]), "task": ms["task"]},
              "outputs": save(rd / "ref.outputs.npz", ref)}
        (rd / "ref.json").write_text(json.dumps(rj, indent=1))
        for c in recipe["cell"]:
            if c.get("script") != TASK_SCRIPT:
                continue
            how = BROKEN.get((m, c["id"]), "good")
            cells[(m, c["id"])] = how
            d = top / "raw" / "t" / "runs" / "t-a" / m / c["id"]
            d.mkdir(parents=True, exist_ok=True)
            run = {"runner": "run_vision_task.py", "started": RUN_STARTED, "ended": "2026-11-10T09:11:00.0000000+09:00",
                   "argv": c["args"], "status": "OK",
                   "power_throttling": {"set_ok": True, "execution_speed_throttling": "off"},
                   "rows": rows_for(ms), "versions": {"onnxruntime": "1.31.1"},
                   "inputs": {"path": "x", "sha256": "0" * 64 if how == "other-inputs" else inputs_sha, "bytes": 1,
                              "n": len(ms["ids"])},
                   "load_s": 1.234, "device": {}, "session_providers": ["X"], "workload": {"task": ms["task"]},
                   "summary": {"inputs": len(ms["ids"]), "median_ms": 11.0, "value": 11.0}}
            if how == "error":
                run.update(status="ERROR", error="Traceback (most recent call last):\nRuntimeError: synthetic failure")
            elif how != "no-outputs":
                run["outputs"] = save(d / "run1.outputs.npz", bad if how == "fail" else good)
            (d / "run1.json").write_text(json.dumps(run, indent=1))
            meta = {"mode": "first", "start": "2026-11-10T09:10:00+09:00", "tries": 1,
                    "power": {"msiUserScenarioMode": 2}, "desktopLocked": False,
                    "engine": {"peakWorkingSetBytes": 123456789}, "command": "python ..."}
            (d / "run1.meta.json").write_text(json.dumps(meta))
    return cells


def run(top, *args):
    r = subprocess.run([sys.executable, *args], cwd=top, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main():
    keep = "--keep" in sys.argv[1:]
    spec = json.loads((ROOT / "pc" / "z5_spec.json").read_text())["models"]
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="zoo-test-")).resolve()   # macOS: /var is /private/var
    top = tmp / "repo"
    files = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "-co", "--exclude-standard"], capture_output=True,
                           check=True).stdout   # the tracked files and the new ones not ignored
    for f in files.decode().split("\0"):
        if f and (ROOT / f).is_file():
            (top / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, top / f)
    cells = write_records(top, spec)
    bad = []
    rc, out = run(top, "tools/results.py", "--runs", str(top / "raw" / "t" / "runs"),
                  "--refs", str(top / "raw" / "t" / "refs"), "--model", *sorted(spec))
    if rc:
        bad.append(f"results.py exit {rc}: {out[-800:]}")
    for (m, cid), how in sorted(cells.items()):
        doc = json.loads((top / "models" / m / "results" / f"{cid}.json").read_text())
        state, got = doc["run"]["state"], doc["line"]["pass"]
        notes = " ".join(x["text"] for x in doc.get("note") or [])
        want = {"good": ("ok", True), "fail": ("ok", False), "error": ("error", None),
                "no-outputs": ("ok", None), "other-inputs": ("ok", None)}[how]
        if (state, got) != want or (how == "other-inputs" and "inputs are not the reference's" not in notes):
            bad.append(f"{m}/{cid} ({how}): state {state}, line {got}, want {want}; notes: {notes[:200]}")
        if how != "good":
            print(f"{m}/{cid} ({how}): state {state}, line {got}; {notes[:120]}")
    for m in sorted(spec):
        comp = top / "models" / m / "reference" / "ref.json"
        cj = json.loads(comp.read_text()) if comp.exists() else {}
        if cj.get("started") != REF_STARTED or not cj.get("task_values"):
            bad.append(f"{m}: the compact reference was not written from the pulled reference")
    for step in (("tools/table.py", "--local"), ("tools/lint.py",)):   # --local: RESULTS_VISION.md is read below
        rc, out = run(top, *step)
        print(f"{step[0]}: exit {rc}; {out.strip().splitlines()[-1] if out.strip() else ''}")
        if rc:
            bad.append(f"{step[0]} exit {rc}: {out[-800:]}")
    table = (top / "RESULTS_VISION.md").read_text()
    if "## Vision, the task cells" not in table:
        bad.append("RESULTS_VISION.md has no task cells table")
    print(f"cells {len(cells)} (broken on purpose {sum(h != 'good' for h in cells.values())}), "
          f"references {len(spec)}, unexpected {len(bad)}")
    for b in bad:
        print("UNEXPECTED", b)
    if keep:
        print("kept", top)
    else:
        shutil.rmtree(tmp)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
