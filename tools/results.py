#!/usr/bin/env python3
"""The PC's run records -> one JSON per cell (models/<model>/results/<cell>.json), the cell's line judged, for the
cells this zoo runs itself (a recipe cell without migrated = true). Then python3 tools/table.py.

    python3 tools/results.py --runs raw/walk/runs --refs raw/walk/refs [--model <model> ...]

Records: <runs>/<sitting>/<model>/<cell>/run<k>.json (the runner's) and run<k>.meta.json (pc/chain.ps1's view: mode,
start, power mode, peak memory). A cell takes its latest timed sitting whose runs ended OK, else its latest first pass.
The line's reference: <refs>/<model>/ref.json (a [[step]] with `reference`, pulled from the PC) or a file of this repo
(a cell's `reference`). A pulled reference is also kept in compact form in models/<model>/reference/ref.json, so the
verdict reads again without raw/.

Lines by kind (the line's rule is the recipe's; these are its computations):
  vision classification  top-1 vs the reference's, inputs whose reference top-1 / top-2 probabilities differ by < 0.01
                         not counted; instruments: cos and max |d| of the logits, top-5 overlap
  llm qa20               the normalized first line of each answer vs the reference's, >= 18 / 20, and the gate prompt
                         ended on an end token within its 16 new tokens
  asr S150               per language: summed errors <= the reference's + 0.5 pt of the same clips, 0 empty answers
                         (tools/fleurs_score.py)
  tts T50                the write-back WER (pc/writeback.py's texts) <= the reference's write-back WER on the same
                         sentences + 1.0 pt, 0 silent waves
  vision task cells      (a cell whose script is pc/run_vision_task.py: the Z5 lane's) the model's own line, by
                         tools/vision_tasks.py on the runner's and the reference's outputs (<stem>.outputs.npz); the
                         cell's inputs must be the reference's (the same inputs.npz sha256)
"""
import argparse
import json
import pathlib
import re
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fleurs_score as FS  # noqa: E402
import zoo  # noqa: E402

LANGS = ["en_us", "cmn_hans_cn", "ja_jp"]
MODEL_LABEL_SET = {"S150": "S150", "T50": "T50"}


def jload(p):
    return json.loads(pathlib.Path(p).read_text(encoding="utf-8-sig"))


# ---- the scorers -------------------------------------------------------------------------------------------------

def score_vision(run_json, run, ref_json, ref):
    import numpy as np
    rrows = {r["id"]: r for r in ref["rows"]}
    counted = agree = not_counted = 0
    overlap = []
    for r in run["rows"]:
        b = rrows.get(r["id"])
        if b is None:
            continue
        overlap.append(len(set(r["top5"]) & set(b["top5"])))
        if b["gap12"] < 0.01:
            not_counted += 1
            continue
        counted += 1
        agree += int(r["top5"][0] == b["top5"][0])
    inst = {"top5_overlap_mean": round(sum(overlap) / len(overlap), 3) if overlap else None,
            "not_counted": not_counted, "nonfinite_inputs": sum(1 for r in run["rows"] if r.get("finite") is False)}
    a, b = pathlib.Path(run_json), pathlib.Path(ref_json)
    la, lb = a.with_name(a.stem + ".logits.npy"), b.with_name(b.stem + ".logits.npy")
    if la.exists() and lb.exists():
        x, y = np.load(la).astype(np.float64), np.load(lb).astype(np.float64)
        ids_run = [r["id"] for r in run["rows"]]
        ids_ref = [r["id"] for r in ref["rows"]]
        if ids_run == ids_ref and x.shape == y.shape:
            cos = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
            inst.update({"cos_min": round(float(cos.min()), 6), "cos_mean": round(float(cos.mean()), 6),
                         "max_abs_diff": round(float(np.abs(x - y).max()), 6)})
    ok = counted > 0 and agree * 100 >= 95 * counted
    detail = {"top1": {"agree": agree, "counted": counted, "bar": "95 %", "pass": bool(ok)}, "pass": bool(ok)}
    return {"top1_agree": agree, "counted": counted, **inst}, detail, \
        f"top-1 {agree}/{counted}" + (f" ({not_counted} not counted)" if not_counted else "")


def norm_answer(t):
    """The first line of an answer, normalized (PLAN section 1b): trimmed, lower case, quotes / backticks / asterisks
    around it and a final . ! ? dropped (repeated until nothing changes), spaces collapsed."""
    lines = (t or "").strip().splitlines()
    s = lines[0] if lines else ""
    prev = None
    while s != prev:
        prev = s
        s = s.strip().lower().strip("\"'`*").rstrip(".!?")
    return " ".join(s.split())


def score_llm(run_json, run, ref_json, ref):
    rrows = {r["id"]: r for r in ref["rows"]}
    agree = full = eos = toks = same_ids = n = 0
    basis = {"ids": 0, "token count": 0}
    rows = []
    gate = None
    for r in run["rows"]:
        b = rrows.get(r["id"])
        if b is None:
            continue
        # the prompt as the runtime saw it: its ids when the runner has them, else its token count (LiteRT-LM, Foundry
        # Local, OpenVINO GenAI's VLMPipeline) against the number of the reference's ids
        if r.get("prompt_ids") is not None:
            same_ids += int(r["prompt_ids"] == b.get("prompt_ids"))
            basis["ids"] += 1
        elif r.get("prompt_tokens") is not None:
            same_ids += int(r["prompt_tokens"] == len(b.get("prompt_ids") or []))
            basis["token count"] += 1
        if r.get("role") == "gate":
            gate = {"eos": bool(r.get("eos")), "new_tokens": r.get("new_tokens"), "max_new": r.get("max_new"),
                    "text": r.get("text"), "ref_eos": bool(b.get("eos")), "pass": bool(r.get("eos"))}
            continue
        n += 1
        a1, b1 = norm_answer(r.get("text")), norm_answer(b.get("text"))
        agree += int(a1 == b1)
        full += int((r.get("text") or "") == (b.get("text") or ""))
        eos += int(bool(r.get("eos")))
        toks += r.get("new_tokens") or 0
        rows.append({"id": r["id"], "cell": a1, "ref": b1, "same": a1 == b1})
    ok = n == 20 and agree >= 18 and bool(gate and gate["pass"])
    detail = {"answers": {"agree": agree, "n": n, "bar": 18, "pass": n == 20 and agree >= 18},
              "gate": gate or {"pass": False, "why": "no gate row"}, "pass": bool(ok)}
    inst = {"agree": agree, "n": n, "full_text_same": full, "eos_within_max": eos, "new_tokens": toks,
            "prompt_ids_same": same_ids, "prompt_basis": {k: v for k, v in basis.items() if v}, "rows": rows,
            "gate_text": ("EOS in " + str(gate["new_tokens"]) if gate and gate["pass"] else
                          f"no EOS in {gate['new_tokens']}" if gate else "no gate row")}
    return inst, detail, f"answers {agree}/{n}, gate {'PASS' if gate and gate['pass'] else 'FAIL'}"


def s150_refs():
    return {c["id"]: c for c in jload(zoo.ROOT / "suites/audio/S150.json")["clips"]}


def score_s150(rows, meta):
    agg = {}
    for r in rows:
        c = meta.get(r["id"])
        if not c:
            continue
        cfg = c["lang"]
        e, n = FS.score(cfg, c["ref"], r.get("text") or "")
        a = agg.setdefault(cfg, {"metric": "WER" if cfg == "en_us" else "CER", "err": 0, "n": 0, "clips": 0, "empty": 0})
        a["err"] += e
        a["n"] += n
        a["clips"] += 1
        a["empty"] += int(not (r.get("text") or "").strip())
    for a in agg.values():
        a["rate_pct"] = round(100.0 * a["err"] / a["n"], 3) if a["n"] else None
    return agg


def line_s150(run_agg, ref_agg, margin=0.5):
    """The audio zoo's line_s150 (windows-audio-zoo tools/results.py): summed errors, no rounded rates compared."""
    out, ok = {}, True
    for cfg, a in run_agg.items():
        b = ref_agg.get(cfg)
        if not b or not b["n"]:
            continue
        p = a["err"] * b["n"] * 100 <= (b["err"] * 100 + margin * b["n"]) * a["n"]
        out[cfg] = {"run": a["rate_pct"], "ref": b["rate_pct"], "bar": round(100.0 * b["err"] / b["n"] + margin, 3), "pass": bool(p)}
        ok &= p
    empty = sum(a["empty"] for a in run_agg.values())
    out["empty"] = {"n": empty, "pass": empty == 0}
    out["pass"] = bool(ok and empty == 0)
    return out


def score_asr(run_json, run, ref_json, ref):
    meta = s150_refs()
    rows = run["rows"]
    ids = {r["id"] for r in rows}
    agg = score_s150(rows, meta)
    ref_agg = score_s150([r for r in ref["rows"] if r["id"] in ids], meta)
    line = line_s150(agg, ref_agg)
    same = sum(1 for r in rows if any(r.get("text") == b.get("text") for b in ref["rows"] if b["id"] == r["id"]))
    words = " / ".join(f"{'en WER' if k == 'en_us' else ('zh CER' if k == 'cmn_hans_cn' else 'ja CER')} {v['rate_pct']:.2f}"
                       for k, v in sorted(agg.items(), key=lambda kv: LANGS.index(kv[0])))
    return {"scores": agg, "reference_scores": ref_agg, "same_text_as_reference": same}, line, words


def score_tts(run_json, run, ref_path, ref):
    meta = s150_refs()
    wb_path = pathlib.Path(run_json).with_name(pathlib.Path(run_json).stem + ".writeback.json")
    if not wb_path.exists():
        return None, None, f"no write-back ({zoo.rel(wb_path)})"
    wb = jload(wb_path)["texts"]
    refwb = ref["writeback"]["texts"]
    e = n = re_ = rn = 0
    for cid, text in wb.items():
        a, k = FS.score("en_us", meta[cid]["ref"], text or "")
        e, n = e + a, n + k
        a2, k2 = FS.score("en_us", meta[cid]["ref"], refwb.get(cid) or "")
        re_, rn = re_ + a2, rn + k2
    silent = (run.get("summary") or {}).get("silent", 0)
    p = e * rn * 100 <= (re_ * 100 + 1.0 * rn) * n
    acc = {"metric": "write-back WER", "err": e, "n": n, "rate_pct": round(100.0 * e / n, 3) if n else None,
           "texts": len(wb), "writeback": zoo.rel(wb_path)}
    line = {"writeback": {"run": acc["rate_pct"], "ref": round(100.0 * re_ / rn, 3) if rn else None,
                          "bar": round(100.0 * re_ / rn + 1.0, 3) if rn else None, "pass": bool(p)},
            "silent": {"n": silent, "pass": not silent}, "pass": bool(p and not silent)}
    return {"rate_pct": acc["rate_pct"], **acc}, line, f"write-back WER {acc['rate_pct']:.2f}"


TASK_SCRIPT = "pc/run_vision_task.py"


def outputs_npz(record_path):
    p = pathlib.Path(record_path)
    return p.with_name(p.stem + ".outputs.npz")


def score_vision_task(model, task, run_json, run, ref_json, ref, line=None):
    """A vision task cell (the Z5 lane): tools/vision_tasks.py on the two outputs files."""
    import numpy as np
    import vision_tasks as VT
    oa, ob = outputs_npz(run_json), outputs_npz(ref_json)
    missing = [zoo.rel(p) for p in (oa, ob) if not p.exists()]
    if missing:
        return None, None, "no outputs file: " + ", ".join(missing)
    ids, ids_ref = VT.ids_of(run), VT.ids_of(ref)
    sa, sb = (run.get("inputs") or {}).get("sha256"), (ref.get("inputs") or {}).get("sha256")
    if ids != ids_ref or not sa or sa != sb:
        return None, None, "the cell's inputs are not the reference's (ids or inputs.npz sha256)"
    with np.load(oa) as zc, np.load(ob) as zr:
        inst, detail, words, vals = VT.score(model, task, zc, zr, ids, line)
        inst = dict(inst, outputs=VT.instruments(zc, zr), per_input=vals)
    return inst, detail, words


def reference_task_values(model, task, ref_path, ref, line=None):
    import numpy as np
    import vision_tasks as VT
    ob = outputs_npz(ref_path)
    if not ob.exists():
        return None
    with np.load(ob) as zr:
        return VT.reference_values(model, task, zr, VT.ids_of(ref), line)


SCORERS = {"vision": score_vision, "llm": score_llm, "asr": score_asr, "tts": score_tts}
METRIC = {"vision": "ms (median of the inputs, one run each)", "llm": "tok/s (new tokens / generate wall, all prompts)",
          "asr": "RTF", "tts": "RTF"}


# ---- one cell ----------------------------------------------------------------------------------------------------

def first_from_command(cmd):
    """The subset of a first pass when the runner's workload does not name it (pc/run_tts_ort.py): the chain's
    command line in run<k>.meta.json, `--first N`."""
    m = re.search(r"(?:^|\s)--first\s+(\d+)", cmd or "")
    return int(m.group(1)) if m else 0


def runs_of(runs_root, model, cell):
    """[(sitting, mode, [(run json, meta)])], newest first (by the meta's start)."""
    out = []
    for sd in sorted(pathlib.Path(runs_root).iterdir()) if pathlib.Path(runs_root).is_dir() else []:
        d = sd / model / cell
        metas = sorted(d.glob("run*.meta.json"))
        if not metas:
            continue
        rs = []
        for m in metas:
            k = m.name[:-len(".meta.json")]
            rs.append((d / f"{k}.json", jload(m)))
        out.append((sd.name, rs[0][1].get("mode"), rs))
    out.sort(key=lambda s: s[2][0][1].get("start") or "", reverse=True)
    return out


def pick(sittings):
    for want in ("timed", "first"):
        for s in sittings:
            if s[1] == want and all(rj.exists() and jload(rj).get("status") == "OK" for rj, _ in s[2]):
                return s
    return sittings[0] if sittings else None


def reference_of(recipe, cell, refs):
    if cell.get("reference"):
        p = zoo.ROOT / cell["reference"]
        return p, jload(p)
    for s in recipe.get("step") or []:
        if s.get("reference"):
            p = pathlib.Path(refs) / pathlib.Path(s["reference"]).relative_to("refs")
            return (p, jload(p)) if p.exists() else (p, None)
    return None, None


def compact_reference(kind, ref):
    keep = {"vision": ("id", "top5", "top5_p", "gap12"), "llm": ("id", "role", "question", "prompt", "prompt_ids",
                                                                  "new_ids", "text", "eos", "new_tokens", "max_new"),
            "asr": ("id", "lang", "seconds", "text")}.get(kind)
    if not keep:
        return None
    return {"runner": ref.get("runner"), "started": ref.get("started"), "ended": ref.get("ended"),
            "status": ref.get("status"), "versions": ref.get("versions"), "workload": ref.get("workload"),
            "form": ref.get("form"), "inputs": ref.get("inputs"), "data_config": ref.get("data_config"),
            "rows": [{k: r.get(k) for k in keep if k in r} for r in ref.get("rows") or []]}


def power_of(meta):
    p = (meta or {}).get("power") or {}
    return {1: "extreme", 2: "balanced"}.get(p.get("msiUserScenarioMode"), "unrecorded"), p


def build(model, recipe, cell, args, cfg):
    kind = recipe["model"]["kind"]
    sittings = runs_of(args.runs, model, cell["id"])
    s = pick(sittings)
    form = next((f for f in recipe.get("form") or [] if f["id"] == cell["form"]), {})
    line_def = next((l for l in recipe.get("line") or [] if l["id"] == cell["line"]), {})
    doc = {"schema": "zoo-cell/1", "model": model, "cell": cell["id"], "kind": kind,
           "runtime": {"id": cell["runtime"], "label": (cfg.get("runtimes") or {}).get(cell["runtime"], cell["runtime"])},
           "device": {"id": cell["device"], "label": (cfg.get("devices") or {}).get(cell["device"], cell["device"])},
           "form": {"id": cell["form"], "label": form.get("short") or form.get("label", cell["form"]),
                    "detail": form.get("label"), "path": form.get("path"), "from": form.get("from")},
           "line": {"id": cell["line"], "kind": line_def.get("kind"), "pass": None},
           "source": {"by": cfg["zoo"]["name"], "record": None}, "note": []}
    if s is None:
        doc["run"] = {"state": "not_run", "mode": None, "started": None, "why": "no run record yet"}
        doc["speed"] = None
        doc["regime"] = None
        doc["note"].append({"rank": 1, "text": "not run yet"})
        if kind in ("asr", "tts"):
            st = "T50" if kind == "tts" else recipe["model"].get("inputs", "").split("/")[-1].split(".")[0]
            doc["audio"] = {"model_label": recipe["model"].get("title"), "set": st, "set_shown": st, "clipset": st}
        else:
            doc[kind] = {}
        return doc, (None, None)
    sitting, mode, rs = s
    rj, meta = rs[0]
    run = jload(rj) if rj.exists() else {}
    err = (run.get("error") or "").strip().splitlines()
    ok = run.get("status") == "OK"
    doc["run"] = {"state": "ok" if ok else "error", "mode": mode, "sitting": sitting, "started": run.get("started"),
                  "ended": run.get("ended"), "runs": len(rs), "tries": meta.get("tries"),
                  "error": run.get("error") if not ok else None,
                  "error_line": (err[-1][:200] if err else ("no runner JSON" if not run else None)) if not ok else None}
    doc["source"].update({"record": zoo.rel(rj), "runner": run.get("runner"), "argv": run.get("argv")})
    vals = [jload(r).get("summary", {}).get("value") for r, _ in rs if r.exists() and jload(r).get("status") == "OK"]
    vals = [v for v in vals if v is not None]
    timed = mode == "timed" and ok and len(vals) == len(rs)
    sm = run.get("summary") or {}
    doc["speed"] = {"metric": METRIC[kind], "value": statistics.median(vals) if timed else None,
                    "first_value": sm.get("value") if not timed else None, "timed": timed, "n": len(vals),
                    "first_token_ms": sm.get("first_token_ms_mean"), "load_s": run.get("load_s"),
                    "warmup_s": run.get("warmup_s")}
    e = meta.get("engine") or {}
    doc["memory"] = {"peak_ws_bytes": e.get("peakWorkingSetBytes") or run.get("peak_working_set_bytes"),
                     "gpu_mib": e.get("gpuMiBPeak")}
    if kind == "llm":
        doc["speed"]["peak_working_set_mb"] = round(doc["memory"]["peak_ws_bytes"] / 2**20) if doc["memory"]["peak_ws_bytes"] else None
        doc["speed"]["ttft_ms"] = (statistics.mean(r["first_token_ms"] for r in run.get("rows") or [] if r.get("first_token_ms"))
                                   if any(r.get("first_token_ms") for r in run.get("rows") or []) else None)
        if not timed:   # an untimed first pass on the shared PC: no speed in the tables (a speed table across
            # runtimes stays on your own disk; the run JSON keeps what the runner measured)
            doc["speed"].update(first_value=None, first_token_ms=None, ttft_ms=None)
    power, pdetail = power_of(meta)
    pt = run.get("power_throttling") or {}
    doc["regime"] = {"qos": "high" if pt.get("set_ok") else "not-raised", "power": power, "timer": "runner",
                     "qos_basis": f"pc/zoolib.py high_qos() first in the runner (run JSON: {pt.get('execution_speed_throttling')}, set_ok {pt.get('set_ok')})",
                     "power_detail": pdetail, "desktop_locked": meta.get("desktopLocked")}
    if mode == "first":
        doc["note"].append({"rank": 1, "text": "first pass, untimed"})
    if not ok:
        doc["note"].append({"rank": 3, "text": "ERROR: " + (doc["run"]["error_line"] or "")[:120]})
    ref_path, ref = reference_of(recipe, cell, args.refs)
    doc["line"]["reference"] = zoo.rel(ref_path) if ref_path and str(ref_path).startswith(str(zoo.ROOT)) else (str(ref_path) if ref_path else None)
    extra = {}
    task_cell = cell.get("script") == TASK_SCRIPT
    if ok and ref is not None and task_cell and ref.get("status") != "OK":
        doc["note"].append({"rank": 2, "text": "the reference step did not end OK: the line is not judged"})
    elif ok and ref is not None:
        if task_cell:
            inst, detail, words = score_vision_task(model, recipe["model"]["task"], rj, run, ref_path, ref, cell.get("line"))
        else:
            inst, detail, words = SCORERS[kind](rj, run, ref_path, ref)
        if detail is None and words:
            doc["note"].append({"rank": 2, "text": words + ": the line is not judged"})
        if detail is not None:
            doc["line"]["pass"] = detail["pass"]
            doc["line"]["detail"] = detail
            doc["line"]["text"] = words
            doc["note"].append({"rank": 0, "text": words})
        extra = inst or {}
    elif ok:
        doc["note"].append({"rank": 2, "text": "no reference record: the line is not judged"})
    versions = run.get("versions")
    if kind == "vision":
        per_input = extra.pop("per_input", None) if task_cell else None
        doc["vision"] = {"task": recipe["model"].get("task"), "scores": extra, "versions": versions,
                         "device_props": run.get("device"), "providers": run.get("session_providers"),
                         "inputs": run.get("inputs"), "model_file": run.get("model")}
        if task_cell:
            doc["vision"].update({"per_input": per_input, "z1_cell": cell.get("z1_cell"),
                                  "workload": run.get("workload"), "outputs_file": run.get("outputs")})
        doc["speed"]["load_ms"] = round(run["load_s"] * 1000, 1) if run.get("load_s") is not None else None
    elif kind == "llm":
        doc["llm"] = {"task": extra, "versions": versions, "workload": run.get("workload")}
    else:
        a = {"model_label": recipe["model"].get("title"), "set": recipe["model"].get("inputs", "").split("/")[-1].split(".")[0],
             "versions": versions, "summary": sm, "workload": run.get("workload")}
        if kind == "asr":
            a["scores"] = extra.get("scores")
            a["reference_scores"] = extra.get("reference_scores")
            a["same_text_as_reference"] = extra.get("same_text_as_reference")
        else:
            a["set"] = "T50"
            a["scores"] = {"rate_pct": extra.get("rate_pct")} if extra else None
            a["writeback"] = extra
        n_first = (run.get("workload") or {}).get("first") or first_from_command(meta.get("command"))
        a["set_shown"] = a["set"]
        a["clipset"] = a["set"] if mode != "first" or not n_first else {
            "S150": f"S150 first {n_first} per language", "T50": f"T50 first {n_first} sentences"}.get(a["set"], a["set"])
        doc["audio"] = a
    return doc, (ref_path, ref)


def set_verdict(doc):
    """line.verdict (tools/zoo.py verdict_text) placed after line.pass"""
    ln = doc.get("line")
    if ln is None:
        return doc
    v = zoo.verdict_text(doc)
    new = {}
    for k, x in ln.items():
        if k != "verdict":
            new[k] = x
        if k == "pass":
            new["verdict"] = v
    new.setdefault("verdict", v)
    doc["line"] = new
    return doc


def verdicts():
    """--verdicts: line.verdict written again in every cell JSON from the rest of the JSON; nothing else changes"""
    n = changed = 0
    for d, c in zoo.all_cells():
        old = (c.get("line") or {}).get("verdict")
        set_verdict(c)
        n += 1
        if (c.get("line") or {}).get("verdict") != old:
            changed += 1
            zoo.write_json(c["_path"], c)
    print(f"verdicts: {n} cells, {changed} written again")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", help="the pulled run folders (tools/pc.sh pull)")
    ap.add_argument("--refs", default="")
    ap.add_argument("--model", nargs="*", default=[])
    ap.add_argument("--cells", default="", help="<model>/<cell>,... : only these cells (default: every own cell)")
    ap.add_argument("--out", default="", help="write the cells (and the compact references) under this folder as "
                    "<out>/<model>/<cell>.json instead of models/ (a walk compared with the committed cells)")
    ap.add_argument("--dry", action="store_true", help="print what would be written, write nothing")
    ap.add_argument("--verdicts", action="store_true", help="only write line.verdict again in every cell JSON")
    args = ap.parse_args()
    if args.verdicts:
        return verdicts()
    if not args.runs:
        ap.error("--runs is required")
    only = {tuple(x.split("/", 1)) for x in args.cells.split(",") if x}
    out = pathlib.Path(args.out).resolve() if args.out else None
    cfg = zoo.config()
    n = 0
    for d in zoo.model_dirs():
        if args.model and d.name not in args.model:
            continue
        r = zoo.recipe(d)
        own = [c for c in r.get("cell") or [] if not c.get("migrated") and (not only or (d.name, c["id"]) in only)]
        for c in own:
            doc, (ref_path, ref) = build(d.name, r, c, args, cfg)
            set_verdict(doc)
            if args.dry:
                print(json.dumps({k: doc.get(k) for k in ("run", "line", "speed", "regime", "note")}, ensure_ascii=False)[:900])
            elif out:
                zoo.write_json(out / d.name / f"{c['id']}.json", doc)
            else:
                zoo.write_json(d / "results" / f"{c['id']}.json", doc)
            n += 1
            print(f"{d.name}/{c['id']}: {doc['run']['state']}"
                  + (f", {doc['run'].get('mode')}" if doc['run'].get('mode') else "")
                  + f", {doc['line'].get('verdict')}")
            # a pulled reference (a [[step]]'s, under --refs), not a repo file (a cell's `reference`): the path's prefix
            # cannot tell them apart when --refs is an absolute path inside the repo
            if not args.dry and ref is not None and ref_path and not c.get("reference"):
                comp = compact_reference(r["model"]["kind"], ref)
                if comp and c.get("script") == TASK_SCRIPT:
                    comp["rows"] = [{k: x.get(k) for k in ("id", "input", "frame", "nonfinite", "head_nonfinite") if k in x}
                                    for x in ref.get("rows") or []]
                    comp.update({"model": ref.get("model"), "files": ref.get("files"), "upstream": ref.get("upstream"),
                                 "outputs": ref.get("outputs"), "task": r["model"].get("task"),
                                 "task_line": c.get("line"),
                                 "task_values": (reference_task_values(d.name, r["model"]["task"], ref_path, ref, c.get("line"))
                                                 if ref.get("status") == "OK" else None)})
                if comp:
                    comp["pulled_from"] = zoo.rel(ref_path)
                    zoo.write_json((out / d.name / "ref.json") if out else (d / "reference" / "ref.json"), comp)
    print(f"results: {n} cells" + (" (dry: nothing written)" if args.dry else " written"))


if __name__ == "__main__":
    main()
