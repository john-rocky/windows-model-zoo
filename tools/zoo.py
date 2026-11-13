"""Shared helpers of the zoo's tools: the settings (zoo.toml), the model directories, their recipes and cell JSONs.

A model directory is models/<model>/ with README.md (the card), recipe.toml (how its forms are made and its cells
run, and the lines they are judged on) and results/<cell>.json (one JSON per cell, schema/cell.schema.json).
"""
import json
import pathlib
import re
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
KINDS = ("vision", "llm", "asr", "tts")
LINE_KINDS = ("task", "internal", "text-integrity", "reference")


def config():
    return tomllib.loads((ROOT / "zoo.toml").read_text(encoding="utf-8"))


def model_dirs():
    return sorted((p for p in MODELS.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))),
                  key=lambda p: p.name.lower()) if MODELS.is_dir() else []


def recipe(model_dir):
    p = pathlib.Path(model_dir) / "recipe.toml"
    return tomllib.loads(p.read_text(encoding="utf-8"))


def cells(model_dir):
    out = []
    for p in sorted((pathlib.Path(model_dir) / "results").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        d["_path"] = p
        out.append(d)
    return out


def all_cells():
    for d in model_dirs():
        for c in cells(d):
            yield d, c


def write_json(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, dict):   # a cell read by cells() carries _path: not written back
        obj = {k: v for k, v in obj.items() if not k.startswith("_")}
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def write_text(path, text):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def regime_key(cell):
    r = cell.get("regime") or {}
    return (r.get("qos") or "unknown", r.get("power") or "unrecorded")


def regime_words(key, cfg=None):
    cfg = cfg or config()
    q, p = key
    return f"{cfg['regime']['qos'].get(q, q)} · {cfg['regime']['power'].get(p, p)}"


def _mark(d, words=""):
    bar = d.get("bar")
    return f" (pass mark {words}{bar})" if bar is not None else ""


def _numbers(ln, run):
    """the line's numbers of a judged cell, from line.detail (a task line) or the copied cell's own words"""
    d = ln.get("detail") or {}
    lid = ln.get("id") or ""
    if "answers" in d:   # qa20
        a, g = d["answers"], d.get("gate") or {}
        out = f"{a['agree']} of {a['n']} answers' first lines equal the reference's (pass mark {a['bar']})"
        return out + (f"; gate {'PASS' if g.get('pass') else 'FAIL'}" if g else "")
    if "ref_boxes" in d:
        return (f"{d['pairs']} boxes paired, of the reference's {d['ref_boxes']} and the cell's {d['cell_boxes']}"
                + _mark(d))
    if "ref_segments" in d:
        return (f"{d['pairs']} segments paired, of the reference's {d['ref_segments']} and the cell's "
                f"{d['cell_segments']}" + _mark(d))
    if "shown_either" in d:
        return f"{d['agree']} of {d['shown_either']} points shown by either side agree" + _mark(d)
    if "within_1px" in d:
        return f"{d['within_1px']} of {d['pixels']} output pixels sample within 1 px of the reference" + _mark(d)
    if "inputs" in d and "agree" in d:
        return f"{d['agree']} of {d['inputs']} counts agree" + _mark(d)
    if "counted" in d and "agree" in d:
        what = "nearest neighbours" if "retrieval" in lid else "top-1 classes"
        nc = d.get("not_counted")
        return (f"{d['agree']} of {d['counted']} {what} equal the reference's"
                + (f" ({nc} not counted; pass mark {d['bar']})" if nc else _mark(d)))
    if "miou" in d:
        return f"mIoU {d['miou']}" + _mark(d, "mIoU at least ")
    if "absrel" in d:
        return f"AbsRel {d['absrel']}" + _mark(d, "AbsRel at most ")
    if "si_sdr_cell_mean" in d:
        return (f"mean SI-SDR {d['si_sdr_cell_mean']} dB against the reference's {d['si_sdr_ref_mean']} dB"
                f" (pass mark {d.get('bar')}; {d.get('bad_outputs', 0)} bad outputs)")
    if ln.get("fails"):
        return "; ".join(ln["fails"])
    if ln.get("text"):
        return ln["text"]
    if ln.get("pass") is False and run.get("why"):
        return run["why"]
    return ""


def verdict_text(cell):
    """line.verdict of a cell JSON: the verdict, the numbers and the line in one string, e.g. `FAIL on qa20: 9 of 20
    answers' first lines equal the reference's (pass mark 18); gate PASS; outside the line: 12 of 20 right by the answer
    key` (the last part is llm.task.key, tools/qa20_key.py). None for a cell with no line (another device's
    record). tools/results.py writes it; tools/lint.py checks every cell against it."""
    ln = cell.get("line")
    if not ln:
        return None
    lid = ln.get("id")
    run = cell.get("run") or {}
    if ln.get("kind") == "reference":
        return f"reference of {lid}: the other cells of this line are compared with this cell's outputs"
    p = ln.get("pass")
    if p is None:
        st = run.get("state") or "no run"
        why = " ".join(str(run.get("why") or run.get("error_line") or "").split())
        if st == "ok" and not why:
            why = next((n["text"] for n in cell.get("note") or [] if n.get("rank") == 2), "")
        return f"no verdict on {lid}: {st}" + (f" ({why[:160]})" if why and why != st else "")
    nums = _numbers(ln, run)
    k = ((cell.get("llm") or {}).get("task") or {}).get("key")
    if k and "answers" in (ln.get("detail") or {}):
        nums += f"; outside the line: {k['right']} of {k['n']} right by the answer key"
    return f"{'PASS' if p else 'FAIL'} on {lid}" + (f": {nums}" if nums else "")


def rel(path):
    """A path as the records name it: relative to the repo, else from the home folder (~/...)."""
    p = pathlib.Path(path).resolve()
    for base, pre in ((ROOT, ""), (pathlib.Path.home(), "~/")):
        try:
            return pre + str(p.relative_to(base))
        except ValueError:
            pass
    return str(p)


# ---- the prompt check of an LLM cell: by the qa20 line's rule, the runtime gets the string the reference step
# rendered, as it is (no template of the runtime, no special token added); tools/results.py gives no verdict when the
# run record does not show that, and tools/lint.py fails a verdict whose record does not show it ----

# runners whose token count covers only the first prefill turn, with a log per prompt that counts the turns:
# pc/run_llm_litert.py writes each prompt's stderr next to its record (<record stem>.prompts/<nn>.stderr.txt)
PREFILL_LOG_RUNNERS = ("run_llm_litert.py",)
PREFILL_TURNS = re.compile(r"Prefill Turns \(Total (\d+) turns\)")


def prefill_turns(record_path, n):
    """{N: how many logs}, N from a prompt log's "Prefill Turns (Total N turns)" line (the largest N when a log has
    several); "no count": the logs without that line; "no log": n minus the number of logs, when there are fewer"""
    logs = sorted(pathlib.Path(record_path).with_suffix(".prompts").glob("*.stderr.txt"))
    out = {}
    for f in logs:
        ks = PREFILL_TURNS.findall(f.read_text(encoding="utf-8", errors="replace"))
        k = str(max(int(x) for x in ks)) if ks else "no count"
        out[k] = out.get(k, 0) + 1
    if len(logs) < n:
        out["no log"] = n - len(logs)
    return dict(sorted(out.items()))


def prompt_numbers(record_path, run, ref):
    """The numbers of the prompt check, from a run record (its path and JSON) and the reference's JSON:
    prompt_ids_same, the number of rows whose prompt token ids equal the reference's (for a runner without ids,
    LiteRT-LM, Foundry Local and OpenVINO GenAI's VLMPipeline: whose prompt token count equals the number of the
    reference's ids); prompt_basis, how many rows had ids and how many a token count; and for a runner in
    PREFILL_LOG_RUNNERS, prefill_turns"""
    rrows = {r["id"]: r for r in ref.get("rows") or []}
    same = 0
    basis = {"ids": 0, "token count": 0}
    for r in run.get("rows") or []:
        b = rrows.get(r["id"])
        if b is None:
            continue
        if r.get("prompt_ids") is not None:
            same += int(r["prompt_ids"] == b.get("prompt_ids"))
            basis["ids"] += 1
        elif r.get("prompt_tokens") is not None:
            same += int(r["prompt_tokens"] == len(b.get("prompt_ids") or []))
            basis["token count"] += 1
    out = {"prompt_ids_same": same, "prompt_basis": {k: v for k, v in basis.items() if v}}
    if run.get("runner") in PREFILL_LOG_RUNNERS:
        out["prefill_turns"] = prefill_turns(record_path, len(rrows))
    return out


def prompt_why(nums, ref):
    """None when the numbers of prompt_numbers show that the runtime got every prompt with the reference's token ids
    (or token count) and, where the logs count the turns, in one prefill turn; else the reasons, in words"""
    n = len({r["id"] for r in ref.get("rows") or []})
    basis = nums.get("prompt_basis") or {}
    out = []
    if not basis:
        out.append("the record gives neither the prompts' token ids nor their token count")
    elif nums.get("prompt_ids_same") != n:
        what = " or ".join({"ids": "token ids", "token count": "token count"}[k] for k in basis)
        out.append(f"the runtime got the reference's {what} for {nums.get('prompt_ids_same')} of {n} prompts")
    t = nums.get("prefill_turns")
    if t is not None and t != {"1": n}:
        more = sum(v for k, v in t.items() if k.isdigit() and int(k) > 1)
        logs = sum(v for k, v in t.items() if k != "no log")
        if more:
            out.append(f"the runtime's logs show a second prefill turn for {more} of {n} prompts")
        if t.get("0"):
            out.append(f"the runtime's logs show no prefill turn for {t['0']} of {n} prompts")
        if t.get("no log"):
            out.append(f"the runtime's logs are missing for {t['no log']} of {n} prompts")
        if t.get("no count"):
            out.append(f"the runtime's logs give no prefill turn count for {t['no count']} of {n} prompts")
        if logs > n:
            out.append(f"{logs} runtime logs for {n} prompts")
    return "; ".join(out) or None


# ---- a small TOML writer (the recipes hold strings, numbers, booleans, lists of those, tables and arrays of tables) ----

def _toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        if any(isinstance(x, dict) for x in v):
            raise TypeError("an inline list of tables")
        inner = ", ".join(_toml_value(x) for x in v)
        return f"[{inner}]" if len(inner) < 100 else "[\n" + "".join(f"  {_toml_value(x)},\n" for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{_toml_key(k)} = {_toml_value(x)}" for k, x in v.items()) + " }"
    raise TypeError(type(v))


def _toml_key(k):
    return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else json.dumps(k)


def toml_dumps(doc, header=""):
    """doc = {table name: dict | list of dicts}; the order of the keys is kept."""
    out = [header.rstrip() + "\n"] if header else []
    for name, body in doc.items():
        blocks = body if isinstance(body, list) else [body]
        for b in blocks:
            out.append(f"\n[[{name}]]\n" if isinstance(body, list) else f"\n[{name}]\n")
            for k, v in b.items():
                if v is None:
                    continue
                out.append(f"{_toml_key(k)} = {_toml_value(v)}\n")
    text = "".join(out).lstrip("\n")
    tomllib.loads(text)  # what is written must read back
    return text

TABLE_ORDER = ("model", "fetch", "form", "step", "line", "cell")


def merge_recipe(model_dir, generated):
    """A migration's recipe merged with what a person added to the same file: the generated [model] keys win, other
    [model] keys stay; [[line]]s the generation does not make stay; [[cell]]s without migrated = true stay; every other
    table ([[fetch]], [[form]], [[step]]) stays. The order of the tables is TABLE_ORDER."""
    path = pathlib.Path(model_dir) / "recipe.toml"
    old = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    out = dict(old)
    model = dict(old.get("model") or {})
    model.update(generated.get("model") or {})
    out["model"] = model
    gl = list(generated.get("line") or [])
    ids = {l["id"] for l in gl}
    out["line"] = gl + [l for l in old.get("line") or [] if l.get("id") not in ids]
    gc = list(generated.get("cell") or [])
    cids = {c["id"] for c in gc}
    out["cell"] = gc + [c for c in old.get("cell") or [] if not c.get("migrated") and c.get("id") not in cids]
    for k, v in generated.items():
        if k not in ("model", "line", "cell"):
            out[k] = v
    return {k: out[k] for k in sorted(out, key=lambda k: (TABLE_ORDER.index(k) if k in TABLE_ORDER else 99, k))
            if out[k] not in (None, [])}

