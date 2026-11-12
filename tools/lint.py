#!/usr/bin/env python3
"""Check every model directory: what is missing, and how its cells stand on their lines.

    python3 tools/lint.py [model ...]        # every model directory when none is named

Per directory (FAIL = something to fix in the directory, with the reason; WARN = told, does not fail):
  card      README.md with the Hub card's front matter (license, base_model, base_model_relation, library_name,
            pipeline_tag) and the generated cells block
  recipe    recipe.toml: [model] kind / source / license; every [[line]] with id, kind, rule and a written time;
            every [[cell]] with id, runtime, device, form, line, and a script in this repo (or migrated = true)
  results   every results/<cell>.json matches schema/cell.schema.json, is a cell of this model and kind, its cell is
            in the recipe, and every recipe cell has a result (a cell nobody ran has run.state = not_run and its reason)
  verdict   every cell's line.verdict is the string tools/zoo.py verdict_text makes from the rest of its JSON
  key       every LLM cell on the qa20 line that ran OK, and the model's compact reference, carry the answers right by
            the answer key that tools/qa20_key.py gives from the record (when the record is in this repo)
  line      a cell's line was written before the cell ran (line.written < run.started); a cell without a start time
            (a record collected from another lane) is a WARN
  fresh     the generated files (the card's block, LICENSES.md; RESULTS_*.md and models/index.json when they are on
            your disk: they are not committed) equal what tools/table.py makes from the JSONs now
Then one line per directory: PASS or FAIL, and the cells' verdicts per line kind (pass / fail / no verdict).
Exit 0 when every directory passes, 1 otherwise.
"""
import collections
import datetime as dt
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import qa20_key  # noqa: E402
import table  # noqa: E402
import zoo  # noqa: E402

CARD_KEYS = ("license", "base_model", "base_model_relation", "library_name", "pipeline_tag")
SCHEMA = json.loads((zoo.ROOT / "schema" / "cell.schema.json").read_text(encoding="utf-8"))
KEY = qa20_key.load_key()


def _is(v, t):
    return {"object": isinstance(v, dict), "array": isinstance(v, list), "string": isinstance(v, str),
            "number": isinstance(v, (int, float)) and not isinstance(v, bool), "integer": isinstance(v, int) and not isinstance(v, bool),
            "boolean": isinstance(v, bool), "null": v is None}[t]


def validate(v, s, path="$"):
    """The JSON Schema keywords schema/cell.schema.json uses: const, enum, type, required, properties, items."""
    errs = []
    if "const" in s and v != s["const"]:
        errs.append(f"{path} = {v!r}, not {s['const']!r}")
    if "enum" in s and v not in s["enum"]:
        errs.append(f"{path} = {v!r}, not one of {s['enum']}")
    t = s.get("type")
    if t and not any(_is(v, x) for x in (t if isinstance(t, list) else [t])):
        return errs + [f"{path} is {type(v).__name__}, not {t}"]
    if isinstance(v, dict):
        errs += [f"{path} lacks {k}" for k in s.get("required", []) if k not in v]
        for k, sub in (s.get("properties") or {}).items():
            if k in v:
                errs += validate(v[k], sub, f"{path}.{k}")
    if isinstance(v, list) and "items" in s:
        for i, x in enumerate(v):
            errs += validate(x, s["items"], f"{path}[{i}]")
    return errs


def parse_time(s):
    if not s:
        return None
    s = re.sub(r"(\.\d{6})\d+", r"\1", str(s))   # PowerShell writes 7 fraction digits
    try:
        t = dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    return t if t.tzinfo else None


def front_matter(text):
    m = re.match(r"\s*---\r?\n([\s\S]*?)\r?\n---", text)
    if not m:
        return None
    out = {}
    for line in m.group(1).splitlines():
        k = re.match(r"^([A-Za-z_]+)\s*:\s*(.*?)\s*$", line)
        if k:
            out[k.group(1)] = k.group(2)
    out["_exception"] = bool(re.search(r"#\s*base-model-exception:\s*\S", m.group(1)))
    return out


def check_dir(d, fresh_files):
    fails, warns = [], []
    name = d.name
    if not zoo.NAME.match(name):
        fails.append(f"directory name {name!r} is not [A-Za-z0-9][A-Za-z0-9._-]*")
    # card
    card = d / "README.md"
    if not card.exists():
        fails.append("no README.md (card)")
    else:
        text = card.read_text(encoding="utf-8")
        fm = front_matter(text)
        if fm is None:
            fails.append("card: no front matter (--- ... ---)")
        else:
            missing = [k for k in CARD_KEYS if not fm.get(k) and not (k.startswith("base_model") and fm["_exception"])]
            if missing:
                fails.append("card: front matter lacks " + ", ".join(missing))
        if table.BEGIN not in text or table.END not in text:
            fails.append("card: no generated cells block (<!-- zoo:cells begin --> ... end -->)")
    # recipe
    try:
        rec = zoo.recipe(d)
    except FileNotFoundError:
        return ["no recipe.toml"], warns, {}
    except Exception as e:  # noqa: BLE001
        return [f"recipe.toml does not parse: {e}"], warns, {}
    md = rec.get("model") or {}
    if md.get("kind") not in zoo.KINDS:
        fails.append(f"recipe: [model] kind {md.get('kind')!r} is not one of {', '.join(zoo.KINDS)}")
    for k in ("source", "license"):
        if not md.get(k):
            fails.append(f"recipe: [model] lacks {k}")
    lines = {}
    for l in rec.get("line") or []:
        lid = l.get("id")
        for k in ("id", "kind", "rule", "written"):
            if not l.get(k):
                fails.append(f"recipe: a [[line]] lacks {k} ({lid})")
        if l.get("kind") and l["kind"] not in zoo.LINE_KINDS:
            fails.append(f"recipe: line {lid} kind {l['kind']!r}")
        if l.get("written") and not parse_time(l["written"]):
            fails.append(f"recipe: line {lid} written {l['written']!r} is not an ISO time with a zone")
        lines[lid] = l
    rcells = {}
    for c in rec.get("cell") or []:
        cid = c.get("id")
        for k in ("id", "runtime", "device", "form"):
            if not c.get(k):
                fails.append(f"recipe: a [[cell]] lacks {k} ({cid})")
        if c.get("line") and c["line"] not in lines:
            fails.append(f"recipe: cell {cid} names line {c['line']!r} that the recipe does not have")
        if not c.get("migrated"):
            s = c.get("script")
            if not s or not (zoo.ROOT / s).exists():
                fails.append(f"recipe: cell {cid} script {s!r} is not in this repo")
            if not isinstance(c.get("args"), list):
                fails.append(f"recipe: cell {cid} has no args list")
            if not (zoo.ROOT / "pc" / "req" / f"{c.get('venv')}.txt").exists():
                fails.append(f"recipe: cell {cid} venv {c.get('venv')!r} has no pc/req/<venv>.txt")
            if c.get("form") not in {f.get("id") for f in rec.get("form") or []}:
                fails.append(f"recipe: cell {cid} form {c.get('form')!r} is not a [[form]]")
        rcells[cid] = c
    for st in rec.get("step") or []:
        for k in ("name", "venv", "script"):
            if not st.get(k):
                fails.append(f"recipe: a [[step]] lacks {k} ({st.get('name')})")
        if st.get("script") and not (zoo.ROOT / st["script"]).exists():
            fails.append(f"recipe: step {st.get('name')} script {st['script']!r} is not in this repo")
        if st.get("venv") and not (zoo.ROOT / "pc" / "req" / f"{st['venv']}.txt").exists():
            fails.append(f"recipe: step {st.get('name')} venv {st['venv']!r} has no pc/req/<venv>.txt")
    for f in rec.get("fetch") or []:
        if f.get("url"):
            if not (f.get("dest") and f.get("path") and isinstance(f.get("size"), int)
                    and re.fullmatch(r"[0-9a-f]{64}", str(f.get("sha256") or ""))):
                fails.append(f"recipe: a [[fetch]] by URL needs dest, path, size and a sha256 ({f.get('url')})")
        elif not f.get("repo") or not re.fullmatch(r"[0-9a-f]{40}", str(f.get("revision") or "")):
            fails.append(f"recipe: a [[fetch]] needs repo and a 40-hex revision ({f.get('repo')})")
    # results
    verdicts = collections.defaultdict(collections.Counter)
    seen = set()
    for c in zoo.cells(d):
        p = c["_path"]
        cid = p.stem
        seen.add(cid)
        where = f"results/{p.name}"
        bad = validate({k: v for k, v in c.items() if not k.startswith("_")}, SCHEMA)
        if bad:
            fails.append(f"{where}: schema: " + "; ".join(bad[:3]) + (f" (+{len(bad) - 3})" if len(bad) > 3 else ""))
        if c.get("model") != name or c.get("cell") != cid:
            fails.append(f"{where}: model / cell ({c.get('model')}, {c.get('cell')}) is not ({name}, {cid})")
        if c.get("kind") != md.get("kind"):
            fails.append(f"{where}: kind {c.get('kind')!r} is not the recipe's {md.get('kind')!r}")
        run = c.get("run") or {}
        ln = c.get("line")
        if ln is None:
            if c.get("role") != "other-device":
                fails.append(f"{where}: no line (only another device's record may have none)")
        else:
            if ln.get("verdict") != zoo.verdict_text(c):
                fails.append(f"{where}: line.verdict is not what the JSON gives (python3 tools/results.py --verdicts)")
            task = (c.get("llm") or {}).get("task")
            if ln.get("id") == "qa20" and task and run.get("state") == "ok":
                rec_path = zoo.ROOT / ((c.get("source") or {}).get("record") or "")
                try:
                    if "key" not in task:
                        fails.append(f"{where}: no llm.task.key (python3 tools/results.py --keys)")
                    elif rec_path.is_file() and task["key"] != qa20_key.score_rows(
                            qa20_key.read_rows(rec_path), KEY, c["source"]["record"]):
                        fails.append(f"{where}: llm.task.key is not what the answer key gives "
                                     "(python3 tools/results.py --keys)")
                except qa20_key.Stop as e:
                    fails.append(f"{where}: {e}")
            if ln.get("id") not in lines:
                fails.append(f"{where}: line {ln.get('id')!r} is not in the recipe")
            elif ln.get("kind") not in (lines[ln["id"]].get("kind"), "reference"):   # the line's reference cell: "reference"
                fails.append(f"{where}: line kind {ln.get('kind')!r} is not the recipe line's {lines[ln['id']].get('kind')!r}")
            lw = parse_time((lines.get(ln.get("id")) or {}).get("written"))
            st = parse_time(run.get("started"))
            if run.get("state") != "not_run":
                if st is None:
                    warns.append(f"{where}: no start time in the record (line-before-result not checked)")
                elif lw and lw >= st:
                    fails.append(f"{where}: line {ln['id']} written {lines[ln['id']]['written']} is not before the run "
                                 f"({run['started']})")
            if c.get("role") is None:
                verdicts[ln.get("kind")][{True: "pass", False: "fail"}.get(ln.get("pass"), "no verdict")] += 1
        if c.get("speed") and not c.get("regime"):
            fails.append(f"{where}: speed without a regime (QoS, power mode)")
        if cid not in rcells:
            fails.append(f"{where}: cell is not in recipe.toml")
    ref_path = d / "reference" / "ref.json"
    if md.get("kind") == "llm" and "qa20" in lines and ref_path.exists():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        try:
            if ref.get("key") != qa20_key.score_rows(ref.get("rows") or [], KEY, f"models/{name}/reference/ref.json"):
                fails.append("reference/ref.json: key is not what the answer key gives (python3 tools/results.py --keys)")
        except qa20_key.Stop as e:
            fails.append(f"reference/ref.json: {e}")
    for cid in rcells:
        if cid not in seen:
            fails.append(f"recipe cell {cid}: no results/{cid}.json (a cell nobody ran: run.state not_run with its reason)")
    # fresh
    rel = f"models/{name}/README.md"
    if rel in fresh_files:
        fails.append("card: the cells block is stale (python3 tools/table.py)")
    return fails, warns, verdicts


def main():
    names = sys.argv[1:]
    dirs = [d for d in zoo.model_dirs() if not names or d.name in names]
    stale = table.stale_files(table.render(table.load_all(), local=True))
    n_fail = 0
    total = collections.defaultdict(collections.Counter)
    for d in dirs:
        fails, warns, verdicts = check_dir(d, set(stale))
        v = "; ".join(f"{k}: " + ", ".join(f"{n} {w}" for w, n in sorted(c.items())) for k, c in sorted(verdicts.items()))
        for k, c in verdicts.items():
            total[k].update(c)
        if fails:
            n_fail += 1
            print(f"FAIL {d.name}: " + " | ".join(fails))
        else:
            print(f"PASS {d.name}" + (f" ({v})" if v else "") + (f" [{len(warns)} WARN]" if warns else ""))
        for w in warns[:3]:
            print(f"  WARN {w}")
        if len(warns) > 3:
            print(f"  WARN ... {len(warns) - 3} more")
    top = [s for s in stale if not s.startswith("models/") or s == "models/index.json"]
    for s in top:
        # a local file (not committed) is written only by --local; deleting it is the other way out
        fix = "python3 tools/table.py --local, or delete it" if s in table.LOCAL else "python3 tools/table.py"
        print(f"FAIL generated file {s} is stale ({fix})")
    print(f"lint: {len(dirs)} model directories, {len(dirs) - n_fail} PASS, {n_fail} FAIL; generated files "
          f"{'fresh' if not top else 'STALE'}; lines: "
          + "; ".join(f"{k} " + ", ".join(f"{n} {w}" for w, n in sorted(c.items())) for k, c in sorted(total.items())))
    sys.exit(1 if (n_fail or top) else 0)


if __name__ == "__main__":
    main()
