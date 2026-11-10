#!/usr/bin/env python3
"""What the PC runs, written from the recipes (models/<model>/recipe.toml) into pc/ (tools/pc.sh push sends it).

    python3 tools/plan.py manifest [model ...]           # pc/hub_manifest.json: every [[fetch]] file pinned (Hub API)
    python3 tools/plan.py steps [model ...]              # pc/steps/setup.json, fetch.json, <model>.json (and for
                                                         # the LLM task cells engines.json, engines-litert.json)
    python3 tools/plan.py first --sitting <name> --cells <model>/<cell>,...      # pc/plans/<name>.json (one run each)
    python3 tools/plan.py timed --sitting <name> --cells ... --control <model>/<cell> [--runs 3] [--known <value>]
                                [--end-by <ISO time>] [--no-windows] [--slots <json>]   # pc/plans/<name>.json (control rows)
    python3 tools/plan.py after --sitting <name>        # pc/steps/after-<name>.json: the cells' "after" steps

Without model names, every model directory that has non-migrated cells. A unit, a step and the placeholders they use
are pc/chain.ps1's and pc/steps.ps1's: {ROOT} the zoo folder on the PC, {HUB} its hub\\ (pc/fetch.py: <owner>__<repo>),
{FIX} its fixtures\\, {HOME} the user folder, {RUN} the unit's run folder (in an "after" step: that cell's run folder
in the named sitting).
"""
import argparse
import fnmatch
import json
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import zoo  # noqa: E402

PC = zoo.ROOT / "pc"


def own_models(names):
    out = []
    for d in zoo.model_dirs():
        if names and d.name not in names:
            continue
        r = zoo.recipe(d)
        if names or any(not c.get("migrated") for c in r.get("cell") or []):
            out.append((d.name, r))
    missing = set(names) - {m for m, _ in out}
    if missing:
        sys.exit(f"no model directory: {', '.join(sorted(missing))}")
    return out


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "windows-model-zoo plan.py"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def manifest(models):
    out, total = [], 0
    for m, r in models:
        for f in r.get("fetch") or []:
            if f.get("url"):   # one plain file by URL, pinned by size and sha256 (pc/fetch.py: <root>\<dest>\<path>)
                if not any(x.get("url") == f["url"] and x.get("dest") == f["dest"] for x in out):
                    out.append({"url": f["url"], "dest": f["dest"], "for": m, "bytes": f["size"],
                                "files": [{"path": f["path"], "size": f["size"], "sha256": f["sha256"]}]})
                    total += f["size"]
                    print(f"{f['url'][:60]:60s} {f['size'] / 2**20:8.2f} MiB (by URL)")
                continue
            repo, sha = f["repo"], f["revision"]
            inc, exc = f.get("include") or [], f.get("exclude") or []
            tree = get(f"https://huggingface.co/api/models/{repo}/tree/{sha}?recursive=true")
            files = []
            for t in tree:
                p = t["path"]
                if t["type"] != "file" or p == ".gitattributes" or p.endswith("README.md"):
                    continue
                if inc and not any(fnmatch.fnmatch(p, g) for g in inc):
                    continue
                if any(fnmatch.fnmatch(p, g) for g in exc):
                    continue
                lfs = t.get("lfs") or {}
                files.append({"path": p, "size": t["size"], "sha256": lfs.get("oid"), "blob": t.get("oid")})
            lost = [g for g in inc if not any(fnmatch.fnmatch(x["path"], g) for x in files)]
            if lost:
                sys.exit(f"STOP: {repo}@{sha[:10]} has no file for {lost}")
            prev = next((x for x in out if x.get("repo") == repo and x.get("revision") == sha), None)
            if prev is not None:   # two models of one repo at one revision: one source with the files of both
                have = {x["path"] for x in prev["files"]}
                more = [x for x in files if x["path"] not in have]
                prev["files"] += more
                prev["bytes"] += sum(x["size"] for x in more)
                prev["for"] += f",{m}"
                total += sum(x["size"] for x in more)
                print(f"{repo:48s} {sha[:10]} +{len(more)} files for {m}")
                continue
            size = sum(x["size"] for x in files)
            total += size
            out.append({"repo": repo, "folder": "", "revision": sha, "for": m, "bytes": size, "files": files})
            print(f"{repo:48s} {sha[:10]} {size / 2**30:6.2f} GiB {len(files):3d} files")
        inp = (r.get("model") or {}).get("inputs")
        if inp and inp.endswith(".json"):
            s = json.loads((zoo.ROOT / inp).read_text(encoding="utf-8"))
            if "images" in s and not any(x.get("url") == s["url"] for x in out):
                files = [{"path": i["path"], "size": i["size"], "sha256": i["sha256"]} for i in s["images"]]
                out.append({"url": s["url"], "dest": "fixtures/vision", "for": m, "bytes": sum(x["size"] for x in files),
                            "files": files})
                total += sum(x["size"] for x in files)
                print(f"{s['url'][:48]:48s} {len(files):3d} files (the vision input set)")
            # an input set taken out of a pinned archive (suites/audio/kws120.json: Speech Commands v0.01's test set)
            if "clips" in s and "url" in s.get("source", {}) and not any(x.get("archive") == s["source"]["url"] for x in out):
                files = [{"path": c["path"], "size": c["bytes"], "sha256": c["sha256"]} for c in s["clips"]]
                out.append({"archive": s["source"]["url"], "archive_bytes": s["source"]["bytes"],
                            "archive_sha256": s["source"]["sha256"], "dest": f"fixtures/{pathlib.Path(inp).stem}",
                            "for": m, "bytes": s["source"]["bytes"], "files": files})
                total += s["source"]["bytes"]
                print(f"{s['source']['url'][:48]:48s} {len(files):3d} files (out of a {s['source']['bytes'] / 2**20:.1f} MiB archive)")
    zoo.write_json(PC / "hub_manifest.json", out)
    print(f"pc/hub_manifest.json: {len(out)} sources, {total / 2**30:.2f} GiB")


def step(name, exe, args, venv=None, timeout=60, ram=None, lock=False, stop=False):
    s = {"name": name, "exe": exe, "args": args, "timeout_min": timeout}
    if venv:
        s["venv"] = venv
    if ram:
        s["ram_gb"] = ram
    if lock:
        s["lock"] = True
    if stop:
        s["stop_on_fail"] = True
    return s


def script_arg(path):
    """recipe path pc/x.py -> {ROOT}\\pc\\x.py"""
    return "{ROOT}\\" + path.replace("/", "\\")


def steps(models):
    venvs, audio = [], False
    for m, r in models:
        for x in (r.get("step") or []) + [c for c in r.get("cell") or [] if not c.get("migrated")]:
            if x.get("venv") and x["venv"] not in venvs:
                venvs.append(x["venv"])
        audio |= (r.get("model") or {}).get("kind") in ("asr", "tts")
    s = [step("python", "uv", ["python", "install", "3.12"], timeout=15, stop=True)]
    for v in venvs:
        if not (PC / "req" / f"{v}.txt").exists():
            sys.exit(f"no pc/req/{v}.txt for the venv {v}")
        s.append(step(f"venv-{v}", "uv", ["venv", f"{{ROOT}}\\.venv-{v}", "--python", "3.12", "--python-preference",
                                          "only-managed", "--allow-existing"], timeout=10))
        s.append(step(f"pip-{v}", "uv", ["pip", "install", "--python", f"{{ROOT}}\\.venv-{v}\\Scripts\\python.exe", "-r",
                                         f"{{ROOT}}\\pc\\req\\{v}.txt"], timeout=45))
        s.append(step(f"versions-{v}", "py", ["{ROOT}\\pc\\venv_versions.py", "--out", f"{{ROOT}}\\logs\\versions-{v}.txt"],
                      venv=v, timeout=5))
    if audio:
        s.append(step("fixtures", "py", ["{ROOT}\\pc\\check_fixtures.py", "--fixtures", "{FIX}", "--sets", "S150"],
                      venv=venvs[0], timeout=5))
    lists = {"setup": s,
             "fetch": [step("python", "uv", ["python", "install", "3.12"], timeout=15, stop=True),
                       step("fetch", "uv", ["run", "--no-project", "--python", "3.12", "--python-preference", "only-managed",
                                            "python", "{ROOT}\\pc\\fetch.py", "--manifest", "{ROOT}\\pc\\hub_manifest.json",
                                            "--dest", "{HUB}", "--root", "{ROOT}"], timeout=240)]}
    # the LLM task cells' engines (pc/engines.py, from the public sources pc/z6_spec.json names): "engines" without the
    # lock, "engines-litert" = the LiteRT-LM build, a step over 8 GB that runs under the lane's own lock (its slot
    # comes from the shared PC's order: launch it then), and the OpenVINO runtime that build fetched
    llm = [m for m, r in models if any(x.get("script") == "pc/z6_setup.py" for x in r.get("step") or [])]
    if llm:
        spec = json.loads((PC / "z6_spec.json").read_text(encoding="utf-8"))
        need = {d for m in llm for d in spec["models"][m].get("engines") or []}

        def eng(what, **k):
            return step(f"engines-{what}", "py", ["{ROOT}\\pc\\engines.py", "--spec", "{ROOT}\\pc\\z6_spec.json", "--root",
                                                  "{ROOT}", "--make", what], venv="genai", **k)
        lst = []
        if any(spec["models"][m].get("foundry_cells") for m in llm):
            lst.append(eng("cuda12", timeout=45))
        if need & {"llama-cpu", "llama-gpu", "llama-cuda"}:
            lst.append(eng("llama", timeout=60))
        if need & {"litert", "openvino"}:
            lst.append(eng("litert-src", timeout=60))
            lists["engines-litert"] = [eng("litert", timeout=30, ram=16, lock=True, stop=True)]
            if "openvino" in need:
                lists["engines-litert"].append(eng("openvino", timeout=10))
        lists["engines"] = lst
    for m, r in models:
        ms = [x for x in r.get("step") or [] if x.get("phase", "before") == "before"]
        if ms:
            lists[m] = [step(x["name"], "py", [script_arg(x["script"])] + list(x.get("args") or []), venv=x["venv"],
                             timeout=x.get("timeout_min", 60), ram=x.get("ram_gb"), lock=bool(x.get("lock")),
                             stop=bool(x.get("stop_on_fail", True))) for x in ms]
    for name, lst in lists.items():
        zoo.write_json(PC / "steps" / f"{name}.json", lst)
        print(f"pc/steps/{name}.json: {len(lst)} steps")


def find_cell(spec):
    m, c = spec.split("/", 1)
    r = zoo.recipe(zoo.MODELS / m)
    for x in r.get("cell") or []:
        if x["id"] == c:
            if x.get("migrated"):
                sys.exit(f"{spec}: a migrated cell (its runner is in its own zoo)")
            return m, r, x
    sys.exit(f"{spec}: no such cell in models/{m}/recipe.toml")


def unit(m, r, c, first):
    args = list(c.get("args") or [])
    if first:   # a first pass runs the cell's subset (e.g. the first 10 clips per language) when it names one
        args += list(c.get("first_args") or (r.get("model") or {}).get("first_args") or [])
    u = {"cell": f"{m}/{c['id']}", "device": c["device"], "venv": c["venv"],
         "script": c["script"].split("/", 1)[1] if c["script"].startswith("pc/") else c["script"],
         "args": args, "timeout_min": c.get("timeout_min", 20), "ram_need_gb": c.get("ram_need_gb", 4),
         "engines_gate": bool(c.get("engines_gate", c["device"] in ("igpu", "rtx")))}
    if c.get("est_sec"):
        u["est_sec"] = c["est_sec"]
    for k in ("cleanup", "foundry"):   # pc/chain.ps1: the runner's --cleanup after a killed run; the Foundry gate
        if c.get(k):
            u[k] = c[k]
    return u


def first(a):
    units = [unit(*find_cell(s), True) for s in a.cells.split(",")]
    zoo.write_json(PC / "plans" / f"{a.sitting}.json", {"sitting": a.sitting, "mode": "first", "units": units})
    print(f"pc/plans/{a.sitting}.json: {len(units)} units (first pass)")


def timed(a):
    cm, cr, cc = find_cell(a.control)
    ctl = unit(cm, cr, cc, False)
    units = [dict(ctl, role="control-warm"), dict(ctl, role="control-head", runs=a.runs)]
    units += [dict(unit(*find_cell(s), False), runs=a.runs) for s in a.cells.split(",")]
    units.append(dict(ctl, role="control-tail"))
    plan = {"sitting": a.sitting, "mode": "timed", "units": units, "windows": not a.no_windows}
    if a.known:
        plan["control_known_value"] = a.known
    if a.end_by:
        plan["end_by"] = a.end_by
    if a.slots:   # fixed slots for other lanes' timings: [{who, release_at, give_up_at}] (pc/chain.ps1)
        plan["slots"] = json.loads(pathlib.Path(a.slots).read_text(encoding="utf-8"))
    zoo.write_json(PC / "plans" / f"{a.sitting}.json", plan)
    print(f"pc/plans/{a.sitting}.json: {len(units)} units (timed, control {a.control})")


def after(a):
    plan = json.loads((PC / "plans" / f"{a.sitting}.json").read_text(encoding="utf-8"))
    out = []
    for u in plan["units"]:
        if u.get("role"):
            continue
        m, cid = u["cell"].split("/", 1)
        for x in zoo.recipe(zoo.MODELS / m).get("step") or []:
            if x.get("phase") == "after" and x.get("cell") == cid:
                run = f"{{ROOT}}\\runs\\{a.sitting}\\{m}\\{cid}"
                out.append(step(f"{x['name']}-{m}-{cid}", "py",
                                [script_arg(x["script"])] + [s.replace("{RUN}", run) for s in x.get("args") or []],
                                venv=x["venv"], timeout=x.get("timeout_min", 30), ram=x.get("ram_gb")))
    zoo.write_json(PC / "steps" / f"after-{a.sitting}.json", out)
    print(f"pc/steps/after-{a.sitting}.json: {len(out)} steps")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="what", required=True)
    for w in ("manifest", "steps"):
        p = sub.add_parser(w)
        p.add_argument("models", nargs="*")
    p = sub.add_parser("first")
    p.add_argument("--sitting", required=True)
    p.add_argument("--cells", required=True)
    p = sub.add_parser("timed")
    p.add_argument("--sitting", required=True)
    p.add_argument("--cells", required=True)
    p.add_argument("--control", required=True)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--known", type=float)
    p.add_argument("--end-by")
    p.add_argument("--no-windows", action="store_true")
    p.add_argument("--slots", help="a JSON file: [{who, release_at, give_up_at}]")
    p = sub.add_parser("after")
    p.add_argument("--sitting", required=True)
    a = ap.parse_args()
    if a.what == "manifest":
        manifest(own_models(a.models))
    elif a.what == "steps":
        steps(own_models(a.models))
    else:
        {"first": first, "timed": timed, "after": after}[a.what](a)


if __name__ == "__main__":
    main()
