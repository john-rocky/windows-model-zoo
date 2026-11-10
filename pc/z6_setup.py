"""The untimed setup of one LLM model's task cells on the PC (a [[step]] before its reference; stdlib only), from
pc/z6_spec.json. Writes logs\\z6_setup_<model>.json and exits 1 when anything does not match.

    python z6_setup.py --spec <root>\\pc\\z6_spec.json --model <model> --root <root>

  engines   <root>\\engines\\<dir> (litert, llama-cpu, llama-gpu, llama-cuda, openvino), made by pc/engines.py from
            their public sources: every .exe / .dll checked against the record of the files the cells ran
            (edge-llm-bench results/raw/2026-11-08-windows-gpu-q06, host.json binaries). The LiteRT-LM tools the
            build compiles (engines.sources.litert.rebuilt) are recorded, not compared: their bytes differ on every
            build of the same commit and flags
  links     the LiteRT-LM and GGUF files pc/fetch.py downloaded (<root>\\hub, each checked there against the Hub's
            sha256) hard-linked into <root>\\work\\... (LiteRT-LM writes its caches beside the model: one folder per
            device), each checked for the byte count of the cells' record
  foundry   the Foundry Local catalog packages, in the user's Foundry cache (`foundry model download <variant>` when
            the folder is not there), checked file by file against the record (size and sha256)
  overlays  the WebGPU overlays of ONNX Runtime GenAI packages (pc/variant_pkg.py: hard links, a new genai_config.json
            with the WebGPU adapterIndex)
  cuda12    the CUDA 12 DLL folder Foundry Local's daemon needs on its PATH (pc/engines.py --make cuda12) is there
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

HOME = os.environ.get("USERPROFILE") or os.path.expanduser("~")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def expand(s, root):
    return (s.replace("{HOME}", HOME).replace("{HUB}", os.path.join(root, "hub")).replace("{ROOT}", root))


def engines(spec, need, root, rep):
    bad = 0
    rebuilt = set(spec["engines"]["sources"]["litert"]["rebuilt"])
    for d in need:
        want = spec["engines"]["dirs"][d]
        dst = pathlib.Path(root) / "engines" / d
        rows = []
        for f in sorted(p for p in dst.iterdir() if p.is_file()) if dst.is_dir() else []:
            if f.suffix.lower() not in (".exe", ".dll"):
                continue
            got, w = sha256(f), want.get(f.name)
            built = d == "litert" and f.name in rebuilt
            ok = built or w is None or got == w
            rows.append({"file": f.name, "sha256": got, "record": w, "rebuilt": built, "ok": ok})
            bad += not ok
        missing = sorted(set(want) - {r["file"] for r in rows})
        bad += len(missing)
        rep["engines"][d] = {"dir": str(dst), "files": len(rows), "missing": missing,
                             "not_matching": [r["file"] for r in rows if not r["ok"]],
                             "rebuilt": {r["file"]: r["sha256"] for r in rows if r["rebuilt"]},
                             "not_in_record": [r["file"] for r in rows if r["record"] is None]}
        print(json.dumps({"engines": d, "files": len(rows), "missing": missing,
                          "not_matching": rep["engines"][d]["not_matching"],
                          "rebuilt": sorted(rep["engines"][d]["rebuilt"])}), flush=True)
    return bad


def links(items, root, rep):
    bad = 0
    for x in items:
        src, dst = pathlib.Path(expand(x["src"], root)), pathlib.Path(expand(x["dst"], root))
        row = {"src": str(src), "dst": str(dst), "bytes": x["bytes"]}
        if not src.is_file() or src.stat().st_size != x["bytes"]:
            row["ok"] = False
            row["why"] = f"source missing or {src.stat().st_size if src.is_file() else 0} bytes, record {x['bytes']}"
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and not os.path.samefile(src, dst):
                dst.unlink()
            if not dst.exists():
                os.link(src, dst)
            row["ok"] = os.path.samefile(src, dst)
            row["sha256"] = sha256(dst)
        bad += not row["ok"]
        rep["links"].append(row)
        print(json.dumps({k: row.get(k) for k in ("dst", "ok", "why")}), flush=True)
    return bad


def daemon_pids():
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq foundrylocald.exe", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    return [int(x.split('","')[1]) for x in out.strip().splitlines() if x.startswith('"foundrylocald')]


def yield_notes(kind):
    """the notes in pcup\\active that match the patterns listed under <kind> in pcup\\zoo_yield.json (a file of the
    shared PC that this repo does not ship: the other lanes' notes this zoo yields to). No file = none; a file that does
    not parse stops the step. ZOO_PCUP names another pcup folder (as in pc\\zlib.ps1)."""
    pcup = pathlib.Path(os.environ.get("ZOO_PCUP") or (pathlib.Path(HOME) / "pcup"))
    f = pcup / "zoo_yield.json"
    if not f.is_file():
        return []
    pats = json.loads(f.read_text(encoding="utf-8-sig")).get(kind) or []
    act = pcup / "active"
    return sorted({p.name for pat in pats if pat for p in act.glob(pat)}) if act.is_dir() else []


def foundry_download(variant, c12):
    """`foundry model download` into the user's Foundry cache. The daemon is one per Windows user and another lane
    may share it: no download while that lane's note is up (foundry_notes in pcup\\zoo_yield.json) or a daemon this
    step did not start runs; the daemon it starts is stopped after"""
    notes = yield_notes("foundry_notes")
    if notes:
        return {"downloaded": False, "why": f"another lane's Foundry note is up ({notes[0]})"}
    if daemon_pids():
        return {"downloaded": False, "why": "a Foundry daemon this step did not start runs"}
    exe = shutil.which("foundry") or "foundry"
    env = dict(os.environ, PATH=c12 + os.pathsep + os.environ.get("PATH", ""))   # the daemon starts with none otherwise
    r = subprocess.run([exe, "model", "download", variant], capture_output=True, text=True, env=env, timeout=3600)
    s = subprocess.run([exe, "server", "stop"], capture_output=True, text=True, env=env, timeout=300)
    return {"downloaded": r.returncode == 0, "command": f"foundry model download {variant}", "exit": r.returncode,
            "output": (r.stdout + r.stderr)[-1500:], "server_stop_exit": s.returncode}


def foundry(items, root, rep, c12):
    bad = 0
    for v in items:
        folder = pathlib.Path(expand(v["folder"], root))
        got = {"downloaded": False, "why": "in the Foundry cache before this step"}
        if not folder.is_dir():
            got = foundry_download(v["variant"], c12)
        wrong = []
        for f in v["files"]:
            p = folder / f["name"]
            if not p.is_file() or p.stat().st_size != f["size"] or sha256(p) != f["sha256"]:
                wrong.append(f["name"])
        extra = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()
                       and str(p.relative_to(folder)) not in {f["name"] for f in v["files"]}) if folder.is_dir() else []
        rep["foundry"].append({"variant": v["variant"], "folder": str(folder), "files": len(v["files"]), "got": got,
                               "not_matching": wrong, "not_in_record": extra})
        bad += len(wrong)
        print(json.dumps({"foundry": v["variant"], "got": got.get("why") or got.get("command"),
                          "not_matching": wrong, "not_in_record": extra}), flush=True)
    return bad


def webgpu_sessions(cfg):
    """place "@webgpu" = every session the source package already puts on WebGPU"""
    place, opts = [], {}
    for comp, v in cfg["model"].items():
        if not isinstance(v, dict):
            continue
        for po in (v.get("session_options") or {}).get("provider_options") or []:
            for name, o in po.items():
                if name.lower() == "webgpu":
                    place.append(f"{comp}=webgpu")
                    opts.update({k: str(x) for k, x in (o or {}).items() if k != "adapterIndex"})
    return ",".join(place), opts


def overlays(items, root, rep):
    bad = 0
    tool = pathlib.Path(root) / "pc" / "variant_pkg.py"
    for o in items:
        src, dst = expand(o["src"], root), expand(o["dst"], root)
        src_cfg = json.loads((pathlib.Path(src) / "genai_config.json").read_text(encoding="utf-8"))
        place, extra = o["place"], {}
        if place == "@webgpu":
            place, extra = webgpu_sessions(src_cfg)
        cmd = [sys.executable, str(tool), "--src", src, "--dst", dst, "--place", place]
        for kv in [f"{k}={v}" for k, v in extra.items()] + o["webgpu_option"]:
            cmd += ["--webgpu_option", kv]
        r = subprocess.run(cmd, capture_output=True, text=True)
        ok = r.returncode == 0 and bool(place)
        row = {"dst": dst, "src": src, "place": place, "carried": extra, "webgpu_option": o["webgpu_option"],
               "command": cmd, "ok": ok, "output": (r.stdout + r.stderr)[-1500:]}
        if ok:
            dst_cfg = json.loads((pathlib.Path(dst) / "genai_config.json").read_text(encoding="utf-8"))
            (pathlib.Path(dst) / "overlay.json").write_text(json.dumps(
                {"src": src, "place": place, "webgpu_option": o["webgpu_option"], "carried": extra, "command": cmd,
                 "src_genai_config": src_cfg, "genai_config": dst_cfg}, indent=1), encoding="utf-8")
        rep["overlays"].append(row)
        bad += not ok
        print(json.dumps({"overlay": dst, "ok": ok, "place": place}), flush=True)
    return bad


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    spec = json.loads(pathlib.Path(a.spec).read_text(encoding="utf-8"))
    ms = spec["models"][a.model]
    c12 = pathlib.Path(expand(spec["cuda12"], a.root))
    rep = {"model": a.model, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "engines": {}, "links": [], "foundry": [],
           "overlays": [], "cuda12": None}
    bad = engines(spec, ms.get("engines") or [], a.root, rep)
    bad += links(ms.get("links") or [], a.root, rep)
    bad += foundry(ms.get("foundry") or [], a.root, rep, str(c12))
    bad += overlays(ms.get("overlays") or [], a.root, rep)
    if ms.get("foundry_cells"):
        rep["cuda12"] = {"dir": str(c12), "dlls": sorted(p.name for p in c12.glob("*.dll")) if c12.is_dir() else None}
        bad += not rep["cuda12"]["dlls"]
    rep["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    rep["not_matching"] = bad
    out = pathlib.Path(a.root) / "logs" / f"z6_setup_{a.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"z6 setup {a.model}: {bad} not matching", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
