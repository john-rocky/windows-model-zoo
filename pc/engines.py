"""The engines of the LLM task cells, made on the PC from their public sources (stdlib only), as pc/z6_spec.json's
engines.sources names them; pc/z6_setup.py then checks every .exe / .dll against the record of the files the cells
ran (engines.dirs). Writes logs\\engines-<make>.json and exits 1 when a part fails.

    python engines.py --spec <root>\\pc\\z6_spec.json --root <root> --make llama|cuda12|litert-src|litert|openvino

  llama       the llama.cpp release zips of b11721 (edge-llm-bench windows/engine-pins.json's pins: URL, size, sha256)
              into <root>\\engines_src, unpacked into <root>\\engines\\llama-cpu, llama-gpu (Vulkan) and llama-cuda
              (with the CUDA runtime zip)
  cuda12      the CUDA 12 DLLs Foundry Local's daemon needs on its PATH (without them it starts no execution provider,
              the CPU's included): pc/req/cuda12.txt's NVIDIA wheels installed by uv into <root>\\cuda12\\pkgs, their
              DLLs copied into <root>\\cuda12\\dll
  litert-src  LiteRT-LM's source at the tag (clone, checkout, the tag's prebuilt Windows DLLs by git lfs), the first
              part of the build script below, without the build (the build takes the PC's lock; this part does not)
  litert      LiteRT-LM publishes no Windows build of its CLI: the tag is built with edge-llm-bench's
              windows/build_litert_lm.ps1 (downloaded at a pinned commit, checked by sha256), with the DirectX shader
              compiler DLLs of the PyPI wheel litert-lm-api 0.19.0. The tools it compiles (sources.litert.rebuilt) get
              other bytes than the record on every build; the DLLs it copies (the tag's prebuilt ones, the wheel's)
              keep theirs
  openvino    the OpenVINO runtime that build fetched, for LiteRT-LM's Intel NPU dispatch (gemma-4-E2B-it's NPU cell)
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
import zipfile


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def say(row):
    print(json.dumps(row), flush=True)


def download(x, path, rep):
    """x = {url, size, sha256}; curl.exe, resumable, cut and resumed under 100 KB/s for 60 s (as pc/fetch.py)"""
    path.parent.mkdir(parents=True, exist_ok=True)

    def ok():
        return path.is_file() and path.stat().st_size == x["size"] and sha256(path) == x["sha256"]

    t0, tries = time.time(), 0
    while not ok() and tries < 6:
        tries += 1
        if path.is_file() and path.stat().st_size >= x["size"]:
            path.unlink()   # too long, or the full length with another hash: from the start
        subprocess.run(["curl.exe", "-sSL", "--fail", "-C", "-", "--speed-limit", "102400", "--speed-time", "60",
                        "--retry", "3", "-o", str(path), x["url"]])
    row = {"url": x["url"], "path": str(path), "size": x["size"], "sha256": x["sha256"], "ok": ok(), "tries": tries,
           "seconds": round(time.time() - t0, 1)}
    rep["downloads"].append(row)
    say(row)
    return row["ok"]


def unzip(zp, dst, members=None):
    """the archive's files into dst; one top folder holding everything is dropped; no absolute or .. path"""
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist() if not n.endswith("/") and (members is None or n in members)]
        parts = [pathlib.PurePosixPath(n).parts for n in names]
        for n, p in zip(names, parts):
            if n.startswith(("/", "\\")) or ":" in n or ".." in p:
                sys.exit(f"{zp}: unsafe path in the archive: {n}")
        strip = members is None and len({p[0] for p in parts}) == 1 and all(len(p) > 1 for p in parts)
        for n, p in zip(names, parts):
            t = dst.joinpath(*(p[1:] if strip or members is not None else p))
            t.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as s, open(t, "wb") as d:
                shutil.copyfileobj(s, d)
    return len(names)


def tool(name, *fallbacks):
    p = shutil.which(name)
    if p:
        return p
    return next((f for f in fallbacks if os.path.isfile(f)), name)


def llama(spec, root, rep):
    src, bad = spec["engines"]["sources"], 0
    for d in ("llama-cpu", "llama-gpu", "llama-cuda"):
        s, dst = src[d], root / "engines" / d
        if dst.exists():
            shutil.rmtree(dst)   # no file of an earlier unpack stays
        n = 0
        for x in [s] + ([s["cudart"]] if s.get("cudart") else []):
            zp = root / "engines_src" / x["url"].rsplit("/", 1)[1]
            if not download(x, zp, rep):
                bad += 1
                continue
            n += unzip(zp, dst)
        rep["made"][d] = {"dir": str(dst), "files": n}
        say({"made": d, "files": n})
    return bad


def cuda12(spec, root, rep):
    uv = tool("uv", os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Links", "uv.exe"))
    dll = pathlib.Path(spec["cuda12"].replace("{ROOT}", str(root)))
    pkgs = dll.parent / "pkgs"
    cmd = [uv, "pip", "install", "--python", sys.executable, "--target", str(pkgs), "--no-deps", "-r",
           str(root / "pc" / "req" / "cuda12.txt")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    dll.mkdir(parents=True, exist_ok=True)
    names = {}
    for p in sorted(pkgs.rglob("*.dll")):
        names.setdefault(p.name, []).append(str(p.relative_to(pkgs)))
        shutil.copy2(p, dll / p.name)
    twice = {k: v for k, v in names.items() if len(v) > 1}
    rep["made"]["cuda12"] = {"command": cmd, "exit": r.returncode, "output": (r.stdout + r.stderr)[-3000:],
                             "dir": str(dll), "dlls": sorted(names), "same_name_twice": twice}
    say({"made": "cuda12", "exit": r.returncode, "dlls": len(names), "same_name_twice": sorted(twice)})
    return int(r.returncode != 0 or not names or bool(twice))


def run(cmd, rep, env=None, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)
    rep["commands"].append({"command": cmd, "exit": r.returncode, "output": (r.stdout + r.stderr)[-2000:]})
    say({"command": " ".join(cmd), "exit": r.returncode})
    return r


def litert_src(spec, root, rep):
    s = spec["engines"]["sources"]["litert"]
    git = tool("git", r"C:\Program Files\Git\cmd\git.exe")
    src = pathlib.Path(s["src"])
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    if not (src / ".git").is_dir():
        if run([git, "-c", "core.longpaths=true", "-c", "core.symlinks=true", "clone", s["repo"], str(src)],
               rep, env).returncode:
            return 1
    for args in (["config", "core.longpaths", "true"], ["config", "core.symlinks", "true"], ["fetch", "--tags", "origin"],
                 ["checkout", "-B", "windows-" + s["tag"], s["tag"]],
                 ["lfs", "pull", "--include", "prebuilt/windows_x86_64/*"]):
        if run([git] + args, rep, env, str(src)).returncode:
            return 1
    head = run([git, "rev-parse", "HEAD"], rep, env, str(src)).stdout.strip()
    rep["made"]["litert-src"] = {"dir": str(src), "tag": s["tag"], "head": head, "record": s["commit"]}
    return int(head != s["commit"])


def litert(spec, root, rep):
    s = spec["engines"]["sources"]["litert"]
    es = root / "engines_src"
    script = es / "build_litert_lm.ps1"
    whl = es / s["dxc_wheel"]["url"].rsplit("/", 1)[1]
    if not (download(s["script"], script, rep) and download(s["dxc_wheel"], whl, rep)):
        return 1
    dxc = es / "dxc"
    unzip(whl, dxc, members=s["dxc_wheel"]["members"])
    dst = root / "engines" / "litert"
    log = root / "logs" / "litert_build.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "-Tag", s["tag"],
           "-Label", s["label"], "-Src", s["src"], "-OutputBase", s["output_base"], "-EngineDir", str(dst),
           "-DxcDir", str(dxc), "-Log", str(log)]
    say({"build": " ".join(cmd)})
    r = subprocess.run(cmd)
    last = ""
    if log.is_file():
        lines = [x for x in log.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]
        last = lines[-1] if lines else ""
    pins = {}
    if (dst / "pins.json").is_file():
        pins = next(iter(json.loads((dst / "pins.json").read_text(encoding="utf-8-sig"))["windows_litert-lm"].values()))
    rep["made"]["litert"] = {"command": cmd, "exit": r.returncode, "log": str(log), "log_last": last,
                             "commit": pins.get("commit"), "record_commit": s["commit"], "pins": pins}
    say({"made": "litert", "exit": r.returncode, "log_last": last, "commit": pins.get("commit")})
    # powershell -File can exit 0 after a native command failed: the log's last line and the tag's commit decide too
    return int(r.returncode != 0 or " staged " not in last or pins.get("commit") != s["commit"])


def openvino(spec, root, rep):
    ob = pathlib.Path(spec["engines"]["sources"]["litert"]["output_base"]) / "external"
    bases = sorted(p / "openvino" / "runtime" for p in ob.glob("*intel_openvino*") if (p / "openvino" / "runtime").is_dir())
    if len(bases) != 1:
        rep["made"]["openvino"] = {"error": f"{len(bases)} OpenVINO runtimes under {ob}", "found": [str(b) for b in bases]}
        say(rep["made"]["openvino"])
        return 1
    b, dst = bases[0], root / "engines" / "openvino"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    files = [p for p in (b / "bin" / "intel64" / "Release").iterdir() if p.is_file()]
    files += [p for p in (b / "3rdparty" / "tbb" / "bin").glob("*.dll") if not p.stem.endswith("_debug")]
    for p in files:
        shutil.copy2(p, dst / p.name)
    rep["made"]["openvino"] = {"from": str(b), "dir": str(dst), "files": sorted(p.name for p in files)}
    say({"made": "openvino", "files": len(files)})
    return 0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--make", required=True, choices=["llama", "cuda12", "litert-src", "litert", "openvino"])
    a = ap.parse_args()
    spec = json.loads(pathlib.Path(a.spec).read_text(encoding="utf-8"))
    root = pathlib.Path(a.root)
    rep = {"make": a.make, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "downloads": [], "commands": [], "made": {}}
    bad = {"llama": llama, "cuda12": cuda12, "litert-src": litert_src, "litert": litert, "openvino": openvino}[a.make](
        spec, root, rep)
    rep["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    rep["failed"] = bad
    out = root / "logs" / f"engines-{a.make}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"engines {a.make}: {'FAIL' if bad else 'OK'}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
