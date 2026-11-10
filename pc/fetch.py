"""Download the Hub files of a pinned manifest (tools/plan.py writes pc/hub_manifest.json) into <root>\\models\\<repo>\\<path> at the pinned revisions and
check every file against the Hub's hash (sha256 = lfs.oid for LFS files, the git blob sha1 for small ones). A source
with "url" (a URL template with {path}) and "dest" (a folder under --root) is a set of plain files pinned by sha256
(the vision input set). A source with "archive" (the URL of a .tar.gz pinned by archive_bytes and archive_sha256) and
"dest" is a set of files taken out of that archive by path (a leading "./" of a member name is not part of its path)
and pinned by sha256 (the keyword clips of the audio models); the archive is deleted once its files match.

    python fetch.py --manifest pc\\hub_manifest.json --dest <zoo folder>\\hub [--only <repo substring> ...]

curl.exe, resumable (-C -), cut and resumed when one connection falls under 100 KB/s for 60 s (seen on this PC: one
connection can crawl while the next is fast). A file whose hash already matches is skipped. Writes
<dest>\\fetch_report.json and exits 1 when any file does not match after 6 tries.
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import time

CURL = "curl.exe" if os.name == "nt" else "curl"   # Windows' own curl.exe on the PC; curl for the tests on a Mac


def blob_sha1(path):
    h = hashlib.sha1()
    size = path.stat().st_size
    h.update(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def ok(path, f):
    if not path.is_file() or path.stat().st_size != f["size"]:
        return False
    return (sha256(path) == f["sha256"]) if f.get("sha256") else (blob_sha1(path) == f["blob"])


def download(url, path, size, check):
    """curl into path (resumed, cut and resumed when slow) until check(path); returns the number of tries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tries = 0
    while not check(path) and tries < 6:
        tries += 1
        if path.is_file() and path.stat().st_size > size:
            path.unlink()
        subprocess.run([CURL, "-sSL", "--fail", "-C", "-", "--speed-limit", "102400", "--speed-time", "60",
                        "--retry", "3", "-o", str(path), url])
    return tries


def fetch_archive(src, root, report):
    """The files of an "archive" source: taken out of the pinned .tar.gz when any of them is missing or differs."""
    folder = root / src["dest"]
    t0 = time.time()
    tries = 0
    if not all(ok(folder / f["path"], f) for f in src["files"]):
        arc = root / "downloads" / src["archive"].rsplit("/", 1)[-1]
        want = {"size": src["archive_bytes"], "sha256": src["archive_sha256"]}
        tries = download(src["archive"], arc, want["size"], lambda p: ok(p, want))
        if ok(arc, want):
            names = {f["path"] for f in src["files"]}
            with tarfile.open(arc, "r:gz") as tar:
                for m in tar:
                    rel = m.name[2:] if m.name.startswith("./") else m.name
                    if m.isfile() and rel in names:
                        out = folder / rel
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_bytes(tar.extractfile(m).read())
    bad = 0
    for f in src["files"]:
        good = ok(folder / f["path"], f)
        bad += not good
        row = {"repo": src["archive"], "revision": None, "path": f["path"], "size": f["size"], "hash": f["sha256"],
               "ok": good, "tries": tries, "seconds": round(time.time() - t0, 1)}
        report.append(row)
        print(json.dumps(row), flush=True)
    if not bad:
        (root / "downloads" / src["archive"].rsplit("/", 1)[-1]).unlink(missing_ok=True)
    return bad


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--root", default="", help="the folder a URL source's dest is under (default: --dest's parent)")
    ap.add_argument("--only", nargs="*", default=[])
    args = ap.parse_args()
    man = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    dest = pathlib.Path(args.dest)
    report, bad = [], 0
    root = pathlib.Path(args.root) if args.root else dest.parent
    for src in man:
        name = src.get("repo", src.get("url", src.get("archive", ""))) + "/" + src.get("folder", "")
        if args.only and not any(o in name for o in args.only):
            continue
        if "archive" in src:
            bad += fetch_archive(src, root, report)
            continue
        for f in src["files"]:
            if "url" in src:
                path = root / src["dest"] / f["path"]
                url = src["url"].replace("{path}", f["path"])
            else:
                path = dest / src["repo"].replace("/", "__") / f["path"]
                url = f"https://huggingface.co/{src['repo']}/resolve/{src['revision']}/{f['path']}"
            t0 = time.time()
            tries = download(url, path, f["size"], lambda p, f=f: ok(p, f))
            good = ok(path, f)
            bad += not good
            row = {"repo": src.get("repo") or src["url"], "revision": src.get("revision"), "path": f["path"],
                   "size": f["size"], "hash": f.get("sha256") or f["blob"], "ok": good, "tries": tries,
                   "seconds": round(time.time() - t0, 1)}
            report.append(row)
            print(json.dumps(row), flush=True)
    dest.mkdir(parents=True, exist_ok=True)   # a manifest of URL sources only (the Z5 lane: the images) makes no hub folder
    (dest / "fetch_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"fetch done: {len(report)} files, {bad} not matching", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
