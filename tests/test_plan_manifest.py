#!/usr/bin/env python3
"""tools/plan.py manifest on fake Hub trees (Mac, no network, no model runs): two models that fetch one repo at one
revision with different `include` lists get one source holding the files of both (each file once); one repo at two
revisions gives two sources; an `include` that matches no file stops. The manifest is written to a scratch folder,
not to pc/. Exit 1 on any unexpected result.

    python3 tests/test_plan_manifest.py
"""
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import plan  # noqa: E402

A, B = "a" * 40, "b" * 40


def lfs(path, size):
    return {"type": "file", "path": path, "size": size, "oid": "0" * 40, "lfs": {"oid": f"{size:064x}"}}


TREES = {
    ("org/Repo-OpenVINO", A): [lfs("m.onnx", 10), lfs("npu/m.blob", 20), lfs("enc.onnx", 30),
                               {"type": "file", "path": "README.md", "size": 5, "oid": "1" * 40},
                               {"type": "directory", "path": "npu", "size": 0, "oid": "2" * 40}],
    ("org/Repo-OpenVINO", B): [lfs("unfixed/m.onnx", 40), lfs("fp16/m.onnx", 50)],
}


def fake_get(url):
    for (repo, sha), tree in TREES.items():
        if url == f"https://huggingface.co/api/models/{repo}/tree/{sha}?recursive=true":
            return tree
    raise AssertionError(f"unexpected URL {url}")


def recipe(*fetches):
    return {"model": {"kind": "vision"}, "fetch": [dict(f) for f in fetches]}


def run(models):
    with tempfile.TemporaryDirectory() as d:
        plan.PC = pathlib.Path(d)
        plan.manifest(models)
        return json.loads((pathlib.Path(d) / "hub_manifest.json").read_text(encoding="utf-8"))


def main():
    plan.get = fake_get
    bad = []

    # 1. one repo, one revision, two models with different include lists -> one source, the union
    out = run([("first", recipe({"repo": "org/Repo-OpenVINO", "revision": A, "include": ["enc.onnx"]})),
               ("second", recipe({"repo": "org/Repo-OpenVINO", "revision": A,
                                  "include": ["m.onnx", "npu/m.blob", "enc.onnx"]}))])
    paths = sorted(f["path"] for f in out[0]["files"]) if len(out) == 1 else None
    if paths != ["enc.onnx", "m.onnx", "npu/m.blob"] or out[0]["bytes"] != 60 or out[0]["for"] != "first,second":
        bad.append(f"union: {out}")

    # 2. the same repo at two revisions (main and a branch commit) -> two sources
    out = run([("first", recipe({"repo": "org/Repo-OpenVINO", "revision": A, "include": ["m.onnx"]},
                                {"repo": "org/Repo-OpenVINO", "revision": B, "include": ["fp16/m.onnx"]})),
               ("second", recipe({"repo": "org/Repo-OpenVINO", "revision": B, "include": ["unfixed/*"]}))])
    got = sorted((s["revision"][:1], tuple(sorted(f["path"] for f in s["files"]))) for s in out)
    if got != [("a", ("m.onnx",)), ("b", ("fp16/m.onnx", "unfixed/m.onnx"))]:
        bad.append(f"two revisions: {got}")

    # 3. no include list: every file but README.md and .gitattributes
    out = run([("first", recipe({"repo": "org/Repo-OpenVINO", "revision": A}))])
    if sorted(f["path"] for f in out[0]["files"]) != ["enc.onnx", "m.onnx", "npu/m.blob"]:
        bad.append(f"no include: {out}")

    # 4. an include that matches nothing stops
    try:
        run([("first", recipe({"repo": "org/Repo-OpenVINO", "revision": A, "include": ["missing.onnx"]}))])
        bad.append("an include with no file did not stop")
    except SystemExit as e:
        if "has no file for" not in str(e):
            bad.append(f"stop text: {e}")

    for b in bad:
        print("FAIL", b)
    print(f"test_plan_manifest: {4 - len(bad)} of 4 cases as expected")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
