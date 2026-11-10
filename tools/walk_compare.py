#!/usr/bin/env python3
"""The walk's cells against this repo's cells of the same names: one line per cell (its verdict and its line's numbers
the same or not) and one per reference (its rows / task values the same or not).

    python3 tools/results.py --runs raw/walk/runs --refs raw/walk/refs --out walk-out --cells <model>/<cell>,...
    python3 tools/walk_compare.py walk-out

tools/results.py --out writes <out>/<model>/<cell>.json and <out>/<model>/ref.json and leaves models/ as it is.
Exits 1 when a cell's verdict differs from this repo's (a difference in the numbers alone is printed, not an exit).
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import zoo  # noqa: E402


def diff_paths(a, b, path=""):
    """the paths under which two JSON values differ (at most 8)"""
    if type(a) is not type(b):
        return [path or "."]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff_paths(a.get(k), b.get(k), f"{path}.{k}")
        return out[:8]
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path} (length {len(a)} / {len(b)})"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff_paths(x, y, f"{path}[{i}]")
        return out[:8]
    return [] if a == b else [path or "."]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = pathlib.Path(sys.argv[1])
    bad = 0
    for p in sorted(out.glob("*/*.json")):
        m = p.parent.name
        walk = json.loads(p.read_text(encoding="utf-8"))
        if p.name == "ref.json":
            mine = zoo.MODELS / m / "reference" / "ref.json"
            if not mine.is_file():
                print(f"{m} reference: this repo has no compact reference")
                continue
            here = json.loads(mine.read_text(encoding="utf-8"))
            keys = [k for k in ("rows", "task_values") if k in here or k in walk]
            d = [x for k in keys for x in diff_paths(walk.get(k), here.get(k), k)]
            print(f"{m} reference: {', '.join(keys)} {'the same' if not d else 'differ at ' + '; '.join(d)}")
            continue
        mine = zoo.MODELS / m / "results" / p.name
        if not mine.is_file():
            print(f"{m}/{p.stem}: not a cell of this repo")
            bad += 1
            continue
        lw, lh = walk.get("line") or {}, json.loads(mine.read_text(encoding="utf-8")).get("line") or {}
        same_v, d = lw.get("pass") == lh.get("pass"), diff_paths(lw.get("detail"), lh.get("detail"), "detail")
        bad += not same_v
        what = "verdict and numbers the same" if same_v and not d else (
            "verdict the same, numbers differ at " + "; ".join(d) if same_v else "VERDICT DIFFERS")
        print(f"{m}/{p.stem}: {what}\n    walk: {lw.get('verdict')}\n    repo: {lh.get('verdict')}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
