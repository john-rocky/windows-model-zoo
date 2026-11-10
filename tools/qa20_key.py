#!/usr/bin/env python3
"""qa20 answers counted with an answer key: a second count next to the qa20 line, which compares each answer's first
line with the reference's (PLAN.md section 1b). The line and each cell's pass / fail do not change.

An answer is right by the key when it contains one of the answers the key accepts for its question (`accept` in
`suites/llm/qa20.key.jsonl`) as a word or a number: no ASCII letter or digit right before or after it, and a number by
its value (25.0 counts for 25; 4.0 does not contain 0). Both are first normalized the same way: Unicode NFKC, lower case
(except q19, which the key marks case-sensitive), and every character other than a letter, a digit or a decimal point
between digits replaced by a space. For q11 and q17, an accepted answer must also come before the other answers the
question offers (the key's `reject`). The answer's form is not judged: a sentence counts as well as one word.

    python3 tools/qa20_key.py <record.json> ...     # run or reference records (rows with id, role, text)
    python3 tools/qa20_key.py --cells               # each LLM task cell's record (source.record) and each reference
    python3 tools/qa20_key.py --rows <record.json>  # one line per question: right or not, and the answer

Each record prints "<record>: <right>/<n> right by key" and the ids not right. It stops with one line (exit 1) when a
record is missing or has no rows, a question row has no text, or a row's id is not in the key. Python 3 standard library only.
"""
import json
import pathlib
import re
import sys
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parents[1]
KEY = ROOT / "suites" / "llm" / "qa20.key.jsonl"
NUMBER = re.compile(r"[0-9]+(?:\.[0-9]+)?")
NUMBER_IN_TEXT = re.compile(r"(?<![A-Za-z0-9.])[0-9]+(?:\.[0-9]+)?(?![A-Za-z0-9.])")


class Stop(Exception):
    """a record or key the count cannot read: one line, exit 1"""


def normalize(text, case=False):
    t = unicodedata.normalize("NFKC", text)
    if not case:
        t = t.lower()
    out = []
    for i, ch in enumerate(t):
        keep = unicodedata.category(ch)[0] in "LNM" or (
            ch == "." and 0 < i < len(t) - 1 and t[i - 1].isdigit() and t[i + 1].isdigit())
        out.append(ch if keep else " ")
    return " ".join("".join(out).split())


def finder(form):
    """a normalized form -> f(normalized text) = the position of its first occurrence, or None"""
    if NUMBER.fullmatch(form):
        value = float(form)
        return lambda t: next((m.start() for m in NUMBER_IN_TEXT.finditer(t) if float(m.group()) == value), None)
    pat = re.compile(r"(?<![A-Za-z0-9.])" + re.escape(form) + r"(?![A-Za-z0-9.])")
    return lambda t: (m.start() if (m := pat.search(t)) else None)


def load_key(path=KEY):
    """{id: {"case": bool, "accept": [finder], "reject": [finder]}}"""
    key = {}
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        case = bool(e.get("case"))
        key[e["id"]] = {"case": case, **{k: [finder(normalize(f, case)) for f in e.get(k) or []]
                                         for k in ("accept", "reject")}}
    return key


def is_right(text, entry):
    t = normalize(text, entry["case"])
    acc = [p for p in (f(t) for f in entry["accept"]) if p is not None]
    rej = [p for p in (f(t) for f in entry["reject"]) if p is not None]
    return bool(acc) and (not rej or min(acc) < min(rej))


def question_rows(rows, key, where):
    """the question rows of a record (the gate row is not a question), each checked against the key"""
    out = []
    for r in rows:
        if r.get("role") == "gate":
            continue
        if r.get("id") not in key:
            raise Stop(f"{where}: row {r.get('id')} is not in the key ({rel(KEY)})")
        if not isinstance(r.get("text"), str):
            raise Stop(f"{where}: row {r.get('id')} has no text")
        out.append(r)
    return out


def score_rows(rows, key, where):
    """{"right": r, "n": n, "wrong": [ids]} over the question rows of a record"""
    qs = question_rows(rows, key, where)
    wrong = [r["id"] for r in qs if not is_right(r["text"], key[r["id"]])]
    return {"right": len(qs) - len(wrong), "n": len(qs), "wrong": wrong}


def rel(p):
    p = pathlib.Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def read_rows(path):
    p = pathlib.Path(path)
    if not p.is_file():
        raise Stop(f"{path}: no record")
    rows = json.loads(p.read_text(encoding="utf-8")).get("rows")
    if not isinstance(rows, list):
        raise Stop(f"{path}: no rows")
    return rows


def line_of(name, s):
    return f"{name}: {s['right']}/{s['n']} right by key" + (f" (not right: {' '.join(s['wrong'])})" if s["wrong"] else "")


def task_cells():
    """(model, cell JSON) of every LLM cell on the qa20 line, in file order"""
    for d in sorted((ROOT / "models").iterdir()):
        for p in sorted((d / "results").glob("task-*.json")):
            c = json.loads(p.read_text(encoding="utf-8"))
            if c.get("kind") == "llm" and (c.get("line") or {}).get("id") == "qa20":
                yield d.name, c


def main(argv):
    if not argv:
        print(__doc__.split("\n\n")[2])
        return 2
    key = load_key()
    if argv[0] == "--rows":
        for path in argv[1:]:
            for r in question_rows(read_rows(path), key, rel(path)):
                mark = "right" if is_right(r["text"], key[r["id"]]) else "NOT  "
                print(f"{r['id']} {mark} {json.dumps(r['text'], ensure_ascii=False)[:150]}")
        return 0
    if argv[0] == "--cells":
        refs = {}
        for model, c in task_cells():
            name = f"{model}/{c['cell']}"
            if (c.get("run") or {}).get("state") != "ok":
                print(f"{name}: no count (run {(c.get('run') or {}).get('state')})")
                continue
            print(line_of(name, score_rows(read_rows(ROOT / c["source"]["record"]), key, c["source"]["record"])))
            if c["line"].get("reference"):
                refs.setdefault(model, c["line"]["reference"])
        for model, ref in refs.items():
            print(line_of(f"{model}/reference ({ref})", score_rows(read_rows(ROOT / ref), key, ref)))
        return 0
    for path in argv:
        print(line_of(rel(path), score_rows(read_rows(path), key, rel(path))))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Stop as e:
        print(f"qa20_key: {e}")
        sys.exit(1)
