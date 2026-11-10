#!/usr/bin/env python3
"""The vision task lines on synthetic outputs (Mac, no model runs). For each model of pc/z5_spec.json: a near copy of
the reference passes the model's line, a broken copy fails it, a non-finite first input is counted by the instruments
(and the line still gives a verdict or a reason, never a crash), and the reference values serialize. For the codec,
the reference's own SI-SDR is high on the aligned line (task-codec-aligned) and far below 0 dB without the offset
(the first line's flaw, PLAN.md Z5 change 1: a near copy passes either way, so the bar is what tells them apart).
Exit 1 on any unexpected result.

    python3 tests/test_vision_tasks.py
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tests")]
import vision_tasks as VT  # noqa: E402
import synth  # noqa: E402
from synth import Z  # noqa: E402


def word(p):
    return "PASS" if p is True else "FAIL" if p is False else "none"


def main():
    spec = json.loads((ROOT / "pc" / "z5_spec.json").read_text())["models"]
    rng = np.random.default_rng(7)
    unexpected = 0
    for m, ms in sorted(spec.items()):
        task, ids, line = ms["task"], ms["ids"], ms.get("line")
        ref, good, bad = synth.gen(m, ms, rng)
        out = []
        for what, cell, want in (("near copy", good, True), ("broken", bad, False)):
            _, detail, words, _ = VT.score(m, task, Z(cell), Z(ref), ids, line=line)
            got = detail["pass"] if detail else None
            ok = got is want
            unexpected += not ok
            out.append(f"{what} {word(got)}" + ("" if ok else f"  <-- want {word(want)} ({words})"))
        if line in VT.CODEC_OFFSET:
            bars = [VT.score(m, task, Z(good), Z(ref), ids, line=ln)[1]["si_sdr_ref_mean"] for ln in (line, None)]
            ok = bars[0] > 10 and bars[1] < 0
            unexpected += not ok
            out.append(f"reference SI-SDR {bars[0]} dB aligned, {bars[1]} dB without the offset"
                       + ("" if ok else "  <-- want > 10 and < 0"))
        nan = {k: (np.full_like(v, np.nan) if k.startswith("n000") else v) for k, v in good.items()}
        _, detail, words, _ = VT.score(m, task, Z(nan), Z(ref), ids, line=line)
        counted = sum(x["nonfinite"] for x in VT.instruments(Z(nan), Z(ref)).values())
        ok = counted > 0 and (detail is not None or bool(words))
        unexpected += not ok
        out.append(f"non-finite first input {word(detail['pass'] if detail else None)}, {counted} counted"
                   + ("" if ok else "  <-- not counted, or no verdict and no reason"))
        json.dumps(VT.reference_values(m, task, Z(ref), ids, line=line))
        print(f"{m} [{task}, {len(ids)} inputs]: " + " | ".join(out))
    print(f"models {len(spec)}, unexpected {unexpected}")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
