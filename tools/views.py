"""The table views, one per kind: which rows, which columns, how a cell's JSON becomes the text of a table cell.

A view renders the rows of one table. tools/table.py splits a kind's cells into tables by (group, line kind, speed
regime): every number in one table comes from one line kind and one regime (QoS x power mode), so no column puts
speeds of different regimes side by side.

vision  wide: one row per model, one column per cell (the Z1 vision zoo's table); the vision task cells (the Z5
        lane's, script pc/run_vision_task.py: each model on its own task line) long: one row per cell
llm     long: one row per cell (the Z2 LLM zoo's table); task-line rows get their own columns
asr/tts long: one row per cell (the Z3 audio zoo's table), one table per clip set
"""

# ---------------------------------------------------------------------------------------------------------- vision

VISION_COLUMNS = [  # (header, cell id)
    ("Galaxy S26 NPU ms", "qhub-s26-npu"), ("X2E NPU ms", "qhub-x2e-npu"),
    ("Intel NPU as-is ms", "ovnpu"), ("Intel NPU precompiled ms", "ovnpu_pre"), ("Intel GPU ms", "ovgpu"),
    ("RTX fp16 ms", "cuda16"), ("CPU ms", "cpu"),
]
VISION_LOADS = [("Intel NPU as-is", "ovnpu"), ("Intel NPU precompiled", "ovnpu_pre"), ("X2E as-is", "qhub-x2e-npu-onnx")]
VISION_LINE_CELLS = [("a", "ovnpu"), ("p", "ovnpu_pre"), ("g", "ovgpu"), ("h", "cuda16"), ("f", "cuda32")]
VISION_NOTE_ORDER = ["ovnpu", "ovnpu_pre", "ovgpu", "cuda16", "cuda32", "cpu"]
VISION_GROUPS = {  # group -> (columns shown, loads shown)
    "main": ([h for h, _ in VISION_COLUMNS], [h for h, _ in VISION_LOADS]),
    "fixed": ([h for h, _ in VISION_COLUMNS[2:]], [h for h, _ in VISION_LOADS[:2]]),
}


def _ms(c):
    if c is None:
        return "-"
    if (c.get("run") or {}).get("state") == "error":
        # another device's record (AI Hub) without a number shows "-" as in its own table; ERR = this PC's cell failed
        return "-" if c.get("role") == "other-device" else "ERR"
    s = c.get("speed") or {}
    v = s.get("value")
    if v is None:
        if s.get("first_value") is not None:   # a first pass only: in brackets, not a timing
            return f"({s['first_value']:.2f})"
        return "-"
    return f"{v:.2f}"


def _line_code(c):
    if c is None:
        return "-"
    if (c.get("run") or {}).get("state") == "error":
        return "E"
    ln = c.get("line") or {}
    if ln.get("pass") is True:
        return "P"
    if ln.get("pass") is False:
        return "F"
    return (ln.get("status") or "-")[0]


def vision_row(model, recipe, cells, group):
    by = {c["cell"]: c for c in cells}
    cols_shown, loads_shown = VISION_GROUPS.get(group, VISION_GROUPS["fixed"])
    cols = []
    for h, cid in VISION_COLUMNS:
        if h not in cols_shown:
            continue
        c = by.get(cid)
        t = _ms(c)
        n = ((c or {}).get("vision") or {}).get("cpu_nodes")
        if t not in ("-", "ERR") and n:
            t += f" (CPU {n})"
        cols.append(t)
    loads = []
    for h, cid in VISION_LOADS:
        if h not in loads_shown:
            continue
        c = by.get(cid)
        v = ((c or {}).get("speed") or {}).get("load_ms") if c and (c.get("run") or {}).get("state") != "error" else None
        loads.append(f"{v / 1000:.2f}" if v is not None else "-")
    par = " ".join(f"{code}:{_line_code(by.get(cid))}" for code, cid in VISION_LINE_CELLS)
    # the note: the model's own notes, then the cells' fragments in the Z1 order (rank, then column order);
    # cells with the same error line share one fragment
    order = {cid: i for i, cid in enumerate(VISION_NOTE_ORDER)}
    frags = [(0, -1, n) for n in (recipe.get("model") or {}).get("notes") or []]
    errs = {}
    for cid in VISION_NOTE_ORDER:
        c = by.get(cid)
        if c and (c.get("run") or {}).get("state") == "error":
            errs.setdefault((c.get("run") or {}).get("error_line") or "?", []).append(cid)
    for i, (e, cs) in enumerate(errs.items()):
        frags.append((2, i, f"{', '.join(cs)} ERROR: {e}"))
    hub_order = {"qhub-s26-npu": 100, "qhub-x2e-npu": 101}
    for c in cells:
        for f in c.get("note") or []:
            frags.append((f["rank"], order.get(c["cell"], hub_order.get(c["cell"], 50)), f["text"]))
    frags.sort(key=lambda x: (x[0], x[1]))
    note = "; ".join(t for _, _, t in frags)
    return [model] + cols + [" / ".join(loads), par, note]


def vision_header(group):
    cols_shown, loads_shown = VISION_GROUPS.get(group, VISION_GROUPS["fixed"])
    return ["model"] + cols_shown + ["load s: " + " / ".join(loads_shown), "parity", "note"]


VISION_TASK_SCRIPT = "pc/run_vision_task.py"
VISION_TASK_CELL_ORDER = ["task-ovnpu", "task-ovnpu_pre", "task-ovgpu", "task-cuda16", "task-cuda32"]
VISION_TASK_HEADER = ["model", "task", "cell", "runtime", "device", "form", "line", "the line's numbers",
                      "first pass ms", "record"]


def vision_task_sort_key(c, model_order):
    cid = c["cell"]
    return (model_order.get(c["model"], 10 ** 6), c["model"].lower(),
            VISION_TASK_CELL_ORDER.index(cid) if cid in VISION_TASK_CELL_ORDER else 99, cid)


def vision_task_row(c, task):
    """One vision task cell: its verdict on its model's line and the line's numbers in words (tools/vision_tasks.py);
    the first-pass ms in brackets (not a timing)."""
    st = (c.get("run") or {}).get("state")
    ln = c.get("line") or {}
    s = c.get("speed") or {}
    rec = (c.get("source") or {}).get("record")
    head = [c["model"], task or "—", f"`{c['cell']}`", c["runtime"]["label"], c["device"]["label"],
            (c.get("form") or {}).get("label") or "—"]
    if st == "not_run":
        return head + ["未", "not run (" + ((c.get("run") or {}).get("why") or "").replace("|", "/") + ")", "—", "—"]
    if st != "ok":
        err = ((c.get("run") or {}).get("error_line") or "").replace("|", "/")[:120]
        return head + ["E", "ERROR: " + err, "—", f"`{rec}`" if rec else "—"]
    verdict = {True: "PASS", False: "FAIL"}.get(ln.get("pass"), "no verdict")
    why = next((n["text"] for n in c.get("note") or [] if n.get("rank") == 2), None)
    words = (ln.get("text") or why or "—").replace("|", "/")
    v = s.get("value") if s.get("value") is not None else s.get("first_value")
    ms = "—" if v is None else (f"{v:.2f}" if s.get("timed") else f"({v:.2f})")
    return head + [verdict, words, ms, f"`{rec}`" if rec else "—"]


# ---------------------------------------------------------------------------------------------------------- llm

LLM_DEVICE_ORDER = ["npu", "igpu", "rtx", "cpu"]
LLM_RUNTIME_ORDER = ["litert-lm", "llama.cpp", "ort-genai", "foundry", "openvino-genai"]


def _fmt(x, nd=1):
    if x is None:
        return "—"
    if isinstance(x, (int, float)):
        return f"{x:,.{nd}f}"
    return str(x)


def llm_row(c):
    """The Z2 table's row (windows-llm-zoo tools/table.py row())."""
    s = c.get("speed") or {}
    st = (c.get("run") or {}).get("state")
    llm = c.get("llm") or {}
    if st == "not_run":
        return [c["model"], c["runtime"]["label"], c["device"]["label"], llm.get("backend") or "—", "—",
                f"未 ({c['run'].get('why')})", "—", "—", "—", "—", "—", "—"]
    if st == "ok" and s.get("value") is not None and s.get("counted", True):
        dec = _fmt(s.get("value"), 2)
        spread = f" (±{_fmt(s.get('spread_pct'), 1)} %, n {s.get('n')})"
        check = (c.get("line") or {}).get("text") or "—"
    elif st == "out":
        dec, spread, check = "out", " (spread > 5 %)", "—"
    else:
        why = (c["run"].get("why") or st or "").replace("|", "/")
        dec, spread, check = "未", f" ({why})", ("FAIL" if (c.get("line") or {}).get("pass") is False else "—")
    mem = s.get("peak_working_set_mb")
    return [c["model"], c["runtime"]["label"], c["device"]["label"], llm.get("backend") or "—", llm.get("package_text") or "—",
            f"{dec}{spread}", _fmt(s.get("prefill_tps")), _fmt(s.get("ttft_ms"), 0), _fmt(mem, 0) if mem else "—", check,
            s.get("sitting") or "—", (c.get("source") or {}).get("record_text") or "—"]


LLM_HEADER = ["model", "runtime", "device", "backend / EP", "package · quantization", "decode tok/s (cold)",
              "prefill tok/s", "TTFT ms", "peak memory (MiB here, see above)", "output check", "sitting", "record"]


def llm_sort_key(c, model_order):
    rt = c["runtime"]["id"]
    dev = c["device"]["id"]
    return (model_order.get(c["model"], 999), c["model"],
            LLM_DEVICE_ORDER.index(dev) if dev in LLM_DEVICE_ORDER else 99,
            LLM_RUNTIME_ORDER.index(rt) if rt in LLM_RUNTIME_ORDER else 99,
            1 if (c.get("run") or {}).get("state") == "not_run" else 0,
            c.get("sort_key") or c["cell"])


LLM_TASK_HEADER = ["model", "runtime", "device", "form", "answers agreeing with the reference", "right by key", "EOS gate", "line",
                   "decode tok/s", "prefill tok/s", "TTFT ms", "peak memory MiB", "record"]


def llm_task_row(c):
    s = c.get("speed") or {}
    ln = c.get("line") or {}
    task = (c.get("llm") or {}).get("task") or {}
    st = (c.get("run") or {}).get("state")
    timed = s.get("timed")
    def sp(v, nd):
        if v is None:
            return "—"
        t = _fmt(v, nd)
        return t if timed else f"({t})"
    if st == "not_run":
        return [c["model"], c["runtime"]["label"], c["device"]["label"], c["form"]["label"], "—", "—", "—", "—", "—", "—", "—", "—",
                f"未 ({(c['run'].get('why') or 'not run').replace('|', '/')})"]
    if st != "ok":
        return [c["model"], c["runtime"]["label"], c["device"]["label"], c["form"]["label"], "—", "—", "—", "—", "—", "—", "—", "—",
                f"`{(c.get('source') or {}).get('record')}` — ERROR: {(c['run'].get('error_line') or '')[:100]}"]
    agree = f"{task.get('agree')}/{task.get('n')}" if task.get("n") else "—"
    k = task.get("key")
    right = f"{k['right']}/{k['n']}" if k else "—"
    gate = task.get("gate_text") or "—"
    return [c["model"], c["runtime"]["label"], c["device"]["label"], c["form"]["label"], agree, right, gate,
            {True: "PASS", False: "FAIL"}.get(ln.get("pass"), "no verdict"), sp(s.get("value") if s.get("value") is not None else s.get("first_value"), 2), sp(s.get("prefill_tps"), 1), sp(s.get("ttft_ms"), 0),
            _fmt(s.get("peak_working_set_mb"), 0), f"`{(c.get('source') or {}).get('record')}`"]


# ---------------------------------------------------------------------------------------------------------- audio

AUDIO_MODEL_ORDER = ["whisper-large-v3-turbo", "Qwen3-ASR-1.7B", "granite-speech-5.0-2b", "parakeet-tdt-0.6b-v3",
                     "Kokoro-82M"]
AUDIO_HEADER = ["model", "runtime", "form", "device", "clips", "WER / CER (%)", "line", "RTF", "first token ms",
                "peak RAM GB / GPU GB", "record", "note"]


def audio_acc_text(c):
    """The Z3 table's accuracy words (windows-audio-zoo tools/table.py acc_text())."""
    a = (c.get("audio") or {}).get("scores") or {}
    st = (c.get("audio") or {}).get("set")
    if st == "S150":
        parts = []
        for cfg, lab in (("en_us", "en WER"), ("cmn_hans_cn", "zh CER"), ("ja_jp", "ja CER")):
            if cfg in a and a[cfg].get("rate_pct") is not None:
                parts.append(f"{lab} {a[cfg]['rate_pct']:.2f}")
        return " / ".join(parts)
    if st == "S220":
        g = a.get("groups") or {}
        parts = []
        for k, lab in (("en_fl", "en-FL WER"), ("ja", "ja CER"), ("eu", "fr+de+es+pt WER")):
            if k in g and g[k].get("rate_pct") is not None:
                parts.append(f"{lab} {g[k]['rate_pct']:.2f}")
        return " / ".join(parts)
    if st == "T50" and a.get("rate_pct") is not None:
        return f"write-back WER {a['rate_pct']:.2f}"
    return ""


def audio_line_text(c):
    ln = (c.get("line") or {}).get("detail")
    if not ln:
        return "—"
    if ln.get("pass"):
        return "PASS"
    bad = [k for k, v in ln.items() if isinstance(v, dict) and not v.get("pass")]
    return "FAIL " + ", ".join(bad)


def audio_mem_text(c):
    m = c.get("memory") or {}
    ws = m.get("peak_ws_bytes")
    out = f"{ws / 2**30:.1f}" if ws else ""
    g = m.get("gpu_mib") or {}
    dd = [v.get("dedicated", 0) for v in g.values() if isinstance(v, dict)]
    sh = [v.get("shared", 0) for v in g.values() if isinstance(v, dict)]
    if dd and max(dd) > 0:
        out += f" / GPU {max(dd) / 1024:.1f} ded"
    elif sh and max(sh) > 0:
        out += f" / GPU {max(sh) / 1024:.1f} shared"
    return out


def audio_row(c):
    """The Z3 table's row (windows-audio-zoo tools/table.py row_this() and its rows of other lanes' cells)."""
    a = c.get("audio") or {}
    note = "; ".join(n["text"] for n in c.get("note") or [])
    rec = f"`{(c.get('source') or {}).get('record')}`" if (c.get("source") or {}).get("record") else "—"
    if a.get("texts"):   # a cell collected from another lane's table: its words as they were (with their quotes)
        t = a["texts"]
        return [a.get("model_label"), c["runtime"]["label"], c["form"]["label"], c["device"]["label"], a.get("set_shown", ""),
                t.get("accuracy", ""), t.get("line", ""), t.get("rtf", ""), t.get("first_token_ms", ""),
                t.get("memory", ""), rec, note]
    ok = (c.get("run") or {}).get("state") == "ok"
    s = c.get("speed") or {}
    rtf, ft = (s.get("value") if s.get("value") is not None else s.get("first_value")), s.get("first_token_ms")
    rtf_t = f"{rtf:.4f}" if (rtf and ok and s.get("timed")) else ("" if not ok else f"({rtf:.4f})" if rtf else "")
    return [a.get("model_label"), c["runtime"]["label"], c["form"]["label"], c["device"]["label"], a.get("set_shown", ""),
            audio_acc_text(c) if ok else "", audio_line_text(c) if ok else "—", rtf_t,
            f"{ft:.0f}" if (ft and ok) else "", audio_mem_text(c) if ok else "", rec, note]
