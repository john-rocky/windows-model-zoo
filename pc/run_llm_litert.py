"""A LiteRT-LM LLM cell: litert_lm_advanced_main (the edge-llm-bench Windows build, v0.19.0) once per prompt, in session
mode, the reference's rendered prompts as they are; one JSON.

    python run_llm_litert.py --exe <litert_lm_advanced_main_windows.exe> --model <file.litertlm> --backend cpu|gpu|npu
        [--adapter igpu|rtx] [--dispatch-dir <dir> --path-prepend <OpenVINO dir>] --ref <ref.json> --out run.json
        [--max-new 48] [--gate-max-new 16]

Each prompt: one process (`--use_session=true`: a Session takes the text as it is, no Jinja; `--sampler_top_k=1` =
greedy; `--max_output_tokens` 48, the gate prompt 16; `--async=false`; `--benchmark` for the token counts; the bench's
`--max_num_tokens=2048` on the CPU and GPU; the NPU: the bench's `--litert_dispatch_lib_dir` and its OpenVINO DLL folder
first on PATH, scripts/windows_ssh_executor.py), the prompt in a UTF-8 file with nothing added. The Session puts the
model's start token in front by itself and stops on a start token in the text (PLAN.md section Z6, the probe of
11-10 07:54): a leading <bos> of the rendered string is dropped. The answer = the process's stdout after "output: "
(the session's text and one line end); its prefill and decode token counts = the "Prefill Turn 1" / "Decode Turn 1"
lines of the benchmark info (stderr). The row's prompt token count is compared with the reference's ids (no ids
here); the run ended on an end token when it decoded fewer tokens than the maximum.
Adapter (the bench's way, windows/run_job.ps1): --adapter igpu sets Windows' per-app graphics preference
GpuPreference=1; on this exe (HKCU\\Software\\Microsoft\\DirectX\\UserGpuPreferences, the value named the exe's full
path) for the cell and removes it after, plus --preferred_device_substr=Intel; --adapter rtx removes any such value
first. The run's "Selected adapter:" line must name Intel / NVIDIA, else the cell is an error.
    python run_llm_litert.py --cleanup --exe <exe>      # remove the preference value (pc/chain.ps1 after a killed run)
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zoolib as al  # noqa: E402

PREF_KEY = r"Software\Microsoft\DirectX\UserGpuPreferences"
ADAPTER_WANT = {"igpu": "Intel", "rtx": "NVIDIA"}


def pref_set(exe, value):
    import winreg
    k = winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, PREF_KEY, 0, winreg.KEY_SET_VALUE)
    winreg.SetValueEx(k, exe, 0, winreg.REG_SZ, value)
    winreg.CloseKey(k)


def pref_remove(exe):
    """-> the value that was there (None when none)"""
    import winreg
    try:
        k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, PREF_KEY, 0, winreg.KEY_QUERY_VALUE | winreg.KEY_SET_VALUE)
    except FileNotFoundError:
        return None
    try:
        old = winreg.QueryValueEx(k, exe)[0]
    except FileNotFoundError:
        winreg.CloseKey(k)
        return None
    winreg.DeleteValue(k, exe)
    winreg.CloseKey(k)
    return old


def seconds(s):
    m = re.match(r"([\d.]+)\s*(ms|us|s)", s)
    if not m:
        return None
    return float(m.group(1)) * {"s": 1.0, "ms": 1e-3, "us": 1e-6}[m.group(2)]


def turn(err, kind):
    m = re.search(kind + r" Turn 1: Processed (\d+) tokens in ([\d.]+\s*(?:ms|us|s))", err)
    return (int(m.group(1)), seconds(m.group(2))) if m else (None, None)


def main():
    al.utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--backend", choices=["cpu", "gpu", "npu"], default="cpu")
    ap.add_argument("--adapter", choices=["", "igpu", "rtx"], default="")
    ap.add_argument("--dispatch-dir", default="")
    ap.add_argument("--path-prepend", default="")
    ap.add_argument("--max-num-tokens", type=int, default=2048)
    ap.add_argument("--ref", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--gate-max-new", type=int, default=16)
    ap.add_argument("--prompt-timeout", type=int, default=1200)
    ap.add_argument("--cleanup", action="store_true")
    args = ap.parse_args()
    exe = str(pathlib.Path(args.exe).resolve())
    if args.cleanup:
        old = pref_remove(exe) if os.name == "nt" else None
        print(json.dumps({"cleanup": "litert", "exe": exe, "removed_preference": old}), flush=True)
        return 0
    rec = al.Record("run_llm_litert.py", args)
    al.banner()
    code = 0
    pref_on = False
    try:
        rec["versions"] = al.versions([])
        rec["exe"] = {"path": exe, "sha256": al.sha256_file(exe)}
        rec["model"] = {"path": args.model, "bytes": os.path.getsize(args.model)}
        ref = json.loads(pathlib.Path(args.ref).read_text(encoding="utf-8"))
        rows = [r for r in ref["rows"] if r.get("prompt")]
        # a killed run leaves one; the bench removes it first too (no such registry off Windows: tests/test_llm_runners.py)
        rec["preference_found"] = pref_remove(exe) if os.name == "nt" else None
        if args.adapter == "igpu" and os.name == "nt":
            pref_set(exe, "GpuPreference=1;")
            pref_on = True
        env = dict(os.environ)
        if args.path_prepend:
            env["PATH"] = args.path_prepend + os.pathsep + env.get("PATH", "")
        work = pathlib.Path(args.out).with_suffix(".prompts")
        work.mkdir(parents=True, exist_ok=True)
        base = [exe, f"--backend={args.backend}", f"--model_path={args.model}", "--async=false", "--use_session=true",
                "--sampler_top_k=1", "--benchmark", "--benchmark_prefill_tokens=0", "--benchmark_decode_tokens=0"]
        if args.backend != "npu":
            base.append(f"--max_num_tokens={args.max_num_tokens}")
        if args.adapter == "igpu":
            base.append("--preferred_device_substr=Intel")
        if args.dispatch_dir:
            base.append(f"--litert_dispatch_lib_dir={args.dispatch_dir}")
        rec["workload"] = {"ref": args.ref, "prompts": len(rows), "backend": args.backend, "adapter": args.adapter or None,
                           "decode": "greedy (session sampler top_k 1)", "max_new_tokens": args.max_new,
                           "gate_max_new_tokens": args.gate_max_new, "one process per prompt": True,
                           "chat_template": "none here (session mode): the reference's rendered prompt without a "
                                            "leading <bos> (the session adds the start token)",
                           "command": base + ["--input_prompt_file=<prompt>", "--max_output_tokens=<48|16>"],
                           "path_prepend": args.path_prepend or None}
        rec.total = len(rows)
        adapters = set()
        for i, r in enumerate(rows):
            mx = args.gate_max_new if r.get("role") == "gate" else args.max_new
            text = r["prompt"]
            form = "string"
            if text.startswith("<bos>"):
                text, form = text[len("<bos>"):], "string without its leading <bos> (the session adds the start token)"
            pf = work / f"{i:02d}.txt"
            pf.write_bytes(text.encode("utf-8"))
            cmd = base + [f"--input_prompt_file={pf}", f"--max_output_tokens={mx}"]
            t0 = time.perf_counter()
            p = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, timeout=args.prompt_timeout, env=env)
            wall_proc = time.perf_counter() - t0
            out = p.stdout.decode("utf-8", "replace").replace("\r\n", "\n")
            err = p.stderr.decode("utf-8", "replace").replace("\r\n", "\n")
            (work / f"{i:02d}.stderr.txt").write_text(err, encoding="utf-8")
            if p.returncode != 0:
                raise RuntimeError(f"prompt {r['id']}: exit {p.returncode}: " + " | ".join(err.strip().splitlines()[-4:]))
            k = out.rfind("output: ")
            if k < 0 or (k > 0 and out[k - 1] != "\n"):
                raise RuntimeError(f"prompt {r['id']}: no 'output: ' line on stdout: {out[-300:]!r}")
            answer = out[k + len("output: "):]
            if answer.endswith("\n"):
                answer = answer[:-1]
            n_in, s_in = turn(err, "Prefill")
            n_out, s_out = turn(err, "Decode")
            sel = re.search(r"Selected adapter: ([^\n]*)", err + "\n" + out)
            if sel:
                adapters.add(sel.group(1).strip())
            if args.adapter and (not sel or ADAPTER_WANT[args.adapter] not in sel.group(1)):
                raise RuntimeError(f"prompt {r['id']}: the run's adapter line is {sel.group(1) if sel else 'missing'}, "
                                   f"not {ADAPTER_WANT[args.adapter]}")
            if n_out is None:
                raise RuntimeError(f"prompt {r['id']}: no Decode Turn 1 line in the benchmark info")
            rec.row({"id": r["id"], "role": r.get("role"), "prompt_tokens": n_in, "prompt_form": form,
                     "new_tokens": n_out, "text": answer, "eos": n_out < mx,
                     "stop": "fewer than max" if n_out < mx else "max_output_tokens", "max_new": mx,
                     "wall_s": (s_in or 0) + (s_out or 0), "first_token_ms": (s_in or 0) * 1000,
                     "process_s": round(wall_proc, 3)})
        rec["adapters_seen"] = sorted(adapters)
        rec["summary"] = rec.summarize("llm")
        rec["status"] = "OK"
    except Exception:
        rec.fail()
        code = 1
    finally:
        if pref_on:
            try:
                pref_remove(exe)
                rec["preference_removed"] = True
            except Exception as e:  # the record says so; the next run of this exe removes it first
                rec["preference_removed"] = f"failed: {e!r}"
    rec.write()
    return code


if __name__ == "__main__":
    sys.exit(main())
