"""A llama.cpp LLM cell: llama-server of the bench's release asset (b11721: CPU, Vulkan or CUDA build) as a child
process, the reference's rendered prompts as token ids, after one untimed warm-up prompt; one JSON.

    python run_llm_llamacpp.py --server <llama-server.exe> --model <file.gguf> [--gpu] [--env KEY=VALUE ...]
        [--device-prefix CUDA|Vulkan] --ref <ref.json> --out run.json [--max-new 48] [--gate-max-new 16]

The server: -m <gguf> -c 2048 -np 1 on 127.0.0.1 (a free port), the bench's GPU flags -ngl 99 -fa on -lv 4 with --gpu,
--env for the server's environment (the bench's Vulkan row: GGML_VK_VISIBLE_DEVICES=0). A GPU cell's server log must
name a device of --device-prefix (CUDA0 / Vulkan0) on the NVIDIA GPU, else the cell is an error. Each prompt
(PLAN.md section Z6): POST /tokenize {content: the rendered string, add_special false, parse_special true} -> the ids
(compared with the reference's), POST /completion {prompt: the ids (an id array gets no BOS), n_predict 48 (the gate
prompt 16), temperature 0, top_k 1, repeat_penalty 1.0, cache_prompt false, return_tokens true}. The row: the ids,
the new ids, the text (content), the evaluated prompt tokens, and whether it stopped on an end token (stop_type eos).
Timing per prompt: the server's own prompt_ms + predicted_ms.
"""
import argparse
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zoolib as al  # noqa: E402


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def call(url, body=None, timeout=1200):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    al.utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--device-prefix", default="")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--gate-max-new", type=int, default=16)
    ap.add_argument("--load-timeout", type=int, default=900)
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()
    rec = al.Record("run_llm_llamacpp.py", args)
    al.banner()
    code = 0
    srv = None
    log_path = pathlib.Path(args.out).with_suffix(".server.txt")
    try:
        rec["versions"] = al.versions([])
        rec["server_exe"] = {"path": args.server, "sha256": al.sha256_file(args.server)}
        rec["model"] = {"path": args.model, "bytes": os.path.getsize(args.model)}
        ref = json.loads(pathlib.Path(args.ref).read_text(encoding="utf-8"))
        rows = [r for r in ref["rows"] if r.get("prompt")]
        port = free_port()
        cmd = [args.server, "-m", args.model, "-c", str(args.ctx), "-np", "1", "--host", "127.0.0.1", "--port", str(port),
               "--no-webui"]
        if args.gpu:   # the bench's GPU rows; -lv 4 as theirs: the log names the device ("using device CUDA0 (...)")
            cmd += ["-ngl", "99", "-fa", "on", "-lv", "4"]
        env = dict(os.environ)
        for kv in args.env:
            k, v = kv.split("=", 1)
            env[k] = v
        rec["server_command"] = cmd
        rec["server_env"] = args.env
        t = time.perf_counter()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logf = open(log_path, "wb")
        srv = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT, env=env)
        base = f"http://127.0.0.1:{port}"
        while True:
            if srv.poll() is not None:
                raise RuntimeError(f"llama-server exited {srv.returncode} while loading: "
                                   + " | ".join(log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-5:]))
            try:
                if call(base + "/health", timeout=5).get("status") == "ok":
                    break
            except (urllib.error.URLError, ConnectionError, OSError, ValueError):
                pass
            if time.perf_counter() - t > args.load_timeout:
                raise RuntimeError(f"llama-server not ready in {args.load_timeout} s")
            time.sleep(0.5)
        rec["load_s"] = round(time.perf_counter() - t, 3)
        log = log_path.read_text(encoding="utf-8", errors="replace")
        rec["server_build"] = (re.search(r"build: ([^\n]+)", log) or [None, None])[1]
        devs = sorted(set(re.findall(r"using device (\S+) \(([^)]*)\)", log)))
        rec["devices"] = [f"{d} ({n})" for d, n in devs]
        if args.gpu:
            ok = [d for d, n in devs if d.startswith(args.device_prefix) and "NVIDIA" in n]
            if not ok:
                raise RuntimeError(f"no {args.device_prefix} device on the NVIDIA GPU in the server log: {rec['devices']}")
        try:
            rec["props"] = {k: v for k, v in call(base + "/props", timeout=30).items() if k in ("build_info", "n_ctx",
                                                                                              "total_slots", "model_path")}
        except Exception as e:  # the record keeps the reason; the cell does not depend on it
            rec["props"] = f"n/a ({e!r})"
        rec["workload"] = {"ref": args.ref, "prompts": len(rows), "gpu": args.gpu,
                           "decode": "greedy (temperature 0, top_k 1, repeat_penalty 1.0)",
                           "max_new_tokens": args.max_new, "gate_max_new_tokens": args.gate_max_new,
                           "chat_template": "none here: /tokenize of the reference's rendered prompt "
                                            "(add_special false, parse_special true), the ids to /completion"}

        def tokenize(text):
            return [int(x) for x in call(base + "/tokenize", {"content": text, "add_special": False,
                                                               "parse_special": True})["tokens"]]

        def complete(ids, mx):
            return call(base + "/completion", {"prompt": ids, "n_predict": mx, "temperature": 0, "top_k": 1,
                                               "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
                                               "presence_penalty": 0.0, "frequency_penalty": 0.0,
                                               "dry_multiplier": 0.0, "cache_prompt": False, "return_tokens": True,
                                               "seed": 0, "stream": False})

        if not args.no_warmup:
            t0 = time.perf_counter()
            complete(tokenize(rows[0]["prompt"]), 8)
            rec["warmup_s"] = round(time.perf_counter() - t0, 3)
        for r in rows:
            mx = args.gate_max_new if r.get("role") == "gate" else args.max_new
            ids = tokenize(r["prompt"])
            t0 = time.perf_counter()
            j = complete(ids, mx)
            wall_http = time.perf_counter() - t0
            tm = j.get("timings") or {}
            new = [int(x) for x in j.get("tokens") or []]
            st = j.get("stop_type")
            rec.row({"id": r["id"], "role": r.get("role"), "prompt_ids": ids, "prompt_tokens": j.get("tokens_evaluated"),
                     "prompt_form": "ids (/tokenize, add_special false, parse_special true)",
                     "new_ids": new, "new_tokens": j.get("tokens_predicted", len(new)), "text": j.get("content"),
                     "eos": st == "eos", "stop": st, "max_new": mx,
                     "wall_s": ((tm.get("prompt_ms") or 0) + (tm.get("predicted_ms") or 0)) / 1000 if tm else wall_http,
                     "first_token_ms": tm.get("prompt_ms"), "http_s": round(wall_http, 3)})
        rec["summary"] = rec.summarize("llm")
        rec["status"] = "OK"
    except Exception:
        rec.fail()
        code = 1
    finally:
        if srv is not None and srv.poll() is None:
            srv.kill()
            try:
                srv.wait(timeout=30)
            except Exception:
                pass
    rec.write()
    return code


if __name__ == "__main__":
    sys.exit(main())
