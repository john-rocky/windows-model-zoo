#!/usr/bin/env python3
"""The Z6 LLM runners against fake engines on the Mac (no model runs): pc/run_llm_litert.py with a fake
litert_lm_advanced_main (session-mode output on stdout, the benchmark info on stderr, the start-token refusal),
pc/run_llm_llamacpp.py with a fake llama-server (/health, /tokenize, /completion, /props), and the package folder
pc/run_llm_foundry.py registers (hard links, the identity template, plain greedy in genai_config). Checks what each
runner reads back: the answer text (also one of two lines), the token counts, the end-token verdict, the prompt form.
Exit 1 on any unexpected result.

    python3 tests/test_llm_runners.py [--keep]
"""
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable

FAKE_LITERT = r'''#!PYTHON
import sys, json, pathlib
a = dict(x[2:].split("=", 1) for x in sys.argv[1:] if x.startswith("--") and "=" in x)
answers = json.loads(pathlib.Path(__file__).with_name("answers.json").read_text())
text = pathlib.Path(a["input_prompt_file"]).read_text(encoding="utf-8")
if text.startswith("<bos>"):
    sys.stderr.write("E0000 litert_lm_advanced_main.cc:405] Failed to run litert_lm: INVALID_ARGUMENT: Input contains bos control token.\n")
    sys.exit(3)
mx = int(a["max_output_tokens"])
ans, n = answers.get(text, ["(unknown prompt)", mx])
n = min(n, mx)
if a.get("backend") == "gpu":
    sys.stderr.write("INFO: Selected adapter: " + ("Intel(R) Arc(TM) Graphics adapterType=Integrated GPU" if a.get("preferred_device_substr") == "Intel" else "NVIDIA GeForce RTX 5080 Laptop GPU") + "\n")
sys.stderr.write("  Prefill Turns (Total 1 turns):\n    Prefill Turn 1: Processed %d tokens in 1.25s duration.\n" % (len(text.split()) + 1))
sys.stderr.write("  Decode Turns (Total 1 turns):\n    Decode Turn 1: Processed %d tokens in 250.5ms duration.\n" % n)
sys.stdout.write("output: " + ans + "\n")
'''

FAKE_SERVER = r'''#!PYTHON
import sys, json, pathlib, http.server
args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
answers = json.loads(pathlib.Path(__file__).with_name("answers.json").read_text())
print("build: 11721 (fake) with fake compiler", flush=True)
if "-ngl" in args:
    print("llama_model_load_from_file_impl: using device CUDA0 (NVIDIA GeForce RTX 5080 Laptop GPU) (0000:01:00.0) - 15000 MiB free", flush=True)
seen = {}
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def reply(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self.reply({"status": "ok"} if self.path == "/health" else {"build_info": "b11721-fake", "n_ctx": 2048, "total_slots": 1})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            assert body["add_special"] is False and body["parse_special"] is True
            ids = [ord(c) for c in body["content"]][:64]
            seen[tuple(ids)] = body["content"]
            return self.reply({"tokens": ids})
        ids = tuple(body["prompt"])
        assert all(isinstance(x, int) for x in ids) and body["temperature"] == 0 and body["cache_prompt"] is False
        ans, n = answers.get(seen.get(ids, ""), ["(warm-up)", body["n_predict"]])
        n = min(n, body["n_predict"])
        stop = "eos" if n < body["n_predict"] else "limit"
        self.reply({"content": ans, "tokens": list(range(100, 100 + n)), "stop_type": stop, "tokens_predicted": n,
                    "tokens_evaluated": len(ids), "timings": {"prompt_ms": 12.5, "predicted_ms": 40.0}})
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def ref_rows():
    rows = []
    for i in range(1, 21):
        rows.append({"id": f"q{i:02d}", "role": "question", "prompt": f"<bos><|turn>user\nquestion {i}<turn|>\n<|turn>model\n",
                     "prompt_ids": list(range(10 + i))})
    rows.append({"id": "gate", "role": "gate", "prompt": "<bos><|turn>user\nReply with only the number: 1+1=?<turn|>\n<|turn>model\n",
                 "prompt_ids": list(range(25))})
    return rows


def fake(path, text):
    path.write_text(text.replace("#!PYTHON", "#!" + PY))
    path.chmod(0o755)


def main():
    keep = "--keep" in sys.argv[1:]
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="zoo-llm-runners-")).resolve()
    bad = []
    rows = ref_rows()
    (tmp / "ref.json").write_text(json.dumps({"rows": rows}))
    # answers keyed by the text the engine gets: LiteRT-LM without the leading <bos>, llama.cpp the whole string
    lit = {r["prompt"][len("<bos>"):]: ["Paris" if r["id"] == "q01" else ("2" if r["role"] == "gate" else
                                        ("line one\nline two" if r["id"] == "q02" else f"answer {r['id']}")),
                                        48 if r["id"] == "q03" else 3] for r in rows}
    eng = tmp / "litert"
    eng.mkdir()
    (eng / "answers.json").write_text(json.dumps(lit))
    fake(eng / "litert_lm_advanced_main_windows.exe", FAKE_LITERT)
    for adapter in ("", "igpu", "rtx"):
        out = tmp / f"litert{adapter or '_cpu'}" / "run1.json"
        cmd = [PY, str(ROOT / "pc" / "run_llm_litert.py"), "--exe", str(eng / "litert_lm_advanced_main_windows.exe"),
               "--model", str(tmp / "ref.json"), "--backend", "gpu" if adapter else "cpu", "--ref", str(tmp / "ref.json"),
               "--out", str(out)] + (["--adapter", adapter] if adapter else [])
        p = subprocess.run(cmd, capture_output=True, text=True)
        j = json.loads(out.read_text()) if out.exists() else {}
        rs = {r["id"]: r for r in j.get("rows") or []}
        want = (j.get("status") == "OK" and len(rs) == 21 and rs["q01"]["text"] == "Paris"
                and rs["q02"]["text"] == "line one\nline two" and rs["q03"]["eos"] is False and rs["q01"]["eos"] is True
                and rs["gate"]["max_new"] == 16 and rs["gate"]["text"] == "2" and rs["q01"]["new_tokens"] == 3
                and rs["q01"]["prompt_tokens"] == len(rows[0]["prompt"][5:].split()) + 1
                and "without its leading <bos>" in rs["q01"]["prompt_form"])
        print(f"litert {adapter or 'cpu'}: exit {p.returncode}, status {j.get('status')}, rows {len(rs)}, adapters {j.get('adapters_seen')}")
        if not want:
            bad.append(f"litert {adapter or 'cpu'}: {j.get('error') or p.stderr[-600:]} {json.dumps(rs.get('q02'))[:300]}")
    # llama.cpp
    lla = {r["prompt"]: v for r, v in zip(rows, lit.values())}
    eng2 = tmp / "llama"
    eng2.mkdir()
    (eng2 / "answers.json").write_text(json.dumps(lla))
    fake(eng2 / "llama-server.exe", FAKE_SERVER)
    for gpu in (False, True):
        out = tmp / f"llama{'_gpu' if gpu else '_cpu'}" / "run1.json"
        cmd = [PY, str(ROOT / "pc" / "run_llm_llamacpp.py"), "--server", str(eng2 / "llama-server.exe"), "--model",
               str(tmp / "ref.json"), "--ref", str(tmp / "ref.json"), "--out", str(out), "--load-timeout", "30"]
        if gpu:
            cmd += ["--gpu", "--device-prefix", "CUDA"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        j = json.loads(out.read_text()) if out.exists() else {}
        rs = {r["id"]: r for r in j.get("rows") or []}
        want = (j.get("status") == "OK" and len(rs) == 21 and rs["q01"]["text"] == "Paris" and rs["q01"]["eos"] is True
                and rs["q03"]["eos"] is False and rs["q03"]["stop"] == "limit" and rs["gate"]["max_new"] == 16
                and rs["q01"]["prompt_ids"][:5] == [ord(c) for c in rows[0]["prompt"][:5]]
                and rs["q01"]["prompt_tokens"] == len(rs["q01"]["prompt_ids"]) and abs(rs["q01"]["wall_s"] - 0.0525) < 1e-9)
        print(f"llama {'gpu' if gpu else 'cpu'}: exit {p.returncode}, status {j.get('status')}, rows {len(rs)}, devices {j.get('devices')}")
        if not want:
            bad.append(f"llama {'gpu' if gpu else 'cpu'}: {j.get('error') or p.stderr[-600:]}")
    # the folder the Foundry runner registers
    spec = importlib.util.spec_from_file_location("rf", ROOT / "pc" / "run_llm_foundry.py")
    rf = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "pc"))
    spec.loader.exec_module(rf)
    pkg = tmp / "pkg"
    (pkg / "decoder").mkdir(parents=True)
    (pkg / "decoder" / "model.onnx.data").write_bytes(b"0" * 1024)
    (pkg / "chat_template.jinja").write_text("{{ bos_token }}<|turn>user ...")
    (pkg / "tokenizer_config.json").write_text(json.dumps({"chat_template": "x", "bos_token": "<bos>"}))
    (pkg / "inference_model.json").write_text(json.dumps({"Name": "catalog:3", "PromptTemplate": None}))
    (pkg / "genai_config.json").write_text(json.dumps({"model": {}, "search": {"do_sample": True, "top_k": 1}}))
    rec = {}
    ch = rf.link_package(pkg, tmp / "work" / "z6-x", "z6-x:1", rec)
    d = tmp / "work" / "z6-x"
    ok = ((d / "chat_template.jinja").read_text() == rf.IDENTITY
          and json.loads((d / "tokenizer_config.json").read_text())["chat_template"] == rf.IDENTITY
          and json.loads((d / "genai_config.json").read_text())["search"]["do_sample"] is False
          and os.path.samefile(d / "decoder" / "model.onnx.data", pkg / "decoder" / "model.onnx.data")
          and not os.path.samefile(d / "genai_config.json", pkg / "genai_config.json")
          and json.loads((d / "inference_model.json").read_text())["Name"] == "z6-x:1"
          and (pkg / "chat_template.jinja").read_text().startswith("{{ bos_token }}"))
    print(f"foundry package: {sorted(ch)}")
    if not ok:
        bad.append(f"foundry package: {ch} {rec}")
    print(f"unexpected {len(bad)}")
    for b in bad:
        print("UNEXPECTED", b)
    if keep:
        print("kept", tmp)
    else:
        shutil.rmtree(tmp)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
