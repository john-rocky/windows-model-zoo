"""One engine of ask_downloads.py: loads one model, then answers prompts read as JSON lines on stdin.

    python worker.py --kind hf    --model-dir <checkpoint folder>
    python worker.py --kind ovg   --model-dir <OpenVINO IR folder> --device NPU|CPU|GPU [--cache-dir <dir>]
    python worker.py --kind llama --server <llama-server.exe> --model <file.gguf>

Each kind runs a prompt the way the zoo's qa20 runner of that runtime does (pc/ref_llm_hf.py, pc/run_llm_ovg.py,
pc/run_llm_llamacpp.py), so a card shows what the zoo's record would hold for the same question: the prompt is the
string the original model's chat template rendered (only the hf worker renders: apply_chat_template with
add_generation_prompt=True and enable_thinking=False), tokenized without special tokens, no chat template of the
runtime, plain greedy decode (no sampling, one beam, repetition penalty 1.0), at most 48 new tokens.

stdin, one JSON object per line:
    {"op": "render", "id": n, "text": "<question>"}   hf only: the rendered prompt
    {"op": "warmup", "id": n, "prompt": "<rendered>"}  one short untimed run (8 new tokens)
    {"op": "ask", "id": n, "prompt": "<rendered>"}     the answer ("max_new": k to stop sooner: the zoo's gate uses 16)
    {"op": "quit"}
stdout, one JSON object per line: {"event": "ready", ...} once loaded, then one reply per request
({"event": "rendered" | "warm" | "answer" | "error", "id": n, ...}). stderr is free text.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

MAX_NEW = 48


def out(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def versions(names):
    from importlib import metadata
    v = {"python": sys.version.split()[0]}
    for n in names:
        try:
            v[n] = metadata.version(n)
        except Exception:
            v[n] = None
    return v


class HF:
    """The original model: transformers on the CPU in fp32 (pc/ref_llm_hf.py)."""

    def __init__(self, a):
        import torch
        import transformers
        self.torch, self.tf = torch, transformers
        self.tok = transformers.AutoTokenizer.from_pretrained(a.model_dir)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.float32).eval()
        eos = self.model.generation_config.eos_token_id
        self.eos = [int(e) for e in (eos if isinstance(eos, (list, tuple)) else [eos]) if e is not None]
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos[0]
        self.info = {"versions": versions(["torch", "transformers", "tokenizers", "numpy"]),
                     "torch_threads": torch.get_num_threads(), "eos_token_ids": self.eos}

    def render(self, text):
        return self.tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)

    def run(self, prompt, mx):
        torch = self.torch
        ids = self.tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        gc = self.tf.GenerationConfig(do_sample=False, num_beams=1, repetition_penalty=1.0, max_new_tokens=mx,
                                      eos_token_id=self.eos, pad_token_id=self.pad)
        with torch.inference_mode():
            o = self.model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=gc)
        new = [int(v) for v in o[0, ids.shape[1]:].tolist()]
        return [int(v) for v in ids[0].tolist()], new, self.tok.decode(new, skip_special_tokens=True)


class OVG:
    """A download as an OpenVINO IR: OpenVINO GenAI's LLMPipeline on one device (pc/run_llm_ovg.py)."""

    def __init__(self, a):
        import openvino_genai as ov_genai
        props = {}
        if a.cache_dir:
            os.makedirs(a.cache_dir, exist_ok=True)
            props["CACHE_DIR"] = a.cache_dir
        self.pipe = ov_genai.LLMPipeline(a.model_dir, a.device, **props)
        self.tok = self.pipe.get_tokenizer()
        self.info = {"versions": versions(["openvino-genai", "openvino", "openvino-tokenizers", "numpy"]),
                     "device": a.device}
        try:
            import openvino as ov
            self.info["device_name"] = ov.Core().get_property(a.device, "FULL_DEVICE_NAME")
        except Exception as e:
            self.info["device_name"] = f"n/a ({type(e).__name__})"

    def config(self, mx):
        c = self.pipe.get_generation_config()
        c.max_new_tokens = mx
        c.do_sample = False
        c.num_beams = 1
        c.repetition_penalty = 1.0
        c.apply_chat_template = False
        return c

    def run(self, prompt, mx):
        enc = self.tok.encode(prompt, add_special_tokens=False)
        res = self.pipe.generate(enc, self.config(mx))
        new = [int(v) for v in list(res.tokens[0])]
        try:
            text = self.tok.decode(new, skip_special_tokens=True)
        except TypeError:
            text = self.tok.decode(new)
        return [int(v) for v in enc.input_ids.data.reshape(-1).tolist()], new, text


class Llama:
    """A download as a GGUF file: llama-server of a llama.cpp release on the CPU (pc/run_llm_llamacpp.py)."""

    def __init__(self, a):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        self.base = f"http://127.0.0.1:{port}"
        log = open(os.path.join(os.path.dirname(os.path.abspath(a.model)), "llama-server.log"), "wb")
        flags = 0x08000000 if os.name == "nt" else 0   # CREATE_NO_WINDOW
        if os.name == "nt":
            # a DLL of llama-server that Windows blocks or cannot find: an exit code, not a system dialog that the
            # server would wait on unseen (the error mode is inherited: SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX)
            import ctypes
            ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
        self.srv = subprocess.Popen([a.server, "-m", a.model, "-c", "2048", "-np", "1", "--host", "127.0.0.1",
                                     "--port", str(port), "--no-webui"], stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, creationflags=flags)
        t = time.perf_counter()
        while True:
            if self.srv.poll() is not None:
                code = self.srv.returncode & 0xFFFFFFFF
                why = (": Windows blocked one of its files (an app control policy such as Smart App Control)"
                       if code == 0xC0E90002 else "")
                raise RuntimeError(f"llama-server exited 0x{code:08X} while loading{why}")
            try:
                if self.call("/health", timeout=5).get("status") == "ok":
                    break
            except (urllib.error.URLError, ConnectionError, OSError, ValueError):
                pass
            if time.perf_counter() - t > 300:
                raise RuntimeError("llama-server not ready in 300 s")
            time.sleep(0.3)
        self.info = {"versions": versions([]), "server": a.server}
        try:
            self.info["build_info"] = self.call("/props", timeout=30).get("build_info")
        except Exception as e:
            self.info["build_info"] = f"n/a ({type(e).__name__})"

    def call(self, path, body=None, timeout=600):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, headers={"Content-Type": "application/json"},
                                     method="POST" if body is not None else "GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def run(self, prompt, mx):
        ids = [int(x) for x in self.call("/tokenize", {"content": prompt, "add_special": False,
                                                        "parse_special": True})["tokens"]]
        j = self.call("/completion", {"prompt": ids, "n_predict": mx, "temperature": 0, "top_k": 1, "top_p": 1.0,
                                      "min_p": 0.0, "repeat_penalty": 1.0, "presence_penalty": 0.0,
                                      "frequency_penalty": 0.0, "dry_multiplier": 0.0, "cache_prompt": False,
                                      "return_tokens": True, "seed": 0, "stream": False})
        return ids, [int(x) for x in j.get("tokens") or []], j.get("content") or ""

    def close(self):
        if self.srv.poll() is None:
            self.srv.kill()
            try:
                self.srv.wait(timeout=20)
            except Exception:
                pass


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["hf", "ovg", "llama"])
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--server", default="")
    ap.add_argument("--model", default="")
    a = ap.parse_args()
    t = time.perf_counter()
    try:
        eng = {"hf": HF, "ovg": OVG, "llama": Llama}[a.kind](a)
    except Exception as e:
        out({"event": "error", "id": None, "error": f"load: {type(e).__name__}: {e}",
             "trace": traceback.format_exc()[-2000:]})
        return 1
    out({"event": "ready", "kind": a.kind, "load_s": round(time.perf_counter() - t, 3), "info": eng.info})
    code = 0
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except ValueError:
                out({"event": "error", "id": None, "error": "not JSON: " + line[:200]})
                continue
            op, rid = req.get("op"), req.get("id")
            if op == "quit":
                break
            try:
                if op == "render":
                    if not hasattr(eng, "render"):
                        raise RuntimeError("only the hf worker renders")
                    out({"event": "rendered", "id": rid, "prompt": eng.render(req["text"])})
                elif op in ("warmup", "ask"):
                    t0 = time.perf_counter()
                    mx = 8 if op == "warmup" else int(req.get("max_new") or MAX_NEW)
                    ids, new, text = eng.run(req["prompt"], mx)
                    s = round(time.perf_counter() - t0, 3)
                    if op == "warmup":
                        out({"event": "warm", "id": rid, "s": s})
                    else:
                        out({"event": "answer", "id": rid, "text": text, "new_ids": new, "prompt_ids": ids, "s": s})
                else:
                    raise RuntimeError(f"unknown op {op!r}")
            except Exception as e:
                out({"event": "error", "id": rid, "error": f"{type(e).__name__}: {e}",
                     "trace": traceback.format_exc()[-2000:]})
    finally:
        if hasattr(eng, "close"):
            eng.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
