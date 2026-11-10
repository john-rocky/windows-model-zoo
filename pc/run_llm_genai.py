"""One engine process of an ONNX Runtime GenAI LLM cell: a GenAI package folder on one execution provider, the
reference's rendered prompts (the suite wrapped by the model's own chat template in the reference step) as they are,
after one untimed warm-up prompt; one JSON.

    python run_llm_genai.py --model-dir <package> --ep cuda|webgpu|openvino|cpu|follow_config [--ov-device GPU|CPU|NPU]
        [--ep-option key=value ...] --ref <ref.json> --out run.json [--max-new 48] [--gate-max-new 16]

EP: the Z2 lane's make_model (windows-llm-zoo pc/run_genai.py): cuda / cpu / openvino / webgpu replace the folder's
providers for every session; follow_config keeps the folder's genai_config (a WebGPU overlay names its adapterIndex
there). Each prompt: the package tokenizer's encode of the rendered string; when that put a start token in front of the
one the string already has (two start tokens at the head), the first is dropped (PLAN.md section Z6), and the ids go to
append_tokens (no chat template of this runtime). Decode = the reference's: plain greedy (do_sample False, num_beams 1,
repetition_penalty 1.0), at most 48 new tokens (the gate prompt 16), stopped by the package's eos ids. A row: the
prompt's ids, the new ids, the text (the new ids without the end token), and whether the last new token is an end
token. Timing per prompt: append_tokens + the generate loop; first token = append_tokens + the first step.
"""
import os

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")  # before any onnxruntime import

import argparse  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zoolib as al  # noqa: E402


def make_model(og, args, rec):
    config = og.Config(args.model_dir)
    names = {"openvino": "OpenVINO", "webgpu": "WebGPU", "cuda": "cuda"}
    if args.ep == "openvino":
        import onnxruntime_ep_openvino as ep
        og.register_execution_provider_library(ep.get_ep_name(), ep.get_library_path())
        config.clear_providers()
        config.append_provider("OpenVINO")
        config.set_provider_option("OpenVINO", "device_type", args.ov_device)
    elif args.ep == "webgpu":
        import onnxruntime_ep_webgpu as ep
        og.register_execution_provider_library(ep.get_ep_name(), ep.get_library_path())
        config.clear_providers()
        config.append_provider("WebGPU")
    elif args.ep == "cuda":
        config.clear_providers()
        config.append_provider("cuda")
    elif args.ep == "cpu":
        config.clear_providers()
    elif args.ep == "follow_config":
        for mod in ("onnxruntime_ep_openvino", "onnxruntime_ep_webgpu"):
            try:
                ep = __import__(mod)
                og.register_execution_provider_library(ep.get_ep_name(), ep.get_library_path())
            except ImportError:
                pass
    for item in args.ep_option:
        key, value = item.split("=", 1)
        config.set_provider_option(names.get(args.ep, args.ep), key, value)
    return og.Model(config)


def main():
    al.utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ep", required=True, choices=["cuda", "webgpu", "openvino", "cpu", "follow_config"])
    ap.add_argument("--ov-device", default="CPU")
    ap.add_argument("--ep-option", action="append", default=[])
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--gate-max-new", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()
    rec = al.Record("run_llm_genai.py", args)
    al.banner()
    code = 0
    try:
        rec["cuda_dll_dirs"] = al.add_cuda_dirs()   # no-op outside the CUDA venv
        import onnxruntime_genai as og
        rec["versions"] = al.versions(["onnxruntime-genai", "onnxruntime-genai-cuda", "onnxruntime", "onnxruntime-gpu",
                                       "onnxruntime-ep-openvino", "onnxruntime-ep-webgpu", "numpy"])
        ref = json.loads(pathlib.Path(args.ref).read_text(encoding="utf-8"))
        rows = [r for r in ref["rows"] if r.get("prompt")]
        cfg = json.loads((pathlib.Path(args.model_dir) / "genai_config.json").read_text(encoding="utf-8"))
        eos = cfg["model"]["eos_token_id"]
        eos = sorted({int(e) for e in (eos if isinstance(eos, list) else [eos])})
        bos = cfg["model"].get("bos_token_id")
        t = time.perf_counter()
        model = make_model(og, args, rec)
        rec["load_s"] = round(time.perf_counter() - t, 3)
        tokenizer = og.Tokenizer(model)
        rec["device_type"] = getattr(model, "device_type", None)
        rec["genai_config_providers"] = {k: (v.get("session_options") or {}).get("provider_options")
                                         for k, v in cfg["model"].items() if isinstance(v, dict) and "session_options" in v}
        rec["genai_config_search"] = cfg.get("search")
        rec["workload"] = {"ref": args.ref, "prompts": len(rows), "ep": args.ep, "ov_device": args.ov_device,
                           "ep_option": args.ep_option,
                           "decode": "greedy (do_sample False, num_beams 1, repetition_penalty 1.0)",
                           "max_new_tokens": args.max_new, "gate_max_new_tokens": args.gate_max_new,
                           "chat_template": "none here: the reference's rendered prompt, the package tokenizer's encode",
                           "eos_token_ids": eos, "bos_token_id": bos}

        def encode(text):
            ids = [int(i) for i in tokenizer.encode(text)]
            if bos is not None and len(ids) >= 2 and ids[0] == ids[1] == int(bos):
                return ids[1:], "encode, the start token it added in front of the string's own dropped"
            return ids, "encode"

        def generate(ids, mx):
            params = og.GeneratorParams(model)
            params.set_search_options(max_length=args.max_length, do_sample=False, num_beams=1, repetition_penalty=1.0)
            g = og.Generator(model, params)
            t0 = time.perf_counter()
            g.append_tokens(ids)
            t1 = time.perf_counter()
            new, first = [], None
            while not g.is_done() and len(new) < mx:
                g.generate_next_token()
                new.append(int(g.get_next_tokens()[0]))
                if first is None:
                    first = time.perf_counter()
            t2 = time.perf_counter()
            del g, params
            return new, t2 - t0, ((first or t2) - t0) * 1000, (t1 - t0) * 1000

        if not args.no_warmup:
            ids, _ = encode(rows[0]["prompt"])
            _, ws, _, _ = generate(ids, 8)
            rec["warmup_s"] = round(ws, 3)
        for r in rows:
            mx = args.gate_max_new if r.get("role") == "gate" else args.max_new
            ids, how = encode(r["prompt"])
            new, wall, first_ms, prefill_ms = generate(ids, mx)
            hit = bool(new) and new[-1] in eos
            rec.row({"id": r["id"], "role": r.get("role"), "prompt_ids": ids, "prompt_form": how,
                     "new_ids": new, "new_tokens": len(new), "text": tokenizer.decode(new[:-1] if hit else new),
                     "eos": hit, "stop": "end token last" if hit else "max_new_tokens", "max_new": mx,
                     "wall_s": wall, "first_token_ms": first_ms, "prefill_ms": prefill_ms})
        rec["summary"] = rec.summarize("llm")
        rec["status"] = "OK"
    except Exception:
        rec.fail()
        code = 1
    rec.write()
    os._exit(code)   # the WebGPU plugin 0.5.0 on the RTX kept the process alive after main() (Z2, 2026-11-06)


if __name__ == "__main__":
    main()
