"""One engine process of an OpenVINO GenAI LLM cell: LLMPipeline on one device, the reference's rendered prompts (the
suite wrapped by the model's own chat template in the reference step) as they are, after one untimed warm-up prompt;
one JSON.

    python run_llm_ovg.py --model-dir <IR folder> --ref <ref.json> --device CPU|GPU|NPU --out run.json
        [--max-new 48] [--gate-max-new 16] [--cache-dir <dir>] [--pipeline auto|llm|vlm]

LLMPipeline (a text IR): the IR's tokenizer encodes the rendered string with add_special_tokens=False, and generate()
gets the ids (no chat template of this runtime: GenerationConfig.apply_chat_template = False). A multimodal IR
(openvino_language_model.xml, Gemma 4: LLMPipeline loads it and then refuses it at generate, "5 inputs", the Z2 lane)
runs in VLMPipeline (--pipeline auto picks it by that file), which takes a string only, no image: the warm-up prompt
decides the string's form by the pipeline's own input token count (perf_metrics) against the reference's ids: the
same count = the rendered string as it is; one over and a leading <bos> = the pipeline adds a start token, so that
<bos> text is left out; any other count = an error (PLAN.md section Z6, the Z6 lane r3: the rendered string without
its <bos> gave one token under the reference's on all 21 prompts). The row carries the pipeline's input token count
instead of ids, and as an instrument whether this IR's tokenizer encodes the string to the reference's ids. Decode = the reference's: plain greedy
(do_sample False, num_beams 1, repetition_penalty 1.0), max_new_tokens 48 (the gate prompt 16). A row: the prompt's
ids (or token count), the new ids (LLMPipeline), the text, and whether the run ended on an end token (the last new
token is one of the config's stop / eos ids, or fewer new tokens than the maximum). Timing per prompt: generate()
wall clock; first token = perf_metrics.get_ttft().
"""
import argparse
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zoolib as al  # noqa: E402


def metric(pm, name):
    try:
        v = getattr(pm, name)()
        return {"mean": float(v.mean), "std": float(v.std)} if hasattr(v, "mean") else float(v)
    except Exception as e:  # not every metric exists for every pipeline
        return f"n/a ({type(e).__name__})"


def main():
    al.utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--gate-max-new", type=int, default=16)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--pipeline", default="auto", choices=["auto", "llm", "vlm"])
    args = ap.parse_args()
    rec = al.Record("run_llm_ovg.py", args)
    al.banner()
    code = 0
    try:
        rec["versions"] = al.versions(["openvino-genai", "openvino", "openvino-tokenizers", "numpy"])
        import openvino_genai as ov_genai
        ref = json.loads(pathlib.Path(args.ref).read_text(encoding="utf-8"))
        rows = [r for r in ref["rows"] if r.get("prompt")]
        props = {}
        if args.cache_dir:
            os.makedirs(args.cache_dir, exist_ok=True)
            props["CACHE_DIR"] = args.cache_dir
        kind = args.pipeline
        if kind == "auto":
            kind = "vlm" if (pathlib.Path(args.model_dir) / "openvino_language_model.xml").is_file() else "llm"
        t = time.perf_counter()
        if kind == "llm":
            pipe = ov_genai.LLMPipeline(args.model_dir, args.device, **props)
        else:
            pipe = ov_genai.VLMPipeline(args.model_dir, args.device, **props)
        rec["load_s"] = round(time.perf_counter() - t, 3)
        rec["pipeline"] = kind
        tok = pipe.get_tokenizer()
        base = pipe.get_generation_config()
        stop = sorted(({int(x) for x in (getattr(base, "stop_token_ids", None) or [])} | {int(base.eos_token_id)}) - {-1})
        rec["workload"] = {"ref": args.ref, "prompts": len(rows), "device": args.device, "pipeline": kind,
                           "decode": "greedy (do_sample False, num_beams 1, repetition_penalty 1.0)",
                           "max_new_tokens": args.max_new, "gate_max_new_tokens": args.gate_max_new,
                           "chat_template": ("none here: the reference's rendered prompt, add_special_tokens=False"
                                             if kind == "llm" else
                                             "none here: the reference's rendered prompt in the form the warm-up "
                                             "decides (vlm_form), apply_chat_template False"),
                           "stop_token_ids": stop}

        def config(mx):
            c = pipe.get_generation_config()
            c.max_new_tokens = mx
            c.do_sample = False
            c.num_beams = 1
            c.repetition_penalty = 1.0
            if not hasattr(c, "apply_chat_template"):
                raise RuntimeError("GenerationConfig has no apply_chat_template: the runtime would wrap the prompt again")
            c.apply_chat_template = False
            return c

        def encode(text):
            try:
                return tok.encode(text, add_special_tokens=False)
            except TypeError:   # an older signature: recorded, the ids are compared with the reference's anyway
                rec["encode_note"] = "Tokenizer.encode without add_special_tokens"
                return tok.encode(text)

        def decode(ids):
            try:
                return tok.decode(ids, skip_special_tokens=True)
            except TypeError:
                return tok.decode(ids)

        form = {"strip_bos": False}

        def vlm_text(text):
            return text[len("<bos>"):] if form["strip_bos"] and text.startswith("<bos>") else text

        def run(text, mx):
            if kind == "vlm":
                t0 = time.perf_counter()
                res = pipe.generate(vlm_text(text), generation_config=config(mx))
                return None, res, time.perf_counter() - t0
            enc = encode(text)
            t0 = time.perf_counter()
            res = pipe.generate(enc, config(mx))
            wall = time.perf_counter() - t0
            return enc, res, wall

        def tok_ids(text, special):
            try:
                return [int(v) for v in tok.encode(text, add_special_tokens=special).input_ids.data.reshape(-1).tolist()]
            except Exception as e:  # the instrument only; the verdict does not use it
                return f"n/a ({e!r})"

        if kind == "vlm":   # the warm-up decides the string's form (the docstring); it runs even with --no-warmup
            r0 = rows[0]
            n_ref = len(r0.get("prompt_ids") or [])
            _, res, ws = run(r0["prompt"], 8)
            n0 = int(res.perf_metrics.get_num_input_tokens())
            if n0 == n_ref:
                vf = "the rendered string as it is (the warm-up's input count = the reference's)"
            elif n0 == n_ref + 1 and r0["prompt"].startswith("<bos>"):
                form["strip_bos"] = True
                _, res, ws2 = run(r0["prompt"], 8)
                n1 = int(res.perf_metrics.get_num_input_tokens())
                if n1 != n_ref:
                    raise RuntimeError(f"warm-up: {n0} input tokens as is, {n1} without the leading <bos>, "
                                       f"the reference's {n_ref}")
                ws += ws2
                vf = "the rendered string without its leading <bos> (the warm-up's count as is was one over)"
            else:
                raise RuntimeError(f"warm-up: the pipeline's input count {n0} for the rendered string, the reference's "
                                   f"{n_ref}: the pipeline changes the prompt")
            rec["vlm_form"] = {"warmup_input_tokens": n0, "reference_tokens": n_ref, "form": vf}
            rec["warmup_s"] = round(ws, 3)
        elif not args.no_warmup:
            _, _, ws = run(rows[0]["prompt"], 8)
            rec["warmup_s"] = round(ws, 3)
        for r in rows:
            mx = args.gate_max_new if r.get("role") == "gate" else args.max_new
            enc, res, wall = run(r["prompt"], mx)
            pm = res.perf_metrics
            ttft = metric(pm, "get_ttft")
            row = {"id": r["id"], "role": r.get("role")}
            if kind == "llm":
                new = [int(v) for v in list(res.tokens[0])]
                last_stop = bool(new) and new[-1] in stop
                n_new = len(new)
                row.update({"prompt_ids": [int(v) for v in enc.input_ids.data.reshape(-1).tolist()],
                            "new_ids": new, "text": decode(new), "prompt_form": "ids (encode, add_special_tokens=False)"})
            else:
                n_new = int(pm.get_num_generated_tokens())
                last_stop = False   # VLMPipeline returns text, not ids: an end token is read as fewer than max
                sent = vlm_text(r["prompt"])
                ref_ids = [int(v) for v in r.get("prompt_ids") or []]
                row.update({"prompt_tokens": int(pm.get_num_input_tokens()), "text": res.texts[0],
                            "prompt_form": rec["vlm_form"]["form"],
                            "tokenizer_ids_same": {f"add_special_tokens={sp}": tok_ids(sent, sp) == ref_ids
                                                   for sp in (True, False)}})
            row.update({"new_tokens": n_new, "eos": bool(last_stop or n_new < mx),
                        "stop": "end token last" if last_stop else ("fewer than max" if n_new < mx else "max_new_tokens"),
                        "max_new": mx, "wall_s": wall,
                        "first_token_ms": ttft["mean"] if isinstance(ttft, dict) else None,
                        "tpot_ms": metric(pm, "get_tpot")})
            rec.row(row)
        rec["summary"] = rec.summarize("llm")
        rec["status"] = "OK"
    except Exception:
        rec.fail()
        code = 1
    rec.write()
    os._exit(code)


if __name__ == "__main__":
    main()
