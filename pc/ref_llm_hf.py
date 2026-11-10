"""The LLM reference (an untimed step; the venv `ref`): transformers fp32 on the CPU, plain greedy, one prompt at a time.

    python ref_llm_hf.py --model-dir <checkpoint folder> --suite <suites\\llm\\qa20.jsonl> --out ref.json
        [--max-new 48] [--gate-max-new 16] [--dtype float32|bfloat16]

Each prompt of the suite is one user turn wrapped by the model's own chat template:
tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
enable_thinking=False). The rendered string goes into the record: a cell's runner passes it as it is (no template of its
own runtime), tokenized with add_special_tokens=False. Decode = plain greedy: do_sample False, num_beams 1,
repetition_penalty 1.0 (a checkpoint's generation_config.json may set a penalty for sampling; both sides drop it),
max_new_tokens 48 (the gate prompt 16); the end tokens = the generation config's eos_token_id. A row: the rendered
prompt, its ids, the new ids, the text (skip_special_tokens) and whether the last new token is an end token.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zoolib as al  # noqa: E402


def main():
    al.utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--gate-max-new", type=int, default=16)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    rec = al.Record("ref_llm_hf.py", args)
    al.banner()
    code = 0
    try:
        rec["versions"] = al.versions(["torch", "transformers", "tokenizers", "numpy"])
        import torch
        import transformers
        if args.threads:
            torch.set_num_threads(args.threads)
        rec["torch_threads"] = torch.get_num_threads()
        suite = al.read_jsonl(args.suite)
        t = time.perf_counter()
        tok = transformers.AutoTokenizer.from_pretrained(args.model_dir)
        model = transformers.AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=getattr(torch, args.dtype)).eval()
        rec["load_s"] = round(time.perf_counter() - t, 3)
        eos = model.generation_config.eos_token_id
        eos_ids = [int(e) for e in (eos if isinstance(eos, (list, tuple)) else [eos]) if e is not None]
        pad = tok.pad_token_id if tok.pad_token_id is not None else eos_ids[0]
        rec["workload"] = {"suite": os.path.basename(args.suite), "prompts": len(suite), "dtype": args.dtype,
                           "device": "cpu", "decode": "greedy (do_sample False, num_beams 1, repetition_penalty 1.0)",
                           "max_new_tokens": args.max_new, "gate_max_new_tokens": args.gate_max_new,
                           "chat_template": "apply_chat_template(add_generation_prompt=True, enable_thinking=False)",
                           "eos_token_ids": eos_ids, "checkpoint_generation_config": model.generation_config.to_dict()}
        for q in suite:
            rendered = tok.apply_chat_template([{"role": "user", "content": q["text"]}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
            ids = tok(rendered, add_special_tokens=False, return_tensors="pt").input_ids
            mx = args.gate_max_new if q.get("role") == "gate" else args.max_new
            # a config of its own, not the checkpoint's (whose sampling keys would ride along): plain greedy
            gc = transformers.GenerationConfig(do_sample=False, num_beams=1, repetition_penalty=1.0, max_new_tokens=mx,
                                               eos_token_id=eos_ids, pad_token_id=pad)
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=gc)
            wall = time.perf_counter() - t0
            new = [int(v) for v in out[0, ids.shape[1]:].tolist()]
            rec.row({"id": q["id"], "role": q.get("role"), "question": q["text"], "prompt": rendered,
                     "prompt_ids": [int(v) for v in ids[0].tolist()], "new_ids": new, "new_tokens": len(new),
                     "text": tok.decode(new, skip_special_tokens=True), "eos": bool(new) and new[-1] in eos_ids,
                     "max_new": mx, "wall_s": wall})
        rec["summary"] = rec.summarize("llm")
        rec["status"] = "OK"
    except Exception:
        rec.fail()
        code = 1
    rec.write()
    os._exit(code)


if __name__ == "__main__":
    main()
