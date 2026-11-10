"""The inputs and the reference of one vision task model (an untimed step; the venv z1-ort).

    python ref_vision_task.py --model <m> --spec <pc\\z5_spec.json> --fixtures <fixtures> --suites <suites>
        --work <work\\m> --out <refs\\m\\ref.json>

1. files: the ones this step reads (the fp32 form, the upstream graph, the downstream head, the host files), each
   checked against the sha256 the spec pins (the vision zoo Z1's records); a difference stops the step. The other
   forms (the fp16 file, the NPU blob) are checked by the cells that run them (pc/run_vision_task.py), so a model whose
   fp16 file has no public copy still gets its reference.
2. inputs: every input of the model's set (suites\\vision\\images.json under fixtures\\vision, or
   suites\\audio\\kws120.json under fixtures\\kws120) through the app's preprocessing, and for a graph that takes another
   graph's output, that graph in fp32 on the ONNX Runtime CPU EP and the app's host step (pc/vtask.py) ->
   <work>\\inputs.npz (keys n<idx>__<feed name>) and <work>\\inputs.json (ids, feeds, the source rows). The reference
   and every cell read these same arrays.
3. reference: the fp32 file on the ONNX Runtime CPU EP over every input, one run each after one untimed warm-up (the
   MoViNet stream and the downstream head as pc/vtask.py) -> <out stem>.outputs.npz and <out> (the rows). For a
   model whose line reads the source (spec store_src: dewarp's picture, the codec's clip), the npz also holds
   n<idx>__src.
"""
import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vtask as V  # noqa: E402
import zoolib as al  # noqa: E402


def build_inputs(ms, spec, eng, args, rec):
    import numpy as np
    fx, su = pathlib.Path(args.fixtures), pathlib.Path(args.suites)
    entry = ms["inputs"][0]
    kind = entry["kind"]
    up = V.OrtEngine(ms["upstream"]["path"], "cpu") if ms.get("upstream") else None
    arrays, feeds, rows, srcs = {}, None, [], {}
    store = ms.get("store_src")
    if ms["set"] == "images":
        suite = json.loads((su / "vision" / "images.json").read_text(encoding="utf-8"))
        src = V.image_rows(suite, ms["ids"], fx / "vision")
    else:
        suite = json.loads((su / "audio" / "kws120.json").read_text(encoding="utf-8"))
        src = V.audio_rows(suite, ms["ids"], fx / "kws120")
    for idx, (iid, item, row) in enumerate(src):
        if kind in ("image", "mlsd4"):   # the model's own image input (MoViNet: input 0 = the frame)
            name = next(n for k, n in enumerate(eng.in_names) if V.feed_index(n, k) == entry["index"])
            f = {name: V.image_feed(entry, item, eng.in_shapes[name])}
        elif kind in ("detr_b", "sam2_dec"):
            ue = ms["upstream"]["inputs"][0]
            un = up.in_names[0]
            outs_up = up.run({un: V.image_feed(ue, item, up.in_shapes[un])})
            if kind == "detr_b":
                parts = V.detr_b(outs_up, spec["host_files"][ms["host_files"][0]]["path"])
            else:
                parts = V.sam2_dec(outs_up[0], spec["host_files"][ms["host_files"][0]]["path"], entry["point"])
            f = {n: V.by_shape(parts, eng.in_shapes[n]) for n in eng.in_names}
        elif kind == "audio":
            x = V.scale_audio(item, ms["audio_scale"])
            n0 = eng.in_names[0]
            f = {n0: x.reshape(eng.in_shapes[n0])}
            row["peak"] = float(np.abs(item).max())
        elif kind in ("upstream", "dac_zq"):
            x = V.scale_audio(item, ms["audio_scale"])
            un = up.in_names[0]
            outs_up = up.run({un: x.reshape(up.in_shapes[un])})
            parts = [outs_up[0]] if kind == "upstream" else V.dac_zq(outs_up[0], spec["host_files"][ms["host_files"][0]]["path"])
            f = {n: V.by_shape(parts, eng.in_shapes[n]) for n in eng.in_names}
            row["peak"] = float(np.abs(item).max())
        else:
            raise ValueError(f"input kind {kind}")
        if store == "image":   # the original picture (uint8 RGB): the Mac's PSNR of the app's unwarped picture
            srcs[V.key(idx, "src")] = np.asarray(item, dtype=np.uint8)
        elif store == "audio":   # the scaled clip the model's graph chain read: the Mac's SI-SDR target
            srcs[V.key(idx, "src")] = V.scale_audio(item, ms["audio_scale"])
        if feeds is None:
            feeds = sorted(f)
        for n, a in f.items():
            arrays[V.key(idx, n)] = np.ascontiguousarray(a, dtype=np.float32)
        rows.append(dict(row, id=iid))
    work = pathlib.Path(args.work)
    info = V.save_npz(work / "inputs.npz", arrays)
    meta = {"model": args.model, "set": ms["set"], "ids": [r["id"] for r in rows], "feeds": feeds,
            "shapes": {n: list(arrays[V.key(0, n)].shape) for n in feeds}, "rows": rows,
            "upstream": ms.get("upstream", {}).get("model"), "npz": info}
    (work / "inputs.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    rec["inputs"] = {"path": info["path"], "n": len(rows), "sha256": info["sha256"], "bytes": info["bytes"],
                     "feeds": meta["shapes"], "set": ms["set"]}
    if up is not None:
        rec["upstream"] = {"model": ms["upstream"]["model"], "providers": up.providers, "load_s": round(up.load_s, 3)}
    return srcs


def main():
    al.utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--suites", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rec = al.Record("ref_vision_task.py", args)
    al.banner()
    code = 0
    try:
        rec["versions"] = al.versions(["onnxruntime", "onnxruntime-ep-openvino", "numpy", "pillow"])
        spec, ms = V.load_spec(args.spec, args.model)
        files = [V.check_file(ms["forms"]["fp32"], f"{args.model} form fp32")]
        if ms.get("upstream"):
            files.append(V.check_file(ms["upstream"], f"upstream {ms['upstream']['model']}"))
        if ms.get("then"):
            files.append(V.check_file(ms["then"], f"downstream {ms['then']['model']}"))
        for h in ms.get("host_files") or []:
            files.append(V.check_file(spec["host_files"][h], f"host file {h}"))
        rec["files"] = files
        rec["model"] = {"path": ms["forms"]["fp32"]["path"], "sha256": ms["forms"]["fp32"]["sha256"]}
        pathlib.Path(args.work).mkdir(parents=True, exist_ok=True)
        eng = V.OrtEngine(ms["forms"]["fp32"]["path"], "cpu")
        rec["load_s"] = round(eng.load_s, 3)
        rec["session_providers"] = eng.providers
        srcs = build_inputs(ms, spec, eng, args, rec)
        then = V.OrtEngine(ms["then"]["path"], "cpu") if ms.get("then") else None
        rec["workload"] = {"inputs": rec["inputs"]["n"], "set": ms["set"], "task": ms["task"],
                           "frames": (ms.get("stream") or {}).get("frames"),
                           "downstream": (ms.get("then") or {}).get("model"), "device": "cpu", "dtype": "float32",
                           "timing": "one run per input, wall clock, host numpy in and out (untimed step)"}
        outs = V.run_set(eng, ms, pathlib.Path(args.work) / "inputs.npz", rec, then=then)
        outs.update(srcs)   # n<idx>__src: the source picture / clip (dewarp, the codec), not a model output
        stem = pathlib.Path(args.out)
        rec["outputs"] = V.save_npz(stem.with_name(stem.stem + ".outputs.npz"), outs)
        rec["summary"] = rec.summarize("vision")
        rec["status"] = "OK"
    except Exception:
        rec.fail()
        code = 1
    rec.write()
    os._exit(code)


if __name__ == "__main__":
    main()
