"""One engine process of a vision task cell: one form of the model (the file the vision zoo Z1 measured) on one cell,
every input of the model's inputs.npz (the reference step's arrays) one run each after one untimed warm-up; one JSON
and the outputs.

    python run_vision_task.py --model <m> --spec <pc\\z5_spec.json> --work <work\\m> --form fp32|fp16|blob
        --engine ort --ep cuda|openvino [--ov-device NPU|GPU.0] --out run.json
    python run_vision_task.py ... --form blob --engine ovblob --out run.json

ort: ONNX Runtime with the CUDA EP (cuda32: fp32, cuda16: Z1's fp16 copy) or the OpenVINO EP plugin through
SessionOptions.add_provider_for_devices (ovnpu: NPU, ovgpu: GPU.0), the ONNX compiled at load. ovblob: OpenVINO
without ONNX Runtime, the NPU blob Z1 compiled and exported (ovnpu_pre), import_model in this process. Each file is
checked against the sha256 the spec pins before use. The MoViNet stream and the downstream head run as pc/vtask.py
(the head in fp32 on the ONNX Runtime CPU EP, as the reference's). Outputs: <out stem>.outputs.npz.
"""
import argparse
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vtask as V  # noqa: E402
import zoolib as al  # noqa: E402


def main():
    al.utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--form", required=True, choices=["fp32", "fp16", "blob"])
    ap.add_argument("--engine", required=True, choices=["ort", "ovblob"])
    ap.add_argument("--ep", choices=["cpu", "cuda", "openvino"])
    ap.add_argument("--ov-device", default="NPU")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rec = al.Record("run_vision_task.py", args)
    al.banner()
    code = 0
    try:
        rec["versions"] = al.versions(["onnxruntime", "onnxruntime-gpu", "onnxruntime-ep-openvino", "openvino", "numpy",
                                       "nvidia-cudnn-cu13", "nvidia-cublas", "nvidia-cuda-runtime"])
        spec, ms = V.load_spec(args.spec, args.model)
        form = ms["forms"][args.form]
        rec["model"] = V.check_file(form, f"{args.model} form {args.form}")
        if ms.get("then"):
            rec["downstream_file"] = V.check_file(ms["then"], f"downstream {ms['then']['model']}")
        inputs = pathlib.Path(args.work) / "inputs.npz"
        rec["inputs"] = {"path": str(inputs), "sha256": V.sha256_file(inputs), "bytes": inputs.stat().st_size}
        dev = {}
        if args.engine == "ovblob":
            eng = V.OvBlobEngine(form["path"], "NPU", dev)
        else:
            eng = V.OrtEngine(form["path"], args.ep, args.ov_device, None, dev)
        dev.pop("_registered", None)
        rec["load_s"] = round(eng.load_s, 3)
        rec["device"] = dev
        rec["session_providers"] = eng.providers
        then = V.OrtEngine(ms["then"]["path"], "cpu") if ms.get("then") else None
        rec["workload"] = {"set": ms["set"], "task": ms["task"], "form": args.form, "engine": args.engine, "ep": args.ep,
                           "ov_device": args.ov_device if args.engine == "ort" and args.ep == "openvino" else None,
                           "frames": (ms.get("stream") or {}).get("frames"),
                           "downstream": (ms.get("then") or {}).get("model"),
                           "timing": "one run per input, wall clock, host numpy in and out (first pass: untimed)"}
        outs = V.run_set(eng, ms, inputs, rec, then=then)
        rec["inputs"]["n"] = len({r.get("input", r["id"]) for r in rec["rows"]})
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
