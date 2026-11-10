"""One engine process of an ONNX Runtime vision cell: one ONNX model on one execution provider, every input of the
model's inputs.npz (the reference step's tensors) one at a time after one untimed warm-up run; one JSON.

    python run_vision_ort.py --model <model.onnx> --inputs <inputs.npz> --ep cpu|cuda|openvino|webgpu
        [--ov-device NPU|GPU] [--ep-option k=v ...] --out run.json

The OpenVINO EP is the plugin EP (onnxruntime-ep-openvino) through SessionOptions.add_provider_for_devices, the ONNX
as it is (the device compiles it at load). Timing per input: session.run wall clock with host numpy in and out.
Outputs: the rows (top-5 classes and probabilities) and <out stem>.logits.npy ([N, classes], float32).
"""
import argparse
import os
import pathlib
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zoolib as al  # noqa: E402


def main():
    al.utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--ep", required=True, choices=["cpu", "cuda", "openvino", "webgpu"])
    ap.add_argument("--ov-device", default="NPU")
    ap.add_argument("--ep-option", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()
    rec = al.Record("run_vision_ort.py", args)
    al.banner()
    code = 0
    try:
        rec["versions"] = al.versions(["onnxruntime", "onnxruntime-gpu", "onnxruntime-ep-openvino",
                                       "onnxruntime-ep-webgpu", "numpy"])
        rec["cuda_dll_dirs"] = al.add_cuda_dirs()
        import numpy as np
        import onnxruntime as ort
        d = np.load(args.inputs)
        ids, x = [str(i) for i in d["ids"]], d["x"].astype(np.float32)
        rec["inputs"] = {"path": args.inputs, "n": len(ids), "shape": list(x.shape), "sha256": al.sha256_file(args.inputs)}
        rec["model"] = {"path": args.model, "sha256": al.sha256_file(args.model)}
        dev = {}
        t = time.perf_counter()
        sess = al.device_session(ort, args.model, args, args.ep, dev)
        rec["load_s"] = round(time.perf_counter() - t, 3)
        dev.pop("registered_ov", None)
        dev.pop("registered_wg", None)
        rec["device"] = dev
        rec["session_providers"] = sess.get_providers()
        iname = sess.get_inputs()[0].name
        rec["workload"] = {"inputs": len(ids), "batch": 1, "input_name": iname, "timing": "session.run wall, host numpy"}
        if not args.no_warmup:
            t0 = time.perf_counter()
            sess.run(None, {iname: x[:1]})
            rec["warmup_s"] = round(time.perf_counter() - t0, 3)
        logits = []
        for i, iid in enumerate(ids):
            t0 = time.perf_counter()
            y = sess.run(None, {iname: x[i:i + 1]})[0]
            wall = time.perf_counter() - t0
            lg = np.asarray(y, dtype=np.float32).reshape(-1)
            logits.append(lg)
            pr = np.exp(lg - lg.max())
            pr /= pr.sum()
            top = np.argsort(-pr)[:5]
            rec.row({"id": iid, "wall_s": wall, "top5": [int(k) for k in top], "top5_p": [float(pr[k]) for k in top],
                     "finite": bool(np.isfinite(lg).all())})
        lpath = pathlib.Path(args.out).with_name(pathlib.Path(args.out).stem + ".logits.npy")
        lpath.parent.mkdir(parents=True, exist_ok=True)
        np.save(lpath, np.stack(logits).astype(np.float32))
        rec["logits"] = {"path": str(lpath), "sha256": al.sha256_file(lpath)}
        rec["summary"] = rec.summarize("vision")
        rec["status"] = "OK"
    except Exception:
        rec.fail()
        code = 1
    rec.write()
    os._exit(code)


if __name__ == "__main__":
    main()
