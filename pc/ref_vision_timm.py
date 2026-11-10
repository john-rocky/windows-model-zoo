"""The reference and the ONNX form of a timm image classifier (an untimed step; the venv `ref`).

    python ref_vision_timm.py --model-name <timm name> --weights <model.safetensors> --images <folder> --manifest
        <suites\\vision\\images.json> --work <folder> --out <ref.json> [--opset 17]

1. inputs: every image of the manifest (checked against its sha256), PIL RGB, timm's eval transform of the model
   (resolve_model_data_config + create_transform(is_training=False), the model card's form) once, stacked into <work>\\inputs.npz (ids, x float32 [N, 3, H, W]): the
   reference and every cell read these same tensors.
2. reference: the model in fp32 on the CPU, one input at a time; <out> = the rows (top-5 classes and probabilities,
   the gap between the first two) and <out stem>.logits.npy ([N, classes], float32).
3. form: torch.onnx.export of the same fp32 model, input [1, 3, H, W] static, opset --opset, to <work>\\model.onnx;
   the TorchScript exporter (dynamo=False) first, the dynamo exporter when that one is gone; the record says which.
"""
import argparse
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zoolib as al  # noqa: E402


def main():
    al.utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    rec = al.Record("ref_vision_timm.py", args)
    al.banner()
    code = 0
    try:
        rec["versions"] = al.versions(["torch", "timm", "pillow", "onnx", "onnxscript", "numpy", "safetensors"])
        import numpy as np
        import timm
        import torch
        from PIL import Image
        if args.threads:
            torch.set_num_threads(args.threads)
        rec["torch_threads"] = torch.get_num_threads()
        work = pathlib.Path(args.work)
        work.mkdir(parents=True, exist_ok=True)
        man = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
        t = time.perf_counter()
        model = timm.create_model(args.model_name, pretrained=True,
                                  pretrained_cfg_overlay={"file": args.weights}).eval()
        rec["load_s"] = round(time.perf_counter() - t, 3)
        cfg = timm.data.resolve_model_data_config(model)   # the model card's form
        tf = timm.data.create_transform(**cfg, is_training=False)
        rec["data_config"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
        rec["transform"] = str(tf)
        ids, xs = [], []
        for im in man["images"]:
            p = pathlib.Path(args.images) / im["path"].replace("/", os.sep)
            if al.sha256_file(p) != im["sha256"]:
                raise RuntimeError(f"{p}: sha256 differs from the manifest")
            ids.append(im["id"])
            xs.append(tf(Image.open(p).convert("RGB")).numpy())
        x = np.stack(xs).astype(np.float32)
        np.savez(work / "inputs.npz", ids=np.array(ids), x=x)
        rec["inputs"] = {"path": str(work / "inputs.npz"), "n": len(ids), "shape": list(x.shape),
                         "sha256": al.sha256_file(work / "inputs.npz")}
        rec["workload"] = {"images": len(ids), "dtype": "float32", "device": "cpu", "batch": 1}
        logits = []
        for i, iid in enumerate(ids):
            t0 = time.perf_counter()
            with torch.inference_mode():
                y = model(torch.from_numpy(x[i:i + 1]))
            wall = time.perf_counter() - t0
            lg = y[0].float().numpy()
            logits.append(lg)
            pr = np.exp(lg - lg.max())
            pr /= pr.sum()
            top = np.argsort(-pr)[:5]
            rec.row({"id": iid, "wall_s": wall, "top5": [int(k) for k in top], "top5_p": [float(pr[k]) for k in top],
                     "gap12": float(pr[top[0]] - pr[top[1]])})
        lpath = pathlib.Path(args.out).with_name(pathlib.Path(args.out).stem + ".logits.npy")
        lpath.parent.mkdir(parents=True, exist_ok=True)
        np.save(lpath, np.stack(logits).astype(np.float32))
        rec["logits"] = {"path": str(lpath), "sha256": al.sha256_file(lpath)}
        # the form: the same fp32 model to ONNX
        onnx_path = work / "model.onnx"
        dummy = torch.from_numpy(x[:1])
        t = time.perf_counter()
        try:
            torch.onnx.export(model, (dummy,), str(onnx_path), opset_version=args.opset, input_names=["input"],
                              output_names=["logits"], dynamo=False)
            exporter = "torch.onnx.export dynamo=False (TorchScript)"
        except Exception as e:  # noqa: BLE001  the TorchScript exporter removed or failing: the dynamo one, recorded
            rec["export_first_try"] = f"{type(e).__name__}: {e}"[:800]
            torch.onnx.export(model, (dummy,), str(onnx_path), opset_version=args.opset, input_names=["input"],
                              output_names=["logits"], dynamo=True)
            exporter = "torch.onnx.export dynamo=True"
        rec["form"] = {"path": str(onnx_path), "exporter": exporter, "opset": args.opset, "input": list(dummy.shape),
                       "export_s": round(time.perf_counter() - t, 3), "bytes": onnx_path.stat().st_size,
                       "sha256": al.sha256_file(onnx_path)}
        rec["summary"] = rec.summarize("vision")
        rec["status"] = "OK"
    except Exception:
        rec.fail()
        code = 1
    rec.write()
    os._exit(code)


if __name__ == "__main__":
    main()
