"""Synthetic outputs for the vision task cells of pc/z5_spec.json (no model runs): per model a reference, a near copy
that should pass the model's task line and a broken copy that should fail it, in the shapes of the models' real
outputs (the npz keys of pc/ref_vision_task.py and pc/run_vision_task.py). Used by test_vision_tasks.py and
test_records.py."""
import numpy as np

import vision_tasks as VT


def key(i, k):
    return f"n{i:03d}__o{k}"


class Z(dict):
    """An np.load-like view of a dict: .files and []."""
    @property
    def files(self):
        return list(self.keys())


def gen(model, ms, rng):
    """ms = the model's entry in pc/z5_spec.json -> (reference, near copy, broken copy), dicts of float32 arrays."""
    task, n = ms["task"], len(ms["ids"])
    frames = (ms.get("stream") or {}).get("frames", 8)
    ref, good, bad = {}, {}, {}

    def put(i, outs, gouts, bouts):
        for k, (a, g, b) in enumerate(zip(outs, gouts, bouts)):
            ref[key(i, k)], good[key(i, k)], bad[key(i, k)] = (a.astype(np.float32), g.astype(np.float32),
                                                              b.astype(np.float32))
    for i in range(n):
        if task == "detection" and model in VT.DETR:
            boxes = np.concatenate([rng.uniform(0.2, 0.8, (1, 300, 2)), rng.uniform(0.05, 0.3, (1, 300, 2))], -1)
            lg = rng.normal(-5, 0.5, (1, 300, 80))
            q = rng.choice(300, 6, replace=False)
            lg[0, q, rng.integers(0, 80, 6)] = 3.0
            put(i, [boxes, lg], [boxes + rng.normal(0, 1e-4, boxes.shape), lg + rng.normal(0, 1e-3, lg.shape)],
                [boxes, lg[:, rng.permutation(300)]])
        elif task == "detection" and model.startswith("ssdlite"):
            outs = []
            for f in (20, 10, 5, 3, 2, 1):
                cls = rng.normal(0, 0.3, (1, 6, 91, f, f))
                cls[:, :, 0] = 5.0
                for _ in range(2 if f >= 5 else 1):
                    a, kk, y, x = rng.integers(0, 6), rng.integers(1, 91), rng.integers(0, f), rng.integers(0, f)
                    cls[0, a, kk, y, x] = 12.0
                outs += [cls.reshape(1, 546, f, f), rng.normal(0, 0.5, (1, 24, f, f))]
            put(i, outs, [o + rng.normal(0, 1e-4, o.shape) for o in outs],
                [o if k % 2 else np.where(o > 11, 0.0, o) for k, o in enumerate(outs)])
        elif task == "detection":   # PP-OCR's probability map
            p = np.zeros((1, 1, 640, 640))
            for _ in range(4):
                y, x = rng.integers(20, 560), rng.integers(20, 500)
                p[0, 0, y:y + 30, x:x + 120] = 0.9
            put(i, [p], [np.clip(p + rng.normal(0, 1e-3, p.shape), 0, 1)], [np.zeros_like(p)])
        elif task == "line-segments":
            o = np.full((1, 9, 256, 256), -10.0)
            for _ in range(5):
                y, x = rng.integers(30, 220), rng.integers(30, 220)
                o[0, 0, y, x] = 5.0
                o[0, 1, y, x], o[0, 2, y, x], o[0, 3, y, x], o[0, 4, y, x] = -15, 0, 15, 0
            put(i, [o], [o + rng.normal(0, 1e-3, o.shape)], [np.roll(o, 20, axis=3)])
        elif task == "keypoints":
            W, H, _ = VT.POSE[model]
            K = {"rtmpose_s_fp16": 17, "rtmw_fp16": 133, "rtm_face_fp16": 98, "rtmhand_fp16": 21,
                 "rtm_animal_fp16": 17}[model]
            sx, sy = rng.normal(0, 0.05, (1, K, W * 2)), rng.normal(0, 0.05, (1, K, H * 2))
            sx[0, np.arange(K), rng.integers(0, W * 2, K)] = 0.9
            sy[0, np.arange(K), rng.integers(0, H * 2, K)] = 0.9
            put(i, [sx, sy], [sx + rng.normal(0, 1e-3, sx.shape), sy + rng.normal(0, 1e-3, sy.shape)],
                [np.roll(sx, 40, axis=2), sy])
        elif task == "mask" and model.startswith("sam2"):
            yy, xx = np.mgrid[0:256, 0:256]
            m = 8.0 - np.hypot(yy - rng.integers(80, 176), xx - rng.integers(80, 176)) / 6
            o = np.stack([m, m - 1, m + 1])[None]
            put(i, [o], [o + rng.normal(0, 1e-3, o.shape)], [-o])
        elif task == "mask":
            yy, xx = np.mgrid[0:352, 0:352]
            o = (1 / (1 + np.exp(-(6.0 - np.hypot(yy - 176, xx - rng.integers(100, 250)) / 12))))[None, None]
            put(i, [o], [np.clip(o + rng.normal(0, 1e-4, o.shape), 0, 1)], [1 - o])
        elif task == "warp":
            g = np.linspace(-0.9, 0.9, 128)
            gx, gy = np.meshgrid(g, g)
            o = np.stack([gx + 0.02 * np.sin(gy * 3), gy])[None]
            put(i, [o], [o + rng.normal(0, 1e-5, o.shape)], [o + 0.05])
            ref[f"n{i:03d}__src"] = rng.integers(0, 256, (300, 200, 3)).astype(np.uint8)
        elif task == "count":
            d = rng.uniform(0, 0.02, (1, 1, 64, 64))
            put(i, [d], [d * (1 + 1e-4)], [d * 1.5])
        elif task == "depth":
            d = rng.uniform(1, 20, (1, 1, 448, 448))
            put(i, [d], [d * (1 + 1e-3)], [d * 1.1])
        elif task == "retrieval":
            base = rng.normal(0, 1, (1, 384)) if i < n // 2 else ref[key(i - n // 2, 0)].mean(1)
            t = base[:, None, :] + rng.normal(0, 1, (1, 1024, 384))
            put(i, [t], [t + rng.normal(0, 1e-3, t.shape)], [rng.normal(0, 1, t.shape)])
        elif task == "classification-stream":
            for f in range(1, frames + 1):
                lg = rng.normal(0, 1, (1, 600))
                lg[0, rng.integers(0, 600)] = 6
                k = f"n{i:03d}__f{f}__o0"
                ref[k] = lg.astype(np.float32)
                good[k] = (lg + rng.normal(0, 1e-3, lg.shape)).astype(np.float32)
                bad[k] = rng.normal(0, 1, (1, 600)).astype(np.float32)
        elif task == "classification":
            lg = rng.normal(0, 1, (1, 12))
            lg[0, rng.integers(0, 12)] = 5
            if model == "w2v2_frontend_fp16":   # features, then the downstream head's logits
                f = rng.normal(0, 1, (1, 49, 768))
                put(i, [f], [f + 1e-3], [f])
                ref[f"n{i:03d}__head"] = lg.astype(np.float32)
                good[f"n{i:03d}__head"] = (lg + 1e-4).astype(np.float32)
                bad[f"n{i:03d}__head"] = np.roll(lg, 1, 1).astype(np.float32)
            else:
                put(i, [lg], [lg + rng.normal(0, 1e-3, lg.shape)], [np.roll(lg, 1, 1)])
        elif task == "codec":
            src = rng.normal(0, 0.1, 16000)
            # the decoder's output starts 8 samples into the clip (15,992 = 16,000 - 8; PLAN.md Z5, change 1)
            o = (src[8:] * 0.9 + rng.normal(0, 0.01, 15992))[None, None]
            put(i, [o], [o + rng.normal(0, 1e-4, o.shape)], [np.zeros_like(o)])
            ref[f"n{i:03d}__src"] = src.astype(np.float32)
        else:
            raise ValueError(f"{model}: no synthetic outputs for task {task}")
    return ref, good, bad
