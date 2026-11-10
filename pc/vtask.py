"""Shared parts of the vision task scripts (pc/ref_vision_task.py, pc/run_vision_task.py): the input sets, the apps'
preprocessing and host steps, the engines of the cells, the MoViNet stream and the downstream head.

The preprocessing, the host steps (DETR graph B's inputs, the SAM2 prompt, the DAC RVQ) and the cells' sessions are
the vision zoo Z1's (john-rocky/windows-zoo@faa4019 pc/make_inputs.py and pc/zoo_run.py), applied to every input of
the model's set instead of one sample. The spec (pc/z5_spec.json, written on the Mac by migrate/z5.py spec) names per
model: its input set and ids, Z1's input entries (geometry, normalization, channel order, upstream graph), the files
of its forms with their sha256, and the stream or downstream step.

Arrays on disk (numpy .npz, uncompressed): inputs.npz keys n<idx>__<feed name>; outputs keys n<idx>__o<k> (one run per
input), n<idx>__f<t>__o0 (MoViNet: the logits after frame t) and n<idx>__head (the downstream head's logits).
"""
import hashlib
import io
import json
import math
import os
import pathlib
import re
import time
import wave

import numpy as np

NORM = {  # Z1 pc/make_inputs.py
    "imagenet": (1 / 255, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "unit": (1 / 255, [0, 0, 0], [1, 1, 1]),
    "minus1to1": (1 / 127.5, [1, 1, 1], [1, 1, 1]),
    "mmpose255": (1.0, [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]),
}
STREAM_DIMS = [2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 4]   # MoViNet: temporal-conv buffer depths (ActionRecognizer.kt)
N_POOL = 16


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def check_file(entry, what):
    """entry = {path, sha256}: the file Z1 measured (or a host file), checked before use. Returns the record row."""
    p = entry["path"]
    if not os.path.isfile(p):
        raise RuntimeError(f"{what}: {p} does not exist")
    got = sha256_file(p)
    if got != entry["sha256"]:
        raise RuntimeError(f"{what}: {p} sha256 {got} differs from the pinned {entry['sha256']}")
    return {"what": what, "path": p, "bytes": os.path.getsize(p), "sha256": got}


def _expand(x, places):
    """{ROOT} {HUB} {FIX} in every "path" of the spec: the zoo folder (the spec's <root>\\pc\\ parent), its hub\\ (pc/fetch.py
    puts <owner>__<repo>\\<path> there) and its fixtures\\"""
    if isinstance(x, dict):
        return {k: (_sub(v, places) if k == "path" and isinstance(v, str) else _expand(v, places)) for k, v in x.items()}
    if isinstance(x, list):
        return [_expand(v, places) for v in x]
    return x


def _sub(s, places):
    for k, v in places.items():
        s = s.replace(k, v)
    return s


def load_spec(path, model):
    path = pathlib.Path(path).resolve()
    root = path.parent.parent
    places = {"{ROOT}": str(root), "{HUB}": str(root / "hub"), "{FIX}": str(root / "fixtures")}
    spec = _expand(json.loads(path.read_text(encoding="utf-8")), places)
    if model not in spec["models"]:
        raise RuntimeError(f"{model}: not in {path}")
    return spec, spec["models"][model]


def key(idx, name):
    return f"n{idx:03d}__{name}"


def feed_index(name, fallback):
    """tf2onnx names the inputs serving_default_args_<i> and the outputs serving_default_output_<j>_output."""
    m = re.search(r"_(\d+)(?:_output)?$", name)
    return int(m.group(1)) if m else fallback


# ---- images (Z1 pc/make_inputs.py geometry / normalize / chw) -----------------------------------------------------

def geometry(im, kind, W, H):
    from PIL import Image
    w, h = im.size
    if kind == "stretch":
        return np.asarray(im.resize((W, H), Image.BILINEAR), np.float32)
    if kind == "center_square":
        s = min(w, h)
        x, y = (w - s) // 2, (h - s) // 2
        return np.asarray(im.crop((x, y, x + s, y + s)).resize((W, H), Image.BILINEAR), np.float32)
    if kind == "center_3x4":
        tgt = W / H
        if w / h > tgt:
            cw = int(round(h * tgt)); x = (w - cw) // 2
            box = (x, 0, x + cw, h)
        else:
            ch = int(round(w / tgt)); y = (h - ch) // 2
            box = (0, y, w, y + ch)
        return np.asarray(im.crop(box).resize((W, H), Image.BILINEAR), np.float32)
    if kind in ("letterbox_white", "letterbox_black"):
        s = min(W / w, H / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        canvas = Image.new("RGB", (W, H), (255, 255, 255) if kind.endswith("white") else (0, 0, 0))
        canvas.paste(im.resize((nw, nh), Image.BILINEAR), ((W - nw) // 2, (H - nh) // 2))
        return np.asarray(canvas, np.float32)
    raise ValueError(f"geometry {kind}")


def normalize(x, norm, order):
    if order == "BGR":
        x = x[..., ::-1]
    scale, mean, std = NORM[norm]
    return ((x * scale - np.array(mean, np.float32)) / np.array(std, np.float32)).astype(np.float32)


def chw(x):
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]).astype(np.float32)


def image_rows(suite, ids, images_dir):
    """The input images (PIL RGB) in the order of ids, each checked against the suite's sha256."""
    from PIL import Image
    by = {i["id"]: i for i in suite["images"]}
    out = []
    for iid in ids:
        e = by[iid]
        p = pathlib.Path(images_dir) / e["path"].replace("/", os.sep)
        if sha256_file(p) != e["sha256"]:
            raise RuntimeError(f"{p}: sha256 differs from the suite")
        out.append((iid, Image.open(p).convert("RGB"), {"path": e["path"], "sha256": e["sha256"]}))
    return out


def image_feed(entry, im, shape):
    """One Z1 image input entry (kind image / mlsd4) -> the NCHW array of the graph input's shape."""
    _, _, H, W = shape
    x = geometry(im, entry["geometry"], W, H)
    if entry["kind"] == "image":
        return chw(normalize(x, entry["norm"], entry["order"]))
    if entry["kind"] == "mlsd4":   # RGB / 127.5 - 1 and the constant fourth channel (MlsdDetector.kt)
        rgb = x / np.float32(127.5) - 1
        ones = np.full(x.shape[:2] + (1,), 1 / 127.5 - 1, np.float32)
        return chw(np.concatenate([rgb, ones], axis=2))
    raise ValueError(f"image kind {entry['kind']}")


# ---- audio ---------------------------------------------------------------------------------------------------------

def read_pcm16(path):
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise RuntimeError(f"{path}: not mono PCM16")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32), w.getframerate()


def audio_rows(suite, ids, clips_dir):
    by = {c["id"]: c for c in suite["clips"]}
    out = []
    for cid in ids:
        c = by[cid]
        p = pathlib.Path(clips_dir) / c["path"].replace("/", os.sep)
        if sha256_file(p) != c["sha256"]:
            raise RuntimeError(f"{p}: sha256 differs from the suite")
        pcm, rate = read_pcm16(p)
        if rate != 16000:
            raise RuntimeError(f"{p}: {rate} Hz")
        out.append((cid, pcm, {"path": c["path"], "sha256": c["sha256"], "label": c["label"],
                               "label_index": c["label_index"]}))
    return out


def scale_audio(pcm, how, n=16000):
    x = np.zeros(n, np.float32)
    x[:min(n, pcm.size)] = pcm[:n]
    if how == "peak_half":     # the keyword app's microphone path: pcm / max(1, peak) * 0.5 (MainActivity.kt)
        peak = max(1.0, float(np.abs(x).max()))
        return (x / np.float32(peak) * np.float32(0.5)).astype(np.float32)
    if how == "pcm32768":
        return (x / np.float32(32768)).astype(np.float32)
    raise ValueError(f"audio scale {how}")


# ---- engines ---------------------------------------------------------------------------------------------------------

class Engine:
    """One model on one cell: run(feed) -> list of float32 outputs in the graph's output order."""
    in_names = out_names = ()
    in_shapes = {}

    def run(self, feed):
        raise NotImplementedError


class OrtEngine(Engine):
    def __init__(self, path, ep, ov_device=None, ep_options=None, dev=None):
        import onnxruntime as ort
        dev = dev if dev is not None else {}
        so = ort.SessionOptions()
        providers = None
        if ep == "cpu":
            providers = ["CPUExecutionProvider"]
        elif ep == "cuda":
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()   # the CUDA / cuDNN DLLs of the nvidia pip wheels (Z1 pc/zoo_run.py)
            providers = [("CUDAExecutionProvider", {}), "CPUExecutionProvider"]
        elif ep == "openvino":
            import onnxruntime_ep_openvino as ov_ep
            if not dev.get("_registered"):
                ort.register_execution_provider_library(ov_ep.get_ep_name(), ov_ep.get_library_path())
                dev["_registered"] = True
            devs = [d for d in ort.get_ep_devices() if d.ep_name == ov_ep.get_ep_name()
                    and d.ep_metadata.get("ov_device") == ov_device]
            if not devs:
                raise RuntimeError(f"no OpenVINO EP device {ov_device}: "
                                   f"{[d.ep_metadata.get('ov_device') for d in ort.get_ep_devices()]}")
            meta = dict(devs[0].ep_metadata)
            dev["openvino_ep"] = {k: meta.get(k) for k in ("FULL_DEVICE_NAME", "ov_device", "version", "os_driver_version",
                                                           "NPU_COMPILER_VERSION", "DEVICE_ARCHITECTURE", "DEVICE_LUID")}
            so.add_provider_for_devices(devs, dict(ep_options or {}))
        else:
            raise ValueError(ep)
        t = time.perf_counter()
        self.sess = (ort.InferenceSession(path, sess_options=so, providers=providers) if providers
                     else ort.InferenceSession(path, sess_options=so))
        self.load_s = time.perf_counter() - t
        self.providers = self.sess.get_providers()
        if ep == "cuda" and "CUDAExecutionProvider" not in self.providers:
            raise RuntimeError(f"CUDA EP not in the session providers: {self.providers}")
        self.in_names = [i.name for i in self.sess.get_inputs()]
        self.in_shapes = {i.name: list(i.shape) for i in self.sess.get_inputs()}
        self.out_names = [o.name for o in self.sess.get_outputs()]

    def run(self, feed):
        return [np.asarray(o, dtype=np.float32) for o in self.sess.run(None, {k: feed[k] for k in self.in_names})]


class OvBlobEngine(Engine):
    """The precompiled NPU blob (Z1's ovnpu_pre cell): OpenVINO without ONNX Runtime, import_model in this process."""

    def __init__(self, path, device="NPU", dev=None):
        import openvino as ov
        dev = dev if dev is not None else {}
        core = ov.Core()
        dev["openvino"] = {"FULL_DEVICE_NAME": core.get_property(device, "FULL_DEVICE_NAME")}
        for k in ("DEVICE_LUID", "NPU_DRIVER_VERSION", "NPU_COMPILER_VERSION", "NPU_COMPILER_TYPE"):
            try:
                dev["openvino"][k] = str(core.get_property(device, k))
            except Exception as e:  # noqa: BLE001
                dev["openvino"][k] = f"<{type(e).__name__}>"
        with open(path, "rb") as f:
            data = f.read()
        t = time.perf_counter()
        try:
            self.compiled = core.import_model(data, device, {})
        except TypeError:   # a release that takes a stream only
            self.compiled = core.import_model(io.BytesIO(data), device, {})
        self.load_s = time.perf_counter() - t
        self.providers = [f"OpenVINO {ov.__version__} import_model {device}",
                          str(self.compiled.get_property("EXECUTION_DEVICES"))]
        self.req = self.compiled.create_infer_request()
        self.in_names = [i.get_any_name() for i in self.compiled.inputs]
        self.in_shapes = {i.get_any_name(): list(i.get_shape()) for i in self.compiled.inputs}
        self.out_names = [o.get_any_name() for o in self.compiled.outputs]

    def run(self, feed):
        res = self.req.infer({k: feed[k] for k in self.in_names}, share_outputs=False)
        return [np.asarray(res[o], dtype=np.float32) for o in self.compiled.outputs]


def ordered_outputs(engine, outs):
    """Outputs by the index in their names (serving_default_output_<j>_output), the graph's order otherwise."""
    idx = [feed_index(n, k) for k, n in enumerate(engine.out_names)]
    if sorted(idx) != list(range(len(idx))):
        return list(outs)
    by = dict(zip(idx, outs))
    return [by[j] for j in range(len(idx))]


# ---- the MoViNet stream (ActionRecognizer.kt) -------------------------------------------------------------------------

def movinet_clip(engine, frame, frames):
    """One still clip: the state from zero, `frames` runs of the same frame; the host shifts the 28 stream buffers and
    adds the 16 pooling sums (the app's runFrame). Returns [logits after frame 1, ..., frame n] and the wall times."""
    names = {feed_index(n, k): n for k, n in enumerate(engine.in_names)}
    if sorted(names) != list(range(47)):
        raise RuntimeError(f"MoViNet inputs: {engine.in_names}")
    state = {k: np.zeros(engine.in_shapes[names[k]], np.float32) for k in range(1, 45)}
    logits, walls = [], []
    for t in range(1, frames + 1):
        feed = {names[0]: frame, names[45]: np.full((1, 1, 1, 1), 1.0 / t, np.float32),
                names[46]: np.ones((1, 1, 1, 1), np.float32)}
        feed.update({names[k]: state[k] for k in range(1, 45)})
        t0 = time.perf_counter()
        outs = ordered_outputs(engine, engine.run(feed))
        walls.append(time.perf_counter() - t0)
        if len(outs) != 28:
            raise RuntimeError(f"MoViNet: {len(outs)} outputs")
        logits.append(outs[0])
        off = 1
        for c, dp in enumerate(STREAM_DIMS):
            for i in range(dp - 1):
                state[off + i] = state[off + i + 1]
            state[off + dp - 1] = outs[1 + c].reshape(state[off + dp - 1].shape)
            off += dp
        for i in range(N_POOL):
            state[29 + i] = state[29 + i] + outs[12 + i].reshape(state[29 + i].shape)
    return logits, walls


# ---- host steps (Z1 pc/make_inputs.py: detr_b, sam2_dec, dac_zq) ----------------------------------------------------

def by_shape(arrays, shape):
    hits = [a for a in arrays if list(a.shape) == list(shape)]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} arrays of shape {shape}: {[a.shape for a in arrays]}")
    return hits[0]


def detr_b(outs_a, host_params_path):
    """D-FINE / RT-DETRv2 graph B inputs from graph A's outputs, as DFine.kt / RtDetr.kt (fp32)."""
    HID, NPROP, NCLS, NQ = 256, 8400, 80, 300
    enc_class = by_shape(outs_a, [1, NPROP, NCLS])[0]
    mem = by_shape(outs_a, [1, NPROP, HID])[0] * np.float32(0.5)   # graph A emits memory_raw * 2
    raw = np.fromfile(host_params_path, dtype="<f4")
    o = 0

    def take(n):
        nonlocal o
        a = raw[o:o + n]; o += n
        return a
    eoW = take(HID * HID).reshape(HID, HID); eoB = take(HID); eoG = take(HID); eoBeta = take(HID)
    bb0W = take(HID * HID).reshape(HID, HID); bb0B = take(HID)
    bb1W = take(HID * HID).reshape(HID, HID); bb1B = take(HID)
    bb2W = take(4 * HID).reshape(4, HID); bb2B = take(4)
    valid = take(NPROP); anchors = take(NPROP * 4).reshape(NPROP, 4)
    assert o == raw.size, (o, raw.size)
    order = np.argsort(-enc_class.max(axis=1), kind="stable")[:NQ]
    x = valid[order, None] * mem[order]
    h = x @ eoW.T + eoB
    mu = h.mean(axis=1, keepdims=True); var = ((h - mu) ** 2).mean(axis=1, keepdims=True)
    target = ((h - mu) / np.sqrt(var + np.float32(1e-5)) * eoG + eoBeta).astype(np.float32)
    h1 = np.maximum(target @ bb0W.T + bb0B, 0)
    h2 = np.maximum(h1 @ bb1W.T + bb1B, 0)
    ref = (h2 @ bb2W.T + bb2B + anchors[order]).astype(np.float32)
    return [mem[None].astype(np.float32), target[None], ref[None]]


def sam2_dec(flat_enc, prompt_path, point):
    IE, F0, F1 = 256 * 64 * 64, 32 * 256 * 256, 64 * 128 * 128
    flat = np.asarray(flat_enc, np.float32).reshape(-1)
    assert flat.size == IE + F0 + F1, flat.size
    p = np.fromfile(prompt_path, dtype="<f4")
    assert p.size == 768, p.size
    g, pe1, nap = p[0:256], p[256:512], p[512:768]
    x, y = point
    ccx = 2 * ((x + 0.5) / 1024) - 1
    ccy = 2 * ((y + 0.5) / 1024) - 1
    proj = (2 * math.pi * (ccx * g[:128].astype(np.float64) + ccy * g[128:].astype(np.float64))).astype(np.float32)
    sparse = np.concatenate([np.sin(proj) + pe1[:128], np.cos(proj) + pe1[128:], nap]).astype(np.float32)
    return [np.concatenate([flat[:IE], sparse, flat[IE:IE + F0], flat[IE + F0:]])[None]]


def dac_zq(z_enc, rvq_path):
    """z_q = RVQ.decode(RVQ.encode(encoder output)), as DacRVQ.kt (12 codebooks, dim 8, size 1024, hidden 1024)."""
    NQ, DIM, SIZE, HID = 12, 8, 1024, 1024
    z = np.asarray(z_enc, np.float32)[0]   # [1024, T]
    raw = np.fromfile(rvq_path, dtype="<f4")
    per = SIZE * DIM + DIM * HID + DIM + HID * DIM + HID
    assert raw.size == NQ * per, raw.size
    residual = z.copy()
    zq = np.zeros_like(z)
    for i in range(NQ):
        b = raw[i * per:(i + 1) * per]
        o = 0
        cb = b[o:o + SIZE * DIM].reshape(SIZE, DIM); o += SIZE * DIM
        wIn = b[o:o + DIM * HID].reshape(DIM, HID); o += DIM * HID
        bIn = b[o:o + DIM]; o += DIM
        wOut = b[o:o + HID * DIM].reshape(HID, DIM); o += HID * DIM
        bOut = b[o:o + HID]
        cbn = cb / np.maximum(np.linalg.norm(cb, axis=1, keepdims=True), 1e-12)
        proj = wIn @ residual + bIn[:, None]
        proj = proj / np.maximum(np.linalg.norm(proj, axis=0, keepdims=True), 1e-12)
        codes = np.argmax(cbn @ proj, axis=0)
        q = wOut @ cb[codes].T + bOut[:, None]
        residual -= q
        zq += q
    return [zq[None].astype(np.float32)]


# ---- one pass of an engine over the prepared inputs (the reference step and the cells) -----------------------------

def run_set(engine, mspec, inputs_path, rec, then=None, warmup=True):
    """Every input of inputs.npz through the engine: returns {outputs key: array} and appends rec rows."""
    meta = json.loads(pathlib.Path(inputs_path).with_suffix(".json").read_text(encoding="utf-8"))
    ids, feeds_names = meta["ids"], meta["feeds"]
    z = np.load(inputs_path)
    out = {}
    stream = mspec.get("stream")
    rec.total = len(ids) * (stream["frames"] if stream else 1)   # the progress lines: "n of total"

    def feed_of(idx):
        return {n: z[key(idx, n)] for n in feeds_names}

    if warmup and ids:   # one untimed run (not kept): the first input, or one frame from zero
        t0 = time.perf_counter()
        if stream:   # inputs.npz holds the frame only (input 0); the state is made by movinet_clip
            movinet_clip(engine, next(iter(feed_of(0).values())), 1)
        else:
            engine.run(feed_of(0))
        rec["warmup_s"] = round(time.perf_counter() - t0, 3)
    for idx, iid in enumerate(ids):
        f = feed_of(idx)
        if stream:
            frame = next(iter(f.values()))
            logits, walls = movinet_clip(engine, frame, stream["frames"])
            for t, (lg, w) in enumerate(zip(logits, walls), start=1):
                out[f"n{idx:03d}__f{t}__o0"] = lg
                rec.row({"id": f"{iid}#f{t}", "input": iid, "frame": t, "wall_s": w,
                         "nonfinite": int((~np.isfinite(lg)).sum())})
            continue
        t0 = time.perf_counter()
        outs = ordered_outputs(engine, engine.run(f))
        wall = time.perf_counter() - t0
        for k, o in enumerate(outs):
            out[key(idx, f"o{k}")] = o
        row = {"id": iid, "wall_s": wall, "nonfinite": [int((~np.isfinite(o)).sum()) for o in outs]}
        if then is not None:   # the downstream head (fp32, CPU EP) on this cell's output
            hn = then.in_names[0]
            h = ordered_outputs(then, then.run({hn: outs[0].reshape(then.in_shapes[hn])}))[0]
            out[key(idx, "head")] = h
            row["head_nonfinite"] = int((~np.isfinite(h)).sum())
        rec.row(row)
    return out


def save_npz(path, arrays):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "arrays": len(arrays)}
