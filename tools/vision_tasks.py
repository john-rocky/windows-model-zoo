#!/usr/bin/env python3
"""The task lines of the vision task cells (PLAN.md section Z5; each rule is the model's [[line]] in its recipe): the
apps' post-processing (john-rocky/LiteRT-Models@96e0bc1f, the Kotlin files the rules name) ported to numpy, applied
to the cell's outputs and to the reference's outputs alike, then the comparison of the line. No model runs here.

Inputs: the runner's outputs (<run stem>.outputs.npz, pc/run_vision_task.py) and the reference's
(<refs>/<model>/ref.outputs.npz, pc/ref_vision_task.py), keys n<idx>__o<k> / n<idx>__f<t>__o0 / n<idx>__head /
n<idx>__src, with the ids in the records' rows. score() returns (instruments, detail, words, values): detail = the
line's verdict and its counts, values = the compact per-input task values (kept in the cell JSON, and the reference's
in models/<m>/reference/ref.json) from which the verdict reads again without the outputs.
"""
import math

import numpy as np

TASKS = ("detection", "line-segments", "keypoints", "mask", "warp", "count", "depth", "retrieval",
         "classification", "classification-stream", "codec")
POSE = {  # model -> (input W, H, the app's show threshold or None = every point shown)
    "rtmpose_s_fp16": (192, 256, 0.3), "rtmw_fp16": (192, 256, 0.3), "rtm_animal_fp16": (256, 256, 0.3),
    "rtmhand_fp16": (256, 256, 0.2), "rtm_face_fp16": (256, 256, None)}
DETR = {"dfine_graphB_fp16", "dfine_graphB_fp16_fixed", "rtdetr_graphB_fp16", "rtdetr_graphB_fp16_fixed"}
CODEC_OFFSET = {"task-codec-aligned": 8}   # line id -> the input sample the decoded output starts at (PLAN.md Z5 change 1)


def _sig32(x):
    x = np.asarray(x, np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        return (np.float32(1) / (np.float32(1) + np.exp(-x))).astype(np.float32)


def outputs_of(z, idx):
    """The model outputs of input idx in order o0, o1, ..."""
    out, k = [], 0
    while f"n{idx:03d}__o{k}" in z:
        out.append(z[f"n{idx:03d}__o{k}"])
        k += 1
    return out


# ---- boxes -------------------------------------------------------------------------------------------------------

def iou_xyxy(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def detr_boxes(o):
    """DFine.kt / RtDetr.kt detect(): per query argmax of 80 logits, sigmoid >= 0.4, per-class NMS at IoU > 0.7.
    Boxes cxcywh in [0, 1] -> xyxy. Returns [(cls, score, x0, y0, x1, y1)] in descending score."""
    boxes = next(a for a in o if a.shape[-1] == 4).reshape(-1, 4).astype(np.float64)
    logits = next(a for a in o if a.shape[-1] == 80).reshape(-1, 80)
    dets = []
    for q in range(logits.shape[0]):
        lg = logits[q]
        if not np.isfinite(lg).any():
            continue
        c = int(np.argmax(np.where(np.isnan(lg), -np.inf, lg)))
        best = float(lg[c])
        score = 1.0 / (1.0 + math.exp(-best)) if best > -700 else 0.0
        if not (score >= 0.4):
            continue
        cx, cy, w, h = boxes[q]
        dets.append((c, score, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    out = []
    for cls in sorted({d[0] for d in dets}):
        g = sorted([d for d in dets if d[0] == cls], key=lambda d: -d[1])
        taken = [False] * len(g)
        for i in range(len(g)):
            if taken[i]:
                continue
            out.append(g[i])
            for j in range(i + 1, len(g)):
                if not taken[j] and iou_xyxy(g[i][2:], g[j][2:]) > 0.7:
                    taken[j] = True
    return sorted(out, key=lambda d: -d[1])


_SSD_ANCHORS = None


def ssd_anchors():
    """SSDLiteDetector.kt buildAnchors(): DefaultBoxGenerator for 320 x 320, level -> row -> col -> anchor, xyxy."""
    global _SSD_ANCHORS
    if _SSD_ANCHORS is not None:
        return _SSD_ANCHORS
    levels, scales = [20, 10, 5, 3, 2, 1], [0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.0]
    out = []
    for li, f in enumerate(levels):
        sk, skp = scales[li], math.sqrt(scales[li] * scales[li + 1])
        wh = [(sk, sk), (skp, skp)]
        for ar in (2, 3):
            sq = math.sqrt(ar)
            wh += [(sk * sq, sk / sq), (sk / sq, sk * sq)]
        wh = [(min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)) for w, h in wh]
        for i in range(f):
            for j in range(f):
                cx, cy = (j + 0.5) / f, (i + 0.5) / f
                for w, h in wh:
                    out.append(((cx - 0.5 * w) * 320, (cy - 0.5 * h) * 320, (cx + 0.5 * w) * 320, (cy + 0.5 * h) * 320))
    _SSD_ANCHORS = np.array(out, np.float64)
    return _SSD_ANCHORS


def ssd_boxes(o):
    """SSDLiteDetector.kt postProcess(): softmax over 91, best of 1..90 >= 0.4, BoxCoder (10, 10, 5, 5), clip to
    320, per-class NMS at IoU > 0.55 on the score-sorted candidates, the first 100. xyxy in the 320 x 320 input."""
    anchors = ssd_anchors()
    clip = math.log(1000.0 / 16.0)
    cands, off = [], 0

    def c(v):
        return min(max(v, 0.0), 320.0)
    for f in (20, 10, 5, 3, 2, 1):
        hw = f * f
        # cls channel (anchor a, class k) = a * 91 + k, box channel (a, j) = a * 4 + j; spatial stride hw
        lg = next(a for a in o if a.shape == (1, 546, f, f)).reshape(6, 91, hw).astype(np.float64).transpose(2, 0, 1)
        bx = next(a for a in o if a.shape == (1, 24, f, f)).reshape(6, 4, hw).astype(np.float64).transpose(2, 0, 1)
        fin = np.isfinite(lg).all(-1)                      # [hw, 6]; an anchor with a non-finite logit gives no box
        with np.errstate(invalid="ignore", over="ignore"):
            m = lg.max(-1)
            k = 1 + np.argmax(lg[..., 1:], -1)             # the first of the largest foreground logits
            best = np.take_along_axis(lg, k[..., None], -1)[..., 0]
            score = np.exp(best - m) / np.exp(lg - m[..., None]).sum(-1)
        for p, a in zip(*np.nonzero(fin & (score >= 0.4))):   # the app's loop order: position, then anchor
            x1, y1, x2, y2 = anchors[off + p * 6 + a]
            aw, ah = x2 - x1, y2 - y1
            acx, acy = x1 + 0.5 * aw, y1 + 0.5 * ah
            dx, dy = bx[p, a, 0] / 10, bx[p, a, 1] / 10
            dw, dh = min(bx[p, a, 2] / 5, clip), min(bx[p, a, 3] / 5, clip)
            pcx, pcy, pw, ph = dx * aw + acx, dy * ah + acy, math.exp(dw) * aw, math.exp(dh) * ah
            cands.append((int(k[p, a]), float(score[p, a]), c(pcx - 0.5 * pw), c(pcy - 0.5 * ph),
                          c(pcx + 0.5 * pw), c(pcy + 0.5 * ph)))
        off += hw * 6
    cands.sort(key=lambda d: -d[1])
    active = [True] * len(cands)
    out = []
    for i in range(len(cands)):
        if not active[i]:
            continue
        out.append(cands[i])
        for j in range(i + 1, len(cands)):
            if active[j] and cands[j][0] == cands[i][0]:
                a, b = cands[i], cands[j]
                ix0, iy0, ix1, iy1 = max(a[2], b[2]), max(a[3], b[3]), min(a[4], b[4]), min(a[5], b[5])
                inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                if inter / ((a[4] - a[2]) * (a[5] - a[3]) + (b[4] - b[2]) * (b[5] - b[3]) - inter + 1e-6) > 0.55:
                    active[j] = False
    return out[:100]


def ppocr_boxes(o):
    """PpocrDetector.kt boxes(): probability > 0.3, 4-connected regions, w and h >= 6, mean >= 0.5, padded by
    0.35 x min(w, h) clamped to 2..24, clipped to 0..639. Returns [(0, mean probability, x0, y0, x1, y1)]."""
    from scipy import ndimage
    prob = np.asarray(o[0], np.float32).reshape(640, 640)
    lab, n = ndimage.label(prob > np.float32(0.3))   # the default structure: 4-connectivity
    out = []
    if n == 0:
        return out
    idx = np.arange(1, n + 1)
    sums = ndimage.sum(prob.astype(np.float64), lab, idx)
    areas = ndimage.sum(np.ones_like(prob, np.float64), lab, idx)
    for k, sl in enumerate(ndimage.find_objects(lab)):
        y0, y1, x0, x1 = sl[0].start, sl[0].stop - 1, sl[1].start, sl[1].stop - 1
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if w < 6 or h < 6 or sums[k] / areas[k] < 0.5:
            continue
        pad = min(max(int(np.float32(min(w, h)) * np.float32(0.35)), 2), 24)
        out.append((0, float(sums[k] / areas[k]), max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, 639), min(y1 + pad, 639)))
    return sorted(out, key=lambda d: -d[1])


def pair_boxes(cell, ref, thr=0.5):
    """One to one, the cell's boxes in descending score, each to the unpaired reference box of its class with the
    largest IoU >= thr. Returns (pairs, ious)."""
    used = [False] * len(ref)
    pairs, ious = 0, []
    for c in sorted(cell, key=lambda d: -d[1]):
        best, bj = thr, -1
        for j, r in enumerate(ref):
            if used[j] or r[0] != c[0]:
                continue
            v = iou_xyxy(c[2:], r[2:])
            if v >= best:
                best, bj = v, j
        if bj >= 0:
            used[bj] = True
            pairs += 1
            ious.append(best)
    return pairs, ious


# ---- line segments (MlsdDetector.kt) -------------------------------------------------------------------------------

def mlsd_lines(o):
    out = np.asarray(o[0], np.float32).reshape(9, 256, 256)
    c = _sig32(out[0])
    pad = np.pad(c, 1, constant_values=-np.inf)
    nb = np.max([pad[1 + dy:257 + dy, 1 + dx:257 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx], axis=0)
    ys, xs = np.nonzero((c > np.float32(0.10)) & ~(nb > c))
    sc = c[ys, xs]
    order = sorted(range(len(sc)), key=lambda i: -float(sc[i]))[:200]   # stable: raster order among equal scores
    lines = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        dxs, dys, dxe, dye = (float(out[k, y, x]) for k in (1, 2, 3, 4))
        if not all(math.isfinite(v) for v in (dxs, dys, dxe, dye)) or math.hypot(dxs - dxe, dys - dye) <= 20:
            continue
        lines.append(((x + dxs) * 2, (y + dys) * 2, (x + dxe) * 2, (y + dye) * 2, float(sc[i])))
    return lines


def pair_segments(cell, ref, tol):
    used = [False] * len(ref)
    pairs = 0
    for s in sorted(cell, key=lambda d: -d[4]):
        best, bj = None, -1
        for j, r in enumerate(ref):
            if used[j]:
                continue
            d1 = max(math.hypot(s[0] - r[0], s[1] - r[1]), math.hypot(s[2] - r[2], s[3] - r[3]))
            d2 = max(math.hypot(s[0] - r[2], s[1] - r[3]), math.hypot(s[2] - r[0], s[3] - r[1]))
            d = min(d1, d2)
            if d <= tol and (best is None or d < best):
                best, bj = d, j
        if bj >= 0:
            used[bj] = True
            pairs += 1
    return pairs


# ---- keypoints (Rtm*Estimator.kt) ---------------------------------------------------------------------------------

def simcc_points(o, W, H):
    a, b = o[0].reshape(o[0].shape[-2], -1), o[1].reshape(o[1].shape[-2], -1)
    if W != H:
        sx, sy = (a, b) if a.shape[-1] == W * 2 else (b, a)
    else:
        sx, sy = a, b
    pts = []
    for k in range(sx.shape[0]):
        xi, yi = int(np.argmax(sx[k])), int(np.argmax(sy[k]))
        xv, yv = float(sx[k][xi]), float(sy[k][yi])
        pts.append((xi / 2.0, yi / 2.0, (xv + yv) / 2.0))
    return pts


# ---- one model's line ------------------------------------------------------------------------------------------------

def ids_of(rec):
    """Input ids in order (a stream's rows are frames: their input ids, first seen)."""
    out = []
    for r in rec.get("rows") or []:
        i = r.get("input", r["id"])
        if i not in out:
            out.append(i)
    return out


def instruments(zc, zr):
    """Per output (o<k>, or head; a stream's frames together): the cosine and max |d| against the reference over the
    values both have finite, and the cell's non-finite count, over every input. Z1's internal measures, not the line."""
    have = set(zc.files)
    groups = {}
    for k in zr.files:
        nm = k.rsplit("__", 1)[1]
        if nm != "src" and k in have:
            groups.setdefault(nm, []).append(k)
    out = {}
    for nm, ks in sorted(groups.items()):
        a = np.concatenate([zc[k].astype(np.float64).reshape(-1) for k in ks])
        b = np.concatenate([zr[k].astype(np.float64).reshape(-1) for k in ks])
        fin = np.isfinite(a) & np.isfinite(b)
        cos = (float((a[fin] * b[fin]).sum() / (np.linalg.norm(a[fin]) * np.linalg.norm(b[fin]) + 1e-300))
               if fin.any() else None)
        out[nm] = {"cos": round(cos, 6) if cos is not None else None,
                   "max_abs": round(float(np.abs(a[fin] - b[fin]).max()), 6) if fin.any() else None,
                   "nonfinite": int((~np.isfinite(a)).sum()), "values": int(a.size)}
    return out


def _rate(a, b):
    return f"{a}/{b}" if b else "0/0"


def score(model, task, zc, zr, ids, line=None):
    """zc, zr = the cell's and the reference's outputs (np.load of the .npz); ids = the inputs in order; line = the
    cell's line id (the codec's offset)."""
    n = len(ids)
    vals = []
    if task == "detection":
        dec = (detr_boxes if model in DETR else ssd_boxes if model.startswith("ssdlite") else ppocr_boxes)
        nr = nc = pairs = 0
        ious = []
        for i in range(n):
            c, r = dec(outputs_of(zc, i)), dec(outputs_of(zr, i))
            p, io = pair_boxes(c, r)
            nr, nc, pairs = nr + len(r), nc + len(c), pairs + p
            ious += io
            vals.append({"id": ids[i], "ref": len(r), "cell": len(c), "pairs": p})
        ok = nr > 0 and pairs * 100 >= 95 * nr and pairs * 100 >= 95 * nc
        detail = {"pairs": pairs, "ref_boxes": nr, "cell_boxes": nc, "bar": "95 % both ways",
                  "pass": bool(ok) if nr else None}
        inst = {"pair_iou_mean": round(float(np.mean(ious)), 4) if ious else None,
                "pair_iou_min": round(float(np.min(ious)), 4) if ious else None}
        words = f"boxes paired {_rate(pairs, nr)} of the reference's, {_rate(pairs, nc)} of the cell's"
        return inst, detail, words, vals
    if task == "line-segments":
        tol = 0.02 * math.hypot(512, 512)
        nr = nc = pairs = 0
        for i in range(n):
            c, r = mlsd_lines(outputs_of(zc, i)), mlsd_lines(outputs_of(zr, i))
            p = pair_segments(c, r, tol)
            nr, nc, pairs = nr + len(r), nc + len(c), pairs + p
            vals.append({"id": ids[i], "ref": len(r), "cell": len(c), "pairs": p})
        ok = nr > 0 and pairs * 100 >= 95 * nr and pairs * 100 >= 95 * nc
        detail = {"pairs": pairs, "ref_segments": nr, "cell_segments": nc, "tol_px": round(tol, 2),
                  "bar": "95 % both ways", "pass": bool(ok) if nr else None}
        return {}, detail, f"segments paired {_rate(pairs, nr)} of the reference's, {_rate(pairs, nc)} of the cell's", vals
    if task == "keypoints":
        W, H, thr = POSE[model]
        tol = 0.02 * math.hypot(W, H)
        either = agree = 0
        dists = []
        for i in range(n):
            c, r = simcc_points(outputs_of(zc, i), W, H), simcc_points(outputs_of(zr, i), W, H)
            e = a = 0
            for (cx, cy, cs), (rx, ry, rs) in zip(c, r):
                vc = True if thr is None else (cs > thr)
                vr = True if thr is None else (rs > thr)
                if not (vc or vr):
                    continue
                e += 1
                d = math.hypot(cx - rx, cy - ry)
                if vc and vr:
                    dists.append(d)
                    a += int(d <= tol)
            either, agree = either + e, agree + a
            vals.append({"id": ids[i], "shown_either": e, "agree": a})
        ok = either > 0 and agree * 100 >= 95 * either
        detail = {"agree": agree, "shown_either": either, "tol_px": round(tol, 2), "show_threshold": thr,
                  "bar": "95 %", "pass": bool(ok) if either else None}
        inst = {"dist_px_median": round(float(np.median(dists)), 3) if dists else None,
                "dist_px_max": round(float(np.max(dists)), 3) if dists else None}
        return inst, detail, f"points agree {_rate(agree, either)} of those shown", vals
    if task == "mask":
        tp = np.zeros(2); fp = np.zeros(2); fn = np.zeros(2); rc = np.zeros(2)
        for i in range(n):
            oc, orr = outputs_of(zc, i)[0], outputs_of(zr, i)[0]
            if model.startswith("sam2"):
                c, r = oc.reshape(-1, 256, 256)[0], orr.reshape(-1, 256, 256)[0]
                thrv = 0.0
            else:
                c, r = oc.reshape(352, 352), orr.reshape(352, 352)
                thrv = 0.5
            rm = (r > thrv).astype(np.int8)
            cm = np.where(np.isfinite(c), (c > thrv).astype(np.int8), -1)
            row = {"id": ids[i]}
            for k in (0, 1):
                t_, f_, n_ = int(((cm == k) & (rm == k)).sum()), int(((cm == k) & (rm != k)).sum()), int(((cm != k) & (rm == k)).sum())
                tp[k] += t_; fp[k] += f_; fn[k] += n_; rc[k] += int((rm == k).sum())
                row[f"c{k}"] = [t_, f_, n_]
            vals.append(row)
        ious = [tp[k] / (tp[k] + fp[k] + fn[k]) for k in (0, 1) if rc[k] > 0]
        miou = float(np.mean(ious)) if ious else None
        ok = miou is not None and miou >= 0.95
        detail = {"miou": round(miou, 5) if miou is not None else None, "iou_background": round(float(tp[0] / max(tp[0] + fp[0] + fn[0], 1)), 5),
                  "iou_object": round(float(tp[1] / max(tp[1] + fp[1] + fn[1], 1)), 5) if rc[1] else None,
                  "bar": 0.95, "pass": bool(ok) if miou is not None else None}
        return {}, detail, f"mIoU {detail['miou']}", vals
    if task == "warp":
        within = total = 0
        psnrs = []
        for i in range(n):
            pc, pr = warp_positions(outputs_of(zc, i)[0]), warp_positions(outputs_of(zr, i)[0])
            d = np.hypot(pc[0] - pr[0], pc[1] - pr[1])
            w = int((np.isfinite(d) & (d <= 1.0)).sum())
            within, total = within + w, total + d.size
            row = {"id": ids[i], "within_1px": w, "pixels": int(d.size)}
            src = f"n{i:03d}__src"
            if src in zr.files:
                ps = psnr(warp_picture(zr[src], pc), warp_picture(zr[src], pr))
                row["psnr_db"] = round(ps, 2) if math.isfinite(ps) else None
                psnrs.append(ps)
            vals.append(row)
        ok = total > 0 and within * 100 >= 95 * total
        detail = {"within_1px": within, "pixels": total, "share": round(within / total, 5) if total else None,
                  "bar": "95 %", "pass": bool(ok)}
        fin = [p for p in psnrs if math.isfinite(p)]
        inst = {"picture_psnr_db_min": round(min(fin), 2) if fin else None, "picture_psnr_db_median": round(float(np.median(fin)), 2) if fin else None}
        return inst, detail, f"sampling within 1 px {detail['share']}", vals
    if task == "count":
        agree = 0
        for i in range(n):
            c, r = float(np.asarray(outputs_of(zc, i)[0], np.float64).sum()), float(np.asarray(outputs_of(zr, i)[0], np.float64).sum())
            tol = max(0.5, 0.02 * r)
            a = math.isfinite(c) and abs(c - r) <= tol
            agree += int(a)
            vals.append({"id": ids[i], "cell": round(c, 3), "ref": round(r, 3), "agree": bool(a)})
        ok = n > 0 and agree * 100 >= 95 * n
        detail = {"agree": agree, "inputs": n, "bar": "95 %", "pass": bool(ok)}
        return {}, detail, f"counts agree {_rate(agree, n)}", vals
    if task == "depth":
        s = 0.0
        cnt = 0
        for i in range(n):
            c, r = outputs_of(zc, i)[0].astype(np.float64).reshape(-1), outputs_of(zr, i)[0].astype(np.float64).reshape(-1)
            v = np.isfinite(r) & (r > 0)
            e = np.where(np.isfinite(c[v]), np.abs(c[v] - r[v]) / r[v], np.inf)
            s += float(e.sum()); cnt += int(v.sum())
            vals.append({"id": ids[i], "absrel": round(float(e.mean()), 6) if e.size else None, "pixels": int(v.sum())})
        absrel = s / cnt if cnt else None
        ok = absrel is not None and absrel <= 0.02
        detail = {"absrel": round(absrel, 6) if absrel is not None and math.isfinite(absrel) else absrel,
                  "pixels": cnt, "bar": 0.02, "pass": bool(ok) if cnt else None}
        return {}, detail, f"AbsRel {detail['absrel']}", vals
    if task == "retrieval":
        def emb(z):
            e = np.stack([outputs_of(z, i)[0].astype(np.float64).reshape(-1, outputs_of(z, i)[0].shape[-1]).mean(0) for i in range(n)])
            return e
        ec, er = emb(zc), emb(zr)

        def sims(e):
            ok_ = np.isfinite(e).all(1)
            with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
                nrm = np.linalg.norm(e, axis=1, keepdims=True)
                en = e / np.where(nrm > 0, nrm, 1)
                s_ = en @ en.T
            s_[~np.isfinite(s_)] = -np.inf
            s_[:, ~ok_] = -np.inf   # a non-finite embedding is nobody's neighbour, and has none (disagrees)
            np.fill_diagonal(s_, -np.inf)
            return s_, ok_
        sc, okc = sims(ec)
        sr, _ = sims(er)
        counted = agree = 0
        for i in range(n):
            o = np.argsort(-sr[i], kind="stable")
            margin = float(sr[i][o[0]] - sr[i][o[1]])
            jc = int(np.argmax(sc[i])) if okc[i] else -1
            cnt_ = margin >= 0.01
            counted += int(cnt_)
            agree += int(cnt_ and jc == int(o[0]))
            vals.append({"id": ids[i], "ref_top1": ids[int(o[0])], "cell_top1": ids[jc] if jc >= 0 else None,
                         "ref_margin": round(margin, 5), "counted": bool(cnt_)})
        ok = counted > 0 and agree * 100 >= 95 * counted
        detail = {"agree": agree, "counted": counted, "not_counted": n - counted, "bar": "95 %",
                  "pass": bool(ok) if counted else None}
        cosv = [float((ec[i] @ er[i]) / (np.linalg.norm(ec[i]) * np.linalg.norm(er[i]) + 1e-300))
                for i in range(n) if np.isfinite(ec[i]).all()]
        inst = {"embedding_cos_min": round(min(cosv), 6) if cosv else None, "nonfinite_embeddings": int((~okc).sum())}
        return inst, detail, f"top-1 neighbour {_rate(agree, counted)}" + (f" ({n - counted} not counted)" if n - counted else ""), vals
    if task in ("classification", "classification-stream"):
        keys = []
        if task == "classification-stream":
            t = 1
            while f"n000__f{t}__o0" in zr.files:
                t += 1
            frames = t - 1
            keys = [(f"n{i:03d}__f{t}__o0", f"{ids[i]}#f{t}") for i in range(n) for t in range(1, frames + 1)]
        else:
            nm = "head" if "n000__head" in zr.files else "o0"
            keys = [(f"n{i:03d}__{nm}", ids[i]) for i in range(n)]
        counted = agree = 0
        for k, iid in keys:
            lc, lr = zc[k].reshape(-1).astype(np.float64), zr[k].reshape(-1).astype(np.float64)
            pr = np.exp(lr - lr.max()); pr /= pr.sum()
            o = np.argsort(-pr, kind="stable")
            gap = float(pr[o[0]] - pr[o[1]])
            jc = int(np.argmax(lc)) if np.isfinite(lc).all() else -1
            cnt_ = gap >= 0.01
            counted += int(cnt_)
            agree += int(cnt_ and jc == int(o[0]))
            vals.append({"id": iid, "ref_top1": int(o[0]), "cell_top1": jc, "ref_gap": round(gap, 5), "counted": bool(cnt_)})
        ok = counted > 0 and agree * 100 >= 95 * counted
        detail = {"agree": agree, "counted": counted, "not_counted": len(keys) - counted, "bar": "95 %",
                  "pass": bool(ok) if counted else None}
        nc_ = len(keys) - counted
        return {}, detail, f"top-1 {_rate(agree, counted)}" + (f" ({nc_} not counted)" if nc_ else ""), vals
    if task == "codec":
        sc_, sr_ = [], []
        bad = 0
        off = CODEC_OFFSET.get(line, 0)
        for i in range(n):
            src = zr[f"n{i:03d}__src"].astype(np.float64).reshape(-1)[off:]
            c, r = outputs_of(zc, i)[0].astype(np.float64).reshape(-1), outputs_of(zr, i)[0].astype(np.float64).reshape(-1)
            m = min(c.size, src.size)
            b_ = (not np.isfinite(c).all()) or not np.any(c)
            bad += int(b_)
            vc = -math.inf if b_ else si_sdr(c[:m], src[:m])
            vr = si_sdr(r[:m], src[:m])
            sc_.append(vc); sr_.append(vr)
            vals.append({"id": ids[i], "si_sdr_cell": round(vc, 3) if math.isfinite(vc) else None, "si_sdr_ref": round(vr, 3)})
        mc, mr = float(np.mean(sc_)), float(np.mean(sr_))
        ok = bad == 0 and mc >= mr - 0.5
        detail = {"si_sdr_cell_mean": round(mc, 3) if math.isfinite(mc) else None, "si_sdr_ref_mean": round(mr, 3),
                  "bar": "reference mean - 0.5 dB", "bad_outputs": bad, "source_offset": off, "pass": bool(ok)}
        snr = []
        for i in range(n):
            c, r = outputs_of(zc, i)[0].astype(np.float64).reshape(-1), outputs_of(zr, i)[0].astype(np.float64).reshape(-1)
            if np.isfinite(c).all():
                snr.append(10 * math.log10((r @ r) / max(((c - r) @ (c - r)), 1e-300)))
        inst = {"snr_vs_reference_db_min": round(min(snr), 2) if snr else None}
        words = (f"SI-SDR {detail['si_sdr_cell_mean']} dB (reference {detail['si_sdr_ref_mean']} dB)" if not bad else
                 f"{bad} clips with a non-finite or all-zero output (reference SI-SDR {detail['si_sdr_ref_mean']} dB)")
        return inst, detail, words, vals
    raise ValueError(f"task {task}")


def reference_values(model, task, zr, ids, line=None):
    """The reference's own task values per input (models/<m>/reference/ref.json): what the app shows for each input
    on the reference, compact."""
    out = []
    if task == "detection":
        dec = (detr_boxes if model in DETR else ssd_boxes if model.startswith("ssdlite") else ppocr_boxes)
        nd = 4 if model in DETR else 1   # DETR boxes in [0, 1]; the others in input pixels
        for i, iid in enumerate(ids):
            b = dec(outputs_of(zr, i))
            out.append({"id": iid, "boxes": [[d[0], round(d[1], 3)] + [round(float(v), nd) for v in d[2:]] for d in b]})
    elif task == "line-segments":
        for i, iid in enumerate(ids):
            out.append({"id": iid, "segments": len(mlsd_lines(outputs_of(zr, i)))})
    elif task == "keypoints":
        W, H, thr = POSE[model]
        for i, iid in enumerate(ids):
            p = simcc_points(outputs_of(zr, i), W, H)
            out.append({"id": iid, "shown": sum(1 for _, _, s in p if thr is None or s > thr),
                        "points": [[round(x, 1), round(y, 1), round(s, 3)] for x, y, s in p]})
    elif task == "mask":
        for i, iid in enumerate(ids):
            o = outputs_of(zr, i)[0]
            m = o.reshape(-1, 256, 256)[0] > 0 if model.startswith("sam2") else o.reshape(352, 352) > 0.5
            out.append({"id": iid, "object_share": round(float(m.mean()), 5)})
    elif task == "warp":
        for i, iid in enumerate(ids):
            sx, sy = warp_positions(outputs_of(zr, i)[0])
            out.append({"id": iid, "source_x_px": [round(float(sx.min()), 2), round(float(sx.max()), 2)],
                        "source_y_px": [round(float(sy.min()), 2), round(float(sy.max()), 2)]})
    elif task == "count":
        for i, iid in enumerate(ids):
            out.append({"id": iid, "count": round(float(outputs_of(zr, i)[0].astype(np.float64).sum()), 3)})
    elif task == "depth":
        for i, iid in enumerate(ids):
            d = outputs_of(zr, i)[0].astype(np.float64).reshape(-1)
            out.append({"id": iid, "median_m": round(float(np.median(d)), 4), "min_m": round(float(d.min()), 4),
                        "max_m": round(float(d.max()), 4)})
    elif task == "retrieval":
        e = np.stack([outputs_of(zr, i)[0].astype(np.float64).reshape(-1, outputs_of(zr, i)[0].shape[-1]).mean(0)
                      for i in range(len(ids))])
        en = e / np.linalg.norm(e, axis=1, keepdims=True)
        s = en @ en.T
        np.fill_diagonal(s, -np.inf)
        for i, iid in enumerate(ids):
            o = np.argsort(-s[i], kind="stable")
            out.append({"id": iid, "top1": ids[int(o[0])], "margin": round(float(s[i][o[0]] - s[i][o[1]]), 5)})
    elif task in ("classification", "classification-stream"):
        if task == "classification-stream":
            t = 1
            while f"n000__f{t}__o0" in zr.files:
                t += 1
            keys = [(f"n{i:03d}__f{f}__o0", f"{iid}#f{f}") for i, iid in enumerate(ids) for f in range(1, t)]
        else:
            nm = "head" if "n000__head" in zr.files else "o0"
            keys = [(f"n{i:03d}__{nm}", iid) for i, iid in enumerate(ids)]
        for k, iid in keys:
            lr = zr[k].reshape(-1).astype(np.float64)
            pr = np.exp(lr - lr.max())
            pr /= pr.sum()
            o = np.argsort(-pr, kind="stable")
            out.append({"id": iid, "top1": int(o[0]), "p1": round(float(pr[o[0]]), 5),
                        "gap": round(float(pr[o[0]] - pr[o[1]]), 5)})
    elif task == "codec":
        for i, iid in enumerate(ids):
            src = zr[f"n{i:03d}__src"].astype(np.float64).reshape(-1)[CODEC_OFFSET.get(line, 0):]
            r = outputs_of(zr, i)[0].astype(np.float64).reshape(-1)
            m = min(r.size, src.size)
            out.append({"id": iid, "si_sdr_db": round(si_sdr(r[:m], src[:m]), 3)})
    else:
        raise ValueError(f"task {task}")
    return out


def si_sdr(est, ref):
    """Scale-invariant SDR in dB, both signals made zero-mean first."""
    ref = ref - ref.mean()
    est = est - est.mean()
    a = float(est @ ref) / max(float(ref @ ref), 1e-300)
    t = a * ref
    e = est - t
    return 10 * math.log10(max(float(t @ t), 1e-300) / max(float(e @ e), 1e-300))


# ---- dewarp (DocumentDewarper.kt) -----------------------------------------------------------------------------------

def warp_positions(bm):
    """Each of the 384 x 384 output pixels: the backward map read bilinearly (sampleBM), as the source position in
    pixels of the 256 x 256 model input. Returns (sx, sy) arrays [384, 384]."""
    BM, OUT = 128, 384
    g = np.asarray(bm, np.float64).reshape(2, BM, BM)
    by = np.arange(OUT) * (BM - 1.0) / (OUT - 1)
    bx = np.arange(OUT) * (BM - 1.0) / (OUT - 1)
    y0 = np.clip(by.astype(int), 0, BM - 1); x0 = np.clip(bx.astype(int), 0, BM - 1)
    y1 = np.minimum(y0 + 1, BM - 1); x1 = np.minimum(x0 + 1, BM - 1)
    fy = (by - y0)[:, None]; fx = (bx - x0)[None, :]
    res = []
    for pl in (0, 1):
        v00 = g[pl][np.ix_(y0, x0)]; v01 = g[pl][np.ix_(y0, x1)]
        v10 = g[pl][np.ix_(y1, x0)]; v11 = g[pl][np.ix_(y1, x1)]
        v = (v00 * (1 - fx) + v01 * fx) * (1 - fy) + (v10 * (1 - fx) + v11 * fx) * fy
        res.append((v + 1.0) * 0.5 * 255.0)
    return res[0], res[1]


def warp_picture(src, pos256):
    """The app's unwarped picture (sampleSrc on the full-resolution source) from positions in the 256 input."""
    img = np.asarray(src, np.float64)
    H, W = img.shape[:2]
    sx = pos256[0] / 255.0 * (W - 1)
    sy = pos256[1] / 255.0 * (H - 1)
    inside = np.isfinite(sx) & np.isfinite(sy) & (sx >= 0) & (sy >= 0) & (sx <= W - 1) & (sy <= H - 1)
    sxc = np.where(inside, sx, 0); syc = np.where(inside, sy, 0)
    x0 = sxc.astype(int); y0 = syc.astype(int)
    x1 = np.minimum(x0 + 1, W - 1); y1 = np.minimum(y0 + 1, H - 1)
    fx = (sxc - x0)[..., None]; fy = (syc - y0)[..., None]
    v = (img[y0, x0] * (1 - fx) + img[y0, x1] * fx) * (1 - fy) + (img[y1, x0] * (1 - fx) + img[y1, x1] * fx) * fy
    v = np.clip(np.floor(v), 0, 255)   # Kotlin toInt() truncates; the values are >= 0
    return np.where(inside[..., None], v, 0.0)


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return math.inf if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)
