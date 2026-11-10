---
license: apache-2.0
base_model: litert-community/PP-OCRv5-LiteRT
base_model_relation: quantized
library_name: onnx
pipeline_tag: image-to-text
---

# ppocr_det_fp16

`ppocr_det_fp16.tflite` from [litert-community/PP-OCRv5-LiteRT](https://huggingface.co/litert-community/PP-OCRv5-LiteRT/tree/f4bcc36c8191) converted to ONNX with tf2onnx 1.17.0, run on one Windows laptop on the Intel NPU, the Intel GPU, the RTX and the CPU. Its cells were copied from the vision zoo [`john-rocky/windows-zoo`](https://github.com/john-rocky/windows-zoo) with that zoo's line; the cells `task-*`, added on 2026-11-10, judge the same files on this model's task line (below).

Upstream licence: apache-2.0; base model: ['PaddlePaddle/PP-OCRv5_mobile_det', 'PaddlePaddle/PP-OCRv5_mobile_rec'].

<!-- zoo:cells begin -->
## Lines

- Line `z1-internal` (internal agreement): every output: cos >= 0.99 against the ONNX Runtime CPU EP fp32 output (yolact: OpenVINO CPU f32), no NaN / Inf; a class output: argmax agreement >= 0.99. Written 2026-11-06T10:40:00+09:00 (john-rocky/windows-zoo PLAN.md section 3).
- Line `z1-tflite` (internal agreement): the ONNX file against the .tflite it was converted from (LiteRT interpreter, CPU, the same inputs): the PLAN section 3 line. Written 2026-11-06T16:30:00+09:00 (john-rocky/windows-zoo PLAN.md, Changes 2026-11-06 16:3x).
- Line `task-detection` (the task line): text detection, the app's boxes (john-rocky/LiteRT-Models@96e0bc1f/ppocr/.../PpocrDetector.kt: probability > 0.3, 4-connected regions, width and height >= 6, mean probability >= 0.5, the box padded by 0.35 x min(w, h) clamped to 2..24 px, in the 640 x 640 input): a cell box pairs with a reference box at IoU >= 0.5 (one class), one to one, the cell's boxes taken in descending mean probability; the pairs are >= 95 % of the reference's boxes and >= 95 % of the cell's boxes, summed over the inputs (a set without a reference box has no verdict). Inputs: the 20 images of suites/vision/images.json, each letterboxed on white to 640 x 640, ImageNet normalization. Reference: the same fp32 ONNX file (the one the vision zoo Z1 measured) on the ONNX Runtime 1.31.1 CPU EP, on the same inputs (Z1's cpu cell). Written 2026-11-10T06:41:00+09:00 (PLAN.md section Z5 (the Z5 lane, written 06:41 JST before any run of its cells)).

## Cells

One JSON per cell (this model on one runtime, device and package) in [`results/`](results/): 13 files. Each one's `line.verdict` gives its verdict, its numbers and its line in one string; how the cell ran is in `recipe.toml` and in the run record the JSON names (`source.record`).
<!-- zoo:cells end -->
