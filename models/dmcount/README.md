---
license: mit
base_model: litert-community/DM-Count-Crowd-LiteRT
base_model_relation: quantized
library_name: onnx
pipeline_tag: image-to-image
---

# dmcount

`dmcount.tflite` from [litert-community/DM-Count-Crowd-LiteRT](https://huggingface.co/litert-community/DM-Count-Crowd-LiteRT/tree/5934338595e4) converted to ONNX with tf2onnx 1.17.0, run on one Windows laptop on the Intel NPU, the Intel GPU, the RTX and the CPU. Its cells were copied from the vision zoo [`john-rocky/windows-zoo`](https://github.com/john-rocky/windows-zoo) with that zoo's line; the cells `task-*`, added on 2026-11-10, judge the same files on this model's task line (below).

Upstream licence: mit.

<!-- zoo:cells begin -->
## Lines

- Line `z1-internal` (internal agreement): every output: cos >= 0.99 against the ONNX Runtime CPU EP fp32 output (yolact: OpenVINO CPU f32), no NaN / Inf; a class output: argmax agreement >= 0.99. Written 2026-11-06T10:40:00+09:00 (john-rocky/windows-zoo PLAN.md section 3).
- Line `z1-tflite` (internal agreement): the ONNX file against the .tflite it was converted from (LiteRT interpreter, CPU, the same inputs): the PLAN section 3 line. Written 2026-11-06T16:30:00+09:00 (john-rocky/windows-zoo PLAN.md, Changes 2026-11-06 16:3x).
- Line `task-count` (the task line): count, the app's number (john-rocky/LiteRT-Models@96e0bc1f/crowdcount/.../CrowdCounter.kt: the sum of the 64 x 64 density map, shown rounded by MainActivity.kt): |cell count - reference count| <= max(0.5, 2 % of the reference count) on >= 95 % of the inputs. Inputs: the 20 images of suites/vision/images.json, each stretched to 512 x 512, ImageNet normalization. Reference: the same fp32 ONNX file (the one the vision zoo Z1 measured) on the ONNX Runtime 1.31.1 CPU EP, on the same inputs (Z1's cpu cell). Written 2026-11-10T06:41:00+09:00 (PLAN.md section Z5 (the Z5 lane, written 06:41 JST before any run of its cells)).

## Cells

One JSON per cell (this model on one runtime, device and package) in [`results/`](results/): 11 files. Each one's `line.verdict` gives its verdict, its numbers and its line in one string; how the cell ran is in `recipe.toml` and in the run record the JSON names (`source.record`).
<!-- zoo:cells end -->
