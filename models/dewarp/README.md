---
license: mit
base_model: litert-community/DewarpNet-LiteRT
base_model_relation: quantized
library_name: onnx
pipeline_tag: image-to-image
---

# dewarp

`dewarp.tflite` from [litert-community/DewarpNet-LiteRT](https://huggingface.co/litert-community/DewarpNet-LiteRT/tree/dfc68e3e462a) converted to ONNX with tf2onnx 1.17.0, run on one Windows laptop on the Intel NPU, the Intel GPU, the RTX and the CPU. Its cells were copied from the vision zoo [`john-rocky/windows-zoo`](https://github.com/john-rocky/windows-zoo) with that zoo's line; the cells `task-*`, added on 2026-11-10, judge the same files on this model's task line (below).

Upstream licence: mit.

<!-- zoo:cells begin -->
## Lines

- Line `z1-internal` (internal agreement): every output: cos >= 0.99 against the ONNX Runtime CPU EP fp32 output (yolact: OpenVINO CPU f32), no NaN / Inf; a class output: argmax agreement >= 0.99. Written 2026-11-06T10:40:00+09:00 (john-rocky/windows-zoo PLAN.md section 3).
- Line `z1-tflite` (internal agreement): the ONNX file against the .tflite it was converted from (LiteRT interpreter, CPU, the same inputs): the PLAN section 3 line. Written 2026-11-06T16:30:00+09:00 (john-rocky/windows-zoo PLAN.md, Changes 2026-11-06 16:3x).
- Line `task-warp` (the task line): unwarp geometry, the app's sampling (john-rocky/LiteRT-Models@96e0bc1f/dewarp/.../DocumentDewarper.kt: each of the 384 x 384 output pixels reads the source where the bilinearly read 128 x 128 backward map points): the source position in pixels of the 256 x 256 model input; the cell's position is within 1.0 px of the reference's for >= 95 % of the output pixels of all inputs. Not the PSNR of the unwarped picture: a sub-pixel shift of a sharp edge moves a pixel's value by tens of levels with no visible change; the PSNR of the app's 384 x 384 picture is an instrument. Inputs: the 20 images of suites/vision/images.json, each stretched to 256 x 256, BGR / 255. Reference: the same fp32 ONNX file (the one the vision zoo Z1 measured) on the ONNX Runtime 1.31.1 CPU EP, on the same inputs (Z1's cpu cell). Written 2026-11-10T06:41:00+09:00 (PLAN.md section Z5 (the Z5 lane, written 06:41 JST before any run of its cells)).

## Cells

One JSON per cell (this model on one runtime, device and package) in [`results/`](results/): 11 files. Each one's `line.verdict` gives its verdict, its numbers and its line in one string; how the cell ran is in `recipe.toml` and in the run record the JSON names (`source.record`).
<!-- zoo:cells end -->
