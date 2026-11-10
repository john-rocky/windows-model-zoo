---
license: apache-2.0
base_model: litert-community/M-LSD-tiny-LiteRT
base_model_relation: quantized
library_name: onnx
pipeline_tag: image-segmentation
---

# mlsd_fp16

`mlsd_fp16.tflite` from [litert-community/M-LSD-tiny-LiteRT](https://huggingface.co/litert-community/M-LSD-tiny-LiteRT/tree/ae7c5698d5bb) converted to ONNX with tf2onnx 1.17.0, run on one Windows laptop on the Intel NPU, the Intel GPU, the RTX and the CPU. Its cells were copied from the vision zoo [`john-rocky/windows-zoo`](https://github.com/john-rocky/windows-zoo) with that zoo's line; the cells `task-*`, added on 2026-11-10, judge the same files on this model's task line (below).

Upstream licence: apache-2.0; base model: navervision/mlsd.

<!-- zoo:cells begin -->
## Lines

- Line `z1-internal` (internal agreement): every output: cos >= 0.99 against the ONNX Runtime CPU EP fp32 output (yolact: OpenVINO CPU f32), no NaN / Inf; a class output: argmax agreement >= 0.99. Written 2026-11-06T10:40:00+09:00 (john-rocky/windows-zoo PLAN.md section 3).
- Line `z1-tflite` (internal agreement): the ONNX file against the .tflite it was converted from (LiteRT interpreter, CPU, the same inputs): the PLAN section 3 line. Written 2026-11-06T16:30:00+09:00 (john-rocky/windows-zoo PLAN.md, Changes 2026-11-06 16:3x).
- Line `task-line-segments` (the task line): line segments, the app's segments (john-rocky/LiteRT-Models@96e0bc1f/mlsd/.../MlsdDetector.kt: the sigmoid of the centre map > 0.10 with no larger value in its 3 x 3 neighbourhood, the 200 highest, the endpoints from the four displacement maps, a segment kept when its endpoints are more than 20 map pixels apart; endpoints in pixels of the 512 x 512 input): a cell segment pairs with a reference segment when both endpoints are within 0.02 of the input's diagonal (14.48 px), in either endpoint order, one to one, the cell's segments taken in descending score; the pairs are >= 95 % of the reference's segments and >= 95 % of the cell's, summed over the inputs. Inputs: the 20 images of suites/vision/images.json, each centre-square cropped to 512 x 512, RGB / 127.5 - 1 and the constant fourth channel 1 / 127.5 - 1. Reference: the same fp32 ONNX file (the one the vision zoo Z1 measured) on the ONNX Runtime 1.31.1 CPU EP, on the same inputs (Z1's cpu cell). Written 2026-11-10T06:41:00+09:00 (PLAN.md section Z5 (the Z5 lane, written 06:41 JST before any run of its cells)).

## Cells

One JSON per cell (this model on one runtime, device and package) in [`results/`](results/): 12 files. Each one's `line.verdict` gives its verdict, its numbers and its line in one string; how the cell ran is in `recipe.toml` and in the run record the JSON names (`source.record`).
<!-- zoo:cells end -->
