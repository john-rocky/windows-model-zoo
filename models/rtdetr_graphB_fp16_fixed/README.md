---
license: apache-2.0
base_model: litert-community/RT-DETRv2-S-LiteRT
base_model_relation: quantized
library_name: onnx
pipeline_tag: object-detection
---

# rtdetr_graphB_fp16_fixed

`rtdetr_graphB_fp16.tflite` from [litert-community/RT-DETRv2-S-LiteRT](https://huggingface.co/litert-community/RT-DETRv2-S-LiteRT/tree/60f16dbbf37d) converted to ONNX with tf2onnx 1.17.0 with the FULLY_CONNECTED bias fix, run on one Windows laptop on the Intel NPU, the Intel GPU, the RTX and the CPU. Its cells were copied from the vision zoo [`john-rocky/windows-zoo`](https://github.com/john-rocky/windows-zoo) with that zoo's line; the cells `task-*`, added on 2026-11-10, judge the same files on this model's task line (below).

Upstream licence: apache-2.0; base model: ['PekingU/rtdetr_v2_r18vd'].

<!-- zoo:cells begin -->
## Lines

- Line `z1-internal` (internal agreement): every output: cos >= 0.99 against the ONNX Runtime CPU EP fp32 output (yolact: OpenVINO CPU f32), no NaN / Inf; a class output: argmax agreement >= 0.99. Written 2026-11-06T10:40:00+09:00 (john-rocky/windows-zoo PLAN.md section 3).
- Line `z1-tflite` (internal agreement): the ONNX file against the .tflite it was converted from (LiteRT interpreter, CPU, the same inputs): the PLAN section 3 line. Written 2026-11-06T16:30:00+09:00 (john-rocky/windows-zoo PLAN.md, Changes 2026-11-06 16:3x).
- Line `task-detection` (the task line): detection, the app's boxes (john-rocky/LiteRT-Models@96e0bc1f/rtdetr/.../RtDetr.kt: per query the class of the largest of the 80 logits, score = its sigmoid, kept at >= 0.4, then per-class NMS at IoU > 0.7): a cell box pairs with a reference box of the same class at IoU >= 0.5, one to one, the cell's boxes taken in descending score; the pairs are >= 95 % of the reference's boxes and >= 95 % of the cell's boxes, summed over the inputs (a set without a reference box has no verdict). Inputs: the 20 images of suites/vision/images.json, each stretched to 640 x 640, RGB / 255, through rtdetr_graphA_fp16_fixed in fp32 on the ONNX Runtime CPU EP and the app's host step (the top 300 queries, host_params.bin) into graph B's three inputs. Reference: the same fp32 ONNX file (the one the vision zoo Z1 measured) on the ONNX Runtime 1.31.1 CPU EP, on the same inputs (Z1's cpu cell). Written 2026-11-10T06:41:00+09:00 (PLAN.md section Z5 (the Z5 lane, written 06:41 JST before any run of its cells)).

## Cells

One JSON per cell (this model on one runtime, device and package) in [`results/`](results/): 8 files. Each one's `line.verdict` gives its verdict, its numbers and its line in one string; how the cell ran is in `recipe.toml` and in the run record the JSON names (`source.record`).
<!-- zoo:cells end -->
