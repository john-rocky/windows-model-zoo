---
license: apache-2.0
base_model: litert-community/DINOv2-ViT-S14-LiteRT
base_model_relation: quantized
library_name: onnx
pipeline_tag: image-feature-extraction
---

# dinov2_s_fp16

`dinov2_s_fp16.tflite` from [litert-community/DINOv2-ViT-S14-LiteRT](https://huggingface.co/litert-community/DINOv2-ViT-S14-LiteRT/tree/5b584fb82b92) converted to ONNX with tf2onnx 1.17.0, run on one Windows laptop on the Intel NPU, the Intel GPU, the RTX and the CPU. Its cells were copied from the vision zoo [`john-rocky/windows-zoo`](https://github.com/john-rocky/windows-zoo) with that zoo's line; the cells `task-*`, added on 2026-11-10, judge the same files on this model's task line (below).

Upstream licence: apache-2.0; base model: facebook/dinov2-small.

<!-- zoo:cells begin -->
## Lines

- Line `z1-internal` (internal agreement): every output: cos >= 0.99 against the ONNX Runtime CPU EP fp32 output (yolact: OpenVINO CPU f32), no NaN / Inf; a class output: argmax agreement >= 0.99. Written 2026-11-06T10:40:00+09:00 (john-rocky/windows-zoo PLAN.md section 3).
- Line `z1-tflite` (internal agreement): the ONNX file against the .tflite it was converted from (LiteRT interpreter, CPU, the same inputs): the PLAN section 3 line. Written 2026-11-06T16:30:00+09:00 (john-rocky/windows-zoo PLAN.md, Changes 2026-11-06 16:3x).
- Line `task-retrieval` (the task line): features, through a downstream task (the model has no task head; the app's own use, john-rocky/LiteRT-Models@96e0bc1f/dinov2/.../Dinov2Features.kt, is a picture of the top 3 principal components of the tokens, whose components flip and swap with small changes of the tokens): image retrieval, each image's embedding = the mean of its 1024 patch tokens; for each input, the most cosine-similar other input under the cell's embeddings is the one under the reference's embeddings on >= 95 % of the inputs, an input whose reference first and second similarities differ by < 0.01 not counted (how many: an instrument). Inputs: the 20 images of suites/vision/images.json, each stretched to 448 x 448, ImageNet normalization. Reference: the same fp32 ONNX file (the one the vision zoo Z1 measured) on the ONNX Runtime 1.31.1 CPU EP, on the same inputs (Z1's cpu cell). Written 2026-11-10T06:41:00+09:00 (PLAN.md section Z5 (the Z5 lane, written 06:41 JST before any run of its cells)).

## Cells

One JSON per cell (this model on one runtime, device and package) in [`results/`](results/): 12 files. Each one's `line.verdict` gives its verdict, its numbers and its line in one string; how the cell ran is in `recipe.toml` and in the run record the JSON names (`source.record`).
<!-- zoo:cells end -->
