---
license: apache-2.0
base_model: litert-community/wav2vec2-keyword-spotting
base_model_relation: quantized
library_name: onnx
pipeline_tag: audio-classification
---

# w2v2_head_fp16

`w2v2_head_fp16.tflite` from [litert-community/wav2vec2-keyword-spotting](https://huggingface.co/litert-community/wav2vec2-keyword-spotting/tree/4f30d050bc19) converted to ONNX with tf2onnx 1.17.0, run on one Windows laptop on the Intel NPU, the Intel GPU, the RTX and the CPU. Its cells were copied from the vision zoo [`john-rocky/windows-zoo`](https://github.com/john-rocky/windows-zoo) with that zoo's line; the cells `task-*`, added on 2026-11-10, judge the same files on this model's task line (below).

Upstream licence: apache-2.0; base model: superb/wav2vec2-base-superb-ks.

<!-- zoo:cells begin -->
## Lines

- Line `z1-internal` (internal agreement): every output: cos >= 0.99 against the ONNX Runtime CPU EP fp32 output (yolact: OpenVINO CPU f32), no NaN / Inf; a class output: argmax agreement >= 0.99. Written 2026-11-06T10:40:00+09:00 (john-rocky/windows-zoo PLAN.md section 3).
- Line `z1-tflite` (internal agreement): the ONNX file against the .tflite it was converted from (LiteRT interpreter, CPU, the same inputs): the PLAN section 3 line. Written 2026-11-06T16:30:00+09:00 (john-rocky/windows-zoo PLAN.md, Changes 2026-11-06 16:3x).
- Line `task-top1` (the task line): keyword classification, the app's argmax of the 12 logits (john-rocky/LiteRT-Models@96e0bc1f/wav2vec2-kws/.../Wav2Vec2Kws.kt): the cell's top-1 equals the reference's on >= 95 % of the clips, a clip whose reference top-1 and top-2 probabilities differ by < 0.01 not counted. Inputs: the 120 clips of suites/audio/kws120.json (Speech Commands v0.01 test, 10 per label), each scaled as the app's microphone path (pcm / max(1, peak) x 0.5), through w2v2_frontend_fp16 in fp32 on the ONNX Runtime CPU EP into the head. Reference: the same fp32 ONNX file (the one the vision zoo Z1 measured) on the ONNX Runtime 1.31.1 CPU EP, on the same inputs (Z1's cpu cell). Written 2026-11-10T06:41:00+09:00 (PLAN.md section Z5 (the Z5 lane, written 06:41 JST before any run of its cells)).

## Cells

One JSON per cell (this model on one runtime, device and package) in [`results/`](results/): 11 files. Each one's `line.verdict` gives its verdict, its numbers and its line in one string; how the cell ran is in `recipe.toml` and in the run record the JSON names (`source.record`).
<!-- zoo:cells end -->
