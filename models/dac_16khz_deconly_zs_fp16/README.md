---
license: mit
base_model: mlboydaisuke/DAC-16kHz-LiteRT
base_model_relation: quantized
library_name: onnx
pipeline_tag: audio-to-audio
---

# dac_16khz_deconly_zs_fp16

`dac_16khz_deconly_zs_fp16.tflite` from [mlboydaisuke/DAC-16kHz-LiteRT](https://huggingface.co/mlboydaisuke/DAC-16kHz-LiteRT/tree/376f41e9b4e6) converted to ONNX with tf2onnx 1.17.0, run on one Windows laptop on the Intel NPU, the Intel GPU, the RTX and the CPU. Its cells were copied from the vision zoo [`john-rocky/windows-zoo`](https://github.com/john-rocky/windows-zoo) with that zoo's line; the cells `task-*`, added on 2026-11-10, judge the same files on this model's task line (below).

Upstream licence: mit.

<!-- zoo:cells begin -->
## Lines

- Line `z1-internal` (internal agreement): every output: cos >= 0.99 against the ONNX Runtime CPU EP fp32 output (yolact: OpenVINO CPU f32), no NaN / Inf; a class output: argmax agreement >= 0.99. Written 2026-11-06T10:40:00+09:00 (john-rocky/windows-zoo PLAN.md section 3).
- Line `z1-tflite` (internal agreement): the ONNX file against the .tflite it was converted from (LiteRT interpreter, CPU, the same inputs): the PLAN section 3 line. Written 2026-11-06T16:30:00+09:00 (john-rocky/windows-zoo PLAN.md, Changes 2026-11-06 16:3x).
- Line `task-codec` (the task line): codec round trip, the app's (john-rocky/LiteRT-Models@96e0bc1f/dac/.../DacCodec.kt: encoder, the RVQ of DacRVQ.kt with dac_rvq.bin, decoder): the SI-SDR of the decoded 15,992 samples against the first 15,992 samples of the input clip; the cell's mean over the clips >= the reference's mean - 0.5 dB, and no clip with a non-finite or all-zero output. Inputs: the 120 clips of suites/audio/kws120.json (pcm / 32768), each through dac_16khz_encoder_fp16 in fp32 on the ONNX Runtime CPU EP and the app's RVQ (encode, decode) into the decoder. Reference: the same fp32 ONNX file (the one the vision zoo Z1 measured) on the ONNX Runtime 1.31.1 CPU EP, on the same inputs (Z1's cpu cell). Written 2026-11-10T06:41:00+09:00 (PLAN.md section Z5 (the Z5 lane, written 06:41 JST before any run of its cells)). Superseded by `task-codec-aligned` (2026-11-10T07:24:02+09:00: compares the decoded samples with the first 15,992 input samples, 8 samples off the decoder's output (PLAN.md section Z5, change 1)).
- Line `task-codec-aligned` (the task line): codec round trip, the app's (john-rocky/LiteRT-Models@96e0bc1f/dac/.../DacCodec.kt: encoder, the RVQ of DacRVQ.kt with dac_rvq.bin, decoder): the SI-SDR of the decoded 15,992 samples against input samples 8 to 15,999 (the decoder's output starts 8 samples into the clip: 16,000 - 15,992, and the lag of the largest correlation on all 120 reference clips); the cell's mean over the clips >= the reference's mean - 0.5 dB, and no clip with a non-finite or all-zero output. Inputs: the 120 clips of suites/audio/kws120.json (pcm / 32768), each through dac_16khz_encoder_fp16 in fp32 on the ONNX Runtime CPU EP and the app's RVQ (encode, decode) into the decoder. Reference: the same fp32 ONNX file (the one the vision zoo Z1 measured) on the ONNX Runtime 1.31.1 CPU EP, on the same inputs (Z1's cpu cell). Written 2026-11-10T07:24:02+09:00 (PLAN.md section Z5, change 1 (the Z5 lane: replaces task-codec, whose comparison was 8 samples off; written after the reference ran, before any cell result was read)).

## Cells

One JSON per cell (this model on one runtime, device and package) in [`results/`](results/): 11 files. Each one's `line.verdict` gives its verdict, its numbers and its line in one string; how the cell ran is in `recipe.toml` and in the run record the JSON names (`source.record`).
<!-- zoo:cells end -->
