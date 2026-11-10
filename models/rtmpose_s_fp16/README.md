---
license: apache-2.0
base_model: litert-community/RTMPose-s-LiteRT
base_model_relation: quantized
library_name: onnx
pipeline_tag: keypoint-detection
---

# rtmpose_s_fp16

`rtmpose_s_fp16.tflite` from [litert-community/RTMPose-s-LiteRT](https://huggingface.co/litert-community/RTMPose-s-LiteRT/tree/9cc07fa96de4) converted to ONNX with tf2onnx 1.17.0, run on one Windows laptop on the Intel NPU, the Intel GPU, the RTX and the CPU. Its cells were copied from the vision zoo [`john-rocky/windows-zoo`](https://github.com/john-rocky/windows-zoo) with that zoo's line; the cells `task-*`, added on 2026-11-10, judge the same files on this model's task line (below).

Upstream licence: apache-2.0; base model: open-mmlab/mmpose.

<!-- zoo:cells begin -->
## Lines

- Line `z1-internal` (internal agreement): every output: cos >= 0.99 against the ONNX Runtime CPU EP fp32 output (yolact: OpenVINO CPU f32), no NaN / Inf; a class output: argmax agreement >= 0.99. Written 2026-11-06T10:40:00+09:00 (john-rocky/windows-zoo PLAN.md section 3).
- Line `z1-tflite` (internal agreement): the ONNX file against the .tflite it was converted from (LiteRT interpreter, CPU, the same inputs): the PLAN section 3 line. Written 2026-11-06T16:30:00+09:00 (john-rocky/windows-zoo PLAN.md, Changes 2026-11-06 16:3x).
- Line `task-keypoints` (the task line): keypoints, the app's points (john-rocky/LiteRT-Models@96e0bc1f/rtmpose/.../RtmPoseEstimator.kt: per point the argmax of the SimCC x and y bins, x = bin / 2, y = bin / 2, score = the mean of the two maxima; shown when score > 0.3 (MainActivity.kt)): of the points either side shows, a point agrees when both show it and the two positions are within 0.02 of the model input's diagonal (6.4 px of 192 x 256); the agreeing points are >= 95 % of the points either side shows, summed over the inputs. Inputs: the images of suites/vision/images.json that show the model's subject (the app takes a crop of one person; chosen by looking at the images before any run): gaze, rtmface, rtmpose, saliency, tddfa (5 images); each centre-cropped to 3:4 and resized to 192 x 256, (x - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375] on 0-255 RGB. Reference: the same fp32 ONNX file (the one the vision zoo Z1 measured) on the ONNX Runtime 1.31.1 CPU EP, on the same inputs (Z1's cpu cell). Written 2026-11-10T06:41:00+09:00 (PLAN.md section Z5 (the Z5 lane, written 06:41 JST before any run of its cells)).

## Cells

One JSON per cell (this model on one runtime, device and package) in [`results/`](results/): 14 files. Each one's `line.verdict` gives its verdict, its numbers and its line in one string; how the cell ran is in `recipe.toml` and in the run record the JSON names (`source.record`).
<!-- zoo:cells end -->
