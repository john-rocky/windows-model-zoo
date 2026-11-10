# PLAN: the lines

A line is a pass / fail rule written before the cells it judges run. Its time is in each recipe's `[[line]].written`,
and `tools/lint.py` checks it against the start of every cell it judges. A rule is not edited after a run: a change is
added under its section with its time, its reason and the value it replaced.

This repo was cut on 2026-11-10 from a private working repo where the cells ran, and where this plan was written in
Japanese. This file is its English text for the parts that concern these 28 models: the forms of section 1 (written
2026-11-10 00:11 JST), section Z5 and section Z6. Z5 and Z6 are the two passes of 2026-11-10 that judged the cells
copied from the vision zoo ([`john-rocky/windows-zoo`](https://github.com/john-rocky/windows-zoo)) and the LLM zoo
([`john-rocky/windows-llm-zoo`](https://github.com/john-rocky/windows-llm-zoo)) again on task lines; the recipes' rules
and the cells cite these section names. The times are those of the original text (JST). File names are this repo's:
`Z5_RESULTS.md` and `Z6_RESULTS.md` there are the local `TASK_PAIRS_VISION.md` and `TASK_PAIRS_LLM.md` here
(`tools/task_pairs_vision.py`, `tools/task_pairs_llm.py`), and the copy tools (`migrate/` there) stayed in that repo.

## 1. Lines

A line is written before any run it judges, with its time in `[[line]].written`. A cell's `line.kind` is `task` (the
model's job), `internal` (agreement of the raw outputs), `text-integrity` (the text is well formed) or `reference` (the
cell is the reference). Internal agreement (cosine, token match, hidden states) stays in a task cell as an instrument
and is not used for its verdict. The reference = the same inputs, the same prompt and the same decoding, with the
original model in fp32 on the CPU (bf16 where fp32 does not fit, and said so).

### 1a. Vision

Each model gets its task value (section Z5 sets them for the 23 vision models here); the forms per task type:

| task type | the value compared (the cell's output and the reference's through the same post-processing) | line (over the whole input set) |
|---|---|---|
| classification | the top-1 class | top-1 equal to the reference's on ≥ 95 % of the inputs (inputs where the reference's top-1 and top-2 probabilities differ by under 0.01 are not counted; how many is an instrument) |
| detection | the boxes after the model's own post-processing (score threshold, NMS) | pairs with the reference's boxes (above the threshold) of the same class at IoU ≥ 0.5, ≥ 95 % both ways (F1) |
| segmentation, matting | the argmax mask (matting: alpha > 0.5) | mIoU with the reference's mask ≥ 0.95 (the mean over the classes the reference has) |
| depth | the depth map (relative depth scaled by the median) | AbsRel ≤ 0.02 |
| pose, points | the point coordinates (SimCC: argmax) | points whose distance over the object's box diagonal is ≤ 0.02, ≥ 95 % (PCK) |
| image to image | the output rounded to uint8 | PSNR ≥ 40 dB against the reference's output |
| text (OCR recognition) | the decoded string | CER against the reference's string ≤ 0.5 pt |
| features only (an encoder with no task head) | — | line kind `internal` (cos ≥ 0.99), with one line of reason in the recipe |

- The input set is the model's recipe's (file URLs and sha256s, preprocessing). A model with a single input has a line
  that can be written but is weak: its card gives the number of inputs.
- Instruments: per output cosine, max |d| and NaN / Inf, the top-5 overlap, the number of inputs not counted.

### 1b. LLM: the qa20 line

A fixed set of questions answered greedily, compared with the reference's answers, and one chat turn that must end on
the end token. [How to read the qa20 line](#how-to-read-the-qa20-line) below says what it counts and what it does not.

- The questions: `suites/llm/qa20.jsonl` (the 20 below, in English, each with a short fixed answer). Each question is
  wrapped as one user turn in the model's own chat template (the reference step keeps in its record the string
  transformers' `apply_chat_template(add_generation_prompt=True, enable_thinking=False)` renders, and the cell's
  runner passes that string as it is: no template of the runtime). Greedy, at most 48 new tokens.
- The comparison: the first line of the answer, normalized (leading and trailing spaces dropped, lower case, `"` `'`
  `` ` `` `*` around it and a final `.` `!` `?` dropped, spaces collapsed to one), agrees when it is the same string as
  the reference's answer normalized the same way.
- **The line**: agreement on ≥ 18 of 20, and the gate prompt "Reply with only the number: 1+1=?" (wrapped the same
  way) ends on the end token (the model's end of turn) within 16 new tokens.
- Instruments: the number of questions whose full answer equals the reference's, the number that ended on the end
  token within 48 tokens, the number of generated tokens.
- The 20 questions (in this order):
  1. What is the capital of France? Answer with one word.
  2. What is 7 + 5? Reply with only the number.
  3. What is 12 * 3? Reply with only the number.
  4. Spell the word "cat" backwards. Reply with only the result.
  5. Which planet is known as the Red Planet? Answer with one word.
  6. How many days are in a week? Reply with only the number.
  7. What is the opposite of "hot"? Answer with one word.
  8. Translate "thank you" into Japanese. Reply with only the translation.
  9. What color do you get by mixing blue and yellow? Answer with one word.
  10. What is the chemical symbol for gold? Reply with only the symbol.
  11. Is 17 a prime number? Answer yes or no.
  12. What is the largest ocean on Earth? Reply with only the name.
  13. Who wrote "Romeo and Juliet"? Reply with only the name.
  14. What is 100 divided by 4? Reply with only the number.
  15. What comes next: 2, 4, 6, 8? Reply with only the number.
  16. At what temperature in Celsius does water freeze? Reply with only the number.
  17. Which is heavier, a kilogram of iron or a kilogram of feathers? Answer "iron", "feathers" or "same".
  18. What is the square root of 81? Reply with only the number.
  19. Write the word "hello" in uppercase. Reply with only the word.
  20. How many legs does a spider have? Reply with only the number.

### Changes to section 1

(2026-11-10 00:45 JST, before any number)

- Section 1b, a clarification (the line is the same): greedy = `do_sample` False, `num_beams` 1, `repetition_penalty`
  1.0. Some checkpoints' `generation_config.json` set a repetition penalty for sampling (1.1 in the one read then), and
  it applies in greedy decoding too, so both sides set 1.0.
- The vision input set: the vision zoo's 23 input files are 20 images (three pairs have the same bytes, by sha256). The
  set holds each image once: `suites/vision/images.json`, with `also` = the same bytes under another app's path.
  Before: 23 images.

## How to read the qa20 line

(Written 2026-11-10 when this repo was cut, after the runs. It explains section 1b and changes nothing in it.)

Twenty short questions, each asking for one word, name or number, go to the cell's package on its runtime and device.
Each is wrapped in the model's own chat template as transformers renders it (`enable_thinking=False`) and passed as that
string or as its token ids, with no template of the runtime and no special token added (section Z6 gives the form per
runtime). Decoding is greedy on both sides (repetition penalty 1.0), at most 48 new tokens. The first line of each
answer, normalized (trimmed, lower case, quotes, backticks or asterisks around it and a final `.` `!` `?` removed, spaces
collapsed), is compared with the first line of the same model's answer in transformers on the same PC's CPU, in fp32
(gemma-4-E4B-it in bf16: its fp32 weights alone would take 29.8 of the PC's 31.5 GiB, "Z6 changes" item 9): the
reference. A cell passes when at least 18 of the 20 agree
and one more prompt, `Reply with only the number: 1+1=?`, ends on the end token within 16 new tokens (only the ending is
checked, not the answer).

The line counts agreement with the reference, not correctness graded by a person. An answer in another form counts as
different (`12 * 3 = 36. so, the answer is 36` against `12 * 3 = 36`), and the reference is not an answer key: the fp32
model answers some questions wrongly itself (Qwen3-0.6B gives `1` for the number of days in a week), and a cell that
answers those rightly counts as different. Each cell is one untimed run. A FAIL says that this package, on that runtime
and device, did not reach the line; it is not a measure of the model or of the runtime in general.

## Z5. The vision zoo's 47 cells that fail its internal line, judged on task lines

(Written 2026-11-10 06:41 JST, before any number of this pass.)

The cells: 23 models, 47 cells = the copied cells of the vision zoo with `line.kind = internal` and `line.pass = false`
(`tools/lint.py`'s "internal 47 fail"; the 8 cells of the TFLite comparison are on another line and not among them).
The copied cells and their line (`z1-internal`) stay. Each new cell has the id `task-<the vision zoo's cell>`, and its
line is the `[[line]]` below in each recipe (`written` = this section's time).

Common to the 23 models:

- The form = the file the vision zoo measured, as it is: on the PC `C:\src\zoo\onnx\<m>.onnx` (fp32),
  `fp16\<m>.onnx` (`cuda16`), `blobs\<m>.blob` (`ovnpu_pre`), checked against the sha256 in the vision zoo's cell JSON
  (`model_sha256`, `fp16_convert.sha256`, `blob`) before use. The runtimes' versions = the vision zoo's pins (ORT
  1.31.1, onnxruntime-ep-openvino 1.8.1, OpenVINO 2026.4.1, onnxruntime-gpu 1.31.1 with the CUDA 13 wheels:
  `pc/req/z1-*.txt`).
- The reference = the same fp32 ONNX on the ORT 1.31.1 CPU EP with the same inputs (the vision zoo's `cpu` form). The
  upstream graphs (DETR's graph A, SAM2's encoder, the wav2vec2 frontend, DAC's encoder) also run once in fp32 on the
  CPU, and the host steps between them (DETR's top-300 and MLP, SAM2's point embedding, DAC's RVQ) are numpy copies of
  the vision zoo's `pc/make_inputs.py`. The upstream of a base model is the base one, of a `_fixed` model the `_fixed`
  one (the vision zoo's rule).
- Preprocessing = the vision zoo's `inputs_spec.json` per model (geometry, normalization, order, with the app code's
  path and line), on each input of the set. Post-processing = a copy of the app's code
  (`john-rocky/LiteRT-Models` 96e0bc1f) on both the cell and the reference.
- The input sets (the images were looked at and chosen before any number): (a) a model that takes the whole picture =
  the 20 images of `suites/vision/images.json`. (b) A top-down pose model that takes a crop of one subject = only the
  images that show the subject (the app passes a crop): a human body = gaze, rtmface, rtmpose, saliency, tddfa (5); a
  face = gaze, hsemotion, rtmface, tddfa (4); a hand = rtmhand (1: a set of one is weak, the card gives the number);
  an animal = dfine, dinov2, rtmanimal (3). (c) Sound = `suites/audio/kws120.json` (the Speech Commands v0.01 test set,
  12 labels × the first 10 files by name = 120 clips, 16 kHz 1 s, CC BY 4.0, the archive's sha256 baa084f6…).
- Pass or fail on the sums over the whole set (not the mean of per-input rates). A non-finite compared value does not
  agree (boxes, points and segments do not pair, a pixel is in another class, a depth error is infinite, a sound's
  SI-SDR is −∞).
- Instruments (not used for the verdict): per output cosine, max |d|, NaN / Inf (the vision zoo's line values), and
  each task's extra values (inputs not counted, the IoU of the pairs, score differences, PSNR and so on).

| model | the task (the app's code) | the input set | the line (id) |
|---|---|---|---|
| dfine_graphB_fp16, dfine_graphB_fp16_fixed, rtdetr_graphB_fp16, rtdetr_graphB_fp16_fixed | detection: per query the class of the largest of 80 logits, score = sigmoid, ≥ 0.4, per-class NMS at IoU > 0.7 (`DFine.kt`, `RtDetr.kt`) | 20 images → graph A (CPU fp32) → host (top-300, host_params.bin) → graph B's 3 inputs | `task-detection`: pairs one to one with the same class and IoU ≥ 0.5 (the cell's boxes in descending score); the pairs ≥ 95 % of the reference's boxes and ≥ 95 % of the cell's |
| ssdlite_mobilenetv3_320_fp16 | detection: softmax over 91, the largest of classes 1 to 90, ≥ 0.4, BoxCoder(10,10,5,5), per-class NMS at IoU > 0.55, the first 100 (`SSDLiteDetector.kt`) | 20 images | `task-detection` (as above) |
| ppocr_det_fp16 | text boxes: probability > 0.3, 4-connected, width and height ≥ 6, mean ≥ 0.5, pad = 0.35 × min(w, h) (2 to 24) (`PpocrDetector.kt`) | 20 images | `task-detection` (one class) |
| mlsd_fp16 | line segments: the centre map's sigmoid > 0.10 and a 3×3 local maximum, the top 200, end points more than 20 apart (in map pixels) (`MlsdDetector.kt`) | 20 images | `task-line-segments`: one to one with both end points within 0.02 of the 512 input's diagonal (14.5 px; either order of the end points), ≥ 95 % both ways |
| rtm_animal_fp16, rtm_face_fp16, rtmhand_fp16, rtmpose_s_fp16, rtmw_fp16 | points: SimCC argmax, x = bin / 2, score = the mean of the two maxima, shown = score > 0.3 (body, animal, whole body) / > 0.2 (hand) / all (face) (`Rtm*Estimator.kt`, `MainActivity.kt`) | the images of (b) | `task-keypoints`: of the points either side shows, those both show within 0.02 of the input's diagonal (192×256 = 6.4 px, 256×256 = 7.24 px) ≥ 95 % |
| sinet | mask: the sigmoid map (the app's overlay), object = map > 0.5 (the matting form), 352×352 | 20 images | `task-mask`: object / background mIoU (the pixels of all images summed, the mean over the classes the reference has) ≥ 0.95 |
| sam2_decoder, sam2_decoder_fixed | mask: mask 0 of the three (`Sam2Segmenter.kt`), object = logit > 0 (`MainActivity.kt`), 256×256 | 20 images → encoder (CPU fp32) → a point prompt at (512, 512) (sam2_prompt.bin) | `task-mask` (as above) |
| dewarp | the flattening geometry: where each pixel of the 384×384 output reads the source (bilinear on the 128×128 backward map) (`DocumentDewarper.kt`) | 20 images | `task-warp`: in pixels of the 256×256 input, output pixels whose cell position is within 1.0 px of the reference's ≥ 95 % (PSNR is an instrument: a shift of a fraction of a pixel at a sharp edge moves the value by tens of levels with no visible difference) |
| dmcount | a count of people: the sum of the density map (`CrowdCounter.kt`; the app rounds it for display) | 20 images | `task-count`: images with \|cell − reference\| ≤ max(0.5, 2 % of the reference) ≥ 95 % |
| metric3d_fp16 | depth (metres; metric, no rescaling) (`MetricDepth.kt`) | 20 images | `task-depth`: AbsRel (the pixels of all images, d_ref > 0) ≤ 0.02 |
| dinov2_s_fp16 | features (no task head; the app draws the tokens' first three principal components, where a small difference can swap components) → the downstream job = image retrieval | 20 images | `task-retrieval`: image embedding = the mean of the 1024 patch tokens; each image's cosine nearest neighbour among the other 19 (the cell's among the cell's, the reference's among the reference's) is the same image ≥ 95 %; images where the reference's first and second cosines differ by under 0.01 are not counted |
| movinet_a0_stream | streaming classification: the state starts at 0, the shift of the 28 stream buffers and the sums of the 16 poolings run on the host, input 45 = 1 / the frame number, 46 = 1 (`ActionRecognizer.kt`), the top-1 per frame | 20 images × a still clip of the same frame 8 times (the cell's state from the cell's outputs) | `task-top1-stream`: of 160 frames, top-1 equal to the reference's ≥ 95 %; frames where the reference's top-1 and top-2 probabilities differ by under 0.01 are not counted |
| w2v2_head_fp16 | keyword: the argmax of 12 logits (`Wav2Vec2Kws.kt`) | kws120 (the app's microphone scale: pcm / max(1, peak) × 0.5) → frontend (CPU fp32) | `task-top1`: of 120 clips, top-1 equal to the reference's ≥ 95 %; probability differences under 0.01 not counted |
| w2v2_frontend_fp16 | features → the downstream job = keyword (the head is `w2v2_head_fp16_fixed`, with the app's weights, in fp32 on the CPU; the base head is a file that lost a bias) | kws120 (as above) | `task-top1` (as above; the cell's features → the reference's head) |
| dac_16khz_deconly_zs_fp16 | the codec's round trip (`DacCodec.kt`: encoder → RVQ (dac_rvq.bin) → decoder) | kws120 (pcm / 32768) → encoder (CPU fp32) → RVQ | `task-codec`: the mean SI-SDR of the 15,992 decoded samples against the input clip ≥ the reference's − 0.5 dB, and no non-finite or all-zero output (changed to `task-codec-aligned` at 07:24: change 1 below) |

- A model with no line: none (all 23 have one above).
- Decided before any number: when a file of the vision zoo's form is not on the PC or its sha256 differs, the same
  sha256 on the Hub (`mlboydaisuke/<repo>-OpenVINO`) is used if there is one; otherwise that cell is left out with its
  reason (one line in the pairs table). A cell that does not run (a compile, load or run failure) has that error as
  its result (`run.state = error`, no verdict).
- On the PC: first passes only (untimed).

### Scoring details

(2026-11-10 07:12 JST, before any run of Z5: where the rules above leave a choice, `tools/vision_tasks.py` makes it.)

- Box pairs: the cell's boxes in descending score, each with the not yet paired reference box of the same class with
  the largest IoU (≥ 0.5). Coordinates as the app has them (DETR = 0 to 1 of the input, SSDLite = 320 px, PP-OCR =
  integer corners in 640 px); IoU on continuous areas. A box of area 0 pairs with none.
- Segment pairs: the cell's segments in descending score, each with the not yet paired reference segment whose end
  point distance (the larger of the two ends, the smaller of the two end orders) is the smallest and within 14.48 px.
- Non-finite values (NaN, inf) count on the side that does not agree: SSDLite makes no box from an anchor with a
  non-finite logit; a point with a non-finite position or score does not agree; a mask pixel is neither object nor
  background; a depth error is infinite; in retrieval the image is nobody's neighbour and has none; a class does not
  agree; the codec counts the clip as bad.
- Classification's "not counted": the reference's softmax top-1 and top-2 differ by under 0.01. The cell's top-1 is the
  argmax of its logits.
- SI-SDR: the mean is subtracted from both signals first (zero-mean), the same formula for the reference and the cell.
- The arithmetic is float64 (the app's is float32): only values exactly at a threshold can come out differently from
  the app, and the same for the reference and the cell.

### Z5 changes

1. 2026-11-10 07:24:02 DAC (`dac_16khz_deconly_zs_fp16`): the line changed from `task-codec` to `task-codec-aligned`.
   Why: the decoder's 15,992 output samples correspond to the input from its 8th sample on (16,000 − 15,992; on all
   120 clips of the reference, the lag of the largest correlation is 8 samples). The 06:41 line compared them with the
   input's first 15,992 samples, a comparison 8 samples off: the reference's own SI-SDR averaged −12.16 dB (+7.73 dB
   shifted by 8 samples). The new line compares with the input's samples 8 to 15,999; the rest is the same (mean ≥ the
   reference's − 0.5 dB, no non-finite or all-zero clip). The numbers looked at were the reference's outputs only; no
   cell's result was read. The 07:21 run of the cell `task-ovgpu` came after the old line and before the new one: it is
   not used for the verdict and runs again after the new line (sitting z5-b). The old line stays in the recipe
   (`superseded_by`).

## Z6. The LLM zoo's 86 cells that pass its text-integrity line, judged on the qa20 line

(Written 2026-11-10 07:58 JST. No number of Z6 yet; neither a reference nor a cell had run on the PC.)

The cells: 5 models, 86 cells = the copied cells of the LLM zoo with `line.kind = text-integrity` and
`line.pass = true`: Qwen3-0.6B 18, Qwen3-1.7B 18, Qwen3-4B 16, gemma-4-E2B-it 19, gemma-4-E4B-it 15; by runtime Foundry
Local 13, LiteRT-LM 15, llama.cpp 14, OpenVINO GenAI 17, ONNX Runtime GenAI 27. The copied cells and their line
(`text-check`) stay. Each new cell has the id `task-<the LLM zoo's cell>`, and its line is each recipe's `qa20`
(`written` = this section's time).

The line = section 1b (the LLM form of `schema/lines.md`) as it is: the 20 questions of `suites/llm/qa20.jsonl`; the
cell's runner passes the string the reference step rendered with transformers'
`apply_chat_template(add_generation_prompt=True, enable_thinking=False)` as it is (no template of the runtime, no
special token added); greedy (`do_sample` False, `num_beams` 1, repetition penalty 1.0); at most 48 new tokens (16 for
the gate prompt); the first line of the answer normalized and equal to the reference's = agreement; **agreement on
≥ 18 of 20, and the gate prompt "Reply with only the number: 1+1=?" ends on the end of turn within 16 tokens**.

The reference = the same model in transformers on the PC's CPU (`pc/ref_llm_hf.py`, venv `ref` = transformers 5.22.2,
torch 2.14.0). fp32: Qwen3-0.6B (weights 1.5 GB in bf16 → 3.0 GB in fp32), Qwen3-1.7B (8.1 GB), Qwen3-4B (16.1 GB),
gemma-4-E2B-it (20.5 GB). **gemma-4-E4B-it is bf16** (fp32 = 32.0 GB does not fit the PC's 31.5 GB of RAM; the Hub's
safetensors are 15.99 GB): said in the recipe and the tables. One reference per model; the cells of all five runtimes
are compared with the same one. The checkpoint = the LLM zoo's recipe's `source_revision`.

The form = what the LLM zoo's cell measured, as it is (and its versions = the LLM zoo's pins):

- OpenVINO GenAI 2026.4.1: the Hub IRs (the revisions of the LLM zoo's `shelf/hub_manifest.json`). The four NPU
  exports = the ones the LLM zoo published, `mlboydaisuke/Qwen3-{0.6B,1.7B,4B}-int4-sym-g128-ov` and
  `mlboydaisuke/gemma-4-E4B-it-int4-sym-cw-ov` (checked against the sha256s of the LLM zoo's export records).
- ONNX Runtime GenAI 0.19.0 (and its CUDA build, OpenVINO EP 1.8.1, WebGPU EP 0.5.0 = the LLM zoo's
  `pc/req/genai*.txt`): onnx-community's Qwen3 folders and `mlboydaisuke/gemma-4-E4B-it-Foundry-Local` (at the LLM
  zoo's revisions); gemma-4-E2B-it = the files of the Foundry catalog package (in the PC's cache). The WebGPU folders
  in the LLM zoo's overlay form (genai_config's `adapterIndex`, providers). The EP choice as the LLM zoo cell's args.
- LiteRT-LM v0.19.0 (edge-llm-bench's Windows build, `litert_lm_advanced_main_windows.exe` sha256 9a8d35ad…) and
  llama.cpp b11721 (the bench's release assets: CPU, Vulkan, CUDA 13.4); the model files = the PC's files of the
  bench (the same names and byte counts as the bench's record, read 11-10 07:5x), hard-linked into the zoo's folder
  (LiteRT-LM writes its caches beside the model: not into the bench's folder).
- Foundry Local 0.13.0: the same catalog variants' files as the LLM zoo's (in the PC's cache) and E4B's cuda-int4-8k
  (Hub). The package files' sha256s checked against the LLM zoo's `raw/setup-walk/logs/foundry_paths.json`.

How the string goes in (the line's "the rendered string as it is, with no special token added" in each runtime's form;
every runtime compares the prompt's ids or token count with the reference's, and the questions where they differ are
counted as an instrument):

- OpenVINO GenAI, Qwen3 (LLMPipeline): the IR tokenizer's `encode(string, add_special_tokens=False)` ids to `generate`,
  `apply_chat_template = False` (`pc/run_llm_ovg.py`). Gemma 4 (the IR is multimodal = `VLMPipeline`; LLMPipeline fails
  in generate with "5 inputs", seen in the LLM zoo): it takes only a string, and the pipeline adds BOS when it encodes
  → the string goes without its leading `<bos>` (`apply_chat_template = False`). The pipeline's input token count
  (perf metrics) is compared with the reference's. (Corrected by change 7.)
- ONNX Runtime GenAI: the package tokenizer's `encode`; when a BOS is added at the head, one more than the reference's
  ids, it is dropped. The ids go to `append_tokens`.
- LiteRT-LM: `--use_session=true` (the Session does not use Jinja), `--sampler_top_k=1` (greedy),
  `--max_output_tokens=48` (gate 16), one process per question, `--async=false`. The Session adds the model's start
  token itself and stops when the string starts with the BOS text → the string goes without its leading `<bos>`. Probe
  (11-10 07:54, the PC's CPU, a sentence that is not in qa20, "Say hello in one word."): Qwen3-0.6B prefilled 18
  tokens = the reference's 18; gemma-4-E2B-it 15 tokens with `<bos>` dropped = the reference's 15, and with `<bos>` kept
  "INVALID_ARGUMENT: Input contains bos control token". The GPU rows in the bench's form (Intel GPU = GpuPreference=1
  for the exe), the NPU row with the bench's `--litert_dispatch_lib_dir`.
- llama.cpp: the same asset's `llama-server` (the bench's cells used the same asset's `llama-cli`). The `/tokenize` ids
  (`add_special` false, `parse_special` true) are compared with the reference's and go to `/completion` as a list of
  ids, `temperature` 0, `n_predict` 48 / 16, `cache_prompt` false. The GPU rows with the bench's `-ngl 99 -fa on`,
  Vulkan = `GGML_VK_VISIBLE_DEVICES=0`, `-c 2048`.
- Foundry Local: its REST has no raw prompt route (`/v1/completions` = 404, and the catalog's chat wraps the messages
  with the folder's `chat_template.jinja`: in the 11-10 07:55 probe an 18-token string became 26 tokens). The same
  package's files are hard-linked into a folder where only `chat_template.jinja` is replaced by a template that returns
  the message content as it is (`{%- for message in messages -%}{{- message['content'] -}}{%- endfor -%}`), registered
  with `foundry model register` (local, id `z6-…:1`, `--task chat-completion`), and the string goes to
  `/catalogs/local/v1/chat/completions` as one user message, `temperature` 0, `max_tokens` 48 / 16. Probe (11-10 07:56,
  the files of Qwen3-0.6B generic-cpu:4): `prompt_tokens` 18 = the reference's 18, the answer `stop`. The usage's
  `prompt_tokens` is compared with the reference's id count (for Gemma 4, one more means the leading `<bos>` is dropped:
  decided by one probe before the cell, written as a change here). The daemon in Foundry Local's form (CUDA 12's DLL
  folder on its PATH; per cell a new daemon: load → 21 questions → unload → stop, the LLM zoo's chain form).

Instruments (not used for the verdict): the number of questions whose full answer equals the reference's, the number
that ended on the end token within 48 tokens, the number of generated tokens, the number whose prompt ids (Foundry: the
count) equal the reference's. The pairs table puts each cell's text-integrity verdict (the copied cell's `text-check`)
and its qa20 verdict on one line.

Decided before any number: when a form's file is not on the PC, or its sha256 differs from the LLM zoo's record, and it
cannot be made from the LLM zoo's pins either, that cell is left out with its reason (one line in the pairs table). A
cell that does not run (a compile, load or run failure) has that error as its result (`run.state = error`, no
verdict). On the PC: first passes only (untimed).

### Z6 changes

(Written 2026-11-10 08:29 JST, before any reference or cell had run on the PC. The line's text `qa20` does not change:
these are details of how the string goes in, decided when the runners were written.)

1. Foundry Local's `<bos>`: per cell, an untimed warm-up (the first question, 8 tokens) reads the daemon's
   `prompt_tokens`. Equal to the reference's id count = as it is; one more and the string starts with `<bos>` = all 21
   scored questions go without the leading `<bos>`; anything else = the registration wraps the prompt = the cell is an
   error (`pc/run_llm_foundry.py`). This is the form of "decided by one probe before the cell" above (the record's
   `bos_rule`).
2. Foundry Local's registration: the Gemma 4 package with `--task vision-language-chat` (the LLM zoo registered the same
   package with this task: its `pc/foundry_setup.ps1`, `pc/plans/walk-foundry-fix3.json`), Qwen3 with
   `chat-completion`. The WebGPU packages (generic-gpu) without `--execution-provider` (the LLM zoo's fix 3: a local
   model with the name WebGpuExecutionProvider gets "unknown execution provider"). When genai_config's `search` has
   `do_sample` true or a repetition penalty other than 1.0, the copy sets `do_sample` false and 1.0 (plain greedy; the
   package's values and the ones used are in the record). The cell does not start while the daemon is not its own
   (another job's Foundry note, or a foundrylocald the cell did not start; the chain waits up to 6 h). After a killed
   run the chain runs the runner's `--cleanup` (unload, unregister, `server stop`; no stop while another job's Foundry
   note is up).
3. ONNX Runtime GenAI: "a BOS added at the head, one more" = the encoded ids start with genai_config's `bos_token_id`
   twice; one is dropped. search = `do_sample` False, `num_beams` 1, `repetition_penalty` 1.0.
4. LiteRT-LM: the end token = fewer decoded tokens than the maximum (no ids come out). The Intel GPU row = as the
   bench, GpuPreference=1 (on the path of the zoo's copy of the exe, for the cell only) and
   `--preferred_device_substr=Intel`; the RTX row with that value removed first. A run whose "Selected adapter:" is not
   Intel / NVIDIA is an error. The NPU row = the bench's folder of OpenVINO DLLs first on PATH (the bench's
   `scripts/windows_ssh_executor.py`). The model file's hard link goes into a folder per device (LiteRT-LM writes its
   GPU cache beside the model with no adapter in its name, the bench's `windows/run_job.ps1`). The engine is copied
   from the bench's folder into the zoo's `engines\`, its .exe and .dll files checked against the bench's 11-08 record
   (host.json's binaries).
5. llama.cpp: `/completion` with, besides temperature 0, top_k 1, top_p 1, min_p 0, repeat_penalty 1.0, penalty 0,
   dry 0; the server with `-np 1`. The GPU rows check the device in the server's log ("using device CUDA0 / Vulkan0
   (NVIDIA …)").
6. Tables: an LLM first pass (untimed) shows no speed in the tables (`tools/results.py`; the tables across runtimes stay
   local). The run JSON keeps the values the runner measured.
7. (Added 2026-11-10 08:44 JST, after the first run of gemma-4-E2B-it's OpenVINO GenAI CPU cell at 08:38; for the cells
   that start after this time.) OpenVINO GenAI's Gemma 4 (VLMPipeline): "the pipeline adds BOS when it encodes → the
   string goes without its leading `<bos>`" above is wrong. With the string so cut, the pipeline's input token count
   was one under the reference's on all 21 questions, and the answers 15 of 20 = that run is not used for the verdict
   (the runner's error; the PC's run is deleted and runs again after this change). Per cell, an untimed warm-up (the
   first question, 8 tokens) reads the pipeline's input token count: equal to the reference's id count = the string as
   it is; one more and the string starts with `<bos>` = the leading `<bos>` dropped (a second warm-up in the dropped
   form must give the reference's count); anything else = the cell is an error (`pc/run_llm_ovg.py`, the form decided
   = the record's `vlm_form`). Instrument (not used for the verdict): whether the IR tokenizer turns the string into the
   reference's ids (`add_special_tokens` True and False, the row's `tokenizer_ids_same`).
8. (Added 2026-11-10 10:08 JST, after the first run of gemma-4-E2B-it's Foundry Local WebGPU package cell
   (`task-foundry_auto_webgpu`) at 09:58 ended in a load error; for the cells that start after this time.) Foundry
   Local 0.13.0, with a generic-gpu package (WebGPU) registered with `--device gpu` and no EP, still puts the main
   decoder on CUDA when CUDA is visible ("using CUDA EP for model", "applying explicit main decoder EP override:
   CUDAExecutionProvider" in the PC's daemon log `foundry_local_2026-11-10.log`). The LLM zoo's Qwen3 generic-gpu
   cells (catalog models) had the same two lines and ran on CUDA (`foundry_local_2026-11-07.log` to `-09.log`,
   including the copied numbers' sitting walk-timed-b) = Z6's three Qwen3 `task-foundry_auto_webgpu` cells have the
   LLM zoo cells' form, and their decoder runs on CUDA, unlike the webgpu in the cell's name (the log lines:
   `raw/z6/foundry_ep_lines.txt`). In gemma-4-E2B-it's package the decoder's other sessions are on WebGPU, and with
   the decoder alone on CUDA the load stops with "Running a model with multiple providers is not supported.
   Encountered WebGPU and CUDA" (Z6 at 09:58 and the LLM zoo on 11-08 02:42). The copied cell's numbers (sitting
   walk-timed-b, the three runs of 11-09 00:38 to 00:42) were run with the daemon started with
   `CUDA_VISIBLE_DEVICES=-1` (the LLM zoo's fix 1: the unit's `daemon_env` in its `pc/plans/walk-timed-b.json`; the log
   reads "using WebGPU EP for model", "override: WebGpuExecutionProvider") = Z6's Foundry cells start the daemon with
   the `daemon_env` of the LLM zoo plan's unit of the copied cell's `speed.sitting` when it has one (the copy tool wrote
   it into the recipe as `--daemon-env`; of the 16 Foundry units of walk-timed-b and walk-timed-b-rerun, this one cell
   only). The 09:58 run stays an error and is not used for the verdict; the PC's run is deleted and runs again after
   this change. If the log of the run again has no "using WebGPU EP for model: z6-gemma-4-e2b-it-generic-gpu:1", that
   run is not used either.
9. (Added 2026-11-10 13:0x to 13:1x JST, when this repo was cut, after the runs; it changes no rule.) The reason given above for
   gemma-4-E4B-it's bf16 reference, "fp32 = 32.0 GB does not fit the PC's 31.5 GB of RAM", compares decimal GB with
   Windows' figure, which is in GiB (`pc/zlib.ps1` reads TotalPhysicalMemory / 1GB for the records). In one unit:
   7,996,156,490 parameters (the Hub's safetensors, BF16) x 4 bytes = 31.98 GB = 29.8 GiB of fp32 weights, on a PC with
   33,840,689,152 bytes = 31.5 GiB = 33.8 GB (TotalPhysicalMemory, read 2026-11-10 13:1x), where Windows and the other
   jobs of the shared PC also run (the bf16 reference's peak working set was 16.56 GB, 0.57 GB over its 15.99 GB of
   weights). No fp32 reference was run (the records hold the bf16 one only); the reference stays bf16.
