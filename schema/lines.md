# Lines

A line is the pass / fail rule a cell is judged on. It is written before any run of the cells it judges, and its time
goes into the recipe's `[[line]] written` (`tools/lint.py` compares it with each cell's start). The task lines below
were written on 2026-11-10 at 00:11 JST, before any number of the template's own cells.

The reference is the same model in fp32 on the PC's CPU (bf16 when it does not fit in fp32, said so in the recipe),
run on the same inputs, with the same prompt and decode. Internal agreement (cosine, max |d|, token agreement) stays
in a cell's JSON as an instrument and does not decide the verdict.

| kind | meaning |
|---|---|
| `task` | the model's job against the reference (the forms below) |
| `internal` | outputs against the reference's outputs (the vision zoo Z1's line: cosine ≥ 0.99 and argmax agreement ≥ 0.99) |
| `text-integrity` | the text is well formed, not correct (the LLM zoo Z2's line: edge-llm-bench's textCheck) |
| `reference` | a cell's role, not a line: the cell is the reference |

## Vision

The input set is the model's `[model] inputs` (a file of `suites/`, each input with its URL and sha256). Both outputs
go through the same post-processing.

| task | value compared | line (over the whole input set) |
|---|---|---|
| classification | the top-1 class | the top-1 equals the reference's on ≥ 95 % of the inputs; an input whose reference top-1 and top-2 probabilities differ by less than 0.01 is not counted (the count is an instrument) |
| detection | the boxes after the model's own post-processing (score threshold, NMS) | the share of boxes that pair with the reference's (same class, IoU ≥ 0.5) is ≥ 95 % both ways (F1) |
| segmentation, matting | the argmax mask (matting: alpha > 0.5) | mIoU against the reference's mask ≥ 0.95 (mean over the classes in the reference) |
| depth | the depth map (relative depth scaled by the median) | AbsRel ≤ 0.02 |
| pose, keypoints | the point coordinates (SimCC: argmax) | ≥ 95 % of the points within 0.02 of the object box's diagonal (PCK) |
| image to image | the output rounded to uint8 | PSNR ≥ 40 dB against the reference's output |
| text recognition (OCR) | the decoded string | CER against the reference's string ≤ 0.5 pt |
| features only (an encoder without a task head) | — | kind `internal` (cosine ≥ 0.99), with the reason in the recipe |

A model with a single input has a set of one: the line can be written but is weak; its card says how many inputs.
Instruments: cosine, max |d| and NaN / Inf per output, the top-5 overlap, the inputs not counted.

## LLM

- The question set `suites/llm/qa20.jsonl`: 20 short English questions whose answers are fixed, and one gate prompt.
  Each question is one user turn in the model's own chat template: the reference step renders it with transformers'
  `apply_chat_template(add_generation_prompt=True, enable_thinking=False)` and records the string; the cell's runner
  passes that string as it is (no template of its own runtime), tokenized without special tokens.
- Decode: greedy (do_sample False, num_beams 1, repetition_penalty 1.0: a checkpoint's generation config may set a
  penalty for sampling; both sides drop it), at most 48 new tokens (the gate prompt 16).
- An answer agrees when its first line, normalized (outer spaces dropped, lower case, outer `"` `'` `` ` `` `*` and a
  final `.` `!` `?` dropped, spaces collapsed), equals the reference's answer normalized the same way.
- **Line:** ≥ 18 of 20 answers agree, and the gate prompt ("Reply with only the number: 1+1=?") ends with the model's
  end-of-turn token within 16 new tokens.
- Instruments: the questions whose whole text equals the reference's, the questions that ended on EOS within 48
  tokens, the tokens generated.

## ASR (the audio zoo Z3's line)

Per language, WER (en) or CER (zh, ja) ≤ the reference's on the same clips and prompt + 0.5 pt, compared on the summed
errors (never on rounded rates), and no empty answer. Scoring: `tools/fleurs_score.py` (pure Python). Reference:
transformers fp32 on the PC's CPU, greedy, the language set per clip. Instruments: the clips whose text equals the
reference's, the RTF.

## TTS (the audio zoo Z3's line)

The write-back WER ≤ the reference's write-back WER + 1.0 pt (the reference = the model's original PyTorch
implementation's speech, written back by the same ASR), and no silent answer (length 0, under 0.2 s, or all silence).
The write-back ASR: OpenVINO GenAI's WhisperPipeline with `OpenVINO/whisper-large-v3-turbo-fp16-ov` on the CPU,
`<|en|>`, the same for every TTS cell, untimed.
