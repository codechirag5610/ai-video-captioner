# Four-Style Video Captioner — AMD Developer Hackathon Act II · Track 2

Generate a caption for each short video clip in **four distinct styles** —
`formal`, `sarcastic`, `humorous-tech`, and `humorous-non-tech` — optimized for
the Track 2 LLM-judge, which scores **accuracy** and **tone**.

## Why this design wins

The judge scores two different things, so we use a **two-stage pipeline** that
optimizes each one separately instead of asking one model to "watch a video and
be funny" (which trades facts for jokes):

```
                 ┌─────────────────────────────────────────────┐
   video ─┬────► │ PREPROCESS (ffmpeg)                          │
          │      │  • adaptive frame sampling                   │
          │      │    (scene-change beats + uniform fallback)   │
          │      │  • audio extraction → ASR (Whisper)          │
          │      └─────────────────────────────────────────────┘
          │                     │ frames + transcript
          │                     ▼
          │      ┌─────────────────────────────────────────────┐
          │      │ STAGE A — UNDERSTAND  (vision model)         │  ← ACCURACY
          │      │  neutral ground truth: setting, subjects,    │
          │      │  events, dialogue, on-screen text (OCR),     │
          │      │  audio, mood, "the notable thing"            │
          │      │  → cached by video hash (protects credits)   │
          │      └─────────────────────────────────────────────┘
          │                     │ ground-truth JSON
          │                     ▼
          │      ┌─────────────────────────────────────────────┐
          └────► │ STAGE B — STYLE  (text model, e.g. Gemma)    │  ← TONE
                 │  4 captions, few-shot per style, non-tech    │
                 │  vocabulary ban, "reference ≥2 clip facts"   │
                 └─────────────────────────────────────────────┘
                                 │
                                 ▼
                 ┌─────────────────────────────────────────────┐
                 │ SELF-CRITIQUE  (local LLM-judge)             │
                 │  score accuracy+tone, check tech vs non-tech │
                 │  distinguishability → regenerate weak styles │
                 └─────────────────────────────────────────────┘
                                 │
                                 ▼                     captions.json
```

- **Model-agnostic.** Every model id lives in [`config/models.yaml`](config/models.yaml).
  Launch day = edit that file, run. No code changes.
- **Gemma-ready.** Point Stage B (and Stage A if a Gemma-3 vision model is
  offered) at Gemma to be eligible for the *Best Use of Gemma* prize.
- **Credit-safe.** Stage A vision output is cached by video content hash, so
  iterating on style prompts never re-pays for understanding.

## Setup

Requirements: Python 3.10+, `ffmpeg` on PATH, a Fireworks API key.

```bash
cp .env.example .env         # then paste your FIREWORKS_API_KEY
pip install -r requirements.txt   # or: pip install -e .
```

## Usage

```bash
# Put clips in ./clips, then:
python -m captioner.cli --input ./clips --output ./output/captions.json
```

Common flags: `--no-judge` (skip the self-critique loop), `--limit N`,
`--max-frames N`, `--keep-work` (keep extracted frames), `-v` (debug logs).

### Docker (submission-ready)

```bash
docker build -t captioner .
docker run --rm --env-file .env \
  -v "$PWD/clips:/app/clips" \
  -v "$PWD/output:/app/output" \
  -v "$PWD/cache:/app/cache" \
  captioner
# → output/captions.json
```

## Output format

```json
{
  "count": 1, "ok": 1, "errors": 0,
  "results": [
    {
      "file": "clip01.mp4",
      "duration_s": 47.0, "n_frames": 12, "has_audio": true, "language": "en",
      "captions": {
        "formal": "...",
        "sarcastic": "...",
        "humorous-tech": "...",
        "humorous-non-tech": "..."
      },
      "ground_truth": { "...": "..." },
      "judge": { "overall": 8.4, "distinguishable": true, "scores": { "...": {} } },
      "error": null
    }
  ]
}
```

## Offline eval loop (iterate before the real judge sees you)

```bash
python scripts/eval.py --input ./clips --out output/eval.json
```

Prints per-style average accuracy/tone and flags weak clips so you know which
prompt to tune next. Use a judge model **different** from the generator to avoid
self-preference bias (set `judge.model` ≠ `style.model` in the config).

## Edge cases handled

Silent clips (ASR skipped, no hallucinated speech) · music-only (mood described,
lyrics not transcribed) · non-English audio (language detected/noted) · rapid
cuts (scene-detection sampling) · text-heavy memes/screen-recordings (OCR field) ·
dark/blurry/low-info clips (low-confidence → Stage B avoids inventing details) ·
weird codecs / vertical / odd resolutions (normalized via ffmpeg) · API
rate-limits/5xx (exponential backoff) · one bad clip never kills the batch
(per-clip error isolation).

## Launch-day checklist

1. Read the revealed model list; set `understand.model`, `style.model`,
   `judge.model`, and `asr.model` in `config/models.yaml`.
2. Confirm the chosen `understand.model` accepts image input; adjust `max_images`.
3. `python scripts/eval.py --input ./clips --limit 3` → sanity-check baseline.
4. Iterate style prompts against the eval loop; decide on fine-tuning (Day 2).
5. `docker build` + run on a clean machine before submitting.

## Project layout

```
config/models.yaml     ← the only file you edit on launch day
src/captioner/
  preprocess.py        ffmpeg: frames + audio + normalization
  client.py            model-agnostic Fireworks (OpenAI-compat) + retries + ASR
  understand.py        Stage A: ground-truth extraction
  styles.py            Stage B: 4-style generation + regeneration
  judge.py             local LLM-judge
  pipeline.py          per-clip orchestration + self-critique loop
  cli.py               batch entry point (Docker ENTRYPOINT)
  cache.py             video-hash cache for Stage A
data/style_examples.py few-shot examples per style
scripts/eval.py        offline generate→judge→aggregate loop
```
