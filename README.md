# Four-Style Video Captioner — AMD Developer Hackathon Act II · Track 2

Generate a caption for each short video clip in **four distinct styles** —
`formal`, `sarcastic`, `humorous-tech`, and `humorous-non-tech` — optimized for
the Track 2 LLM-judge, which scores **accuracy** and **tone**.

## Why this design wins

The judge scores two different things, so we **separate perception from style**
instead of asking one model to "watch a video and be funny" (which trades facts
for jokes). Because the clip set is fixed and visible, the engine is
**best-of-N generation + judge selection**: generate several candidates per
style, keep only the one that lands.

```
   video ──► ┌─────────────────────────────────────────────┐
             │ PREPROCESS (ffmpeg)                          │
             │  • adaptive frame sampling                   │
             │    (scene-change beats + uniform fallback)   │
             │  • audio extraction → ASR (Whisper)          │
             └─────────────────────────────────────────────┘
                           │ frames + transcript
                           ▼
             ┌─────────────────────────────────────────────┐
             │ STAGE A — FACT SHEET  (vision model)         │  ← ACCURACY
             │  neutral facts: setting, subjects, events,   │
             │  dialogue, on-screen text (OCR), audio,      │
             │  mood, notable + an `uncertain` no-go list   │
             │  → cached by video hash (protects credits)   │
             └─────────────────────────────────────────────┘
                           │ fact sheet
                           ▼
             ┌─────────────────────────────────────────────┐
             │ STAGE 3 — COMEDY MATERIAL  (text model)      │
             │  grounded absurd/ironic angles + tech-angle; │
             │  skipped for formal → cached with fact sheet │
             └─────────────────────────────────────────────┘
                           │ fact sheet + comedic angles
                           ▼
             ┌──────────────────────────────────────────────┐
             │ STAGE 4 — STYLE GEN  (text model, e.g. Gemma) │  ← TONE
             │  N candidates × 4 styles, per-style temp,     │
             │  style cards + few-shot, "reference ≥2 clip   │
             │  facts", non-tech vocabulary ban              │
             └──────────────────────────────────────────────┘
                           │ N candidates per style
                           ▼
             ┌──────────────────────────────────────────────┐
             │ STAGE 5 — JUDGE & SELECT  (diff. model family)│
             │  score accuracy+tone+distinct+fit, pick best  │
             │  per style; distinctness vs already-chosen;   │
             │  weak winner → 1 bounded regen w/ critique     │
             └──────────────────────────────────────────────┘
                           │
                           ▼                       captions.json
```

- **Best-of-N is the engine.** N candidates per style at per-style temperature
  (low for formal, high for humor), scored against the fact sheet; the judge
  keeps the winner. Variance is where the good jokes come from.
- **Accuracy anchored to tone.** Every caption must reference ≥2 concrete facts
  and may never use anything on the fact sheet's `uncertain` list — the guard
  against the #1 failure mode, hallucinated humor.
- **Distinct by construction.** Styles are selected in order and each judge pass
  sees the already-chosen captions, so sarcastic ≠ humorous-non-tech and
  tech-humor stays identifiably technical. A deterministic banned-word guard
  (with inflections) forces a regen if a tech term leaks into the non-tech style.
- **Model-agnostic.** Every model id lives in [`config/models.yaml`](config/models.yaml).
  Launch day = edit that file, run. No code changes.
- **Gemma-ready.** Point Stage 3/4 (and Stage A if a Gemma-3 vision model is
  offered) at Gemma for the *Best Use of Gemma* prize; judge with a different
  family to avoid self-preference bias.
- **Credit-safe.** Stage A and Stage 3 outputs are cached by video content hash,
  so iterating on style prompts never re-pays for understanding.

## Setup

Requirements: Python 3.10+, `ffmpeg` (and `ffprobe`) on PATH, a Fireworks API key.

```bash
cp .env.example .env         # then paste your FIREWORKS_API_KEY
pip install -e .             # installs deps + the `captioner` command
```

> The package lives under `src/`, so a bare `python -m captioner.cli` only works
> after `pip install -e .`. If you'd rather not install, run everything with
> `PYTHONPATH=src python -m captioner.cli ...`. (`pip install -r requirements.txt`
> installs the runtime deps but **not** the `captioner` entry point.)

## Usage

```bash
# Put clips in ./clips, then:
captioner --input ./clips --output ./output/captions.json
# equivalently, without installing:
#   PYTHONPATH=src python -m captioner.cli --input ./clips --output ./output/captions.json
```

Flags: `--config/-c PATH` (default `config/models.yaml`), `--cache-dir DIR`
(default `cache`), `--max-frames N`, `--no-judge` (skip the self-critique loop),
`--limit N` (first N clips), `--keep-work` (keep extracted frames/audio),
`--verbose/-v` (debug logs).

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
      "video_hash": "a1b2c3d4e5f6a7b8",
      "duration_s": 47.0, "n_frames": 12, "has_audio": true, "language": "en",
      "captions": {
        "formal": "...",
        "sarcastic": "...",
        "humorous-tech": "...",
        "humorous-non-tech": "..."
      },
      "ground_truth": {
        "setting": "...", "subjects": ["..."], "events": ["0:04 - ..."],
        "dialogue_summary": "...", "visible_text": "...", "audio_description": "...",
        "mood": "...", "notable": "...", "uncertain": ["..."], "confidence": 0.9
      },
      "comedy_material": [
        { "element": "...", "why_funny": "...", "tech_angle": "..." }
      ],
      "selection": {
        "formal":            { "accuracy": 9, "tone": 8, "distinct": 9, "fit": 8, "n_candidates": 4, "critique": "" },
        "sarcastic":         { "accuracy": 8, "tone": 9, "distinct": 8, "fit": 8, "n_candidates": 4, "critique": "" },
        "humorous-tech":     { "accuracy": 8, "tone": 9, "distinct": 9, "fit": 9, "n_candidates": 4, "critique": "" },
        "humorous-non-tech": { "accuracy": 9, "tone": 8, "distinct": 8, "fit": 8, "n_candidates": 4, "critique": "" }
      },
      "error": null
    }
  ]
}
```

`selection` holds the winning candidate's judge scores per style (accuracy, tone,
distinct, fit) plus how many candidates were generated and any regeneration
critique. It is omitted entirely under `--no-judge` (which takes the first
candidate per style). `comedy_material` is empty when `comedy.enabled: false`. On
failure, `error` holds `"<ExceptionType>: <message>"` and `captions` are empty
strings — the batch continues regardless.

## Offline eval loop (iterate before the real judge sees you)

```bash
python scripts/eval.py --input ./clips --out output/eval.json
```

Prints a **per-clip × per-style score sheet** (accuracy/tone grid), per-style
averages, and flags weak clips so you know exactly which cell to tune next. Use a
judge model **different** from the generator to avoid self-preference bias (set
`judge.model` ≠ `style.model` in the config).

To sanity-check the pipeline without spending a credit (preprocessing, prompt
construction, cache, JSON parsing), run the offline smoke test — it needs
`ffmpeg` and the sample clips in `./clips` but no API key:

```bash
python tests/smoke_offline.py
```

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
config/models.yaml       ← the only file you edit on launch day
src/captioner/
  config.py              loads models.yaml + .env into typed config
  preprocess.py          ffmpeg: frame sampling + audio + normalization
  client.py              model-agnostic Fireworks (OpenAI-compat) + retries + ASR call
  asr.py                 audio transcription (fireworks | local | none)
  understand.py          Stage A: fact-sheet extraction (+ uncertain list)
  comedy.py              Stage 3: grounded comedy-material extraction
  styles.py              Stage 4: best-of-N generation + style cards + non-tech guard
  judge.py               Stage 5: candidate scoring + winner selection
  pipeline.py            per-clip orchestration (best-of-N engine + bounded regen)
  cli.py                 batch entry point (Docker ENTRYPOINT)
  cache.py               video-hash cache for Stage A + Stage 3
data/style_examples.py   few-shot exemplars per style
scripts/eval.py          offline generate→judge→score-sheet loop
tests/smoke_offline.py   no-API smoke test (preprocess, prompts, cache, parsing, engine)
```
