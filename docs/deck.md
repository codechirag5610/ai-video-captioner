---
marp: true
paginate: true
theme: uncover
class: invert
style: |
  section { font-size: 26px; text-align: left; justify-content: flex-start; padding: 56px 70px; }
  h1 { font-size: 44px; color: #86efac; }
  h2 { font-size: 33px; color: #86efac; }
  strong { color: #fca5a5; }
  code { background: #1e293b; color: #e2e8f0; }
  pre { font-size: 19px; }
  ul { line-height: 1.45; }
  section.lead { text-align: center; justify-content: center; }
  footer { color: #64748b; font-size: 15px; }
footer: 'Four-Style Video Captioner · AMD Developer Hackathon Act II · Track 2'
---

<!-- _class: lead invert -->
<!-- _paginate: false -->

# Four-Style Video Captioner

## Gemma watches the video. Gemma writes the caption.

Chirag Sharma · Bisman Singh
AMD Developer Hackathon Act II · Track 2

---

## The task, and what actually scores

One clip in, four captions out: **formal, sarcastic, humorous_tech, humorous_non_tech**.

An LLM judge scores every caption twice:
- **Accuracy**: does the caption reflect what is actually in the video?
- **Style**: is the tone unmistakable?

Asking one model to "watch a video and be funny" trades facts for jokes.
So we never ask that.

---

## Architecture: perception and style, separated

```
video ─ ffmpeg ─► 12 uniform frames + audio ─► Whisper (local ASR)
                       │
                       ▼
   GEMMA 4 26B-A4B ── neutral FACT SHEET ──────────► ACCURACY
   (a native video-understanding model:              setting, subjects, events,
    frames are its home turf)                        dialogue, visible text,
                       │                             plus an "uncertain" no-go list
                       ▼
   GEMMA 4 31B ─────── comedy angles + N candidate
   (the writer)        captions per style, in parallel ─► STYLE
                       │
                       ▼
   GLM judge (a RIVAL model family, blind) picks each winner;
   weak winners get one bounded, budget-gated regeneration
```

---

## Gemma is the core, not a garnish

- **Every fact and every caption is Gemma's work.** Gemma 4 26B-A4B reads the
  frames; Gemma 4 31B finds the comedy and writes all four styles.
- The judge is deliberately **not** Gemma: a rival family referees Gemma's
  captions blind. Stronger evidence than Gemma grading itself.
- Served via the Gemini API, with a Gemma-on-OpenRouter fallback tier, then
  Fireworks as last resort. **Every fallback is logged** to `run_report.json`
  next to the results, so the Gemma claim is auditable, not aspirational.
- The config reserved these stages for Gemma from day one.

---

## Grounding beats hallucination

- The fact sheet carries an **"uncertain" list**: anything the vision pass
  could not verify is a no-go for every later stage.
- Comedy angles must cite the fact sheet. The writer must anchor each caption
  in a concrete, verified detail.
- **humorous_non_tech is hard-guarded**: a deterministic vocabulary check
  (with plural and punctuation dodges covered) rejects any caption with tech
  jargon before it can ship.
- Whisper output is filtered for the classic "Thanks for watching" style
  hallucinations before it can enter the facts.

---

## Reliability engineering (a timeout scores zero)

- **Watchdog**: a valid `results.json` is force-flushed before the 10-minute
  wall, no matter what a network call is doing.
- **Per-clip deadlines** with a degradation ladder: skip ASR, then skip comedy,
  then single-candidate fast path — a slow clip gets a plainer caption, never
  a zero.
- **No caption is ever empty**: judge failure falls back to the best candidate,
  pipeline failure to a deterministic caption built from the fact sheet, and a
  final backfill guarantees every requested style ships text.
- Downloads capped by wall-clock and size; all four styles generate in parallel.

---

## Real output (example clip: city traffic, autumn boulevard)

- **formal**: "Heavy traffic flows along a multi-lane avenue in a modern East
  Asian city, with motion blur indicating continuous movement."
- **sarcastic**: "Three minutes of cinematic motion blur so traffic can feel
  urgent about remaining exactly the same."
- **humorous_tech**: "Aggressive motion blur to render traffic that literally
  just continues — peak overengineering."
- **humorous_non_tech**: "The blue bus got one second of fame before the blur
  ate it, and I have never related to a vehicle more."

---

## Validated end to end

- Full harness contract: reads `/input/tasks.json`, downloads each
  `video_url`, writes `/output/results.json` with exact style keys.
- Three example clips: **all four styles filled on every clip**, correct keys,
  ~54s per clip with four-way parallelism — comfortable inside the budget at
  the hidden set's scale.
- Container: linux/amd64, ffmpeg + Whisper baked in, well under the size limit.

---

<!-- _class: lead invert -->

## Four-Style Video Captioner

Gemma perceives. Gemma writes. A rival model referees.

github.com/codechirag5610/ai-video-captioner

Thank you
