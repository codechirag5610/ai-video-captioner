# Track 2: Four-Style Video Captioning — End-to-End Design

## 1. What the judging actually rewards, and the strategy that follows

Scoring is **LLM-Judge on two axes: accuracy and tone.** That one line dictates the whole design.

**Accuracy** means the caption must describe what actually happens in the clip. The single biggest failure mode in this track will be teams whose "funny" captions invent things that are not in the video. Under comedy pressure, models hallucinate. An LLM judge with the video description in front of it will punish that on the accuracy axis. So: **every joke must be grounded in something verifiably in the clip.** Conveniently, this is also how good comedy works. Specificity is funny. "A cat knocks a glass off a counter while making direct eye contact" beats a generic cat joke on both axes at once.

**Tone** means the four styles must each land AND be distinct from each other. The realistic risk is style collapse: sarcastic and humorous drift together, tech-humor becomes non-tech humor with the word "server" in it. Distinctness is a first-class design goal, not a nice-to-have.

Three structural facts that shape everything:

1. **The clip set is fixed and visible.** Unlike Track 1's hidden test set, you generate captions for clips you can watch. There is no generalization risk. Iterating your pipeline until its outputs are excellent *on these exact clips* is not overfitting, it is the game. Best-of-N generation with judge selection is therefore the core loop.
2. **Token cost is a non-issue.** A fixed set of 30s–2min clips, even with heavy best-of-N and judge passes, costs a few dollars against $50+ in Fireworks credits. This is the opposite economy from Track 1: spend tokens lavishly on quality. (If volume ever matters, Fireworks batch inference is 50% off.)
3. **Fireworks has no native video input on the serverless API.** Vision models accept **images only: max 30 per request, base64 total under 10MB, URL images under 5MB each**. Frame extraction is therefore a mandatory pipeline stage, not an optimization. Fireworks does have **audio transcription endpoints (Whisper family)** on a separate audio base URL.

Core strategy in one line: **separate perception from style.** Build one maximally accurate, structured account of what happens in the clip (the fact sheet), then generate all four styles from that single fact sheet, then let an LLM judge pick the best of N candidates per style. Accuracy is won in stage one, tone in stage two, consistency by construction.

## 2. Pipeline architecture

```
video clip
  -> Stage 0  ingest & sample      ffmpeg + scene detection -> 8–16 keyframes w/ timestamps
  -> Stage 1  audio                extract track -> Whisper transcript (skip if no speech)
  -> Stage 2  perception           VLM: frames + transcript -> structured FACT SHEET
  -> Stage 3  comedy material      LLM pass: what is inherently absurd/notable here?
  -> Stage 4  style generation     4 styles x N candidates from fact sheet + style cards
  -> Stage 5  judge & select       LLM judge scores accuracy + tone -> pick winner per style
  -> Stage 6  final QA             fact-check winners against fact sheet, regenerate failures
```

### Stage 0: Ingest and frame sampling

- Extract frames with **ffmpeg**; run **scene/shot detection** (PySceneDetect `detect-content`, or ffmpeg's `select='gt(scene,0.3)'`) to find cuts.
- Research on frame sampling for VLMs consistently finds **shot-aware sampling beats uniform sampling**: partition the video into shots, then take a representative frame per shot plus any high-change moment within the shot. The thing you must not miss is the *payoff moment* (the fall, the reveal, the reaction). Uniform sampling misses brief decision-critical events.
- Practical budget for 30s–2min clips: **8–16 frames**, comfortably under the 30-image API cap. Downscale to ~768px longest side, JPEG q≈80, to stay well under the 10MB base64 limit.
- For clips with few cuts (single continuous shot), fall back to uniform 1 fps capped at 12, plus a frame-difference pass to catch the action spike.
- **Label every frame with its timestamp** and keep them in order. Temporal ordering is how the VLM narrates a sequence of events instead of a bag of images.

### Stage 1: Audio (the differentiator most teams will skip)

- Extract audio with ffmpeg; run a cheap voice-activity / energy check. If speech is present, transcribe via **Fireworks' Whisper endpoint**. If it is music/SFX only, note that ("upbeat music, crash sound at 0:41") rather than transcribing.
- Why this matters: in many short clips, the comedy or the entire meaning is in what is said or the sound (the scream, the deadpan commentary, the crash off-screen). Frames-only pipelines describe a different, poorer video. Teams that skip audio will produce captions that are subtly wrong, and the accuracy axis will catch it.

### Stage 2: Perception -> fact sheet (accuracy is won here)

One VLM call: ordered timestamped frames + transcript + instructions to produce a **structured fact sheet**, not prose:

- setting/location; subjects (people, animals, objects) with distinguishing details
- **event sequence with timestamps** (what happens, in order, including the payoff)
- any **on-screen text / captions / UI**, transcribed verbatim
- audio summary (who says what, key sounds)
- notable/unusual details (the things a human would point at)
- explicit `uncertain:` list for anything the model cannot verify from frames

The `uncertain` list is load-bearing: it is what keeps later stages from confidently jokes-ifying a guess.

Model choice at kickoff: whatever VLM the organizers reveal; Fireworks' catalog includes Qwen2.5-VL, InternVL3, Llama 3.2 Vision, Phi-3.5-vision, and multimodal Gemma 3. If choice is free, use the strongest available for this stage; perception quality is the ceiling on everything downstream. Run it at temp 0.

### Stage 3: Comedy material extraction

Cheap but high-leverage LLM pass over the fact sheet: "list the inherently absurd, ironic, or notable elements here; for each, why it is funny; note tech-angle potential." This separates *finding the funny* from *writing the joke*, so the style generators are aiming at pre-validated material instead of free-associating (which is where hallucination creeps in).

### Stage 4: Style generation from style cards

All four styles are generated **from the same fact sheet** (never from raw frames), each with its own **style card**: definition, voice, do/don't list, and 2–3 few-shot exemplars *you write by hand for unrelated videos*. Hand-written exemplars are the single best lever on tone quality; write them once, reuse for every clip.

The style cards (summarized; full prompts in §5):

- **Formal**: neutral, precise, professional summary. No humor, no editorializing, no exclamation marks. Reads like a broadcast description or an alt-text on a serious site. The trap: leaking mild amusement. Formal is judged on being genuinely straight.
- **Sarcastic**: dry, deadpan, mock-impressed or mock-sympathetic. Says the opposite of what it means, understates disasters, "ah yes, exactly the right moment to let go." Wit through irony, not jokes. The trap: drifting into generic humor; sarcasm must have an *edge* and a *target* (the situation, not protected groups).
- **Humorous (tech)**: comedy through tech/engineering metaphor. The cat is running an unauthorized load test; the fall is a rollback without a backup; the dog has achieved eventual consistency. The trap: bolting one tech word onto a generic joke. The *mechanism* of the joke must be the tech concept mapping onto the event.
- **Humorous (non-tech)**: everyday observational comedy, accessible to anyone; family-group-chat energy. Relatable framing, exaggeration, mock-drama. The trap: overlapping with sarcastic; this style is warm and playful where sarcastic is dry and cutting.

Generate **N=3–5 candidates per style** at temp ~0.9 for the humor styles, ~0.3 for formal. Cost is trivial; variance is where the good jokes come from.

Length discipline: a caption is 1–2 sentences unless the task spec says otherwise. Judges reward density; rambling dilutes both tone and accuracy.

### Stage 5: Judge and select (best-of-N is the engine)

An LLM judge scores every candidate on a rubric mirroring the real judging: **accuracy against the fact sheet** (contradictions? inventions?) and **tone fidelity against the style card** (is it this style, is it distinct from the other three, does it land?). Pick the top candidate per style; if all N fail a bar, regenerate with the judge's critique appended.

Two anti-self-delusion rules:
- **Judge with a different model family than you generate with** (e.g. generate with Gemma, judge with Qwen or Llama), so you do not tune to one model's taste.
- **Spot-check with a second judge model** on a sample. If two judges disagree wildly on tone, a human (you) breaks the tie. Four people watching 20 clips and sanity-checking winners is an hour well spent; the final submission quality is the product.

### Stage 6: Final QA

Deterministic pass over the four winners per clip: fact-check each against the fact sheet one last time (cheap LLM call: "does this caption assert anything not in the fact sheet?"), verify style distinctness across the four (no shared punchline or shared opening), verify format compliance with whatever submission schema is specified. Cache everything keyed by clip hash so re-runs are free and reproducible.

## 3. Prompts

### 3.1 Perception / fact sheet (Stage 2, temp 0)

```
You are analyzing a short video via {k} frames sampled in order, with timestamps,
plus an audio transcript. Produce a precise factual account. Do not speculate;
if something cannot be verified from the frames or audio, list it under "uncertain".

Frames: [frame 1, t=0:00] ... [frame k, t=1:37]
Audio transcript (may be empty): {transcript}

Output exactly this structure:
SETTING: <location, environment, context>
SUBJECTS: <each person/animal/object that matters, with distinguishing details>
EVENTS: <numbered sequence with timestamps, including the outcome/payoff moment>
ON-SCREEN TEXT: <verbatim, or "none">
AUDIO: <who says what; key sounds; or "music only" / "no audio">
NOTABLE: <unusual, ironic, or attention-grabbing details a human would remark on>
UNCERTAIN: <anything ambiguous; do not let these appear as facts above>
```

### 3.2 Comedy material (Stage 3, temp 0.7)

```
Here is a factual account of a short video:
{fact_sheet}

List 3-6 elements with comedic potential. For each:
- the element (must be from the account above, cite the event number)
- why it is funny (irony, absurdity, timing, relatability, scale mismatch)
- tech-metaphor potential: a software/engineering concept this maps onto, if any
Do not invent details not present in the account.
```

### 3.3 Style generation (Stage 4; one call per style, N samples)

```
Write {n} candidate captions for a short video, in the style defined below.

FACTS (the only things you may reference):
{fact_sheet}

COMEDIC MATERIAL (pre-approved angles):
{comedy_material}     # omitted for the formal style

STYLE CARD: {one of the four cards below}

Rules:
- 1-2 sentences per caption.
- Every claim must be supported by FACTS. No invented details.
- Each candidate should take a different angle.
Output as a numbered list, captions only.
```

**Formal card:**
```
STYLE: Formal
Voice: neutral, precise, professional; a broadcast description.
Do: state setting, subject, and key event objectively; past or present tense, consistent.
Don't: no humor, no irony, no exclamation marks, no editorializing, no slang.
Example (unrelated video): "A delivery courier slips on an icy driveway while
carrying a package, recovers his footing, and completes the delivery."
```

**Sarcastic card:**
```
STYLE: Sarcastic
Voice: dry, deadpan, mock-impressed; says the opposite of what it means; understates chaos.
Do: irony with a clear target (the situation/decision), faux praise, flat delivery.
Don't: no "haha" energy, no puns for their own sake, no meanness toward identity groups.
Example (unrelated video): "Truly inspired decision to carry all nine grocery bags
in one trip. The driveway ice really rewarded that confidence."
```

**Humorous tech card:**
```
STYLE: Humorous (tech)
Voice: comedy built from software/engineering metaphors mapped onto the event.
Do: make the tech concept the MECHANISM of the joke (rollback, race condition,
load test, garbage collection, prod incident, retry loop, edge case).
Don't: generic joke + random tech word; jargon so deep only SREs laugh; more than
one metaphor per caption.
Example (unrelated video): "Courier hit an unhandled ice exception, executed a
mid-air retry, and still shipped to prod. Five nines of delivery."
```

**Humorous non-tech card:**
```
STYLE: Humorous (non-tech)
Voice: warm observational comedy; family-group-chat energy; relatable exaggeration.
Do: mock-drama, universal experiences, playful narration of the payoff.
Don't: no tech references at all; no dry irony (that is the sarcastic lane);
no cruelty.
Example (unrelated video): "Man versus driveway: driveway takes round one, but
he sticks the landing like it was choreography and the package never doubted him."
```

### 3.4 Judge (Stage 5, temp 0, different model family)

```
You are judging video captions. Here is the ground-truth account of the video:
{fact_sheet}

Style definition: {style_card}
The other three styles' current best captions (for distinctness check): {others}

Candidates:
{numbered candidates}

Score each candidate 1-10 on:
ACCURACY: consistent with the account; asserts nothing not in it (any invented
detail caps this at 3).
TONE: matches the style card's voice; would a reader label it as this style?
DISTINCT: clearly different in voice from the other three styles' captions.
FUNNY/FIT: for humor styles, does it land? for formal, is it clean and complete?

Output: per candidate, four scores + one-line justification. Then: "WINNER: <n>"
and, if all candidates score below 7 on any axis, "REGENERATE: <what to fix>".
```

### 3.5 Refinement (only when the judge says REGENERATE)

```
The previous candidates failed judging. Critique: {judge_critique}
Rewrite {n} new candidates fixing exactly that issue. Same FACTS, same STYLE CARD,
same rules.
```

## 4. Fine-tuning: permitted, but earn it first

The track explicitly allows fine-tuning. The honest ROI call:

- **Do not start here.** Prompted generation + best-of-N + judge selection on a strong instruct model will likely clear the bar, and it is tunable in minutes rather than hours.
- Fine-tune **only if** the eval loop shows a persistent style gap that prompting cannot close (most likely candidate: the sarcastic/humorous separation, or a consistently flat tech-humor voice).
- If so: **LoRA SFT on Fireworks** ($0.50/1M training tokens, fine-tuned model serves at base-model price). Training data: build it from your own pipeline. Take your judge-approved winners plus hand-curated exemplars, format as (fact sheet -> styled caption) pairs, a few hundred examples per style is plenty for a style LoRA. That is a half-day task with the pipeline already producing the data.
- A style LoRA on a small text model is also a legitimately good demo/writeup beat ("we trained our own captioner on our own judge-filtered outputs"), which matters less on a leaderboard but still colors perception.

## 5. The Gemma angle

The $6k Gemma bonus spans tracks, and Gemma is served on Fireworks (Gemma 3 is multimodal). This track is the cheapest place to earn eligibility: use **Gemma 3 27B as the style-generation model** (and optionally as the Stage 2 VLM), judge with a non-Gemma model. That gives you a true "built on Gemma" story with zero architectural compromise. Verify at kickoff which Gemma variants the organizers count.

## 6. Kickoff-day playbook

1. Read the actual task spec: exact output format, how many clips, submission mechanism, which models are revealed/required, and precisely how the LLM judge is described. If they publish the judge rubric or judge model, mirror it in Stage 5 verbatim.
2. Download the clip set; run Stage 0/1 across all clips immediately (it is pure ffmpeg/Whisper plumbing, no tuning).
3. Generate fact sheets for every clip; **hand-verify a sample of 5** against the actual videos. Fix perception before touching style.
4. Write the four style cards' few-shot exemplars by hand (30 minutes, highest-leverage writing of the week).
5. Run the full loop on 3 clips end to end; eyeball; then run the full set.
6. Iterate: judge scores -> regenerate weak styles -> re-judge. Keep a per-clip, per-style score sheet so you can see exactly where the weak cells are.
7. Human pass on final winners. Submit early if the leaderboard allows resubmission; iterate after.

## 7. Pitfalls that quietly lose this track

- **Hallucinated humor.** The number one killer. Every invention costs accuracy points. The fact-sheet architecture exists to prevent exactly this; never generate style from raw frames.
- **Style collapse.** Sarcastic and humorous-non-tech merging; tech humor that is a generic joke wearing a lanyard. The distinctness check in the judge rubric and the "don't" lines in the style cards are the guardrails.
- **Missing the payoff frame.** Uniform sampling that skips the 0.8s where the thing actually happens. Shot-aware sampling plus a frame-difference spike check.
- **Ignoring audio.** Captions that miss the punchline because it was spoken.
- **On-screen text blindness.** Memes, signs, and UI text often carry the joke; the fact sheet explicitly extracts them.
- **Judge sycophancy to your own generator.** Cross-family judging, second-judge spot checks, human tiebreaks.
- **Format non-compliance.** A perfect caption in the wrong schema scores zero. Validate against the spec in Stage 6.
- **Over-length captions.** Density reads as wit; rambling reads as filler on both axes.
- **NSFW/edge content in clips.** If a clip contains something sensitive, keep the humor styles gentle and the target situational; the judge will not reward punching down, and neither will the organizers.

## 8. Build order

1. Stage 0/1 plumbing (ffmpeg, scene detection, Whisper). Runs on any video today; no dependency on kickoff.
2. Fact-sheet generation + hand-verification loop on sample videos you pick yourself (grab short clips of the likely genre and dry-run everything).
3. Style cards + exemplars, generation, judge, selection loop.
4. Score-sheet harness (per clip x per style grid).
5. Kickoff: swap in revealed models, run the playbook.
6. Optional: style LoRA if, and only if, the score sheet shows a stubborn gap.