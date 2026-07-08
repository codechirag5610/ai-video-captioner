"""Offline smoke test: exercises everything that does NOT hit the Fireworks API.

Proves preprocessing, frame encoding, prompt construction, JSON parsing, and the
config/style wiring all work before we spend a single credit.

    python tests/smoke_offline.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from captioner import preprocess as P
from captioner import understand, styles, comedy
from captioner.client import _parse_json
from captioner.cache import video_hash, Cache

FAILS = []


def check(name, cond):
    print(f"  {'✓' if cond else '✗'} {name}")
    if not cond:
        FAILS.append(name)


def main():
    clips = sorted((ROOT / "clips").glob("*.mp4"))
    assert clips, "no test clips found in ./clips"

    print("\n[1] probe + preprocess")
    with tempfile.TemporaryDirectory() as tmp:
        for clip in clips:
            pre = P.preprocess(clip, Path(tmp) / clip.stem, max_frames=8)
            silent = "silent" in clip.name
            check(f"{clip.name}: duration>0", pre.probe.duration > 0)
            check(f"{clip.name}: frames extracted", len(pre.frames) > 0)
            check(f"{clip.name}: audio matches expectation",
                  (pre.audio_path is None) == silent)
            if pre.frames:
                uri = P.encode_frame(pre.frames[0].path, max_edge=512)
                check(f"{clip.name}: frame encodes to data URI",
                      uri.startswith("data:image/jpeg;base64,") and len(uri) > 100)

            # Stage A message construction (no API call)
            msgs = understand.build_messages(pre, {"text": "", "language": None}, _spec_understand())
            has_image = any(
                isinstance(c, dict) and c.get("type") == "image_url"
                for c in msgs[1]["content"]
            )
            check(f"{clip.name}: Stage A builds vision message", has_image)

    print("\n[2] video hashing + cache")
    with tempfile.TemporaryDirectory() as tmp:
        c = Cache(Path(tmp))
        h = video_hash(clips[0])
        c.put(h, "understand", "sig", {"mood": "test"})
        check("cache round-trips", c.get(h, "understand", "sig") == {"mood": "test"})
        check("cache miss on wrong sig", c.get(h, "understand", "other") is None)

    print("\n[3] Stage B best-of-N prompt + style cards + few-shot")
    gt = {
        "setting": "kitchen", "subjects": ["a cat"], "events": ["0:01 - jumps", "0:03 - falls"],
        "dialogue_summary": "", "visible_text": "", "audio_description": "silence",
        "mood": "chaotic", "notable": "the cat misjudges the jump",
        "uncertain": ["the breed of the cat"], "confidence": 0.9,
    }
    # per-style prompt build (n=4), for each of the four styles
    for skey in styles.STYLES:
        msgs = styles._build_style_prompt(skey, gt, comedy_text="1. cat vs gravity", n=4)
        prompt = msgs[1]["content"]
        check(f"{skey}: asks for 4 candidates", '"candidates"' in prompt and "4 strings" in prompt)
        check(f"{skey}: injects ground truth", "cat" in prompt and "misjudges" in prompt)
        check(f"{skey}: warns off uncertain items", "UNCERTAIN" in prompt and "breed" in prompt)
    # style-specific behaviors
    nt = styles._build_style_prompt("humorous_non_tech", gt, "", 3)[1]["content"]
    check("non-tech prompt bans tech words", "deploy" in nt and "server" in nt)
    formal = styles._build_style_prompt("formal", gt, "SHOULD NOT APPEAR", 3)[1]["content"]
    check("formal prompt omits comedic material", "SHOULD NOT APPEAR" not in formal)
    check("candidate parser reads list",
          styles._parse_candidates({"candidates": ["a", "b", "c"]}, 4) == ["a", "b", "c"])

    print("\n[4] comedy-material rendering")
    block = comedy._render_material([
        {"element": "cat misjudges jump", "why_funny": "overconfidence", "tech_angle": "failed load test"},
        {"element": "glass into sink", "why_funny": "collateral", "tech_angle": ""},
    ])
    check("comedy block cites elements + tech angle",
          "cat misjudges jump" in block and "load test" in block)

    print("\n[5] non-tech leak detector (incl. inflections + false-positive guard)")
    _, clean = styles.sanitize_non_tech("A cat leaps and slides into the sink.")
    check("clean non-tech passes", clean)
    _, dirty = styles.sanitize_non_tech("The cat deployed itself to prod and crashed.")
    check("inflected tech word 'deployed' flagged", not dirty)
    for w in ["servers", "coding", "debugging", "compiled", "uploads"]:
        _, d = styles.sanitize_non_tech(f"The dog {w} something.")
        check(f"inflection '{w}' flagged", not d)
    # short/risky stems must NOT false-positive on ordinary words
    for phrase in ["He ate an apple in the air.", "The circle was aimmense.",
                   "She rammed the door and ran up the ramp."]:
        _, c = styles.sanitize_non_tech(phrase)
        check(f"no false positive: {phrase!r}", c)

    print("\n[6] robust JSON parser")
    check("parses fenced json", _parse_json('```json\n{"a": 1}\n```') == {"a": 1})
    check("parses embedded json", _parse_json('Sure! {"a": 2} done') == {"a": 2})

    print("\n[7] best-of-N engine end-to-end (mock client, no network)")
    _test_best_of_n(gt)


def _test_best_of_n(gt):
    """Drive pipeline._best_of_n with a fake client to exercise generation,
    judge-selection, distinctness threading, and the non-tech regen loop."""
    from captioner import pipeline
    from captioner.config import Config, ComedyConfig
    from captioner.styles import STYLE_LABELS

    class FakeClient:
        def __init__(self):
            self.calls = {"gen": 0, "judge": 0}
            self.leaked_once = False

        def chat_json(self, spec, messages, **kw):
            prompt = messages[-1]["content"]
            # Judge calls ask to score "CANDIDATES"; generation asks for "candidates" JSON.
            if "Score EACH candidate" in prompt:
                self.calls["judge"] += 1
                # If the candidates contain a banned word, score fine but let the
                # deterministic leak guard trigger the regen.
                return {
                    "candidates": [{"accuracy": 9, "tone": 9, "distinct": 9, "fit": 9,
                                    "justification": "grounded and on-style"}],
                    "winner": 1, "regenerate": "",
                }
            self.calls["gen"] += 1
            # First non-tech generation leaks a banned word; regen is clean.
            if "NO technology vocabulary" in prompt and not self.leaked_once:
                self.leaked_once = True
                return {"candidates": ["The cat deployed itself off the counter."]}
            return {"candidates": [f"A caption referencing the jump and the fall #{self.calls['gen']}"]}

    fake = FakeClient()
    cfg = Config.__new__(Config)  # bypass loader; set only what _best_of_n reads
    cfg.style = _spec_style()
    cfg.judge = _spec_style()
    from captioner.config import CritiqueConfig
    cfg.critique = CritiqueConfig(enabled=True, min_score=7.0, max_retries=1)

    captions, selection = pipeline._best_of_n(gt, "1. cat vs gravity", cfg, fake, do_judge=True)

    labels = [STYLE_LABELS[s] for s in ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]]
    check("all four styles produced", all(captions.get(l) for l in labels))
    check("selection recorded per style", all(l in selection for l in labels))
    check("judge invoked per style (+1 regen for non-tech leak)", fake.calls["judge"] == 5)
    nt = captions[STYLE_LABELS["humorous_non_tech"]]
    _, clean = styles.sanitize_non_tech(nt)
    check("non-tech leak was regenerated clean", clean)


def _spec_style():
    from captioner.config import ModelSpec
    return ModelSpec(model="fake", n_candidates=2, temperature_formal=0.3, temperature_humor=0.9)

    print("\n" + ("ALL SMOKE TESTS PASSED ✅" if not FAILS else f"FAILURES: {FAILS}"))
    return 1 if FAILS else 0


def _spec_understand():
    from captioner.config import ModelSpec
    return ModelSpec(model="x", supports_vision=True, max_images=8, image_max_edge=512)


if __name__ == "__main__":
    sys.exit(main())
