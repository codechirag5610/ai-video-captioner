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
from captioner import understand, styles
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

    print("\n[3] Stage B prompt + few-shot wiring")
    gt = {
        "setting": "kitchen", "subjects": ["a cat"], "events": ["0:01 - jumps", "0:03 - falls"],
        "dialogue_summary": "", "visible_text": "", "audio_description": "silence",
        "mood": "chaotic", "notable": "the cat misjudges the jump", "confidence": 0.9,
    }
    msgs = styles._build_prompt(gt)
    prompt = msgs[1]["content"]
    check("prompt names all 4 styles",
          all(lbl in prompt for lbl in ["formal", "sarcastic", "humorous-tech", "humorous-non-tech"]))
    check("prompt includes banned-word list", "deploy" in prompt and "server" in prompt)
    check("prompt includes few-shot examples", "EXAMPLE 1" in prompt)
    check("prompt injects ground truth", "cat" in prompt and "misjudges" in prompt)

    print("\n[4] non-tech leak detector")
    _, clean = styles._sanitize_non_tech("A cat leaps and slides into the sink.")
    check("clean non-tech passes", clean)
    _, dirty = styles._sanitize_non_tech("The cat deployed itself to prod and crashed.")
    check("tech words flagged", not dirty)

    print("\n[5] robust JSON parser")
    check("parses fenced json", _parse_json('```json\n{"a": 1}\n```') == {"a": 1})
    check("parses embedded json", _parse_json('Sure! {"a": 2} done') == {"a": 2})

    print("\n" + ("ALL SMOKE TESTS PASSED ✅" if not FAILS else f"FAILURES: {FAILS}"))
    return 1 if FAILS else 0


def _spec_understand():
    from captioner.config import ModelSpec
    return ModelSpec(model="x", supports_vision=True, max_images=8, image_max_edge=512)


if __name__ == "__main__":
    sys.exit(main())
