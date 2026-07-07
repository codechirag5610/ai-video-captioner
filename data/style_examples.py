"""Few-shot examples per style.

Each example is a (ground_truth_summary, caption) pair. These TEACH the register
and the accuracy-anchoring habit (every caption references concrete clip
details). They are intentionally generic so they generalize to any test clip.

Keep captions to 1-2 sentences: judges reward punchy, on-target lines.
"""

# A compact ground-truth blurb we show the model as the "input" in each shot.
_GT1 = ("A cat leaps onto a kitchen counter aiming for a glass of water, misjudges "
        "the jump, and slides off, knocking the glass into the sink. Owner gasps off-camera.")
_GT2 = ("A developer at a standing desk stares at a screen showing a failing CI "
        "pipeline (red X). They sip coffee, refresh the page twice, and the build "
        "goes green on its own. They shrug at the camera.")
_GT3 = ("Timelapse of a person assembling flat-pack furniture. Halfway through they "
        "hold up two leftover screws, look confused, then shove the shelf against "
        "the wall so it stands anyway.")

FEW_SHOT = {
    "formal": [
        (_GT1, "A domestic cat attempts to reach a glass of water on a kitchen counter, "
               "misjudges the leap, and slides off, sending the glass into the sink."),
        (_GT2, "A software developer monitors a failing continuous-integration pipeline; "
               "after two page refreshes the build passes without intervention."),
        (_GT3, "An individual assembles flat-pack furniture in timelapse, finishes with two "
               "surplus screws, and stabilizes the completed shelf against a wall."),
    ],
    "sarcastic": [
        (_GT1, "Ah yes, flawless landing. The glass clearly started it, and the sink was "
               "just collateral damage in this cat's ongoing war on gravity."),
        (_GT2, "Nothing says 'senior engineer' like refreshing a red build twice until the "
               "computer fixes itself and you get to take the credit."),
        (_GT3, "Two leftover screws? Purely decorative. The shelf leaning on the wall for "
               "dear life is a bold design choice, truly."),
    ],
    "humorous_tech": [
        (_GT1, "Cat pushed to prod without tests, immediately hit a null-gravity exception, "
               "and rolled the glass straight into the sink. Classic uncaught fall-through."),
        (_GT2, "The build was red, he refreshed twice, and it went green on its own — flaky "
               "test or divine intervention, ship it before CI changes its mind."),
        (_GT3, "Assembled the shelf, ended with two orphaned screws and undefined behavior, "
               "then resolved the merge conflict by shoving it against the wall. Works in prod."),
    ],
    "humorous_non_tech": [
        (_GT1, "The cat had one job, aimed for the water, and instead invented an extreme "
               "sport where the finish line is the kitchen sink."),
        (_GT2, "He stared, he sipped, he refreshed — and the problem quietly fixed itself, "
               "the way all my problems absolutely never do."),
        (_GT3, "Finished the furniture with two spare screws and a shelf that only stands "
               "because the wall agreed to be its best friend."),
    ],
}
