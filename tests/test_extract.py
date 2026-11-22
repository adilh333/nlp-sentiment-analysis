from fos_triage.extract import intake_from_text

DECISION = """
DRN-0000001

The complaint

Ms A complains Example Insurance Ltd declined a home claim after a pipe leaked and damaged the kitchen.

What happened

Ms A held a buildings policy. In March 2024 a pipe leaked under the sink and ruined the kitchen units. The insurer said the pipe had worn out gradually and declined the claim. Example Insurance Ltd didn't uphold the complaint. Our adjudicator upheld the complaint. The kitchen units were oak.

What I've decided - and why

I've decided to uphold the complaint because the damage was sudden.

My final decision

I uphold the complaint and require £400 compensation.
"""


def test_decision_section_is_removed():
    intake = intake_from_text(DECISION)
    text = intake.model_text.lower()
    assert "pipe leaked" in text
    assert "kitchen" in text
    assert "i've decided" not in text
    assert "investigator" not in text
    assert "compensation" not in text or "£400" not in text
    assert "uphold" not in text
    assert intake.leak_flags == []


def test_empty_text():
    intake = intake_from_text("  ")
    assert intake.model_text == ""
