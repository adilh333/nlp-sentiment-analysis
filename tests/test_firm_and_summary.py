from fos_triage.firm_type import firm_type
from fos_triage.summarize import extractive_summary, key_issues


def test_firm_types():
    assert firm_type("Barclays Bank UK PLC") == "bank_or_building_society"
    assert firm_type("Aviva Insurance Limited") == "insurer"
    assert firm_type("Startline Motor Finance Limited") == "motor_finance"
    assert firm_type("Monzo Bank Ltd") == "bank_or_building_society"
    assert firm_type("Revolut Ltd") == "payments_or_emoney"
    assert firm_type(None) == "unknown"


def test_summary_uses_the_narrative():
    text = (
        "Ms A complains the insurer declined a home claim after a pipe leaked under the sink. "
        "The leak ruined the kitchen units and the ceiling below the bathroom. "
        "She had held the buildings policy for four years and had made no earlier claim. "
        "An engineer later said the pipe joint had failed suddenly rather than worn out."
    )
    summary = extractive_summary(text, n_sentences=2)
    assert "pipe" in summary.lower() or "kitchen" in summary.lower()
    issues = key_issues(text)
    assert issues
    assert all(" " in issue or len(issue) > 3 for issue in issues)
