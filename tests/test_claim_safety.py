from pathlib import Path


def test_master_protocol_preserves_claim_boundaries() -> None:
    text = Path("ACCOUNT_PULSE_MASTER_PROMPT.md").read_text(encoding="utf-8")
    assert "predictive decision science" in text.lower()
    assert "never causal lift" in text.lower()
    assert "not enterprise CRM" in text


def test_no_locked_receipt_exists_before_freeze() -> None:
    assert not Path("LOCKED_START_RECEIPT.json").exists()
