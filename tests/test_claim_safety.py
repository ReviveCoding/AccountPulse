from pathlib import Path


def test_master_protocol_preserves_claim_boundaries() -> None:
    text = Path("ACCOUNT_PULSE_MASTER_PROMPT.md").read_text(encoding="utf-8")
    assert "predictive decision science" in text.lower()
    assert "never causal lift" in text.lower()
    assert "not enterprise CRM" in text


def test_tracka_receipt_is_bound_to_frozen_protocol() -> None:
    receipt = Path("LOCKED_START_RECEIPT.json").read_text(encoding="utf-8")
    freeze = Path("FREEZE_MANIFEST.yaml").read_text(encoding="utf-8")
    assert "AP-V1-TRACKA-20260923-A1" in receipt
    assert "status: FROZEN" in freeze
    assert "locked_outcomes_opened: false" in freeze
