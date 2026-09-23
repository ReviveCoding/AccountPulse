# Business Decision Card

- **Decision:** which eligible opportunities/accounts receive limited Sales or Marketing attention now.
- **Prediction timestamp:** the scoring `as_of` time; every input must be available no later than it.
- **Primary persona:** PREASSIGN, excluding agent, manager, and sales-team identity.
- **Secondary persona:** POSTASSIGN, allowing assignment facts already known at scoring time; no causal salesperson interpretation.
- **Capacity:** 10% primary; 5%, 20%, and 30% sensitivity.
- **Utility:** future realized value captured, subject to win-capture, calibration, robustness, system, and claim-safety gates.
- **Primary metric:** ValueCapture@10%.
- **Co-primary metric:** WinCapture@10%.
- **Action:** prioritize in descending eligible score; uncertain cases may be flagged for manual review under a POLICY-selected abstention rule.
- **Non-action:** AccountPulse does not estimate the causal effect of outreach and does not prescribe an online treatment policy.
- **Evidence:** Track A is fictitious public CRM; B/C/D have their separate public benchmark boundaries. None is enterprise production evidence.
