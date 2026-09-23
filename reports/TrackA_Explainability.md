# Track A Explainability

base_protocol: `AP-V1-TRACKA-20260923-A1`
diagnostic_protocol: `AP-V1-TRACKA-20260923-A1-DIAG1`
result_label: `POST_LOCK_DIAGNOSTIC_ONLY`
post_lock: `true`
decision_eligible: `false`


Native SHAP/contribution inference and grouped decision-specific permutation were not executed because CUDA is unavailable; silently moving the frozen GPU models to CPU is prohibited. The artifacts explicitly record this `INFRASTRUCTURE_FAILURE` status.

Local explanation targets remain predefined as top missed whales, false priorities, and largest B1/AP-EV disagreements. No explanation was used for feature selection or model alteration. PDP/ICE was not executed and, when prospectively run, must be labeled **MODEL RESPONSE, NOT CAUSAL EFFECT**; correlated-feature combinations may be unrealistic.

## GPU Infrastructure-Recovery Supplement

DIAG1 initially could not execute GPU-dependent explainability and its `INFRASTRUCTURE_FAILURE` record above remains unchanged. Later WSL/PyTorch CUDA qualification succeeded on `NVIDIA GeForce RTX 4090 Laptop GPU`. GPU1 used only frozen read-only inference with `device=cuda:0`; no model training, updating, recalibration, feature selection, or scientific retuning occurred.

Native B4 TreeSHAP passed strict additivity on POLICY and LOCKED. See [the separate GPU supplement](TrackA_GPU_Explainability_Supplement.md). All attributions are model explanations, not causal effects.
