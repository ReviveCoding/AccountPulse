# Desktop Study

## Decision context

Scarce sales and marketing capacity makes probability-only ranking incomplete: two opportunities with similar win probability can differ in expected value and time to realization. AccountPulse therefore evaluates decomposed timing, terminal cause, and value while retaining transparent direct baselines. Capacity capture—not generic accuracy—is the decision-facing endpoint, but calibration and robustness remain non-compensatory safeguards.

## Evidence strategy

The design triangulates four deliberately separate evidence types. Maven supplies a feature-rich but fictitious CRM benchmark. Olist tests whether the method remains useful with sparse, real first-contact information. UCI Online Retail II provides longitudinal sequences, text, and bipartite interactions for representation learning. Randomized Criteo data demonstrates why propensity and incrementality are different questions. These sources improve method coverage; none establishes enterprise production impact.

## Method rationale

Strong priors, heuristics, regularized linear models, boosted trees, valid survival/competing-risk models, value regressors, and LambdaMART set the minimum bar. AP-EV decomposes a score into interpretable timing, Won probability, Won value, and discount components. Decomposition may improve decision alignment but may also compound estimation error, so it is tested rather than presumed superior. Calibration candidates and abstention are similarly challengers, not guaranteed improvements.

## Evaluation design

Chronological FIT/VALIDATION/POLICY/LOCKED partitions emulate future deployment and prevent random-split optimism. Entity-disjoint cold evaluations expose memorization. Horizon maturity prevents unknown futures from becoming negatives. Point-in-time joins forbid future facts. Clustered bootstrap respects repeated decision units. A one-shot locked evaluation follows a complete protocol freeze.

## Operational design

Batch scoring is the v1.0 serving target. MLflow records runs and logical model families; DuckDB over canonical Parquet is the evidence layer. A single GPU lock prevents concurrent heavy CUDA jobs and every GPU-eligible stage refuses silent CPU fallback. Historical shadow replay validates delayed-label mechanics without being called online production deployment.

## Current evidence status

The qualified bootstrap proves the local CUDA stack works. At protocol initialization, no canonical raw Track A-D files were found in the repository or searched Downloads locations, so empirical source status remains pending acquisition and cannot support model-performance claims yet.
