# Estimand Contract

## Predictive estimands

At an authorized prediction timestamp, estimate horizon-specific P(Won), terminal event-time distribution with Won/Lost competing states, conditional Won value, realized value, and a capacity ranking. The primary policy estimand is observed realized value captured in the top 10% under the frozen scoring rule; WinCapture@10% is co-primary.

Track A labels distinguish `MATURE_POSITIVE`, `MATURE_NEGATIVE`, and `CENSORED`. Lost is a terminal competing event. Track B value is fixed-horizon post-contact marketplace value. Track C estimands are next purchase within 30/90 days, time to next purchase, and future 90-day spend.

## Causal estimand

Only verified randomized Track D supports treatment-effect evaluation. It compares response ranking against incremental-effect ranking at fixed capacities. Its results stay in a separate causal leaderboard and do not make AccountPulse causal.

## Sequential allocation estimand

No online, adaptive treatment-policy estimand is identified in v1.0. Historical shadow replay assesses operational mechanics, not deployed policy value.

## Observation and inference units

Opportunity/lead predictions use account-cluster or lead-aware inference as specified; lifecycle predictions use customer clusters. Rankings are evaluated in realistic temporal capacity cohorts. Hindsight oracle regret is descriptive only.
