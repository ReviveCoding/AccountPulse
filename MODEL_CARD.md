# Model Card

## Intended use

Research evaluation of predictive ranking under fixed capacity on public benchmarks. Not causal targeting, production deployment, or salesperson-effect measurement.

## Evaluated models

Track A: business priors/heuristic, Logistic, CatBoost GPU, XGBoost CUDA, Cox, XGBoost AFT, discrete competing terminal model, Gamma/Tweedie/XGBoost value models, D0-D4, CUDA LambdaMART, AP-EV, calibration, PREASSIGN/POSTASSIGN comparison, and frozen ablations. Track B/C models remain as previously reported. Separate Track D uses response and uplift learners with held-out AIPW policy evaluation.

## Selected decision

`RETAIN_BASELINE`. On Track-A LOCKED, the business heuristic achieved ValueCapture@10% = 0.292 and AP-EV = 0.242; difference −0.050, 95% CI [−0.145, 0.081]. WinCapture@10% was 0.122 versus 0.128. AP-EV failed G5; isotonic slope 0.244 failed G7; no T2 cold accounts made G9 not evaluable. The prior Track-B/C R2 retention decision is unchanged.

## Limitations

Maven is fictitious public data and has no production traffic. Track-A has only 403 LOCKED rows, no cold accounts, and severe temporal probability shift; RSF was not run because no reliable implementation was installed. The Criteo causal bridge remains advertising-specific and cannot validate transportability to CRM, Olist, or UCI. Track-C Cox remains a scientific failure.
