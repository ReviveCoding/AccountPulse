# Model Card

## Intended use

Research evaluation of predictive ranking under fixed capacity on public benchmarks. Not causal targeting, production deployment, or salesperson-effect measurement.

## Evaluated models

Track B: historical value prior, regularized Logistic, CUDA XGBoost, raw/Platt/isotonic calibration, value and timing regressors, simple EV, and sparse AP-EV. Track C: RFM, Logistic, CUDA XGBoost, Cox, three-seed BF16 Temporal Transformer, PIT GraphSAGE, cached MiniLM text embeddings + MLP, early/fixed/adaptive fusion, and modality ablations. Separate Track D: treated-response, S-learner, T-learner, transformed-outcome, and propensity/outcome nuisance models with held-out AIPW policy evaluation.

## Selected decision

`RETAIN_BASELINE`. UCI RFM ValueCapture@10% = 0.573; fixed fusion = 0.575, difference = 0.003, 95% CI [-0.010, 0.019]. Adaptive fusion AUC = 0.869, Brier = 0.154, calibration slope = 0.913.

## Limitations

No complete Track-A source, no Track-A LambdaMART/AP-EV/competing-risk/locked result, and no production traffic. The Criteo causal bridge is advertising-specific and cannot validate transportability to CRM, Olist, or UCI. Olist predictors are extremely sparse. UCI customer histories and values are not B2B CRM opportunities. Cold performance is materially weaker. Cox failed scientifically and is ineligible.
