"""Track-A development models, policy selection, and reusable frozen scorer."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from catboost import CatBoostClassifier
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import GammaRegressor, LogisticRegression, TweedieRegressor
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from accountpulse.contracts import capacity_metrics
from accountpulse.evidence import EVIDENCE, write_table
from accountpulse.gpu import GPUSlot, require_cuda
from accountpulse.paths import REPO
from accountpulse.tracka_data import (
    POSTASSIGN_CATEGORICAL,
    PREASSIGN_CATEGORICAL,
    PREASSIGN_NUMERIC,
    PROTOCOL,
)

SEED = 20260923
CAPACITIES = [0.05, 0.10, 0.20, 0.30]
MODEL_DIR = REPO / "artifacts/models/tracka"
BUNDLE_PATH = MODEL_DIR / "tracka_frozen_bundle.joblib"


def _preprocessor(postassign: bool = False) -> ColumnTransformer:
    categorical = PREASSIGN_CATEGORICAL + (POSTASSIGN_CATEGORICAL if postassign else [])
    return ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                PREASSIGN_NUMERIC,
            ),
        ],
        verbose_feature_names_out=False,
    )


def _tie_break(score: np.ndarray, identifiers: pd.Series) -> np.ndarray:
    hashes = np.array(
        [int.from_bytes(hashlib.sha256(str(x).encode()).digest()[:8], "big") for x in identifiers],
        dtype=np.float64,
    )
    hashes /= np.iinfo(np.uint64).max
    return np.asarray(score, dtype=float) + hashes * max(1.0, float(np.max(np.abs(score)))) * 1e-12


def _calibration_fit(raw: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    clipped = np.clip(raw, 1e-6, 1 - 1e-6)
    platt = LogisticRegression(C=1e6, random_state=SEED).fit(logit(clipped).reshape(-1, 1), y)
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(raw, y)
    return {"platt": platt, "isotonic": isotonic}


def _calibrate(raw: np.ndarray, method: str, calibrators: dict[str, Any]) -> np.ndarray:
    raw = np.clip(np.asarray(raw, dtype=float), 1e-6, 1 - 1e-6)
    if method == "raw":
        return raw
    if method == "platt":
        return calibrators["platt"].predict_proba(logit(raw).reshape(-1, 1))[:, 1]
    if method == "isotonic":
        return np.asarray(calibrators["isotonic"].predict(raw), dtype=float)
    raise ValueError(method)


def _calibration_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    design = np.column_stack([np.ones(len(p)), logit(p)])
    fit = LogisticRegression(C=1e6, fit_intercept=False, random_state=SEED).fit(design, y)
    top = np.argsort(-p)[: max(1, int(np.ceil(0.1 * len(p))))]
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "calibration_intercept": float(fit.coef_[0, 0]),
        "calibration_slope": float(fit.coef_[0, 1]),
        "top_decile_predicted": float(p[top].mean()),
        "top_decile_observed": float(y[top].mean()),
    }


def _rank_metrics(
    frame: pd.DataFrame, score: np.ndarray, model: str, cohort: str
) -> list[dict[str, Any]]:
    stable = _tie_break(score, frame["opportunity_id"])
    rows = []
    for capacity in CAPACITIES:
        metric = capacity_metrics(
            stable,
            frame["realized_value"].to_numpy(float),
            frame["won"].to_numpy(int),
            capacity,
        )
        rows.append(
            {
                "protocol_id": PROTOCOL,
                "track": "A",
                "cohort": cohort,
                "model": model,
                "capacity": capacity,
                "value_capture": metric.value_capture,
                "win_capture": metric.win_capture,
                "realized_value": metric.realized_value,
                "oracle_regret": metric.oracle_regret,
                "selected_count": metric.selected_count,
            }
        )
    return rows


def _account_bootstrap(
    frame: pd.DataFrame,
    challenger: np.ndarray,
    baseline: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    groups = frame["account_key"].astype(str).unique()
    account_keys = frame["account_key"].astype(str).to_numpy()
    indices = {key: np.flatnonzero(account_keys == key) for key in groups}
    rng = np.random.default_rng(seed)
    value_diff = np.empty(replicates)
    win_diff = np.empty(replicates)
    values = frame["realized_value"].to_numpy(float)
    wins = frame["won"].to_numpy(int)
    for iteration in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        take = np.concatenate([indices[key] for key in sampled])
        ch = capacity_metrics(challenger[take], values[take], wins[take], 0.10)
        ba = capacity_metrics(baseline[take], values[take], wins[take], 0.10)
        value_diff[iteration] = ch.value_capture - ba.value_capture
        win_diff[iteration] = ch.win_capture - ba.win_capture
    return {
        "replicates": replicates,
        "cluster": "account_key",
        "value_capture_difference": {
            "estimate": float(
                capacity_metrics(challenger, values, wins, 0.10).value_capture
                - capacity_metrics(baseline, values, wins, 0.10).value_capture
            ),
            "ci_95": [float(np.quantile(value_diff, 0.025)), float(np.quantile(value_diff, 0.975))],
            "probability_positive": float((value_diff > 0).mean()),
        },
        "win_capture_difference": {
            "estimate": float(
                capacity_metrics(challenger, values, wins, 0.10).win_capture
                - capacity_metrics(baseline, values, wins, 0.10).win_capture
            ),
            "ci_95": [float(np.quantile(win_diff, 0.025)), float(np.quantile(win_diff, 0.975))],
        },
    }


def _xgb_classifier(multiclass: bool = False) -> xgb.XGBClassifier:
    extra = (
        {"objective": "multi:softprob", "num_class": 3}
        if multiclass
        else {"objective": "binary:logistic"}
    )
    return xgb.XGBClassifier(
        n_estimators=220,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=2.0,
        tree_method="hist",
        device="cuda",
        random_state=SEED,
        n_jobs=1,
        **extra,
    )


def _xgb_regressor() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=240,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        tree_method="hist",
        device="cuda",
        random_state=SEED,
        n_jobs=1,
    )


@dataclass
class FrozenTrackAScorer:
    preprocessor: Any
    post_preprocessor: Any
    classifiers: dict[str, Any]
    calibrators: dict[str, Any]
    timing_model: Any
    competing_model: Any
    conditional_win: Any
    conditional_calibrators: dict[str, Any]
    value_models: dict[str, Any]
    direct_value_model: Any
    aft_model: Any
    ranker: Any
    postassign_model: Any
    empirical_timing: np.ndarray
    selected_config: dict[str, Any]
    strongest_baseline: str
    fit_value_thresholds: list[float]

    def _encoded(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.preprocessor.transform(frame), dtype=np.float32)

    def _conditional_components(
        self, encoded: np.ndarray, calibration: str, value_model: str
    ) -> tuple[np.ndarray, np.ndarray]:
        win_columns = []
        value_columns = []
        for time_bin in (1, 2):
            augmented = np.column_stack([encoded, np.full(len(encoded), time_bin)])
            raw_win = self.conditional_win.predict_proba(augmented)[:, 1]
            win_columns.append(_calibrate(raw_win, calibration, self.conditional_calibrators))
            raw_value = self.value_models[value_model].predict(augmented)
            if value_model == "xgboost":
                raw_value = np.expm1(raw_value)
            value_columns.append(np.maximum(raw_value, 0.0))
        return np.column_stack(win_columns), np.column_stack(value_columns)

    def apev(self, frame: pd.DataFrame, override: dict[str, Any] | None = None) -> np.ndarray:
        cfg = {**self.selected_config, **(override or {})}
        encoded = self._encoded(frame)
        if cfg["timing"] == "discrete_xgb":
            timing = self.timing_model.predict_proba(encoded)[:, 1:3]
        else:
            timing = np.repeat(self.empirical_timing.reshape(1, 2), len(frame), axis=0)
        win, value = self._conditional_components(encoded, cfg["win"], cfg["value"])
        horizon_mask = np.array([30, 60]) <= cfg["horizon"]
        discount = np.exp(-cfg["rho"] * np.array([15.0, 45.0]))
        if cfg.get("minus_timing"):
            timing = np.repeat(np.array([[0.5, 0.5]]), len(frame), axis=0)
        if cfg.get("minus_value"):
            value = np.ones_like(value)
        if cfg.get("minus_discount"):
            discount = np.ones(2)
        return (
            timing[:, horizon_mask]
            * win[:, horizon_mask]
            * value[:, horizon_mask]
            * discount[horizon_mask]
        ).sum(axis=1)

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        encoded = self._encoded(frame)
        raw = {name: model.predict_proba(encoded)[:, 1] for name, model in self.classifiers.items()}
        selected_classifier = self.selected_config["classifier"]
        probability = _calibrate(
            raw[selected_classifier],
            self.selected_config["fixed_horizon_calibration"],
            self.calibrators,
        )
        timing_probability = self.timing_model.predict_proba(encoded)
        terminal_probability = 1 - timing_probability[:, 0]
        expected_days = (
            timing_probability[:, 1] * 15
            + timing_probability[:, 2] * 45
            + timing_probability[:, 0] * 60
        )
        conditional_value = np.maximum(
            np.expm1(
                self.value_models["xgboost"].predict(
                    np.column_stack([encoded, np.ones(len(frame))])
                )
            ),
            0,
        )
        direct_value = np.maximum(np.expm1(self.direct_value_model.predict(encoded)), 0)
        predictions = pd.DataFrame(
            {
                "opportunity_id": frame["opportunity_id"].to_numpy(),
                "account_key": frame["account_key"].to_numpy(),
                "engage_date": frame["engage_date"].to_numpy(),
                "B0_historical_prior": frame["account_prior_win_rate"].to_numpy(float),
                "B1_business_heuristic": (
                    frame["sales_price"].fillna(0).to_numpy(float)
                    * frame["product_prior_win_rate"].to_numpy(float)
                    * frame["sector_prior_win_rate"].to_numpy(float)
                ),
                "B2_logistic": raw["B2_logistic"],
                "B3_catboost_gpu": raw["B3_catboost_gpu"],
                "B4_xgboost_cuda": raw["B4_xgboost_cuda"],
                "D0_p_won": probability,
                "D1_predicted_realized_value": direct_value,
                "D2_probability_times_value": probability * conditional_value,
                "D3_survival_urgency": terminal_probability / np.maximum(expected_days, 1),
                "D4_discounted_ev": probability
                * conditional_value
                * np.exp(-0.005 * expected_days),
                "LambdaMART": self.ranker.predict(encoded),
                "AccountPulse_EV": self.apev(frame),
                "win_probability": probability,
                "expected_terminal_days": expected_days,
                "expected_value_given_win": conditional_value,
            }
        )
        post_encoded = np.asarray(self.post_preprocessor.transform(frame), dtype=np.float32)
        predictions["POSTASSIGN_xgboost"] = self.postassign_model.predict_proba(post_encoded)[:, 1]
        return predictions


def run_development() -> dict[str, Any]:
    started = time.perf_counter()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(EVIDENCE / "maven_development.parquet")
    fit = data[data["split"] == "FIT"].copy()
    validation = data[data["split"] == "VALIDATION"].copy()
    policy = data[data["split"] == "POLICY"].copy()
    if (data["split"] == "LOCKED").any():
        raise ValueError("Protected LOCKED labels entered development")

    preprocessor = _preprocessor(False)
    x_fit = np.asarray(preprocessor.fit_transform(fit), dtype=np.float32)
    x_validation = np.asarray(preprocessor.transform(validation), dtype=np.float32)
    x_policy = np.asarray(preprocessor.transform(policy), dtype=np.float32)
    y_fit = fit["won"].to_numpy(int)
    y_validation = validation["won"].to_numpy(int)
    y_policy = policy["won"].to_numpy(int)

    logistic = LogisticRegression(C=0.2, max_iter=2000, class_weight="balanced", random_state=SEED)
    logistic.fit(x_fit, y_fit)
    classifiers: dict[str, Any] = {"B2_logistic": logistic}
    gpu_started = time.perf_counter()
    with GPUSlot(timeout=10):
        require_cuda()
        catboost = CatBoostClassifier(
            iterations=250,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="Logloss",
            task_type="GPU",
            devices="0",
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
        )
        catboost.fit(x_fit, y_fit)
        boosted = _xgb_classifier()
        boosted.fit(x_fit, y_fit)
        classifiers.update({"B3_catboost_gpu": catboost, "B4_xgboost_cuda": boosted})

        timing_model = _xgb_classifier(multiclass=True)
        timing_model.fit(x_fit, fit["time_bin"].to_numpy(int))
        competing_model = _xgb_classifier(multiclass=True)
        competing_target = np.select([fit["won"].eq(1), fit["lost"].eq(1)], [1, 2], default=0)
        competing_model.fit(x_fit, competing_target)

        direct_value_model = _xgb_regressor()
        direct_value_model.fit(x_fit, np.log1p(fit["realized_value"].to_numpy(float)))

        terminal_fit = fit[fit["event_within_horizon"]].copy()
        terminal_x = x_fit[fit["event_within_horizon"].to_numpy(bool)]
        terminal_augmented = np.column_stack([terminal_x, terminal_fit["time_bin"].to_numpy(int)])
        conditional_win = _xgb_classifier()
        conditional_win.fit(terminal_augmented, terminal_fit["won"].to_numpy(int))
        won_fit = terminal_fit[terminal_fit["won"] == 1]
        won_x = terminal_augmented[terminal_fit["won"].to_numpy(bool)]
        xgb_value = _xgb_regressor()
        xgb_value.fit(won_x, np.log1p(won_fit["realized_value"].to_numpy(float)))

        aft_train = xgb.DMatrix(x_fit)
        observed = fit["event_within_horizon"].to_numpy(bool)
        lower = fit["event_time"].to_numpy(float)
        upper = np.where(observed, lower, np.inf)
        aft_train.set_float_info("label_lower_bound", lower)
        aft_train.set_float_info("label_upper_bound", upper)
        aft_model = xgb.train(
            {
                "objective": "survival:aft",
                "eval_metric": "aft-nloglik",
                "aft_loss_distribution": "normal",
                "aft_loss_distribution_scale": 1.0,
                "tree_method": "hist",
                "device": "cuda",
                "max_depth": 4,
                "eta": 0.05,
                "seed": SEED,
            },
            aft_train,
            num_boost_round=220,
        )

        values_positive = fit.loc[fit["realized_value"] > 0, "realized_value"]
        thresholds = [float(x) for x in values_positive.quantile([0.25, 0.5, 0.75])]
        relevance = np.where(
            fit["realized_value"].to_numpy(float) > 0,
            np.digitize(fit["realized_value"].to_numpy(float), thresholds) + 1,
            0,
        )
        qid_text = fit["engage_date"].dt.to_period("M").astype(str)
        rare = qid_text.value_counts().loc[lambda s: s < 2].index
        qid_text = qid_text.where(~qid_text.isin(rare), "EARLY_COMBINED")
        order = np.argsort(qid_text.to_numpy(), kind="stable")
        qids = pd.factorize(qid_text.iloc[order], sort=True)[0]
        ranker = xgb.XGBRanker(
            objective="rank:ndcg",
            n_estimators=220,
            max_depth=4,
            learning_rate=0.05,
            tree_method="hist",
            device="cuda",
            random_state=SEED,
            n_jobs=1,
        )
        ranker.fit(x_fit[order], relevance[order], qid=qids)

        post_preprocessor = _preprocessor(True)
        post_x_fit = np.asarray(post_preprocessor.fit_transform(fit), dtype=np.float32)
        postassign_model = _xgb_classifier()
        postassign_model.fit(post_x_fit, y_fit)
    gpu_seconds = time.perf_counter() - gpu_started

    gamma = GammaRegressor(alpha=0.1, max_iter=2000).fit(won_x, won_fit["realized_value"])
    tweedie = TweedieRegressor(power=1.5, alpha=0.1, link="log", max_iter=2000).fit(
        won_x, won_fit["realized_value"]
    )

    cox_frame = pd.DataFrame(x_fit)
    cox_frame["duration"] = fit["event_time"].to_numpy(float)
    cox_frame["event"] = fit["event_within_horizon"].to_numpy(int)
    nonconstant = [column for column in cox_frame.columns[:-2] if cox_frame[column].std() > 1e-8]
    cox_columns = nonconstant[: min(45, len(nonconstant))]
    cox = CoxPHFitter(penalizer=0.5)
    cox_status = "COMPLETE"
    try:
        cox.fit(cox_frame[[*cox_columns, "duration", "event"]], "duration", "event")
    except Exception as exc:
        cox_status = f"SCIENTIFIC_FAILURE:{type(exc).__name__}"

    validation_raw = {
        name: model.predict_proba(x_validation)[:, 1] for name, model in classifiers.items()
    }
    validation_rows = []
    for name, scores in validation_raw.items():
        metric = capacity_metrics(
            _tie_break(scores, validation["opportunity_id"]),
            validation["realized_value"].to_numpy(float),
            y_validation,
            0.10,
        )
        validation_rows.append(
            {
                "model": name,
                "value_capture_10": metric.value_capture,
                "win_capture_10": metric.win_capture,
            }
        )
    validation_selection = pd.DataFrame(validation_rows).sort_values(
        ["value_capture_10", "win_capture_10", "model"], ascending=[False, False, True]
    )
    selected_classifier = str(validation_selection.iloc[0]["model"])
    calibrators = _calibration_fit(validation_raw[selected_classifier], y_validation)
    calibration_rows = []
    policy_raw = classifiers[selected_classifier].predict_proba(x_policy)[:, 1]
    for method in ("raw", "platt", "isotonic"):
        p = _calibrate(policy_raw, method, calibrators)
        calibration_rows.append(
            {
                "protocol_id": PROTOCOL,
                "cohort": "POLICY",
                "method": method,
                **_calibration_metrics(y_policy, p),
            }
        )
    calibration = pd.DataFrame(calibration_rows).sort_values(["brier", "log_loss", "method"])
    selected_fixed_calibration = str(calibration.iloc[0]["method"])

    validation_terminal = validation[validation["event_within_horizon"]]
    val_terminal_x = x_validation[validation["event_within_horizon"].to_numpy(bool)]
    val_terminal_augmented = np.column_stack(
        [val_terminal_x, validation_terminal["time_bin"].to_numpy(int)]
    )
    conditional_raw = conditional_win.predict_proba(val_terminal_augmented)[:, 1]
    conditional_calibrators = _calibration_fit(
        conditional_raw, validation_terminal["won"].to_numpy(int)
    )

    empirical_timing = (
        fit["time_bin"].value_counts(normalize=True).reindex([1, 2], fill_value=0).to_numpy(float)
    )
    provisional = FrozenTrackAScorer(
        preprocessor=preprocessor,
        post_preprocessor=post_preprocessor,
        classifiers=classifiers,
        calibrators=calibrators,
        timing_model=timing_model,
        competing_model=competing_model,
        conditional_win=conditional_win,
        conditional_calibrators=conditional_calibrators,
        value_models={"gamma": gamma, "tweedie": tweedie, "xgboost": xgb_value},
        direct_value_model=direct_value_model,
        aft_model=aft_model,
        ranker=ranker,
        postassign_model=postassign_model,
        empirical_timing=empirical_timing,
        selected_config={},
        strongest_baseline="",
        fit_value_thresholds=thresholds,
    )

    search = yaml.safe_load((REPO / "configs/track_a.yaml").read_text())["apev_search"]
    candidate_rows = []
    candidate_scores: dict[str, np.ndarray] = {}
    for horizon in search["horizons"]:
        for rho in search["rho"]:
            for timing in search["timing"]:
                for win in search["win"]:
                    for value in search["value"]:
                        candidate = {
                            "horizon": horizon,
                            "rho": rho,
                            "timing": timing,
                            "win": win,
                            "value": value,
                        }
                        name = f"H{horizon}_rho{rho}_{timing}_{win}_{value}"
                        score = provisional.apev(policy, candidate)
                        metric = capacity_metrics(
                            _tie_break(score, policy["opportunity_id"]),
                            policy["realized_value"].to_numpy(float),
                            y_policy,
                            0.10,
                        )
                        candidate_rows.append(
                            {
                                "candidate": name,
                                **candidate,
                                "value_capture_10": metric.value_capture,
                                "win_capture_10": metric.win_capture,
                            }
                        )
                        candidate_scores[name] = score
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["value_capture_10", "win_capture_10", "candidate"], ascending=[False, False, True]
    )
    winner = candidates.iloc[0]
    selected_config = {
        "horizon": int(winner["horizon"]),
        "rho": float(winner["rho"]),
        "timing": str(winner["timing"]),
        "win": str(winner["win"]),
        "value": str(winner["value"]),
        "classifier": selected_classifier,
        "fixed_horizon_calibration": selected_fixed_calibration,
    }
    provisional.selected_config = selected_config

    policy_predictions = provisional.score(policy)
    validation_predictions = provisional.score(validation)
    eligible_baselines = [
        "B0_historical_prior",
        "B1_business_heuristic",
        "B2_logistic",
        "B3_catboost_gpu",
        "B4_xgboost_cuda",
        "D0_p_won",
        "D1_predicted_realized_value",
        "D2_probability_times_value",
        "D3_survival_urgency",
        "D4_discounted_ev",
        "LambdaMART",
    ]
    ranking_rows: list[dict[str, Any]] = []
    for cohort, frame, predictions in (
        ("VALIDATION", validation, validation_predictions),
        ("POLICY", policy, policy_predictions),
    ):
        for model in [*eligible_baselines, "AccountPulse_EV", "POSTASSIGN_xgboost"]:
            ranking_rows.extend(
                _rank_metrics(frame, predictions[model].to_numpy(float), model, cohort)
            )
    ranking = pd.DataFrame(ranking_rows)
    at_ten = ranking[(ranking["cohort"] == "POLICY") & (ranking["capacity"] == 0.10)]
    strongest_baseline = str(
        at_ten[at_ten["model"].isin(eligible_baselines)]
        .sort_values(["value_capture", "win_capture", "model"], ascending=[False, False, True])
        .iloc[0]["model"]
    )
    provisional.strongest_baseline = strongest_baseline

    # Leakage diagnostics use VALIDATION only and never enter model selection.
    leakage = []
    l0 = validation_predictions["B4_xgboost_cuda"].to_numpy(float)
    leakage.append(
        {
            "variant": "L0",
            "status": "ELIGIBLE",
            **_rank_metrics(validation, l0, "L0", "VALIDATION")[1],
        }
    )
    invalid = validation.copy()
    invalid["future_close_value"] = invalid["realized_value"]
    invalid["future_won"] = invalid["won"]
    l1 = invalid["future_close_value"].to_numpy(float) + invalid["future_won"].to_numpy(float) * 1e6
    leakage.append(
        {
            "variant": "L1",
            "status": "INELIGIBLE_LEAKAGE_DIAGNOSTIC",
            **_rank_metrics(validation, l1, "L1", "VALIDATION")[1],
        }
    )
    full_account = data.groupby("account_key")["won"].mean()
    l2 = validation["account_key"].map(full_account).fillna(data["won"].mean()).to_numpy(float)
    leakage.append(
        {
            "variant": "L2",
            "status": "INELIGIBLE_LEAKAGE_DIAGNOSTIC",
            **_rank_metrics(validation, l2, "L2", "VALIDATION")[1],
        }
    )
    future_value = data.groupby("account_key")["realized_value"].sum()
    l3 = validation["account_key"].map(future_value).fillna(0).to_numpy(float)
    leakage.append(
        {
            "variant": "L3",
            "status": "INELIGIBLE_LEAKAGE_DIAGNOSTIC",
            **_rank_metrics(validation, l3, "L3", "VALIDATION")[1],
        }
    )
    leakage_frame = pd.DataFrame(leakage)

    # Timing/survival evidence on POLICY without using LOCKED.
    survival_rows: list[dict[str, Any]] = []
    timing_prob = timing_model.predict_proba(x_policy)
    for horizon, positive_probability in (
        (30, timing_prob[:, 1]),
        (60, timing_prob[:, 1] + timing_prob[:, 2]),
    ):
        observed_terminal = (
            policy["event_within_horizon"].to_numpy(bool)
            & (policy["event_time"].to_numpy(float) <= horizon)
        ).astype(int)
        survival_rows.append(
            {
                "protocol_id": PROTOCOL,
                "model": "B8_discrete_terminal",
                "metric": f"AUC@{horizon}",
                "value": float(roc_auc_score(observed_terminal, positive_probability)),
            }
        )
        survival_rows.append(
            {
                "protocol_id": PROTOCOL,
                "model": "B8_discrete_terminal",
                "metric": f"Brier@{horizon}",
                "value": float(brier_score_loss(observed_terminal, positive_probability)),
            }
        )
    aft_prediction = aft_model.predict(xgb.DMatrix(x_policy))
    survival_rows.append(
        {
            "protocol_id": PROTOCOL,
            "model": "B7_xgboost_aft",
            "metric": "Harrell_concordance_descriptive",
            "value": float(
                concordance_index(
                    policy["event_time"], aft_prediction, policy["event_within_horizon"]
                )
            ),
        }
    )
    if cox_status == "COMPLETE":
        cox_policy = pd.DataFrame(x_policy)[cox_columns]
        cox_risk = cox.predict_partial_hazard(cox_policy).to_numpy()
        survival_rows.append(
            {
                "protocol_id": PROTOCOL,
                "model": "B5_cox_ph",
                "metric": "Harrell_concordance_descriptive",
                "value": float(
                    concordance_index(
                        policy["event_time"], -cox_risk, policy["event_within_horizon"]
                    )
                ),
            }
        )
    survival_rows.append(
        {
            "protocol_id": PROTOCOL,
            "model": "B6_random_survival_forest",
            "metric": "status",
            "value": np.nan,
            "status": "NOT_RUN_NO_RELIABLE_INSTALLED_IMPLEMENTATION",
        }
    )

    bootstrap = _account_bootstrap(
        policy,
        policy_predictions["AccountPulse_EV"].to_numpy(float),
        policy_predictions[strongest_baseline].to_numpy(float),
        5000,
        SEED,
    )
    write_table("maven_ranking_metrics_development", ranking)
    write_table("maven_calibration_metrics_development", pd.DataFrame(calibration_rows))
    write_table("maven_apev_search", candidates)
    write_table("maven_leakage_diagnostics", leakage_frame)
    write_table("maven_survival_metrics_development", pd.DataFrame(survival_rows))
    write_table(
        "maven_predictions_development",
        pd.concat(
            [
                validation_predictions.assign(split="VALIDATION").merge(
                    validation[
                        ["opportunity_id", "won", "lost", "realized_value", "terminal_state"]
                    ],
                    on="opportunity_id",
                ),
                policy_predictions.assign(split="POLICY").merge(
                    policy[["opportunity_id", "won", "lost", "realized_value", "terminal_state"]],
                    on="opportunity_id",
                ),
            ],
            ignore_index=True,
        ),
    )
    (EVIDENCE / "maven_policy_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2) + "\n", encoding="utf-8"
    )
    joblib.dump(provisional, BUNDLE_PATH)
    bundle_hash = hashlib.sha256(BUNDLE_PATH.read_bytes()).hexdigest()

    result = {
        "protocol_id": PROTOCOL,
        "selected_classifier_from_validation": selected_classifier,
        "selected_fixed_horizon_calibration_from_policy": selected_fixed_calibration,
        "selected_apev_config_from_policy": selected_config,
        "strongest_policy_baseline": strongest_baseline,
        "policy_apev_value_capture_10": float(
            at_ten.loc[at_ten["model"] == "AccountPulse_EV", "value_capture"].iloc[0]
        ),
        "policy_baseline_value_capture_10": float(
            at_ten.loc[at_ten["model"] == strongest_baseline, "value_capture"].iloc[0]
        ),
        "policy_bootstrap": bootstrap,
        "bundle_path": str(BUNDLE_PATH.relative_to(REPO)),
        "bundle_sha256": bundle_hash,
        "gpu_training_seconds": gpu_seconds,
        "total_seconds": time.perf_counter() - started,
        "cox_status": cox_status,
        "rsf_status": "NOT_RUN_NO_RELIABLE_INSTALLED_IMPLEMENTATION",
        "locked_outcomes_accessed": False,
    }
    (EVIDENCE / "maven_development_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    mlflow.set_tracking_uri(f"sqlite:///{EVIDENCE / 'mlflow.db'}")
    mlflow.set_experiment("accountpulse-track-a-development")
    with mlflow.start_run(run_name=PROTOCOL):
        mlflow.log_params(
            {
                "protocol_id": PROTOCOL,
                "persona": "PREASSIGN",
                "horizon_days": 60,
                "selected_classifier": selected_classifier,
                "strongest_policy_baseline": strongest_baseline,
                **{f"apev_{key}": value for key, value in selected_config.items()},
            }
        )
        mlflow.log_metrics(
            {
                "policy_apev_value_capture_10": result["policy_apev_value_capture_10"],
                "policy_baseline_value_capture_10": result["policy_baseline_value_capture_10"],
                "gpu_training_seconds": gpu_seconds,
            }
        )
        mlflow.log_artifact(str(EVIDENCE / "maven_development_result.json"))
    return result
