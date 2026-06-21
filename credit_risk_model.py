"""
Probability-of-default model on a synthetic consumer loan portfolio:
generates loan-level data with realistic, economically sensible
relationships between borrower characteristics and default risk, trains
a logistic regression baseline alongside a gradient boosting model, and
evaluates both the way a credit risk team actually would, not just by
which one has the higher accuracy number.

Three things specifically distinguish this from a model that just
produces predictions:
  - a held-out test set, never touched during training
  - calibration, not just rank-ordering: if the model says 10% PD, do
    roughly 10% of those loans actually default in the test set
  - a sign check on every feature: does the model's learned relationship
    match the economic relationship the data was generated with, e.g.
    does a higher debt-to-income ratio actually increase predicted risk

This is the one project in the broader portfolio built specifically to
demonstrate classical ML evaluation, as a counterpart to the
deterministic pricing engines and the AI governance work elsewhere.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "credit_score", "dti", "ltv", "loan_amount", "interest_rate",
    "employment_years", "delinquencies_2yr",
]

# Sign each feature should have on default risk once standardized --
# used later to check the model learned the right direction, not just a
# number that happens to score well.
EXPECTED_SIGN = {
    "credit_score": -1,        # higher score, lower risk
    "dti": +1,                 # higher debt-to-income, higher risk
    "ltv": +1,                 # higher loan-to-value, higher risk
    "loan_amount": 0,          # no strong prior either way
    "interest_rate": +1,       # priced-in risk should still show up
    "employment_years": -1,    # longer tenure, lower risk
    "delinquencies_2yr": +1,   # prior delinquencies, higher risk
}


def generate_loan_portfolio(n: int = 5000, seed: int = 23) -> pd.DataFrame:
    """Synthetic loan-level data with a deliberately realistic, known
    data-generating process: default risk is a logistic function of
    standardized borrower features, so the 'true' relationship is known
    and can be checked against what the model actually learns."""
    rng = np.random.default_rng(seed)

    credit_score = np.clip(rng.normal(680, 60, n), 300, 850)
    dti = np.clip(rng.normal(0.35, 0.12, n), 0.02, 0.95)
    ltv = np.clip(rng.normal(0.75, 0.15, n), 0.10, 1.10)
    loan_amount = np.clip(rng.lognormal(mean=np.log(22_000), sigma=0.5, size=n), 1_000, 150_000)
    employment_years = np.clip(rng.exponential(5.0, n), 0, 35)
    delinquencies_2yr = rng.poisson(0.35, n)

    # Rate is priced partly off credit score (inverse) plus noise, the
    # way a real underwriting model would set it.
    interest_rate = np.clip(
        0.18 - 0.00012 * credit_score + rng.normal(0, 0.015, n), 0.04, 0.30
    )

    def standardize(x):
        return (x - x.mean()) / x.std()

    logit = (
        -6.0
        + 1.7 * standardize(dti)
        + 1.1 * standardize(ltv)
        - 1.5 * standardize(credit_score)
        + 0.7 * standardize(delinquencies_2yr)
        - 0.5 * standardize(employment_years)
        + 0.5 * standardize(interest_rate)
        + rng.normal(0, 3.5, n)  # unexplained idiosyncratic risk
    )
    default_prob = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, default_prob)

    return pd.DataFrame({
        "credit_score": credit_score,
        "dti": dti,
        "ltv": ltv,
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "employment_years": employment_years,
        "delinquencies_2yr": delinquencies_2yr,
        "default": default,
    })


def split_and_scale(df: pd.DataFrame, test_size: float = 0.3, seed: int = 23):
    X = df[FEATURE_NAMES]
    y = df["default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=FEATURE_NAMES, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_NAMES, index=X_test.index)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_models(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 23) -> dict:
    logistic = LogisticRegression(max_iter=1000, random_state=seed)
    logistic.fit(X_train, y_train)

    gbm = GradientBoostingClassifier(random_state=seed, n_estimators=150, max_depth=3, learning_rate=0.05)
    gbm.fit(X_train, y_train)

    return {"logistic": logistic, "gbm": gbm}


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> dict:
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    return {
        "auc": roc_auc_score(y_test, probs),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "probs": probs,
    }


def calibration_table(y_test: pd.Series, probs: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Predicted PD versus actual default rate by bucket -- the check
    that matters most for a model whose output is meant to be used as a
    probability, not just a ranking."""
    actual_freq, predicted_value = calibration_curve(y_test, probs, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({
        "predicted_pd": predicted_value,
        "actual_default_rate": actual_freq,
    })


def feature_signs(model, model_name: str, X_test: pd.DataFrame = None) -> pd.DataFrame:
    """Direction the model actually learned for each feature, checked
    against the economically expected direction it was generated with.

    For logistic regression the coefficient sign is the direct answer.
    Gradient boosting's feature_importances_ are unsigned magnitudes, so
    its effective direction is read off the correlation between each
    feature and the model's own predicted probabilities on the test
    set, which is a fair, real answer rather than an unscored 'n/a'.
    """
    if model_name == "logistic":
        importances = model.coef_[0]
        learned_signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in importances]
    else:
        importances = model.feature_importances_
        predicted_probs = model.predict_proba(X_test)[:, 1]
        learned_signs = [
            int(np.sign(np.corrcoef(X_test[name], predicted_probs)[0, 1]))
            for name in FEATURE_NAMES
        ]

    rows = []
    for name, value, learned_sign in zip(FEATURE_NAMES, importances, learned_signs):
        expected = EXPECTED_SIGN[name]
        matches = "yes" if (learned_sign == expected or expected == 0) else "no"
        rows.append({"feature": name, "importance_or_coef": value, "matches_expected_direction": matches})
    return pd.DataFrame(rows).sort_values("importance_or_coef", key=abs, ascending=False)


def run_full_evaluation(n: int = 5000, seed: int = 23) -> dict:
    df = generate_loan_portfolio(n=n, seed=seed)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df, seed=seed)
    models = train_models(X_train, y_train, seed=seed)

    results = {}
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        calibration = calibration_table(y_test, metrics["probs"])
        signs = feature_signs(model, name, X_test=X_test)
        results[name] = {"metrics": metrics, "calibration": calibration, "signs": signs}

    return {
        "df": df,
        "default_rate": float(df["default"].mean()),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "results": results,
    }
