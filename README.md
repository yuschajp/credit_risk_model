# Credit Risk Model

A probability-of-default model on a synthetic consumer loan portfolio: a logistic regression baseline evaluated side by side with a gradient boosting model, on a held-out test set, with calibration and feature-direction checks rather than a single accuracy number presented as the whole story.

## Why this exists

The three other engines in this broader portfolio (a ledger and reconciliation engine, a rates and derivatives engine, a collateral and margin engine) are deterministic financial math and rule-based logic, not statistical models trained on data. This project exists specifically to demonstrate that other half: real machine learning, with a genuine train/test split, real evaluation metrics, and an honest comparison between an interpretable baseline and a more complex model, rather than reaching for whichever one sounds more impressive.

## Architecture

1. Synthetic loan portfolio generator, with a known data-generating process so the "true" relationship between features and default risk is known in advance — *built*
2. Train/test split with stratification on the rare class, and feature scaling fit only on the training set — *built*
3. Two models trained on the same data: logistic regression and gradient boosting — *built*
4. Evaluation layer: AUC, precision/recall/F1 at a stated threshold, a confusion matrix, calibration by decile, and a feature-direction check against the known data-generating process — *built*
5. Reports & dashboard summarizing all of the above — *built*

## Design decisions

The data-generating process is deliberately known rather than left as a black box: default risk is a logistic function of standardized borrower features (debt-to-income, loan-to-value, credit score, delinquency history, employment tenure, interest rate, loan amount) plus unexplained idiosyncratic noise. Knowing the true relationship in advance is what makes the feature-direction check meaningful later: a model can be checked against the actual economics it was generated from, not just against its own accuracy score.

The default rate (just under 11%) and the resulting AUC (around 0.80 for both models) were both deliberately tuned to be realistic rather than impressive. An earlier version of this same generator produced a 27% default rate and a 0.91 AUC, both well outside what a real consumer credit model would show; a portfolio piece claiming numbers like that would read as either overfit or unrealistic to anyone who has actually evaluated a credit model before, which would hurt credibility rather than help it.

Both models are trained on identically scaled features and evaluated against the same held-out test set, never touched during training. Logistic regression coefficients give a direct, signed read on each feature's learned direction. Gradient boosting's `feature_importances_` are unsigned magnitudes, so that model's effective direction is read off the correlation between each feature and its own predicted probabilities on the test set instead, a real, computed answer rather than an unscored gap in the comparison.

Calibration is checked explicitly because it answers a different question than AUC does. AUC measures whether the model ranks riskier loans higher than safer ones; calibration measures whether a loan the model scores at 10% predicted default risk actually defaults close to 10% of the time. A model can rank well and still be miscalibrated, and a probability of default that doesn't mean what it says isn't usable for pricing or reserving even if it discriminates well.

## Product framing

The user here is a credit risk or underwriting team deciding whether, and how, to trust a model's output, not a data scientist optimizing a leaderboard metric. The success metric isn't AUC in isolation, it's whether the predicted probability is trustworthy enough to price or reserve against (calibration) and whether the model's reasoning matches reality well enough to defend to a model risk or compliance reviewer (the feature-direction check). The guardrail is the held-out test set and the explicit threshold discussion: precision and recall are reported at 0.5 specifically so that number can be challenged, not treated as a finding, since the right operating threshold is a business decision about the relative cost of declining a good borrower versus approving a bad one, not a default left untouched because it's the library's.

## Getting started

Requires numpy, pandas, and scikit-learn, since this project is specifically about real statistical modeling rather than dependency-free deterministic math.

```bash
pip install numpy pandas scikit-learn
```

```bash
git clone <your-repo-url>
cd credit-risk-model
python3 demo.py
python3 dashboard.py    # then open dashboard.html in a browser
```

Expected output from `demo.py` (abbreviated):

```
Portfolio size: 5000 loans
Overall default rate: 10.8%
Train / test split: 3500 / 1500

LOGISTIC
AUC:       0.801
Precision: 63.2%  (at 0.5 threshold)
Recall:    14.8%  (at 0.5 threshold)

GBM
AUC:       0.790
Precision: 60.7%  (at 0.5 threshold)
Recall:    10.5%  (at 0.5 threshold)
```

## Project structure

```
credit_risk_model.py    # data generation, train/test split, model training, evaluation
dashboard.py              # generates a static HTML report from a full evaluation run
demo.py                      # end-to-end evaluation example, printed to the console
README.md
```

## Status

All five planned phases are built: data generation with a known ground-truth process, train/test split, both models, the full evaluation layer, and the dashboard. A natural next step would be adding a cost-weighted threshold selection, picking the operating point that minimizes expected loss given stated costs for a false positive versus a false negative, rather than stopping at reporting precision and recall at 0.5.
