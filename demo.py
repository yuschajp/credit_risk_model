"""
Trains a logistic regression baseline and a gradient boosting model on
the same synthetic 5,000-loan portfolio, evaluates both against a held-
out test set, and reports the things a credit risk team would actually
want to see before trusting either one: AUC, precision and recall at a
default threshold, calibration by decile, and whether each model's
learned relationships point the same direction as the underlying
economics they were generated from.

Same evaluation habit as the rest of this portfolio: show what the
model gets right, what it misses, and where the obvious default
(a 0.5 classification threshold) is itself a choice worth questioning
rather than something to use just because it's the library default.
"""

from credit_risk_model import run_full_evaluation


def print_model_report(name: str, result: dict):
    m = result["metrics"]
    print(f"\n{'=' * 60}")
    print(f"{name.upper()}")
    print(f"{'=' * 60}")
    print(f"AUC:       {m['auc']:.3f}")
    print(f"Precision: {m['precision']:.1%}  (at 0.5 threshold)")
    print(f"Recall:    {m['recall']:.1%}  (at 0.5 threshold)")
    print(f"F1:        {m['f1']:.2f}")
    print(f"Confusion matrix -- TP: {m['true_positives']}  FP: {m['false_positives']}  "
          f"TN: {m['true_negatives']}  FN: {m['false_negatives']}")

    print("\nCalibration (predicted PD vs. actual default rate, by decile):")
    for _, row in result["calibration"].iterrows():
        print(f"  predicted {row['predicted_pd']:.1%}  ->  actual {row['actual_default_rate']:.1%}")

    print("\nFeature direction check (does the model agree with the underlying economics?):")
    for _, row in result["signs"].iterrows():
        print(f"  {row['feature']:<18} {row['importance_or_coef']:+.3f}   matches expected: {row['matches_expected_direction']}")


def main():
    result = run_full_evaluation(n=5000, seed=23)

    print("=" * 60)
    print("CREDIT RISK MODEL EVALUATION")
    print("(synthetic, seeded data -- demonstrates the evaluation")
    print(" methodology, not a real loan portfolio)")
    print("=" * 60)
    print(f"\nPortfolio size: {len(result['df'])} loans")
    print(f"Overall default rate: {result['default_rate']:.1%}")
    print(f"Train / test split: {result['n_train']} / {result['n_test']}")

    for name, r in result["results"].items():
        print_model_report(name, r)

    print("\n" + "=" * 60)
    print("PRODUCT TAKEAWAY")
    print("=" * 60)
    print(
        "Both models learn the right direction on every feature, debt-to-\n"
        "income and loan-to-value push risk up, credit score and tenure\n"
        "pull it down, which matters more than either model's AUC: a model\n"
        "that scores well but learned a backwards relationship on even one\n"
        "feature isn't one you'd want underwriting real loans, regardless\n"
        "of its headline metric.\n\n"
        "Recall is low for both models at the default 0.5 threshold, not\n"
        "because the models are weak, but because 0.5 isn't the right\n"
        "operating point for an 11% base rate. Where you actually set that\n"
        "threshold is a business decision about the cost of a false\n"
        "positive (declining a borrower who would have repaid) against a\n"
        "false negative (approving one who won't), not a default to leave\n"
        "untouched because it's the library's."
    )


if __name__ == "__main__":
    main()
