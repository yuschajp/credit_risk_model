"""
Generates a static HTML dashboard from the credit risk model: portfolio
summary, a side-by-side comparison of the logistic regression and
gradient boosting models, calibration by decile for each, and a check
of whether each model's learned feature directions agree with the
underlying economics, all on one page you open directly in a browser,
no server required.

Run with: python3 dashboard.py
Then open dashboard.html in any browser.
"""

import html

from credit_risk_model import run_full_evaluation


def render_table(headers, rows):
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = ""
    for row in rows:
        cells = "".join(f"<td>{html.escape('' if v is None else str(v))}</td>" for v in row)
        body += f"<tr>{cells}</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_data():
    result = run_full_evaluation(n=5000, seed=23)

    comparison_rows = []
    calibration_tables = {}
    sign_tables = {}
    for name, r in result["results"].items():
        m = r["metrics"]
        label = "Logistic regression" if name == "logistic" else "Gradient boosting"
        comparison_rows.append((
            label, f"{m['auc']:.3f}", f"{m['precision']:.1%}", f"{m['recall']:.1%}", f"{m['f1']:.2f}",
        ))
        calibration_tables[label] = [
            (f"{row['predicted_pd']:.1%}", f"{row['actual_default_rate']:.1%}")
            for _, row in r["calibration"].iterrows()
        ]
        sign_tables[label] = [
            (row["feature"], f"{row['importance_or_coef']:+.3f}", row["matches_expected_direction"])
            for _, row in r["signs"].iterrows()
        ]

    return {
        "n_loans": len(result["df"]),
        "default_rate": result["default_rate"],
        "n_train": result["n_train"],
        "n_test": result["n_test"],
        "comparison_rows": comparison_rows,
        "calibration_tables": calibration_tables,
        "sign_tables": sign_tables,
    }


def build_html(data):
    calibration_cards = ""
    for label, rows in data["calibration_tables"].items():
        calibration_cards += f'<div style="margin-top: 16px;"><strong style="font-size: 13px; color: #1B3A5C;">{html.escape(label)}</strong></div>'
        calibration_cards += render_table(["Predicted PD", "Actual default rate"], rows)

    sign_cards = ""
    for label, rows in data["sign_tables"].items():
        sign_cards += f'<div style="margin-top: 16px;"><strong style="font-size: 13px; color: #1B3A5C;">{html.escape(label)}</strong></div>'
        sign_cards += render_table(["Feature", "Importance / coefficient", "Matches expected direction"], rows)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Credit Risk Model Dashboard</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; background: #f5f6f8; color: #1a1a1a; margin: 0; padding: 32px; }}
  h1 {{ color: #1B3A5C; margin-bottom: 4px; }}
  .subtitle {{ color: #5a5a5a; margin-bottom: 28px; }}
  .source-link {{ display: inline-block; margin-bottom: 20px; font-size: 13px; }}
  .source-link a {{ color: #1B3A5C; text-decoration: none; font-weight: 600; }}
  .source-link a:hover {{ text-decoration: underline; }}
  .card {{ background: #ffffff; border-radius: 8px; padding: 20px 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .card h2 {{ color: #1B3A5C; font-size: 16px; margin-top: 0; border-bottom: 1px solid #e2e2e2; padding-bottom: 8px; }}
  .figure-row {{ }}
  .figure-block {{ display: inline-block; margin-right: 48px; vertical-align: top; }}
  .figure {{ font-size: 24px; font-weight: 600; color: #1B3A5C; }}
  .figure-label {{ font-size: 12px; color: #5a5a5a; margin-bottom: 4px; }}
  .note {{ font-size: 13px; color: #5a5a5a; margin-bottom: 14px; }}
  .takeaway {{ font-size: 13px; line-height: 1.6; color: #2a2a2a; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }}
  th {{ text-align: left; background: #1B3A5C; color: #fff; padding: 8px 10px; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #ececec; }}
  tr:nth-child(even) td {{ background: #fafafa; }}
</style>
</head>
<body>
  <h1>Credit Risk Model</h1>
  <div class="subtitle">Probability of default: logistic regression vs. gradient boosting, evaluated side by side</div>
  <div class="source-link"><a href="https://github.com/yuschajp/credit_risk_model">View source on GitHub &rarr;</a></div>

  <div class="card">
    <h2>Portfolio summary</h2>
    <div class="note">Synthetic, seeded data &mdash; demonstrates the evaluation methodology, not a real loan portfolio.</div>
    <div class="figure-row">
      <div class="figure-block">
        <div class="figure-label">Loans in portfolio</div>
        <div class="figure">{data['n_loans']:,}</div>
      </div>
      <div class="figure-block">
        <div class="figure-label">Overall default rate</div>
        <div class="figure">{data['default_rate']:.1%}</div>
      </div>
      <div class="figure-block">
        <div class="figure-label">Train / test split</div>
        <div class="figure">{data['n_train']:,} / {data['n_test']:,}</div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Model comparison</h2>
    {render_table(["Model", "AUC", "Precision", "Recall", "F1"], data["comparison_rows"])}
    <div class="note" style="margin-top: 10px;">Precision and recall are both measured at a 0.5 classification threshold, which is a starting point for comparison, not the right operating point for an 11% base rate in production.</div>
  </div>

  <div class="card">
    <h2>Calibration: predicted PD vs. actual default rate, by decile</h2>
    {calibration_cards}
  </div>

  <div class="card">
    <h2>Feature direction check</h2>
    <div class="note">Does each model's learned relationship agree with the economic direction the data was generated with?</div>
    {sign_cards}
    <div style="margin-top: 20px;"><strong style="font-size: 13px; color: #1B3A5C;">Product takeaway</strong></div>
    <div class="takeaway">Both models learn the right direction on every feature, which matters more than either model's AUC: a model that scores well but learned a backwards relationship on even one feature isn't one you'd want underwriting real loans. Recall is low for both at the default threshold not because the models are weak, but because 0.5 isn't the right operating point for an 11% base rate &mdash; where that threshold actually gets set is a business decision about the cost of a false positive against a false negative, not something to leave at the library's default.</div>
  </div>
</body>
</html>"""


def main():
    data = build_data()
    output = build_html(data)
    with open("dashboard.html", "w") as f:
        f.write(output)
    print("Wrote dashboard.html -- open it in your browser to view it.")


if __name__ == "__main__":
    main()
