import os, json, joblib, numpy as np, pandas as pd

FEATURE_COLUMNS = [
    "loan_amnt","int_rate","installment","annual_inc","dti","delinq_2yrs",
    "fico_range_low","open_acc","pub_rec","revol_bal","revol_util","total_acc",
    "mort_acc","pub_rec_bankruptcies","log_annual_inc","loan_to_income",
    "home_ownership_MORTGAGE","home_ownership_OWN","home_ownership_RENT",
    "verification_status_Source Verified","verification_status_Verified",
    "purpose_credit_card","purpose_debt_consolidation","purpose_home_improvement",
    "purpose_major_purchase","purpose_other","initial_list_status_w",
    "application_type_Joint App",
]

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))

def input_fn(request_body, request_content_type):
    data = json.loads(request_body)
    if isinstance(data, dict): data = [data]
    df = pd.DataFrame(data)
    for col in FEATURE_COLUMNS:
        if col not in df.columns: df[col] = 0
    return df[FEATURE_COLUMNS]

def predict_fn(input_data, model):
    proba = model.predict_proba(input_data)[:, 1]
    return {"predictions": (proba >= 0.5).astype(int).tolist(), "default_probability": proba.tolist()}

def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
