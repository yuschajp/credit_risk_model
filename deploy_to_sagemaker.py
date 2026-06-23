import argparse, json, os, tarfile, tempfile
import boto3, joblib, numpy as np, pandas as pd, sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ENDPOINT_NAME   = "credit-risk-model-endpoint"
MODEL_NAME      = "credit-risk-gbm"
BUCKET_PREFIX   = "credit-risk-model"
SKLEARN_VERSION = "1.2-1"
INSTANCE_TYPE   = "ml.m5.large"

session    = sagemaker.Session()
role       = "arn:aws:iam::543704476633:role/SageMakerExecutionRole"
bucket     = session.default_bucket()
s3_client  = boto3.client("s3", region_name=session.boto_region_name)
sm_client  = boto3.client("sagemaker", region_name=session.boto_region_name)
sm_runtime = boto3.client("sagemaker-runtime", region_name=session.boto_region_name)

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

def engineer_features(df):
    df = df.copy()
    df["log_annual_inc"] = np.log1p(df["annual_inc"])
    df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)
    df = pd.get_dummies(df, columns=["home_ownership","verification_status","purpose","initial_list_status","application_type"])
    for col in FEATURE_COLUMNS:
        if col not in df.columns: df[col] = 0
    return df[FEATURE_COLUMNS]

def build_and_save_model(model_dir):
    print("Generating training data...")
    np.random.seed(42)
    n = 5000
    raw = pd.DataFrame({
        "loan_amnt": np.random.uniform(1000,40000,n),
        "int_rate": np.random.uniform(5,30,n),
        "installment": np.random.uniform(50,1500,n),
        "annual_inc": np.random.lognormal(10.8,0.6,n),
        "dti": np.random.uniform(0,40,n),
        "delinq_2yrs": np.random.poisson(0.3,n),
        "fico_range_low": np.random.randint(580,850,n),
        "open_acc": np.random.randint(1,30,n),
        "pub_rec": np.random.poisson(0.1,n),
        "revol_bal": np.random.uniform(0,50000,n),
        "revol_util": np.random.uniform(0,100,n),
        "total_acc": np.random.randint(2,60,n),
        "mort_acc": np.random.poisson(1,n),
        "pub_rec_bankruptcies": np.random.poisson(0.05,n),
        "home_ownership": np.random.choice(["MORTGAGE","OWN","RENT"],n),
        "verification_status": np.random.choice(["Not Verified","Source Verified","Verified"],n),
        "purpose": np.random.choice(["credit_card","debt_consolidation","home_improvement","major_purchase","other"],n),
        "initial_list_status": np.random.choice(["w","f"],n),
        "application_type": np.random.choice(["Individual","Joint App"],n,p=[0.9,0.1]),
    })
    logit = -6 + 0.08*raw["int_rate"] + 0.03*raw["dti"] - 0.015*(raw["fico_range_low"]-650) + 0.5*raw["delinq_2yrs"] + 0.3*raw["pub_rec"]
    raw["loan_status"] = (np.random.rand(n) < 1/(1+np.exp(-logit))).astype(int)
    X = engineer_features(raw)
    y = raw["loan_status"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    print("Training model...")
    model = Pipeline([("scaler",StandardScaler()),("gbm",GradientBoostingClassifier(n_estimators=200,max_depth=4,learning_rate=0.05,subsample=0.8,random_state=42))])
    model.fit(X_train,y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    print(f"AUC: {auc:.4f}")
    joblib.dump(model, os.path.join(model_dir,"model.joblib"))
    print("Model saved.")

def package_and_upload(model_dir):
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = os.path.join(tmp,"model.tar.gz")
        with tarfile.open(tar_path,"w:gz") as tar:
            tar.add(os.path.join(model_dir,"model.joblib"), arcname="model.joblib")
            tar.add("code/inference.py", arcname="code/inference.py")
        s3_key = f"{BUCKET_PREFIX}/model.tar.gz"
        s3_client.upload_file(tar_path, bucket, s3_key)
        uri = f"s3://{bucket}/{s3_key}"
        print(f"Uploaded → {uri}")
        return uri

def deploy_endpoint(s3_uri):
    print(f"Deploying endpoint {ENDPOINT_NAME}...")
    SKLearnModel(model_data=s3_uri, role=role, entry_point="inference.py",
                 source_dir="code", framework_version=SKLEARN_VERSION,
                 name=MODEL_NAME).deploy(
        initial_instance_count=1, instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME)
    print(f"✅ Endpoint live: {ENDPOINT_NAME}")

def test_endpoint():
    sample = {"loan_amnt":15000,"int_rate":18.5,"installment":416,"annual_inc":75000,
               "dti":22.0,"delinq_2yrs":0,"fico_range_low":680,"open_acc":8,"pub_rec":0,
               "revol_bal":12000,"revol_util":55.0,"total_acc":18,"mort_acc":1,
               "pub_rec_bankruptcies":0,"log_annual_inc":11.22,"loan_to_income":0.2,
               "home_ownership_MORTGAGE":1,"home_ownership_OWN":0,"home_ownership_RENT":0,
               "verification_status_Source Verified":0,"verification_status_Verified":1,
               "purpose_credit_card":0,"purpose_debt_consolidation":1,
               "purpose_home_improvement":0,"purpose_major_purchase":0,"purpose_other":0,
               "initial_list_status_w":1,"application_type_Joint App":0}
    resp = sm_runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                      ContentType="application/json",
                                      Accept="application/json",
                                      Body=json.dumps(sample))
    result = json.loads(resp["Body"].read())
    print(f"\nVerdict: {'DEFAULT' if result['predictions'][0]==1 else 'NO DEFAULT'}")
    print(f"P(default): {result['default_probability'][0]:.4f}\n")

def delete_endpoint():
    sm_client.delete_endpoint(EndpointName=ENDPOINT_NAME)
    print("Endpoint deleted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["deploy","test","delete"], default="deploy")
    args = parser.parse_args()
    if args.mode == "deploy":
        with tempfile.TemporaryDirectory() as model_dir:
            build_and_save_model(model_dir)
            s3_uri = package_and_upload(model_dir)
        deploy_endpoint(s3_uri)
        test_endpoint()
    elif args.mode == "test":
        test_endpoint()
    elif args.mode == "delete":
        delete_endpoint()
