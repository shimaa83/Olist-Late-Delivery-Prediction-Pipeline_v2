import logging

import yaml

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient

import src.task3.config as config
from .models import (
    build_logistic_regression,
    build_random_forest,
    evaluate_model,
    load_artifacts,
    save_model,
    tune_xgboost,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("mlflow_sqlite_train")

MODEL_REGISTRY_NAME = "Olist_Delay_Model"
DVC_FILE = "../data/olist.db.dvc"  # عدّل المسار لو الملف .dvc في مجلد تاني


def get_dvc_hash(dvc_file: str) -> str:
    """يقرأ md5 hash وباقي معلومات الملف المتتبع من ملف .dvc"""
    try:
        with open(dvc_file, "r") as f:
            data = yaml.safe_load(f)
        out = data["outs"][0]
        return out.get("md5") or out.get("hash", "unknown")
    except FileNotFoundError:
        logger.warning(f"ملف DVC غير موجود: {dvc_file} — هيتسجل tag بقيمة 'unknown'")
        return "unknown"
    except (KeyError, IndexError):
        logger.warning(f"تعذّر قراءة الـ hash من {dvc_file} — تأكد من صيغة الملف")
        return "unknown"


def log_dvc_info(dvc_file: str, data_hash: str):
    """يسجل الـ DVC hash كـ tag ويرفع ملف .dvc نفسه كـ artifact في الـ run الحالي"""
    mlflow.set_tag("dvc_data_hash", data_hash)
    mlflow.set_tag("dvc_file", dvc_file)
    try:
        mlflow.log_artifact(dvc_file)
    except FileNotFoundError:
        pass


def run_training_pipeline():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Olist_Order_Delay_Prediction")

    data_hash = get_dvc_hash(DVC_FILE)
    logger.info(f"DVC data hash (olist.db): {data_hash}")

    logger.info("تحميل مصفوفات البيانات...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_artifacts()

    # 1. Logistic Regression
    with mlflow.start_run(run_name="Logistic_Regression"):
        log_dvc_info(DVC_FILE, data_hash)

        lr_model = build_logistic_regression()
        lr_model.fit(X_train, y_train)
        metrics = evaluate_model("Logistic Regression", lr_model, X_val, y_val)

        mlflow.log_params(lr_model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(lr_model, artifact_path="model")

    # 2. Random Forest
    with mlflow.start_run(run_name="Random_Forest"):
        log_dvc_info(DVC_FILE, data_hash)

        rf_model = build_random_forest()
        rf_model.fit(X_train, y_train)
        metrics = evaluate_model("Random Forest", rf_model, X_val, y_val)

        mlflow.log_params(rf_model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(rf_model, artifact_path="model")

    # 3. XGBoost Tuned & Registry
    with mlflow.start_run(run_name="XGBoost_Tuned_Best"):
        log_dvc_info(DVC_FILE, data_hash)

        best_xgb, best_params = tune_xgboost(X_train, y_train)

        val_metrics = evaluate_model("XGBoost Tuned (Val)", best_xgb, X_val, y_val)
        test_metrics = evaluate_model("XGBoost Tuned (Test)", best_xgb, X_test, y_test)
        test_metrics_logged = {f"test_{k}": v for k, v in test_metrics.items()}

        mlflow.log_params(best_params)
        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(test_metrics_logged)

        logger.info(
            f"تسجيل النموذج في MLflow Model Registry باسم: {MODEL_REGISTRY_NAME}"
        )
        mlflow.xgboost.log_model(
            xgb_model=best_xgb,
            artifact_path="model",
            registered_model_name=MODEL_REGISTRY_NAME,
        )

        save_model(best_xgb, filepath=config.BEST_MODEL_PATH)

    # 4. Transition to Production
    client = MlflowClient()
    latest_versions = client.get_latest_versions(MODEL_REGISTRY_NAME, stages=["None"])

    if latest_versions:
        latest_version = latest_versions[-1].version
        logger.info(f"ترقية النموذج Version {latest_version} إلى حالة Production...")

        client.transition_model_version_stage(
            name=MODEL_REGISTRY_NAME,
            version=latest_version,
            stage="Production",
            archive_existing_versions=True,
        )


if __name__ == "__main__":
    run_training_pipeline()
