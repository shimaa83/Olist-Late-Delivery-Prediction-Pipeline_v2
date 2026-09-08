import joblib
import src.task3.config
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier


def load_artifacts():
    """تحميل مصفوفات البيانات والمتغيرات من المسارات المحددة في config."""
    X_train = joblib.load(src.task3.config.X_TRAIN_PATH)
    X_val = joblib.load(src.task3.config.X_VAL_PATH)
    X_test = joblib.load(src.task3.config.X_TEST_PATH)

    y_train = joblib.load(src.task3.config.Y_TRAIN_PATH)
    y_val = joblib.load(src.task3.config.Y_VAL_PATH)
    y_test = joblib.load(src.task3.config.Y_TEST_PATH)

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(model_name: str, model, X_val, y_val):
    """حساب وتقييم مقاييس الأداء."""
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred, zero_division=0)
    recall = recall_score(y_val, y_val_pred, zero_division=0)
    f1 = f1_score(y_val, y_val_pred, zero_division=0)
    roc_auc = roc_auc_score(y_val, y_val_prob)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def build_logistic_regression(random_state: int = src.task3.config.RANDOM_STATE):
    return LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=random_state
    )


def build_random_forest(
    n_estimators: int = 300,
    random_state: int = src.task3.config.RANDOM_STATE,
    n_jobs: int = src.task3.config.N_JOBS,
):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=n_jobs,
    )


def calculate_scale_pos_weight(y_train) -> float:
    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()
    return negative / positive if positive > 0 else 1.0


def tune_xgboost(
    X_train,
    y_train,
    n_iter: int = 5,
    cv: int = 3,
    random_state: int = src.task3.config.RANDOM_STATE,
    n_jobs: int = src.task3.config.N_JOBS,
):
    scale_pos_weight = calculate_scale_pos_weight(y_train)

    base_xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        enable_categorical=True,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=n_jobs,
    )

    random_search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=src.task3.config.XGB_PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="f1",
        cv=cv,
        verbose=2,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    random_search.fit(X_train, y_train)
    return random_search.best_estimator_, random_search.best_params_


def save_model(model, filepath=src.task3.config.BEST_MODEL_PATH):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
