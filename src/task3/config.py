from pathlib import Path

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
# ==========================================
# 1. المسارات الأساسية (Directories)
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = BASE_DIR / "logs"

# ==========================================
# 2. مسارات البيانات الخام والمعالجة (Data Paths)
# ==========================================
DB_PATH = DATA_DIR / "olist.db"  # أو olist.sqlite
MASTER_CSV_PATH = DATA_DIR / "olist_final_table.csv"
LABELED_DATA_PATH = DATA_DIR / "olist_labels_table.csv"

# مسارات البيانات المقسمة (Data Splits)
TRAIN_DATA_PATH = DATA_DIR / "train_df.csv"
VAL_DATA_PATH = DATA_DIR / "validation_df.csv"
TEST_DATA_PATH = DATA_DIR / "test_df.csv"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_train_df.csv"

# ==========================================
# 3. مسارات نواتج المعالجة (Artifacts Paths)
# ==========================================
CAT_IMPUTER_PATH = ARTIFACTS_DIR / "cat_imputer.joblib"
CAT_ENCODER_PATH = ARTIFACTS_DIR / "cat_encoder.joblib"
NUM_IMPUTER_PATH = ARTIFACTS_DIR / "num_imputer.joblib"
NUM_SCALER_PATH = ARTIFACTS_DIR / "num_scaler.joblib"

X_TRAIN_PATH = ARTIFACTS_DIR / "X_train_final.joblib"
X_VAL_PATH = ARTIFACTS_DIR / "X_val_final.joblib"
X_TEST_PATH = ARTIFACTS_DIR / "X_test_final.joblib"

Y_TRAIN_PATH = ARTIFACTS_DIR / "y_train.joblib"
Y_VAL_PATH = ARTIFACTS_DIR / "y_val.joblib"
Y_TEST_PATH = ARTIFACTS_DIR / "y_test.joblib"

FEATURE_COLS_PATH = ARTIFACTS_DIR / "feature_cols.joblib"
CATEGORICAL_FEATURES_PATH = ARTIFACTS_DIR / "categorical_features.joblib"
NUMERICAL_FEATURES_PATH = ARTIFACTS_DIR / "numerical_features.joblib"
FINAL_FEATURE_NAMES_PATH = ARTIFACTS_DIR / "final_feature_names.joblib"

# ==========================================
# 4. مسارات النماذج والتقارير (Models & Reports)
# ==========================================
MODEL_PATH = MODELS_DIR / "delay_model_pipeline.joblib"
LOGISTIC_REGRESSION_PATH = MODELS_DIR / "logistic_regression.joblib"
RANDOM_FOREST_PATH = MODELS_DIR / "random_forest.joblib"
XGBOOST_PATH = MODELS_DIR / "xgboost.joblib"

# ✅ التعديل الرئيسي: من .joblib إلى .onnx
BEST_MODEL_PATH = MODELS_DIR / "best_xgb_model.onnx"

# تقارير الأداء والرسم البياني
METRICS_PATH = REPORTS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = FIGURES_DIR / "roc_curve.png"
FEATURE_IMPORTANCE_PATH = FIGURES_DIR / "feature_importance.png"

# ==========================================
# 5. السجلات والتطبيقات (Logs & API Configs)
# ==========================================
LOG_FILE_PATH = LOGS_DIR / "app.log"
LOG_LEVEL = "INFO"

# إعدادات السيرفر والـ API
API_HOST = "0.0.0.0"
API_PORT = 8000

# ==========================================
# 6. إعدادات تدريب وتقسيم البيانات (Train & Split Configs)
# ==========================================
RANDOM_STATE = 42
N_JOBS = -1

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==========================================
# 7. أسماء الأعمدة والمتغيرات (Column Schema)
# ==========================================
TARGET_COL = "delay_or_not"
TIME_SORT_COL = "order_purchase_timestamp"
DELIVERY_DATE_COL = "order_delivered_customer_date"
ESTIMATED_DATE_COL = "order_estimated_delivery_date"

DATE_COLS = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

LOG_TRANSFORM_COLS = [
    "total_item_price",
    "total_freight",
    "total_payment_value",
]

CATEGORICAL_FEATURES = [
    "customer_state",
    "seller_state",
    "customer_zip_code_prefix",
    "seller_zip_code_prefix",
    "payment_types_used",
]

NUMERICAL_FEATURES = [
    "purchase_year",
    "purchase_month",
    "purchase_dayofweek",
    "is_weekend",
    "is_same_state",
    "zip_prefix_diff",
    "total_items",
    "num_sellers",
    "total_item_price",
    "total_freight",
    "avg_item_price",
    "num_payment_installments_types",
    "max_installments",
    "total_payment_value",
    "num_categories",
    "avg_product_weight_g",
]

FEATURE_COLS = [
    "purchase_year",
    "purchase_month",
    "purchase_dayofweek",
    "is_weekend",
    "customer_state",
    "seller_state",
    "customer_zip_code_prefix",
    "seller_zip_code_prefix",
    "is_same_state",
    "zip_prefix_diff",
    "total_items",
    "num_sellers",
    "total_item_price",
    "total_freight",
    "avg_item_price",
    "num_payment_installments_types",
    "max_installments",
    "total_payment_value",
    "payment_types_used",
    "num_categories",
    "avg_product_weight_g",
]

VALID_ORDER_STATUS = "delivered"

# ==========================================
# 8. الفراطامترات للبحث (Hyperparameter Grids)
# ==========================================
XGB_PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 150, 200, 300],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 10],
    "gamma": [0, 0.1, 0.3, 0.5],
}

# ==========================================
# 9. التأكد من إنشاء المجلدات عند التشغيل
# ==========================================
for directory in [
    DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    ARTIFACTS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    LOGS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
