import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import src.task3.config as config


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """تطبيق هندسة الميزات على البيانات (Log transform, date features, diff features)."""
    df = df.copy()

    # تحويل وقت الشراء إلى Datetime
    df[config.TIME_SORT_COL] = pd.to_datetime(df[config.TIME_SORT_COL], errors="coerce")

    # تحويل القيم المالية معالجة للانحراف (Log1p)
    for col in config.LOG_TRANSFORM_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # إنشاء ميزة تطابق الولاية (Same state)
    if "customer_state" in df.columns and "seller_state" in df.columns:
        df["is_same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)

    # إنشاء ميزة الفرق في الرمز البريدي
    if (
        "customer_zip_code_prefix" in df.columns
        and "seller_zip_code_prefix" in df.columns
    ):
        df["zip_prefix_diff"] = np.abs(
            df["customer_zip_code_prefix"] - df["seller_zip_code_prefix"]
        )

    # استخراج ميزات التاريخ
    df["purchase_year"] = df[config.TIME_SORT_COL].dt.year
    df["purchase_month"] = df[config.TIME_SORT_COL].dt.month
    df["purchase_dayofweek"] = df[config.TIME_SORT_COL].dt.dayofweek
    df["is_weekend"] = df["purchase_dayofweek"].isin([5, 6]).astype(int)

    return df


def prepare_datasets():
    """تحميل مجموعات البيانات وتطبيق Feature Engineering عليها واستخلاص الهدف والميزات."""
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(config.VAL_DATA_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)

    train_df = apply_feature_engineering(train_df)
    val_df = apply_feature_engineering(val_df)
    test_df = apply_feature_engineering(test_df)

    X_train = train_df[config.FEATURE_COLS].copy()
    X_val = val_df[config.FEATURE_COLS].copy()
    X_test = test_df[config.FEATURE_COLS].copy()

    y_train = train_df[config.TARGET_COL].copy()
    y_val = val_df[config.TARGET_COL].copy()
    y_test = test_df[config.TARGET_COL].copy()

    return X_train, X_val, X_test, y_train, y_val, y_test


def process_and_save_features():
    """تجهيز المعالجة الشاملة للميزات النصية والرقمية وحفظ كافة الـ Artifacts."""
    print("=== 1. Loading and Transforming Datasets ===")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_datasets()

    print("=== 2. Processing Categorical Features ===")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train_cat = cat_imputer.fit_transform(X_train[config.CATEGORICAL_FEATURES])
    X_val_cat = cat_imputer.transform(X_val[config.CATEGORICAL_FEATURES])
    X_test_cat = cat_imputer.transform(X_test[config.CATEGORICAL_FEATURES])

    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X_train_cat = cat_encoder.fit_transform(X_train_cat)
    X_val_cat = cat_encoder.transform(X_val_cat)
    X_test_cat = cat_encoder.transform(X_test_cat)

    print("=== 3. Processing Numerical Features ===")
    num_imputer = SimpleImputer(strategy="median")
    X_train_num = num_imputer.fit_transform(X_train[config.NUMERICAL_FEATURES])
    X_val_num = num_imputer.transform(X_val[config.NUMERICAL_FEATURES])
    X_test_num = num_imputer.transform(X_test[config.NUMERICAL_FEATURES])

    num_scaler = StandardScaler()
    X_train_num = num_scaler.fit_transform(X_train_num)
    X_val_num = num_scaler.transform(X_val_num)
    X_test_num = num_scaler.transform(X_test_num)

    print("=== 4. Stacking Features ===")
    X_train_final = hstack([X_train_num, X_train_cat])
    X_val_final = hstack([X_val_num, X_val_cat])
    X_test_final = hstack([X_test_num, X_test_cat])

    print("X_train_final shape:", X_train_final.shape)
    print("X_val_final shape:", X_val_final.shape)
    print("X_test_final shape:", X_test_final.shape)

    print("=== 5. Saving Artifacts ===")
    # حفظ المحولات والمعالجات
    joblib.dump(cat_imputer, config.CAT_IMPUTER_PATH)
    joblib.dump(cat_encoder, config.CAT_ENCODER_PATH)
    joblib.dump(num_imputer, config.NUM_IMPUTER_PATH)
    joblib.dump(num_scaler, config.NUM_SCALER_PATH)

    # حفظ قوائم الميزات وأسمائها النهائية
    joblib.dump(config.CATEGORICAL_FEATURES, config.CATEGORICAL_FEATURES_PATH)
    joblib.dump(config.NUMERICAL_FEATURES, config.NUMERICAL_FEATURES_PATH)
    joblib.dump(config.FEATURE_COLS, config.FEATURE_COLS_PATH)

    cat_feature_names = cat_encoder.get_feature_names_out(config.CATEGORICAL_FEATURES)
    final_feature_names = config.NUMERICAL_FEATURES + list(cat_feature_names)
    joblib.dump(final_feature_names, config.FINAL_FEATURE_NAMES_PATH)

    # حفظ المصفوفات الناتجة والـ Targets
    joblib.dump(X_train_final, config.X_TRAIN_PATH)
    joblib.dump(X_val_final, config.X_VAL_PATH)
    joblib.dump(X_test_final, config.X_TEST_PATH)

    joblib.dump(y_train, config.Y_TRAIN_PATH)
    joblib.dump(y_val, config.Y_VAL_PATH)
    joblib.dump(y_test, config.Y_TEST_PATH)

    print(f"All artifacts saved successfully to '{config.ARTIFACTS_DIR}'")


if __name__ == "__main__":
    process_and_save_features()
