import numpy as np
import pandas as pd

# استيراد كافة المسارات والإعدادات من config.py
import src.task3.config as config


def load_dataset(file_path: config.Path = config.TRAIN_DATA_PATH) -> pd.DataFrame:
    """تحميل البيانات وتحويل أعمدة التواريخ المحددة في config.DATE_COLS."""
    df = pd.read_csv(file_path)
    df[config.DATE_COLS] = df[config.DATE_COLS].apply(pd.to_datetime, errors="coerce")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """تصفية الطلبات المقبولة، حذف القيم المفقودة، واستبعاد المعرفات."""
    # 1. تصفية الطلبات المكتملة بناءً على VALID_ORDER_STATUS
    if "order_status" in df.columns:
        df = df[df["order_status"] == config.VALID_ORDER_STATUS]

    # 2. حذف القيم المفقودة
    df_clean = df.dropna().reset_index(drop=True)

    # 3. استبعاد المعرفات الثابتة والأعمدة التي تم الفلترة بناءً عليها
    drop_cols = ["order_id", "customer_id", "order_status"]
    return df_clean.drop(columns=[col for col in drop_cols if col in df.columns])


def apply_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """تطبيق Log1p Transformation على القيم المالية ذات الانحراف الشديد."""
    df = df.copy()
    log_cols = ["total_item_price", "total_freight", "total_payment_value"]
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    return df


def analyze_delay_rates(df: pd.DataFrame) -> pd.Series:
    """حساب نسبة التأخير حسب الولاية بناءً على config.TARGET_COL."""
    if "customer_state" in df.columns and config.TARGET_COL in df.columns:
        return (
            df.groupby("customer_state")[config.TARGET_COL]
            .mean()
            .mul(100)
            .round(2)
            .sort_values(ascending=False)
        )
    return pd.Series(dtype=float)


def run_eda_pipeline() -> pd.DataFrame:
    print(f"=== 1. Loading Dataset from: {config.TRAIN_DATA_PATH} ===")
    df = load_dataset()
    print(f"Original Shape: {df.shape}")

    print("\n=== 2. Cleaning Data ===")
    df_clean = clean_data(df)
    print(f"Cleaned Shape: {df_clean.shape}")

    print(
        f"\n=== 3. Delay Rate (%) by Customer State (Target: {config.TARGET_COL}) ==="
    )
    delay_summary = analyze_delay_rates(df_clean)
    print(delay_summary)

    print("\n=== 4. Applying Feature Transformations ===")
    df_transformed = apply_transformations(df_clean)

    # حفظ البيانات المعالجة في المسار الجديد CLEANED_DATA_PATH
    df_transformed.to_csv(config.CLEANED_DATA_PATH, index=False)
    print(f"\nSaved processed dataset to: {config.CLEANED_DATA_PATH}")

    return df_transformed


if __name__ == "__main__":
    processed_df = run_eda_pipeline()
