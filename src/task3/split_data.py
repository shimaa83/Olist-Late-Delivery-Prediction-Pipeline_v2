import pandas as pd

from src.task3 import config


def split_summary(df: pd.DataFrame, name: str) -> None:
    print(f"\n{name}")
    print("=" * 50)

    # Number of rows
    print("Rows:", len(df))

    # Date range
    print(
        "Date range:",
        df[config.TIME_SORT_COL].min(),
        "→",
        df[config.TIME_SORT_COL].max(),
    )

    # Label distribution
    print("\nLabel distribution:")
    print(df[config.TARGET_COL].value_counts().sort_index())

    # Label percentage
    print("\nLabel percentage:")
    print(
        df[config.TARGET_COL]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )


def main():
    # --- الجزء الأول: تحميل البيانات ومعالجتها وإنشاء Target Label ---
    df = pd.read_csv(config.MASTER_CSV_PATH)

    # تحويل أعمدة التواريخ إلى Datetime
    for col in config.DATE_COLS:
        df[col] = pd.to_datetime(df[col])

    # تصفية الطلبات المسلمة فقط
    df = df[df["order_status"] == config.VALID_ORDER_STATUS]

    # حساب أيام التأخير والعمود المستهدف delay_or_not
    df["no_delay_days"] = (
        df[config.DELIVERY_DATE_COL] - df[config.ESTIMATED_DATE_COL]
    ).dt.total_seconds() / (24 * 3600)

    df[config.TARGET_COL] = (
        df[config.DELIVERY_DATE_COL] > df[config.ESTIMATED_DATE_COL]
    ).astype(int)

    # طباعة نسبة عدم التوازن (Imbalance)
    print("Class distribution:")
    print(df[config.TARGET_COL].value_counts(normalize=True))

    # حفظ الجدول النهائي مع الـ Labels
    df.to_csv(config.LABELED_DATA_PATH, index=False)

    # --- الجزء الثاني: تقسيم البيانات زمنيًا (Time-Based Split) ---
    df = pd.read_csv(config.LABELED_DATA_PATH)

    for col in config.DATE_COLS:
        df[col] = pd.to_datetime(df[col])

    # التأكد من عدد القيم الفارغة في أوقات التسليم
    print(
        "Total number of null values in the delivery date column:",
        df[config.DELIVERY_DATE_COL].isna().sum(),
    )
    print(
        "Total number of null values in the estimated delivery date column:",
        df[config.ESTIMATED_DATE_COL].isna().sum(),
    )

    # حذف الصفوف التي تحتوي على قيم فارغة في تاريخ التسليم
    df_clean = df.dropna(subset=[config.DELIVERY_DATE_COL]).copy()

    # ترتيب البيانات زمنيًا بناءً على وقت الشراء
    df_clean = df_clean.sort_values(by=config.TIME_SORT_COL)

    # تحديد أحجام التقسيم بناءً على النسب المعرفة
    total_rows = len(df_clean)
    train_size = int(total_rows * config.TRAIN_RATIO)
    validation_size = int(total_rows * config.VAL_RATIO)

    train_df = df_clean.iloc[:train_size]
    validation_df = df_clean.iloc[train_size : train_size + validation_size]
    test_df = df_clean.iloc[train_size + validation_size :]

    # طباعة أبعاد المجموعات
    print("train_df shape:", train_df.shape)
    print("validation_df shape:", validation_df.shape)
    print("test_df shape:", test_df.shape)

    # طباعة ملخص كل مجموعة
    split_summary(train_df, "df_train")
    split_summary(validation_df, "df_validation")
    split_summary(test_df, "df_test")

    # حفظ المجموعات في ملفات CSV منفصلة
    train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
    validation_df.to_csv(config.VAL_DATA_PATH, index=False)
    test_df.to_csv(config.TEST_DATA_PATH, index=False)


if __name__ == "__main__":
    main()
