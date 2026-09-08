import sqlite3

import pandas as pd

from src.task3.config import DB_PATH, MASTER_CSV_PATH


def get_db_connection(db_path=DB_PATH) -> sqlite3.Connection:
    """إنشاء اتصال بقاعدة البيانات SQLite باستخدام المسار المعتمد في config.py."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found at: {db_path.resolve()}")
    return sqlite3.connect(db_path)


def get_table_names(conn: sqlite3.Connection) -> list[str]:
    """استرجاع أسماء كافة الجداول الموجودة في قاعدة البيانات."""
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql_query(query, conn)["name"].tolist()
    return tables


def analyze_table(
    conn: sqlite3.Connection, table_name: str, pk_cols: list[str] | None = None
) -> None:
    """تحليل جدول معين وعرض الإحصائيات الخاصة به."""
    df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)

    print("\n========================================")
    print(f" Table: {table_name}")
    print("========================================")
    print(f"• Total Rows: {len(df):,}")
    print(f"• Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"• Duplicates across all columns: {df.duplicated().sum():,}")

    if pk_cols:
        print(
            f"• Duplicates in Primary Key ({pk_cols}): {df.duplicated(subset=pk_cols).sum():,}"
        )

    missing = df.isnull().sum()
    print("\n• Missing Values:")
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values.")

    print("\n• Summary Statistics (df.describe):")
    print(df.describe(include="all"))

    print("\n• First 3 Rows (df.head(3)):")
    print(df.head(3))


def build_orders_master_table(conn: sqlite3.Connection) -> pd.DataFrame:
    """تجميع ودمج بيانات الطلبات، العملاء، عناصر الطلب، المدفوعات، والمنتجات في DataFrame واحد."""
    # 1. Orders & Customers join
    df_orders = pd.read_sql_query(
        """
        SELECT
            o.order_id,
            o.customer_id,
            c.customer_zip_code_prefix,
            c.customer_state,
            o.order_status,
            o.order_purchase_timestamp,
            o.order_delivered_customer_date,
            o.order_estimated_delivery_date
        FROM olist_orders_dataset o
        LEFT JOIN olist_customers_dataset c
            ON o.customer_id = c.customer_id
    """,
        conn,
    )

    # 2. Order items aggregation & join with sellers
    df_items_agg = pd.read_sql_query(
        """
        SELECT
            i.order_id,
            COUNT(i.order_item_id) AS total_items,
            COUNT(DISTINCT i.seller_id) AS num_sellers,
            SUM(i.price) AS total_item_price,
            SUM(i.freight_value) AS total_freight,
            AVG(i.price) AS avg_item_price,
            MIN(s.seller_state) AS seller_state,
            MIN(s.seller_zip_code_prefix) AS seller_zip_code_prefix
        FROM olist_order_items_dataset i
        LEFT JOIN olist_sellers_dataset s
            ON i.seller_id = s.seller_id
        GROUP BY i.order_id
    """,
        conn,
    )

    # 3. Order payments aggregation
    df_payments_agg = pd.read_sql_query(
        """
        SELECT
            order_id,
            COUNT(payment_sequential) AS num_payment_installments_types,
            MAX(payment_installments) AS max_installments,
            SUM(payment_value) AS total_payment_value,
            GROUP_CONCAT(DISTINCT payment_type) AS payment_types_used
        FROM olist_order_payments_dataset
        GROUP BY order_id
    """,
        conn,
    )

    # 4. Products aggregation & join with order_items
    df_products_agg = pd.read_sql_query(
        """
        SELECT
            i.order_id,
            COUNT(DISTINCT p.product_category_name) AS num_categories,
            AVG(p.product_weight_g) AS avg_product_weight_g
        FROM olist_order_items_dataset i
        LEFT JOIN olist_products_dataset p
            ON i.product_id = p.product_id
        GROUP BY i.order_id
    """,
        conn,
    )

    # 5. Final join of all aggregated dataframes
    final_table = (
        df_orders.merge(df_items_agg, on="order_id", how="left")
        .merge(df_payments_agg, on="order_id", how="left")
        .merge(df_products_agg, on="order_id", how="left")
    )

    return final_table


def run_full_pipeline() -> None:
    """تشغيل خط المعالجة بالكامل (التحليل، دمج البيانات، وحفظ الملف النهائي)."""
    conn = get_db_connection()

    table_pks = {
        "olist_customers_dataset": ["customer_id"],
        "olist_orders_dataset": ["order_id"],
        "olist_order_items_dataset": ["order_id", "order_item_id"],
        "olist_order_payments_dataset": ["order_id", "payment_sequential"],
        "olist_order_reviews_dataset": ["review_id"],
        "olist_products_dataset": ["product_id"],
        "olist_sellers_dataset": ["seller_id"],
        "olist_geolocation_dataset": None,
    }

    try:
        tables = get_table_names(conn)
        print("Tables found in database:")
        for t in tables:
            print(f" - {t}")

        # 1. تحليل كل جدول بشكل مستقل
        for table_name, pk_cols in table_pks.items():
            if table_name in tables:
                analyze_table(conn, table_name, pk_cols=pk_cols)

        # 2. إنشاء الجدول المدمج
        print("\n========================================")
        print(" Building Master Orders Table")
        print("========================================")
        df_master = build_orders_master_table(conn)
        print(f"• Master Table Shape: {df_master.shape}")

        # 3. حفظ النتيجة في CSV داخل مجلد data مباشرة عبر config.py
        df_master.to_csv(MASTER_CSV_PATH, index=False)
        print(f"\n Saved master table successfully to: {MASTER_CSV_PATH}")

    finally:
        conn.close()


if __name__ == "__main__":
    run_full_pipeline()
