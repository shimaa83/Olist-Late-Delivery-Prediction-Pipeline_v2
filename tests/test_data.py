import sqlite3
from pathlib import Path
import pandas as pd
import pytest

from src.task3.data import build_orders_master_table, get_db_connection, get_table_names


@pytest.fixture
def mock_db(tmp_path):
    """إنشاء قاعدة بيانات مؤقتة وبجداول وبيانات وهمية للاختبار."""
    db_path = tmp_path / "test_olist.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. إنشاء الجداول الوهمية
    cursor.execute(
        "CREATE TABLE olist_orders_dataset (order_id TEXT, customer_id TEXT,"
        " order_status TEXT, order_purchase_timestamp TEXT,"
        " order_delivered_customer_date TEXT, order_estimated_delivery_date"
        " TEXT);"
    )
    cursor.execute(
        "CREATE TABLE olist_customers_dataset (customer_id TEXT,"
        " customer_zip_code_prefix TEXT, customer_state TEXT);"
    )
    cursor.execute(
        "CREATE TABLE olist_order_items_dataset (order_id TEXT, order_item_id"
        " INTEGER, product_id TEXT, seller_id TEXT, price REAL, freight_value"
        " REAL);"
    )
    cursor.execute(
        "CREATE TABLE olist_order_payments_dataset (order_id TEXT,"
        " payment_sequential INTEGER, payment_installments INTEGER, payment_value"
        " REAL, payment_type TEXT);"
    )
    cursor.execute(
        "CREATE TABLE olist_products_dataset (product_id TEXT,"
        " product_category_name TEXT, product_weight_g REAL);"
    )
    cursor.execute(
        "CREATE TABLE olist_sellers_dataset (seller_id TEXT, seller_state TEXT,"
        " seller_zip_code_prefix TEXT);"
    )

    # 2. إدخال بيانات تجريبية بسيطة
    cursor.execute("INSERT INTO olist_customers_dataset VALUES ('c1', '12345', 'CA');")
    cursor.execute(
        "INSERT INTO olist_orders_dataset VALUES ('o1', 'c1', 'delivered',"
        " '2023-01-01', '2023-01-05', '2023-01-06');"
    )
    cursor.execute(
        "INSERT INTO olist_order_items_dataset VALUES ('o1', 1, 'p1', 's1', 100.0,"
        " 10.0);"
    )
    cursor.execute(
        "INSERT INTO olist_order_payments_dataset VALUES ('o1', 1, 1, 110.0,"
        " 'credit_card');"
    )
    cursor.execute(
        "INSERT INTO olist_products_dataset VALUES ('p1', 'electronics', 500.0);"
    )
    cursor.execute("INSERT INTO olist_sellers_dataset VALUES ('s1', 'NY', '54321');")

    conn.commit()
    conn.close()
    return db_path


def test_get_db_connection_not_found():
    """اختبار رفع خطأ FileNotFoundError إذا لم يوجد ملف القاعدة."""
    with pytest.raises(FileNotFoundError):
        get_db_connection(Path("non_existent_database.db"))


def test_get_table_names(mock_db):
    """اختبار استرجاع أسماء الجداول بشكل صحيح."""
    conn = get_db_connection(mock_db)
    tables = get_table_names(conn)
    conn.close()

    assert "olist_orders_dataset" in tables
    assert "olist_customers_dataset" in tables
    assert "olist_order_items_dataset" in tables


def test_build_orders_master_table(mock_db):
    """اختبار دمج الجداول والتأكد من صحة النتائج وحجم البيانات."""
    conn = get_db_connection(mock_db)
    df_master = build_orders_master_table(conn)
    conn.close()

    # التأكد أن النتيجة DataFrame وليست فارغة
    assert isinstance(df_master, pd.DataFrame)
    assert not df_master.empty
    assert "order_id" in df_master.columns
    assert "total_item_price" in df_master.columns

    # التأكد من صحة الحسابات للبيانات المدخلة
    item_price = df_master.loc[
        df_master["order_id"] == "o1", "total_item_price"
    ].values[0]
    assert item_price == 100.0
