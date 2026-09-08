import pandas as pd
import pytest

import src.task3.config
from src.task3.features import apply_feature_engineering, process_and_save_features


@pytest.fixture
def mock_feature_env(tmp_path, monkeypatch):
    train_p = tmp_path / "train.csv"
    val_p = tmp_path / "val.csv"
    test_p = tmp_path / "test.csv"

    monkeypatch.setattr(src.task3.config, "TRAIN_DATA_PATH", train_p)
    monkeypatch.setattr(src.task3.config, "VAL_DATA_PATH", val_p)
    monkeypatch.setattr(src.task3.config, "TEST_DATA_PATH", test_p)
    monkeypatch.setattr(src.task3.config, "ARTIFACTS_DIR", tmp_path)

    monkeypatch.setattr(src.task3.config, "CAT_IMPUTER_PATH", tmp_path / "cat_imp.pkl")
    monkeypatch.setattr(src.task3.config, "CAT_ENCODER_PATH", tmp_path / "cat_enc.pkl")
    monkeypatch.setattr(src.task3.config, "NUM_IMPUTER_PATH", tmp_path / "num_imp.pkl")
    monkeypatch.setattr(
        src.task3.config, "NUM_SCALER_PATH", tmp_path / "num_scaler.pkl"
    )
    monkeypatch.setattr(
        src.task3.config, "CATEGORICAL_FEATURES_PATH", tmp_path / "cat_feats.pkl"
    )
    monkeypatch.setattr(
        src.task3.config, "NUMERICAL_FEATURES_PATH", tmp_path / "num_feats.pkl"
    )
    monkeypatch.setattr(
        src.task3.config, "FEATURE_COLS_PATH", tmp_path / "feat_cols.pkl"
    )
    monkeypatch.setattr(
        src.task3.config, "FINAL_FEATURE_NAMES_PATH", tmp_path / "final_feat_names.pkl"
    )
    monkeypatch.setattr(src.task3.config, "X_TRAIN_PATH", tmp_path / "X_train.pkl")
    monkeypatch.setattr(src.task3.config, "X_VAL_PATH", tmp_path / "X_val.pkl")
    monkeypatch.setattr(src.task3.config, "X_TEST_PATH", tmp_path / "X_test.pkl")
    monkeypatch.setattr(src.task3.config, "Y_TRAIN_PATH", tmp_path / "y_train.pkl")
    monkeypatch.setattr(src.task3.config, "Y_VAL_PATH", tmp_path / "y_val.pkl")
    monkeypatch.setattr(src.task3.config, "Y_TEST_PATH", tmp_path / "y_test.pkl")

    monkeypatch.setattr(src.task3.config, "TIME_SORT_COL", "order_purchase_timestamp")
    monkeypatch.setattr(src.task3.config, "LOG_TRANSFORM_COLS", ["total_item_price"])
    monkeypatch.setattr(src.task3.config, "CATEGORICAL_FEATURES", ["customer_state"])
    monkeypatch.setattr(src.task3.config, "NUMERICAL_FEATURES", ["total_item_price"])
    monkeypatch.setattr(
        src.task3.config,
        "FEATURE_COLS",
        ["customer_state", "total_item_price", "order_purchase_timestamp"],
    )
    monkeypatch.setattr(src.task3.config, "TARGET_COL", "delay_or_not")

    df = pd.DataFrame(
        {
            "order_purchase_timestamp": ["2023-01-01 10:00:00", "2023-01-07 12:00:00"],
            "total_item_price": [100.0, 200.0],
            "customer_state": ["CA", "NY"],
            "seller_state": ["CA", "TX"],
            "customer_zip_code_prefix": [12345, 54321],
            "seller_zip_code_prefix": [12345, 11111],
            "delay_or_not": [0, 1],
        }
    )

    df.to_csv(train_p, index=False)
    df.to_csv(val_p, index=False)
    df.to_csv(test_p, index=False)


def test_apply_feature_engineering():
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": ["2023-01-01 10:00:00"],
            "total_item_price": [10.0],
            "customer_state": ["CA"],
            "seller_state": ["CA"],
            "customer_zip_code_prefix": [1000],
            "seller_zip_code_prefix": [500],
        }
    )

    src.task3.config.TIME_SORT_COL = "order_purchase_timestamp"
    src.task3.config.LOG_TRANSFORM_COLS = ["total_item_price"]

    res_df = apply_feature_engineering(df)
    assert "is_same_state" in res_df.columns


def test_process_and_save_features(mock_feature_env):
    process_and_save_features()
    assert src.task3.config.X_TRAIN_PATH.exists()
