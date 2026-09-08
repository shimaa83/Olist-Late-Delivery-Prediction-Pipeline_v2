import pandas as pd
import pytest

import src.task3.config
from src.task3.split_data import main, split_summary


@pytest.fixture
def mock_pipeline_env(tmp_path, monkeypatch):
    master_path = tmp_path / "master.csv"
    labeled_path = tmp_path / "labeled.csv"
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    test_path = tmp_path / "test.csv"

    monkeypatch.setattr(src.task3.config, "MASTER_CSV_PATH", master_path)
    monkeypatch.setattr(src.task3.config, "LABELED_DATA_PATH", labeled_path)
    monkeypatch.setattr(src.task3.config, "TRAIN_DATA_PATH", train_path)
    monkeypatch.setattr(src.task3.config, "VAL_DATA_PATH", val_path)
    monkeypatch.setattr(src.task3.config, "TEST_DATA_PATH", test_path)

    # التأكد من ثبات المتغيرات المعرفة في config
    monkeypatch.setattr(
        src.task3.config,
        "DATE_COLS",
        [
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    monkeypatch.setattr(
        src.task3.config, "DELIVERY_DATE_COL", "order_delivered_customer_date"
    )
    monkeypatch.setattr(
        src.task3.config, "ESTIMATED_DATE_COL", "order_estimated_delivery_date"
    )
    monkeypatch.setattr(src.task3.config, "VALID_ORDER_STATUS", "delivered")
    monkeypatch.setattr(src.task3.config, "TARGET_COL", "delay_or_not")
    monkeypatch.setattr(src.task3.config, "TIME_SORT_COL", "order_purchase_timestamp")
    monkeypatch.setattr(src.task3.config, "TRAIN_RATIO", 0.6)
    monkeypatch.setattr(src.task3.config, "VAL_RATIO", 0.2)

    data = {
        "order_status": ["delivered"] * 10,
        "order_purchase_timestamp": pd.date_range(
            start="2023-01-01", periods=10, freq="D"
        ),
        "order_delivered_customer_date": pd.date_range(
            start="2023-01-03", periods=10, freq="D"
        ),
        "order_estimated_delivery_date": pd.date_range(
            start="2023-01-02", periods=10, freq="D"
        ),
    }
    df = pd.DataFrame(data)
    df.to_csv(master_path, index=False)

    return train_path, val_path, test_path


def test_split_summary(capsys):
    df = pd.DataFrame(
        {
            src.task3.config.TIME_SORT_COL: pd.date_range(
                start="2023-01-01", periods=3, freq="D"
            ),
            src.task3.config.TARGET_COL: [0, 1, 0],
        }
    )
    split_summary(df, "Sample Group")
    captured = capsys.readouterr()
    assert "Sample Group" in captured.out


def test_main_pipeline_execution(mock_pipeline_env):
    train_path, val_path, test_path = mock_pipeline_env
    main()
    assert train_path.exists()
    assert val_path.exists()
    assert test_path.exists()
