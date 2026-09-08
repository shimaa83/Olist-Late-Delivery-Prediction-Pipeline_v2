import logging
import sys

import great_expectations as gx
import pandas as pd

from src.task3 import config

logger = logging.getLogger("validate_data")


def build_expectations(validator):
    validator.expect_column_values_to_not_be_null("order_id")

    validator.expect_column_values_to_be_in_set(config.TARGET_COL, [0, 1])

    # 3. الأسعار لازم تكون موجبة ومنطقية
    if "price" in validator.active_batch.data.dataframe.columns:
        validator.expect_column_values_to_be_between(
            "price", min_value=0, max_value=100000
        )

    # 4. الجدول مينفعش يكون فاضي
    validator.expect_table_row_count_to_be_between(min_value=1)

    # 5. مفيش صفوف مكررة بالكامل (اختياري - علّق لو مش محتاجه)
    # validator.expect_compound_columns_to_be_unique(list(validator.active_batch.data.dataframe.columns))

    return validator


def validate_dataframe(df: pd.DataFrame, dataset_name: str) -> bool:
    """
    يشغّل الـ Expectations على أي DataFrame ويرجّع True/False.
    لو فيه فشل، بيطبع تفاصيل الـ Expectation اللي فشلت.
    """
    context = gx.get_context(mode="ephemeral")

    data_source = context.data_sources.add_pandas(name=f"{dataset_name}_source")
    data_asset = data_source.add_dataframe_asset(name=f"{dataset_name}_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        f"{dataset_name}_batch"
    )
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    validator = context.get_validator(batch=batch)
    validator = build_expectations(validator)

    result = validator.validate()

    if not result.success:
        logger.error(f"❌ فشل الـ validation على: {dataset_name}")
        for r in result.results:
            if not r.success:
                logger.error(
                    f"  - {r.expectation_config.type}: {r.expectation_config.kwargs}"
                )
    else:
        logger.info(f"✅ الـ validation نجح على: {dataset_name}")

    return result.success


def validate_all(fail_on_error: bool = True):
    """يتنادى قبل load_artifacts() في train_mlflow.py."""
    datasets = {
        "train_df": config.TRAIN_DATA_PATH,
        "validation_df": config.VAL_DATA_PATH,
        "test_df": config.TEST_DATA_PATH,
    }

    all_passed = True
    for name, path in datasets.items():
        df = pd.read_csv(path)
        passed = validate_dataframe(df, name)
        all_passed = all_passed and passed

    if not all_passed and fail_on_error:
        logger.error("توقف الـ pipeline: فشل الـ data validation.")
        sys.exit(1)

    return all_passed


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s"
    )
    validate_all()
