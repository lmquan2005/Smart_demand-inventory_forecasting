from pathlib import Path

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StringType,
    IntegerType,
    DateType,
    DoubleType,
)


SILVER_PATH = Path("dataset/processed/silver")

TABLES = {
    "sales": {
        "path": SILVER_PATH / "silver_sales",
        "required_columns": [
            "id", "item_id", "dept_id", "cat_id",
            "store_id", "state_id", "d", "sales"
        ],
        "key_columns": ["id", "d"],
    },
    "calendar": {
        "path": SILVER_PATH / "silver_calendar",
        "required_columns": ["d", "date", "wm_yr_wk"],
        "key_columns": ["d"],
    },
    "sell_prices": {
        "path": SILVER_PATH / "silver_sell_prices",
        "required_columns": [
            "store_id", "item_id", "wm_yr_wk", "sell_price"
        ],
        "key_columns": ["store_id", "item_id", "wm_yr_wk"],
    },
}


@pytest.mark.parametrize("table_name", TABLES.keys())
def test_silver_table_exists_and_has_data(spark, table_name):
    """Kiểm tra bảng Silver tồn tại và có dữ liệu."""

    config = TABLES[table_name]
    path = str(config["path"])

    df = spark.read.format("delta").load(path)

    # Kiểm tra schema
    actual_columns = set(df.columns)
    missing_columns = set(config["required_columns"]) - actual_columns

    assert not missing_columns, (
        f"{table_name}: thiếu các cột {missing_columns}"
    )

    # Kiểm tra bảng không rỗng
    assert df.limit(1).count() > 0, (
        f"{table_name}: bảng không có dữ liệu"
    )

    print(f"\n===== SILVER {table_name.upper()} =====")
    df.printSchema()
    df.show(5, truncate=False)


@pytest.mark.parametrize("table_name", TABLES.keys())
def test_silver_table_has_no_null_keys(spark, table_name):
    """Kiểm tra các cột khóa không chứa NULL."""

    config = TABLES[table_name]
    df = spark.read.format("delta").load(str(config["path"]))

    key_columns = config["key_columns"]

    null_conditions = [
        F.sum(
            F.when(F.col(column).isNull(), 1).otherwise(0)
        ).alias(column)
        for column in key_columns
    ]

    result = df.agg(*null_conditions).first().asDict()

    for column, null_count in result.items():
        null_count = null_count or 0

        assert null_count == 0, (
            f"{table_name}: cột {column} có {null_count} giá trị NULL"
        )


def test_silver_calendar_date_type(spark):
    """Kiểm tra kiểu dữ liệu của bảng Calendar."""

    df = spark.read.format("delta").load(
        str(TABLES["calendar"]["path"])
    )

    schema = {field.name: field.dataType for field in df.schema.fields}

    assert isinstance(schema["date"], DateType)
    assert isinstance(schema["wm_yr_wk"], IntegerType)


def test_silver_sales_data_types(spark):
    """Kiểm tra kiểu dữ liệu của Sales sau khi unpivot."""

    df = spark.read.format("delta").load(
        str(TABLES["sales"]["path"])
    )

    schema = {field.name: field.dataType for field in df.schema.fields}

    assert isinstance(schema["d"], StringType)
    assert isinstance(schema["sales"], IntegerType)


def test_silver_prices_data_types(spark):
    """Kiểm tra kiểu dữ liệu của Sell Prices."""

    df = spark.read.format("delta").load(
        str(TABLES["sell_prices"]["path"])
    )

    schema = {field.name: field.dataType for field in df.schema.fields}

    assert isinstance(schema["wm_yr_wk"], IntegerType)
    assert isinstance(schema["sell_price"], DoubleType)