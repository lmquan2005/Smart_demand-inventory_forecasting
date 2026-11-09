
from pathlib import Path

import pytest
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BRONZE_PATH = PROJECT_ROOT / "dataset" / "processed" / "bronze"

BRONZE_TABLES = {
    "bronze_sales": BRONZE_PATH / "bronze_sales",
    "bronze_calendar": BRONZE_PATH / "bronze_calendar",
    "bronze_sell_prices": BRONZE_PATH / "bronze_sell_prices",
}


@pytest.mark.parametrize(
    "table_name, table_path",
    BRONZE_TABLES.items()
)
def test_bronze_table(spark, table_name, table_path):

    print(f"\n{'=' * 70}")
    print(f"TEST TABLE: {table_name}")
    print(f"{'=' * 70}")

    # 1. Check path
    assert table_path.exists(), (
        f"Bronze table does not exist: {table_path}"
    )

    # 2. Read Delta
    df = (
        spark.read
        .format("delta")
        .load(str(table_path))
    )

    # 3. Display schema
    print("\n[SCHEMA]")
    df.printSchema()

    # 4. Count records
    total_records = df.count()

    print(f"\n[TOTAL RECORDS]: {total_records:,}")

    assert total_records > 0, (
        f"{table_name} is empty!"
    )

    # 5. Display preview
    print("\n[DATA PREVIEW]")
    df.show(10, truncate=False)

    # 6. Null statistics
    print("\n[NULL STATISTICS]")

    null_stats = df.select([
        F.count(
            F.when(F.col(c).isNull(), 1)
        ).alias(c)
        for c in df.columns
    ])

    null_stats.show(truncate=False)

    print(f"\n[PASSED] {table_name}")
