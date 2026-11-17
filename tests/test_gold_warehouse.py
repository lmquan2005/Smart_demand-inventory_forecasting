import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    LongType,
)

from src.warehouse.dimension_builder import DimensionBuilder
from src.warehouse.fact_inventory_builder import FactInventoryBuilder
from src.warehouse.fact_sales_builder import FactSalesBuilder
from src.warehouse.schema import GOLD_SCHEMAS


def _gold_inputs(spark):
    sales = spark.createDataFrame(
        [
            ("ITEM_1", "DEPT_1", "FOODS", "STORE_1", "CA", "d_1", 2),
        ],
        [
            "item_id", "dept_id", "cat_id", "store_id", "state_id",
            "d", "sales",
        ],
    )
    calendar = spark.createDataFrame(
        [
            (
                "d_1", "2011-01-29", 11101, "Saturday", 1, 1, 2011,
                "Event 1", "Type 1", "Event 2", "Type 2", 1, 0, 0,
            ),
        ],
        [
            "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2",
            "snap_CA", "snap_TX", "snap_WI",
        ],
    ).withColumn("date", F.to_date("date"))
    prices = spark.createDataFrame(
        [("ITEM_1", "STORE_1", 11101, 2.5)],
        ["item_id", "store_id", "wm_yr_wk", "sell_price"],
    )
    inventory = spark.createDataFrame(
        [
            (
                "ITEM_1", "DEPT_1", "FOODS", "STORE_1", "CA", "d_1",
                2.0, 1.5, 3, 5.0, 1.0, 4.0, 4.5, 0.0, 0, 0,
            ),
        ],
        [
            "item_id", "dept_id", "cat_id", "store_id", "state_id", "day",
            "units_sold", "avg_daily_demand", "lead_time_days",
            "opening_stock", "replenishment_qty", "closing_stock",
            "reorder_point", "lost_sales", "stockout_flag", "overstock_flag",
        ],
    )
    return sales, calendar, prices, inventory


def test_gold_builders_apply_schema_and_resolve_daily_keys(spark):
    sales, calendar, prices, inventory = _gold_inputs(spark)
    dimensions = DimensionBuilder()
    dim_date = dimensions.build_dim_date(calendar)
    dim_product = dimensions.build_dim_product(sales, inventory)
    dim_store = dimensions.build_dim_store(sales, inventory)

    fact_sales = FactSalesBuilder().build(
        sales, prices, dim_date, dim_product, dim_store
    )
    fact_inventory = FactInventoryBuilder().build(
        inventory, dim_date, dim_product, dim_store
    )

    assert dim_date.first()["date_key"] == 20110129
    assert dim_date.first()["m5_day"] == "d_1"
    assert fact_sales.first()["units_sold"] == 2
    assert fact_sales.first()["sell_price"] == 2.5
    assert fact_sales.first()["revenue"] == 5.0
    assert fact_inventory.first()["date_key"] == 20110129
    assert fact_inventory.first()["units_sold"] == 2.0

    dataframes = {
        "dim_date": dim_date,
        "dim_product": dim_product,
        "dim_store": dim_store,
        "fact_sales_daily": fact_sales,
        "fact_inventory_daily": fact_inventory,
    }
    for table_name, dataframe in dataframes.items():
        assert dataframe.columns == GOLD_SCHEMAS[table_name].fieldNames()
        assert [
            field.dataType for field in dataframe.schema.fields
        ] == [
            field.dataType for field in GOLD_SCHEMAS[table_name].fields
        ]

    assert isinstance(dim_date.schema["date_key"].dataType, IntegerType)
    assert isinstance(dim_date.schema["full_date"].dataType, DateType)
    assert isinstance(dim_product.schema["product_key"].dataType, LongType)
    assert isinstance(fact_sales.schema["units_sold"].dataType, LongType)
    assert isinstance(
        fact_inventory.schema["units_sold"].dataType,
        DoubleType,
    )


def test_fact_sales_rejects_duplicate_business_grain(spark):
    sales, calendar, prices, inventory = _gold_inputs(spark)
    duplicate_sales = sales.unionByName(sales)
    dimensions = DimensionBuilder()
    dim_date = dimensions.build_dim_date(calendar)
    dim_product = dimensions.build_dim_product(sales, inventory)
    dim_store = dimensions.build_dim_store(sales, inventory)

    with pytest.raises(ValueError, match="duplicate rows"):
        FactSalesBuilder().build(
            duplicate_sales,
            prices,
            dim_date,
            dim_product,
            dim_store,
        )
