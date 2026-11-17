from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)


GOLD_SCHEMAS = {
    "dim_date": StructType([
        StructField("date_key", IntegerType(), False),
        StructField("m5_day", StringType(), False),
        StructField("full_date", DateType(), False),
        StructField("wm_yr_wk", IntegerType(), True),
        StructField("weekday", StringType(), True),
        StructField("day_of_week", IntegerType(), True),
        StructField("month", IntegerType(), True),
        StructField("year", IntegerType(), True),
        StructField("event_name_1", StringType(), True),
        StructField("event_type_1", StringType(), True),
        StructField("event_name_2", StringType(), True),
        StructField("event_type_2", StringType(), True),
        StructField("snap_ca", IntegerType(), True),
        StructField("snap_tx", IntegerType(), True),
        StructField("snap_wi", IntegerType(), True),
    ]),
    "dim_product": StructType([
        StructField("product_key", LongType(), False),
        StructField("item_id", StringType(), False),
        StructField("dept_id", StringType(), True),
        StructField("cat_id", StringType(), True),
    ]),
    "dim_store": StructType([
        StructField("store_key", LongType(), False),
        StructField("store_id", StringType(), False),
        StructField("state_id", StringType(), True),
    ]),
    "fact_sales_daily": StructType([
        StructField("date_key", IntegerType(), False),
        StructField("product_key", LongType(), False),
        StructField("store_key", LongType(), False),
        StructField("units_sold", LongType(), True),
        StructField("sell_price", DoubleType(), True),
        StructField("revenue", DoubleType(), True),
    ]),
    "fact_inventory_daily": StructType([
        StructField("date_key", IntegerType(), False),
        StructField("product_key", LongType(), False),
        StructField("store_key", LongType(), False),
        StructField("units_sold", DoubleType(), True),
        StructField("avg_daily_demand", DoubleType(), True),
        StructField("lead_time_days", IntegerType(), True),
        StructField("opening_stock", DoubleType(), True),
        StructField("replenishment_qty", DoubleType(), True),
        StructField("closing_stock", DoubleType(), True),
        StructField("reorder_point", DoubleType(), True),
        StructField("lost_sales", DoubleType(), True),
        StructField("stockout_flag", IntegerType(), True),
        StructField("overstock_flag", IntegerType(), True),
    ]),
}


def apply_gold_schema(dataframe: DataFrame, table_name: str) -> DataFrame:
    """Select and cast a Gold dataframe to its declared column order and types."""
    if table_name not in GOLD_SCHEMAS:
        raise ValueError(f"Unknown Gold table: {table_name}")

    schema = GOLD_SCHEMAS[table_name]
    missing_columns = [
        field.name for field in schema.fields
        if field.name not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{table_name} is missing columns: {missing_columns}"
        )

    return dataframe.select(*[
        F.col(field.name).cast(field.dataType).alias(field.name)
        for field in schema.fields
    ])
