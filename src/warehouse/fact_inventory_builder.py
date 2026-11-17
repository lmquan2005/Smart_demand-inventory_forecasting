from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .dataframe_utils import (
    assert_no_nulls,
    assert_unique,
    require_columns,
)
from .schema import apply_gold_schema


class FactInventoryBuilder:
    def build(
        self,
        silver_inventory: DataFrame,
        dim_date: DataFrame,
        dim_product: DataFrame,
        dim_store: DataFrame,
    ) -> DataFrame:
        inventory_columns = [
            "item_id",
            "store_id",
            "day",
            "units_sold",
            "avg_daily_demand",
            "lead_time_days",
            "opening_stock",
            "replenishment_qty",
            "closing_stock",
            "reorder_point",
            "lost_sales",
            "stockout_flag",
            "overstock_flag",
        ]
        require_columns(
            silver_inventory,
            "silver_inventory",
            inventory_columns,
        )
        assert_no_nulls(
            silver_inventory,
            "silver_inventory",
            ["item_id", "store_id", "day"],
        )
        assert_unique(
            silver_inventory,
            "silver_inventory",
            ["item_id", "store_id", "day"],
        )

        inventory = silver_inventory.alias("inventory")
        calendar = dim_date.select("m5_day", "date_key").alias("calendar")
        product = dim_product.select("item_id", "product_key").alias("product")
        store = dim_store.select("store_id", "store_key").alias("store")

        joined = (
            inventory.join(
                calendar,
                F.col("inventory.day") == F.col("calendar.m5_day"),
                "left",
            )
            .join(
                product,
                F.col("inventory.item_id") == F.col("product.item_id"),
                "left",
            )
            .join(
                store,
                F.col("inventory.store_id") == F.col("store.store_id"),
                "left",
            )
        )
        result = joined.select(
            F.col("calendar.date_key").alias("date_key"),
            F.col("product.product_key").alias("product_key"),
            F.col("store.store_key").alias("store_key"),
            *[
                F.col(f"inventory.{column}").alias(column)
                for column in inventory_columns[3:]
            ],
        )
        assert_no_nulls(
            result,
            "fact_inventory_daily",
            ["date_key", "product_key", "store_key"],
        )
        assert_unique(
            result,
            "fact_inventory_daily",
            ["date_key", "product_key", "store_key"],
        )
        return apply_gold_schema(result, "fact_inventory_daily")
