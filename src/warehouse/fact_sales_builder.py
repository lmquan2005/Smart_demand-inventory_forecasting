from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .dataframe_utils import (
    assert_no_nulls,
    assert_unique,
    require_columns,
)
from .schema import apply_gold_schema


class FactSalesBuilder:
    def build(
        self,
        silver_sales: DataFrame,
        silver_sell_prices: DataFrame,
        dim_date: DataFrame,
        dim_product: DataFrame,
        dim_store: DataFrame,
    ) -> DataFrame:
        require_columns(
            silver_sales,
            "silver_sales",
            ["item_id", "store_id", "d", "sales"],
        )
        require_columns(
            silver_sell_prices,
            "silver_sell_prices",
            ["item_id", "store_id", "wm_yr_wk", "sell_price"],
        )
        assert_no_nulls(
            silver_sales,
            "silver_sales",
            ["item_id", "store_id", "d"],
        )
        assert_unique(
            silver_sales,
            "silver_sales",
            ["item_id", "store_id", "d"],
        )
        assert_unique(
            silver_sell_prices,
            "silver_sell_prices",
            ["item_id", "store_id", "wm_yr_wk"],
        )

        sales = silver_sales.alias("sales")
        calendar = dim_date.select(
            "m5_day", "date_key", "wm_yr_wk"
        ).alias("calendar")
        product = dim_product.select("item_id", "product_key").alias("product")
        store = dim_store.select("store_id", "store_key").alias("store")
        prices = silver_sell_prices.alias("prices")

        joined = (
            sales.join(
                calendar,
                F.col("sales.d") == F.col("calendar.m5_day"),
                "left",
            )
            .join(
                product,
                F.col("sales.item_id") == F.col("product.item_id"),
                "left",
            )
            .join(
                store,
                F.col("sales.store_id") == F.col("store.store_id"),
                "left",
            )
            .join(
                prices,
                (F.col("sales.item_id") == F.col("prices.item_id"))
                & (F.col("sales.store_id") == F.col("prices.store_id"))
                & (F.col("calendar.wm_yr_wk") == F.col("prices.wm_yr_wk")),
                "left",
            )
        )

        result = joined.select(
            F.col("calendar.date_key").alias("date_key"),
            F.col("product.product_key").alias("product_key"),
            F.col("store.store_key").alias("store_key"),
            F.col("sales.sales").cast("long").alias("units_sold"),
            F.col("prices.sell_price").cast("double").alias("sell_price"),
            (
                F.col("sales.sales").cast("double")
                * F.col("prices.sell_price").cast("double")
            ).alias("revenue"),
        )
        assert_no_nulls(
            result,
            "fact_sales_daily",
            ["date_key", "product_key", "store_key"],
        )
        assert_unique(
            result,
            "fact_sales_daily",
            ["date_key", "product_key", "store_key"],
        )
        return apply_gold_schema(result, "fact_sales_daily")