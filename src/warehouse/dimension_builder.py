from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from .dataframe_utils import assert_no_nulls, assert_unique, require_columns
from .schema import apply_gold_schema


class DimensionBuilder:
    def build_dim_date(self, silver_calendar: DataFrame) -> DataFrame:
        columns = [
            "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2",
            "snap_CA", "snap_TX", "snap_WI",
        ]
        require_columns(silver_calendar, "silver_calendar", columns)
        assert_no_nulls(silver_calendar, "silver_calendar", ["d", "date"])

        dates = silver_calendar.select(
            F.col("d").alias("m5_day"),
            F.date_format("date", "yyyyMMdd").cast("int").alias("date_key"),
            F.col("date").alias("full_date"),
            F.col("wm_yr_wk"),
            F.col("weekday"),
            F.col("wday").alias("day_of_week"),
            F.col("month"),
            F.col("year"),
            F.col("event_name_1"),
            F.col("event_type_1"),
            F.col("event_name_2"),
            F.col("event_type_2"),
            F.col("snap_CA").alias("snap_ca"),
            F.col("snap_TX").alias("snap_tx"),
            F.col("snap_WI").alias("snap_wi"),
        )
        assert_unique(dates, "dim_date", ["m5_day"])
        assert_unique(dates, "dim_date", ["date_key"])
        return apply_gold_schema(dates, "dim_date")

    @staticmethod
    def _build_surrogate_dimension(
        dataframe: DataFrame,
        table_name: str,
        key_column: str,
        attributes: list,
        surrogate_key: str,
    ) -> DataFrame:
        assert_no_nulls(dataframe, table_name, [key_column])

        unique_rows = dataframe.select(key_column, *attributes).distinct()
        conflicts = (
            unique_rows.groupBy(key_column)
            .count()
            .filter(F.col("count") > 1)
            .limit(1)
            .count()
        )
        if conflicts:
            raise ValueError(
                f"{table_name} has conflicting attributes for {key_column}"
            )

        window = Window.orderBy(F.col(key_column))
        dimension = unique_rows.withColumn(
            surrogate_key,
            F.row_number().over(window).cast("long"),
        ).select(surrogate_key, key_column, *attributes)

        assert_unique(dimension, table_name, [surrogate_key])
        assert_unique(dimension, table_name, [key_column])
        return dimension

    def build_dim_product(
        self,
        silver_sales: DataFrame,
        silver_inventory: DataFrame,
    ) -> DataFrame:
        attributes = ["dept_id", "cat_id"]
        required = ["item_id", *attributes]
        require_columns(silver_sales, "silver_sales", required)
        require_columns(silver_inventory, "silver_inventory", required)

        products = (
            silver_sales.select(*required)
            .unionByName(silver_inventory.select(*required))
        )
        dimension = self._build_surrogate_dimension(
            products,
            "dim_product",
            "item_id",
            attributes,
            "product_key",
        )
        return apply_gold_schema(dimension, "dim_product")

    def build_dim_store(
        self,
        silver_sales: DataFrame,
        silver_inventory: DataFrame,
    ) -> DataFrame:
        attributes = ["state_id"]
        required = ["store_id", *attributes]
        require_columns(silver_sales, "silver_sales", required)
        require_columns(silver_inventory, "silver_inventory", required)

        stores = (
            silver_sales.select(*required)
            .unionByName(silver_inventory.select(*required))
        )
        dimension = self._build_surrogate_dimension(
            stores,
            "dim_store",
            "store_id",
            attributes,
            "store_key",
        )
        return apply_gold_schema(dimension, "dim_store")