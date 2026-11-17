from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType


class InventoryTransformer:

    def clean_string(self, df: DataFrame) -> DataFrame:

        for field in df.schema.fields:

            if isinstance(field.dataType, StringType):

                col_name = field.name

                df = df.withColumn(
                    col_name,
                    F.when(
                        F.trim(F.col(col_name)) == "",
                        F.lit(None).cast(StringType())
                    ).otherwise(
                        F.trim(F.col(col_name))
                    )
                )

        return df

    def transform(self, df: DataFrame) -> DataFrame:

        print("Transforming inventory data...")

        df = self.clean_string(df)

        required_columns = [
            "item_id",
            "store_id",
            "day",
        ]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        integer_columns = [
            "lead_time_days",
            "stockout_flag",
            "overstock_flag",
        ]

        for col_name in integer_columns:

            if col_name in df.columns:
                df = df.withColumn(
                    col_name,
                    F.col(col_name).cast(IntegerType())
                )

        double_columns = [
            "units_sold",
            "avg_daily_demand",
            "opening_stock",
            "replenishment_qty",
            "closing_stock",
            "reorder_point",
            "lost_sales",
        ]

        for col_name in double_columns:

            if col_name in df.columns:
                df = df.withColumn(
                    col_name,
                    F.col(col_name).cast(DoubleType())
                )

        for col_name in required_columns:
            df = df.filter(
                F.col(col_name).isNotNull()
            )

        return df