
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, DoubleType


class SellPricesTransformer:

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

        print("Transforming sell prices...")

        # 1. Clean strings
        df = self.clean_string(df)

        # 2. Validate required columns
        required_columns = [
            "store_id",
            "item_id",
            "wm_yr_wk",
            "sell_price"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        # 3. Cast types
        df = (
            df
            .withColumn(
                "wm_yr_wk",
                F.col("wm_yr_wk").cast(IntegerType())
            )
            .withColumn(
                "sell_price",
                F.col("sell_price").cast(DoubleType())
            )
        )

        # 4. Remove records with missing business keys
        key_columns = [
            "store_id",
            "item_id",
            "wm_yr_wk"
        ]

        for col_name in key_columns:

            df = df.filter(
                F.col(col_name).isNotNull()
            )

        return df
