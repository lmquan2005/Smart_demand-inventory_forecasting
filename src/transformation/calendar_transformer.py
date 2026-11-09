
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType


class CalendarTransformer:

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

        print("Transforming calendar data...")

        # 1. Clean strings
        df = self.clean_string(df)

        # 2. Cast integer columns
        integer_columns = [
            "wm_yr_wk",
            "wday",
            "month",
            "year",
            "snap_CA",
            "snap_TX",
            "snap_WI"
        ]

        for col_name in integer_columns:

            if col_name in df.columns:

                df = df.withColumn(
                    col_name,
                    F.col(col_name).cast(IntegerType())
                )

        # 3. Cast date
        if "date" in df.columns:

            df = df.withColumn(
                "date",
                F.to_date(F.col("date"), "yyyy-MM-dd")
            )

        # 4. Validate day identifier
        if "d" not in df.columns:
            raise ValueError("Calendar column 'd' is missing.")

        df = df.filter(
            F.col("d").isNotNull()
        )

        return df
