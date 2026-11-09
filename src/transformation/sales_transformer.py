
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType


class SalesTransformer:

    def clean_string(self, df: DataFrame) -> DataFrame:
        """Trim string columns and convert empty strings to NULL."""

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
        """Transform M5 sales from wide format to long format."""

        print("Transforming sales data...")

        # 1. Identify daily sales columns
        sales_columns = sorted(
            [
                col for col in df.columns
                if col.startswith("d_")
                and col[2:].isdigit()
            ],
            key=lambda col: int(col[2:])
        )

        if not sales_columns:
            raise ValueError("No daily sales columns found.")

        # 2. Identify metadata columns
        id_columns = [
            col for col in df.columns
            if col not in sales_columns
        ]

        # 3. Clean strings
        df = self.clean_string(df)

        # 4. Wide -> Long
        result = df.unpivot(
            ids=id_columns,
            values=sales_columns,
            variableColumnName="d",
            valueColumnName="sales"
        )

        # 5. Cast types
        result = (
            result
            .withColumn("d", F.col("d").cast(StringType()))
            .withColumn("sales", F.col("sales").cast(IntegerType()))
        )

        # 6. Validate required identifiers
        required_columns = [
            "id",
            "item_id",
            "store_id",
            "d"
        ]

        for col_name in required_columns:

            if col_name in result.columns:
                result = result.filter(
                    F.col(col_name).isNotNull()
                )

        # 7. Select output columns
        output_columns = [
            "id",
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
            "d",
            "sales"
        ]

        result = result.select(
            *[
                col for col in output_columns
                if col in result.columns
            ]
        )

        return result
