
from pathlib import Path
import csv

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StringType,
    NumericType,
)

from src.common.spark_manager import get_spark


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BRONZE_PATH = PROJECT_ROOT / "dataset" / "processed" / "bronze"
REPORT_PATH = PROJECT_ROOT / "reports" / "bronze_data_quality_report.csv"

# Business keys of M5 dataset
TABLE_CONFIG = {
    "bronze_sales": {
        "path": BRONZE_PATH / "bronze_sales",
        "key_columns": ["id"],
        "required_columns": [
            "id", "item_id", "dept_id",
            "cat_id", "store_id", "state_id"
        ],
    },

    "bronze_calendar": {
        "path": BRONZE_PATH / "bronze_calendar",
        "key_columns": ["d"],
        "required_columns": ["d", "date", "wm_yr_wk"],
    },

    "bronze_sell_prices": {
        "path": BRONZE_PATH / "bronze_sell_prices",
        "key_columns": ["store_id", "item_id", "wm_yr_wk"],
        "required_columns": [
            "store_id", "item_id", "wm_yr_wk", "sell_price"
        ],
    },
}

# Number of columns processed in one Spark aggregation
BATCH_SIZE = 100


# ============================================================
# DATA QUALITY REPORT
# ============================================================

class DataQualityReport:

    def __init__(self, spark):
        self.spark = spark
        self.results = []

    def add_result(
        self,
        table,
        check,
        column,
        value,
        status,
        details=""
    ):
        self.results.append({
            "table": table,
            "check": check,
            "column": column,
            "value": value,
            "status": status,
            "details": details,
        })

    # --------------------------------------------------------
    # 1. CHECK TABLE VOLUME
    # --------------------------------------------------------

    def check_volume(self, df, table_name):

        row_count = df.count()
        column_count = len(df.columns)

        self.add_result(
            table_name,
            "row_count",
            "*",
            row_count,
            "PASS" if row_count > 0 else "FAIL",
            "Total number of records"
        )

        self.add_result(
            table_name,
            "column_count",
            "*",
            column_count,
            "PASS" if column_count > 0 else "FAIL",
            "Total number of columns"
        )

        return row_count

    # --------------------------------------------------------
    # 2. CHECK SCHEMA
    # --------------------------------------------------------

    def check_schema(self, df, table_name, required_columns):

        actual_columns = set(df.columns)

        missing_columns = [
            col for col in required_columns
            if col not in actual_columns
        ]

        self.add_result(
            table_name,
            "required_columns",
            "*",
            len(missing_columns),
            "PASS" if not missing_columns else "FAIL",
            (
                "All required columns exist"
                if not missing_columns
                else f"Missing: {missing_columns}"
            )
        )

    # --------------------------------------------------------
    # 3. CHECK NULL AND EMPTY STRING
    # --------------------------------------------------------

    def check_completeness(self, df, table_name):

        schema = {
            field.name: field.dataType
            for field in df.schema.fields
        }

        columns = df.columns

        for start in range(0, len(columns), BATCH_SIZE):

            batch = columns[start:start + BATCH_SIZE]

            expressions = []

            for col in batch:

                expressions.append(
                    F.count(
                        F.when(F.col(col).isNull(), 1)
                    ).alias(f"null__{col}")
                )

                if isinstance(schema[col], StringType):

                    expressions.append(
                        F.count(
                            F.when(
                                F.trim(F.col(col)) == "",
                                1
                            )
                        ).alias(f"empty__{col}")
                    )

            metrics = df.agg(*expressions).first().asDict()

            for col in batch:

                null_count = metrics[f"null__{col}"]

                self.add_result(
                    table_name,
                    "null_count",
                    col,
                    null_count,
                    "PASS" if null_count == 0 else "WARN",
                    "Number of NULL values"
                )

                if isinstance(schema[col], StringType):

                    empty_count = metrics[f"empty__{col}"]

                    self.add_result(
                        table_name,
                        "empty_string_count",
                        col,
                        empty_count,
                        "PASS" if empty_count == 0 else "WARN",
                        "Number of empty strings"
                    )

    # --------------------------------------------------------
    # 4. CHECK DUPLICATE BUSINESS KEYS
    # --------------------------------------------------------

    def check_duplicates(
        self,
        df,
        table_name,
        key_columns
    ):

        missing = [
            col for col in key_columns
            if col not in df.columns
        ]

        if missing:
            self.add_result(
                table_name,
                "duplicate_key",
                ",".join(key_columns),
                "N/A",
                "FAIL",
                f"Missing key columns: {missing}"
            )
            return

        duplicate_count = (
            df.groupBy(*key_columns)
            .count()
            .filter(F.col("count") > 1)
            .count()
        )

        self.add_result(
            table_name,
            "duplicate_key",
            ",".join(key_columns),
            duplicate_count,
            "PASS" if duplicate_count == 0 else "FAIL",
            "Number of duplicated business keys"
        )

    # --------------------------------------------------------
    # 5. CHECK KEY NULL VALUES
    # --------------------------------------------------------

    def check_key_nulls(
        self,
        df,
        table_name,
        key_columns
    ):

        existing_keys = [
            col for col in key_columns
            if col in df.columns
        ]

        if not existing_keys:
            return

        expressions = [
            F.count(
                F.when(F.col(col).isNull(), 1)
            ).alias(col)
            for col in existing_keys
        ]

        metrics = df.agg(*expressions).first().asDict()

        for col in existing_keys:

            null_count = metrics[col]

            self.add_result(
                table_name,
                "key_null_count",
                col,
                null_count,
                "PASS" if null_count == 0 else "FAIL",
                "NULL values in business key"
            )

    # --------------------------------------------------------
    # 6. CHECK NEGATIVE VALUES
    # --------------------------------------------------------

    def check_negative_values(self, df, table_name):

        # Sales demand columns: d_1, d_2, ...
        if table_name == "bronze_sales":

            numeric_columns = [
                col for col in df.columns
                if col.startswith("d_")
                and isinstance(
                    df.schema[col].dataType,
                    NumericType
                )
            ]

        elif table_name == "bronze_sell_prices":

            numeric_columns = [
                "sell_price"
            ] if "sell_price" in df.columns else []

        else:
            numeric_columns = []

        for start in range(
            0,
            len(numeric_columns),
            BATCH_SIZE
        ):

            batch = numeric_columns[start:start + BATCH_SIZE]

            expressions = [
                F.count(
                    F.when(F.col(col) < 0, 1)
                ).alias(col)
                for col in batch
            ]

            metrics = df.agg(*expressions).first().asDict()

            for col in batch:

                negative_count = metrics[col]

                self.add_result(
                    table_name,
                    "negative_values",
                    col,
                    negative_count,
                    "PASS" if negative_count == 0 else "FAIL",
                    "Number of negative values"
                )

    # --------------------------------------------------------
    # 7. RUN ALL CHECKS
    # --------------------------------------------------------

    def run_table(self, table_name, config):

        table_path = config["path"]

        print("\n" + "=" * 80)
        print(f"DATA QUALITY CHECK: {table_name}")
        print("=" * 80)

        if not table_path.exists():

            self.add_result(
                table_name,
                "table_exists",
                "*",
                0,
                "FAIL",
                f"Path does not exist: {table_path}"
            )

            print("[FAILED] Table does not exist.")
            return

        try:

            df = (
                self.spark.read
                .format("delta")
                .load(str(table_path))
            )

            print("\n[SCHEMA]")
            df.printSchema()

            print("\n[DATA PREVIEW]")
            df.show(5, truncate=False)

            print("\n[RUNNING QUALITY CHECKS]")

            row_count = self.check_volume(df, table_name)

            self.check_schema(
                df,
                table_name,
                config["required_columns"]
            )

            self.check_completeness(df, table_name)

            self.check_key_nulls(
                df,
                table_name,
                config["key_columns"]
            )

            self.check_duplicates(
                df,
                table_name,
                config["key_columns"]
            )

            self.check_negative_values(df, table_name)

            print(f"\nTotal records: {row_count:,}")
            print("[COMPLETED] Quality checks finished.")

        except Exception as e:

            self.add_result(
                table_name,
                "execution",
                "*",
                "N/A",
                "FAIL",
                str(e)
            )

            print(f"[ERROR] {e}")

    # --------------------------------------------------------
    # 8. EXPORT CSV REPORT
    # --------------------------------------------------------

    def export_report(self):

        REPORT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        fieldnames = [
            "table",
            "check",
            "column",
            "value",
            "status",
            "details"
        ]

        with open(
            REPORT_PATH,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(self.results)

        print(f"\nReport exported: {REPORT_PATH}")

    # --------------------------------------------------------
    # 9. PRINT SUMMARY
    # --------------------------------------------------------

    def print_summary(self):

        print("\n" + "=" * 80)
        print("BRONZE DATA QUALITY SUMMARY")
        print("=" * 80)

        statuses = ["PASS", "WARN", "FAIL"]

        for status in statuses:

            count = sum(
                1 for result in self.results
                if result["status"] == status
            )

            print(f"{status}: {count}")

        print("-" * 80)

        for table_name in TABLE_CONFIG:

            table_results = [
                result for result in self.results
                if result["table"] == table_name
            ]

            failed = sum(
                r["status"] == "FAIL"
                for r in table_results
            )

            warnings = sum(
                r["status"] == "WARN"
                for r in table_results
            )

            print(
                f"{table_name}: "
                f"{failed} FAIL | {warnings} WARN"
            )

        print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    spark = get_spark()

    try:

        report = DataQualityReport(spark)

        for table_name, config in TABLE_CONFIG.items():

            report.run_table(
                table_name,
                config
            )

        report.print_summary()
        report.export_report()

    finally:
        spark.stop()


if __name__ == "__main__":
    main()

