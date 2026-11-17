import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SILVER_PATH = PROJECT_ROOT / "dataset" / "processed" / "silver"
REPORT_PATH = PROJECT_ROOT / "reports" / "silver_schema_analysis.csv"

TABLE_CONFIG = {
    "silver_sales": {
        "warehouse_table_type": "fact",
        "grain": "one row per product/store/day",
        "business_keys": ["id", "d"],
    },
    "silver_calendar": {
        "warehouse_table_type": "dimension",
        "grain": "one row per M5 day",
        "business_keys": ["d"],
    },
    "silver_sell_prices": {
        "warehouse_table_type": "fact",
        "grain": "one row per product/store/week",
        "business_keys": ["store_id", "item_id", "wm_yr_wk"],
    },
    "silver_inventory": {
        "warehouse_table_type": "fact",
        "grain": "one row per product/store/day",
        "business_keys": ["item_id", "store_id", "day"],
    },
}

DIMENSION_KEY_COLUMNS = {
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "d",
    "date",
    "wm_yr_wk",
}

MEASURE_COLUMNS = {
    "sales",
    "sell_price",
    "units_sold",
    "avg_daily_demand",
    "lead_time_days",
    "opening_stock",
    "replenishment_qty",
    "closing_stock",
    "reorder_point",
    "lost_sales",
}

INDICATOR_COLUMNS = {
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "stockout_flag",
    "overstock_flag",
}

REPORT_COLUMNS = [
    "table_name",
    "warehouse_table_type",
    "grain",
    "column_name",
    "spark_data_type",
    "nullable_in_schema",
    "warehouse_role",
    "candidate_business_key",
    "candidate_key_columns",
    "silver_path",
]


class SilverSchemaAnalyzer:

    def __init__(
        self,
        spark,
        silver_path: Path = SILVER_PATH,
        report_path: Path = REPORT_PATH,
    ):
        self.spark = spark
        self.silver_path = Path(silver_path)
        self.report_path = Path(report_path)

    @staticmethod
    def _warehouse_role(column_name: str, business_keys: set) -> str:
        if column_name in business_keys:
            return "business_key"
        if column_name in INDICATOR_COLUMNS:
            return "business_indicator"
        if column_name in DIMENSION_KEY_COLUMNS:
            return "dimension_key_candidate"
        if column_name in MEASURE_COLUMNS:
            return "measure"
        return "descriptive_attribute"

    def analyze(self) -> list:
        report_rows = []

        for table_name, config in TABLE_CONFIG.items():
            table_path = self.silver_path / table_name

            if not table_path.exists():
                raise FileNotFoundError(table_path)

            dataframe = (
                self.spark.read
                .format("delta")
                .load(str(table_path))
            )
            columns = set(dataframe.columns)
            business_keys = set(config["business_keys"])
            missing_keys = business_keys - columns

            if missing_keys:
                raise ValueError(
                    f"{table_name} is missing candidate key columns: "
                    f"{sorted(missing_keys)}"
                )

            for field in dataframe.schema.fields:
                report_rows.append({
                    "table_name": table_name,
                    "warehouse_table_type": config["warehouse_table_type"],
                    "grain": config["grain"],
                    "column_name": field.name,
                    "spark_data_type": field.dataType.simpleString(),
                    "nullable_in_schema": field.nullable,
                    "warehouse_role": self._warehouse_role(
                        field.name,
                        business_keys,
                    ),
                    "candidate_business_key": (
                        field.name in business_keys
                    ),
                    "candidate_key_columns": ", ".join(
                        config["business_keys"]
                    ),
                    "silver_path": str(table_path),
                })

        return report_rows

    def write_report(self, report_rows: list) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

        with self.report_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as report_file:
            writer = csv.DictWriter(
                report_file,
                fieldnames=REPORT_COLUMNS,
            )
            writer.writeheader()
            writer.writerows(report_rows)

        print(f"Silver schema analysis written to: {self.report_path}")

    def run(self) -> list:
        report_rows = self.analyze()
        self.write_report(report_rows)
        return report_rows


def main():
    from src.common.spark_manager import get_spark

    spark = get_spark()

    try:
        SilverSchemaAnalyzer(spark).run()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
