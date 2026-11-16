
from pathlib import Path
import shutil

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from ..common.spark_manager import get_spark


class InventorySimulator:
    """
    Simulate daily inventory based on M5 historical sales.

    Input:
        sales_train_evaluation.csv

    Output:
        dataset/simulated/inventory_simulation.csv
    """

    def __init__(
        self,
        sales_path=None,
        output_path=None,
        random_seed=42,
    ):
        self.spark = get_spark()

        self.project_root = Path(__file__).resolve().parents[2]

        self.sales_path = Path(
            sales_path or (
                self.project_root
                / "dataset"
                / "raw"
                / "sales_train_evaluation.csv"
            )
        )

        self.output_path = Path(
            output_path or (
                self.project_root
                / "dataset"
                / "simulated"
                / "inventory_simulation.csv"
            )
        )

        self.random_seed = random_seed

        self.review_period = 7
        self.overstock_coverage_days = 21

    def load_sales(self) -> DataFrame:
        """Read original M5 sales data."""

        if not self.sales_path.exists():
            raise FileNotFoundError(
                f"Sales dataset not found: {self.sales_path}"
            )

        return (
            self.spark.read
            .option("header", True)
            .option("inferSchema", False)
            .csv(str(self.sales_path))
        )

    def transform_sales(self, sales: DataFrame) -> DataFrame:
        """
        Convert M5 wide sales format into long format.

        Each row represents one product-store-day.
        """

        day_columns = sorted(
            [
                col for col in sales.columns
                if col.startswith("d_")
                and col[2:].isdigit()
            ],
            key=lambda col: int(col[2:])
        )

        if not day_columns:
            raise ValueError("No M5 day columns found.")

        selected_days = day_columns

        metadata = [
            col for col in [
                "item_id",
                "dept_id",
                "cat_id",
                "store_id",
                "state_id",
            ]
            if col in sales.columns
        ]

        # Build Spark stack expression dynamically.
        stack_args = []

        for col in selected_days:
            stack_args.extend([
                f"'{col}'",
                f"CAST(`{col}` AS DOUBLE)"
            ])

        stack_expr = (
            f"stack({len(selected_days)}, "
            f"{', '.join(stack_args)}) "
            "AS (day, units_sold)"
        )

        long_sales = sales.selectExpr(
            *metadata,
            stack_expr
        )

        return (
            long_sales
            .withColumn(
                "day_idx",
                F.regexp_extract("day", r"d_(\d+)", 1).cast("int")
            )
            .withColumn(
                "units_sold",
                F.coalesce(F.col("units_sold"), F.lit(0.0))
            )
            .filter(F.col("units_sold") >= 0)
        )

    def calculate_demand(self, sales: DataFrame) -> DataFrame:
        """Calculate average daily demand per product and store."""

        return sales.groupBy(
            "item_id", "store_id"
        ).agg(
            F.avg("units_sold").alias("avg_daily_demand")
        )

    def simulate_inventory(self, sales: DataFrame) -> DataFrame:
        """Generate inventory movements for each product and store."""

        demand = self.calculate_demand(sales)

        df = sales.join(
            demand,
            on=["item_id", "store_id"],
            how="left"
        )

        # Generate deterministic simulation parameters.
        # These values remain consistent for each product-store pair.

        product_hash = F.xxhash64(
            F.col("item_id"),
            F.col("store_id"),
            F.lit(self.random_seed)
        )

        df = (
            df
            .withColumn(
                "lead_time_days",
                (F.pmod(product_hash, F.lit(6)) + 2).cast("int")
            )
            .withColumn(
                "safety_stock_days",
                (
                    F.pmod(product_hash, F.lit(5)) + 3
                ).cast("int")
            )
            .withColumn(
                "initial_stock",
                F.ceil(
                    F.col("avg_daily_demand")
                    * (
                        F.col("lead_time_days")
                        + F.col("safety_stock_days")
                    )
                    * (
                        F.lit(0.8)
                        + F.pmod(product_hash, F.lit(5)) * F.lit(0.1)
                    )
                ).cast("double")
            )
        )

        # Determine replenishment schedule.
        first_day = min(
            int(row["day_idx"])
            for row in df.select("day_idx").distinct().collect()
        )

        df = df.withColumn(
            "day_seq",
            F.col("day_idx") - F.lit(first_day)
        )

        df = df.withColumn(
            "replenishment_qty",
            F.when(
                (F.col("day_seq") > 0)
                & (
                    F.pmod(
                        F.col("day_seq"),
                        F.lit(self.review_period)
                    ) == 0
                ),
                F.ceil(
                    F.col("avg_daily_demand")
                    * F.lit(self.review_period)
                )
            ).otherwise(F.lit(0))
        )

        # Calculate cumulative sales and replenishment
        # before the current day.

        window = (
            Window
            .partitionBy("item_id", "store_id")
            .orderBy("day_idx")
            .rowsBetween(
                Window.unboundedPreceding,
                -1
            )
        )

        df = (
            df
            .withColumn(
                "previous_sales",
                F.coalesce(
                    F.sum("units_sold").over(window),
                    F.lit(0.0)
                )
            )
            .withColumn(
                "previous_replenishment",
                F.coalesce(
                    F.sum("replenishment_qty").over(window),
                    F.lit(0.0)
                )
            )
        )

        # Opening stock = initial stock
        # + previous replenishment - previous demand.

        df = df.withColumn(
            "opening_stock",
            F.greatest(
                F.col("initial_stock")
                + F.col("previous_replenishment")
                - F.col("previous_sales"),
                F.lit(0.0)
            )
        )

        # Available inventory during the day.

        df = df.withColumn(
            "available_stock",
            F.col("opening_stock")
            + F.col("replenishment_qty")
        )

        # Calculate lost sales and closing inventory.

        df = (
            df
            .withColumn(
                "lost_sales",
                F.greatest(
                    F.col("units_sold")
                    - F.col("available_stock"),
                    F.lit(0.0)
                )
            )
            .withColumn(
                "closing_stock",
                F.greatest(
                    F.col("available_stock")
                    - F.col("units_sold"),
                    F.lit(0.0)
                )
            )
        )

        # Reorder point = expected demand during lead time.

        df = df.withColumn(
            "reorder_point",
            F.ceil(
                F.col("avg_daily_demand")
                * F.col("lead_time_days")
            )
        )

        # Inventory status.

        df = (
            df
            .withColumn(
                "stockout_flag",
                F.when(F.col("lost_sales") > 0, 1).otherwise(0)
            )
            .withColumn(
                "overstock_flag",
                F.when(
                    F.col("closing_stock")
                    > (
                        F.col("avg_daily_demand")
                        * self.overstock_coverage_days
                    ),
                    1
                ).otherwise(0)
            )
        )

        return df.select(
            *[
                F.col(c)
                for c in [
                    "item_id",
                    "dept_id",
                    "cat_id",
                    "store_id",
                    "state_id",
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
                if c in df.columns
            ]
        )

    def save(self, dataframe):
        output_path = (
            self.project_root
            / "dataset"
            / "simulated"
            / "inventory_simulation"
        )

        (
            dataframe.write
            .mode("overwrite")
            .option("header", "true")
            .csv(str(output_path))
        )

        print(f"Inventory data saved to: {output_path}")
        
    def run(self) -> DataFrame:
        """Execute the complete inventory simulation pipeline."""

        print("=" * 60)
        print("START INVENTORY SIMULATION")
        print("=" * 60)

        sales = self.load_sales()

        print("[1/4] Sales dataset loaded.")

        transformed = self.transform_sales(sales)

        print("[2/4] Sales transformed into long format.")

        inventory = self.simulate_inventory(transformed)

        print("[3/4] Inventory simulation completed.")

        self.save(inventory)

        print("[4/4] Inventory CSV saved successfully.")

        return inventory
