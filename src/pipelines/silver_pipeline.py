
from pathlib import Path

from src.common.spark_manager import get_spark

from src.transformation.sales_transformer import SalesTransformer
from src.transformation.calendar_transformer import CalendarTransformer
from src.transformation.price_transformer import SellPricesTransformer
from src.transformation.inventory_transformer import InventoryTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BRONZE_PATH = PROJECT_ROOT / "dataset" / "processed" / "bronze"
SILVER_PATH = PROJECT_ROOT / "dataset" / "processed" / "silver"


class SilverPipeline:

    def __init__(self, spark):

        self.spark = spark

        self.sales_transformer = SalesTransformer()
        self.calendar_transformer = CalendarTransformer()
        self.prices_transformer = SellPricesTransformer()
        self.inventory_transformer = InventoryTransformer()

    def read_bronze(self, table_name):

        path = BRONZE_PATH / table_name

        if not path.exists():
            raise FileNotFoundError(path)

        return (
            self.spark.read
            .format("delta")
            .load(str(path))
        )

    def write_silver(self, df, table_name):

        path = SILVER_PATH / table_name

        (
            df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .save(str(path))
        )

        print(f"Written: {table_name}")

    def run(self):

        SILVER_PATH.mkdir(parents=True, exist_ok=True)

        print("\n========== SILVER PIPELINE START ==========")

        # SALES
        bronze_sales = self.read_bronze("bronze_sales")

        silver_sales = self.sales_transformer.transform(
            bronze_sales
        )

        self.write_silver(
            silver_sales,
            "silver_sales"
        )

        # CALENDAR
        bronze_calendar = self.read_bronze("bronze_calendar")

        silver_calendar = self.calendar_transformer.transform(
            bronze_calendar
        )

        self.write_silver(
            silver_calendar,
            "silver_calendar"
        )

        # SELL PRICES
        bronze_prices = self.read_bronze("bronze_sell_prices")

        silver_prices = self.prices_transformer.transform(
            bronze_prices
        )

        self.write_silver(
            silver_prices,
            "silver_sell_prices"
        )

        # INVENTORY
        bronze_inventory = self.read_bronze("bronze_inventory")

        silver_inventory = self.inventory_transformer.transform(
            bronze_inventory
        )

        self.write_silver(
            silver_inventory,
            "silver_inventory"
        )

        print("\n========== SILVER PIPELINE COMPLETED ==========")


def main():

    spark = get_spark()

    try:
        pipeline = SilverPipeline(spark)
        pipeline.run()

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
