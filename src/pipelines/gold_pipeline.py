from pathlib import Path

from src.common.spark_manager import get_spark
from src.warehouse.dimension_builder import DimensionBuilder
from src.warehouse.fact_inventory_builder import FactInventoryBuilder
from src.warehouse.fact_sales_builder import FactSalesBuilder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SILVER_PATH = PROJECT_ROOT / "dataset" / "processed" / "silver"
GOLD_PATH = PROJECT_ROOT / "dataset" / "processed" / "gold"


class GoldPipeline:
    def __init__(
        self,
        spark,
        silver_path: Path = SILVER_PATH,
        gold_path: Path = GOLD_PATH,
    ):
        self.spark = spark
        self.silver_path = Path(silver_path)
        self.gold_path = Path(gold_path)
        self.dimension_builder = DimensionBuilder()
        self.sales_builder = FactSalesBuilder()
        self.inventory_builder = FactInventoryBuilder()

    def read_silver(self, table_name: str):
        path = self.silver_path / table_name
        if not path.exists():
            raise FileNotFoundError(path)
        return self.spark.read.format("delta").load(str(path))

    def write_gold(self, dataframe, table_name: str) -> None:
        path = self.gold_path / table_name
        (
            dataframe.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .save(str(path))
        )
        print(f"Written Gold table: {table_name}")

    def run(self) -> dict:
        self.gold_path.mkdir(parents=True, exist_ok=True)

        silver_sales = self.read_silver("silver_sales")
        silver_calendar = self.read_silver("silver_calendar")
        silver_prices = self.read_silver("silver_sell_prices")
        silver_inventory = self.read_silver("silver_inventory")

        dimensions = {
            "dim_date": self.dimension_builder.build_dim_date(
                silver_calendar
            ),
            "dim_product": self.dimension_builder.build_dim_product(
                silver_sales,
                silver_inventory,
            ),
            "dim_store": self.dimension_builder.build_dim_store(
                silver_sales,
                silver_inventory,
            ),
        }
        facts = {
            "fact_sales_daily": self.sales_builder.build(
                silver_sales,
                silver_prices,
                dimensions["dim_date"],
                dimensions["dim_product"],
                dimensions["dim_store"],
            ),
            "fact_inventory_daily": self.inventory_builder.build(
                silver_inventory,
                dimensions["dim_date"],
                dimensions["dim_product"],
                dimensions["dim_store"],
            ),
        }

        for table_name, dataframe in dimensions.items():
            self.write_gold(dataframe, table_name)
        for table_name, dataframe in facts.items():
            self.write_gold(dataframe, table_name)

        return {**dimensions, **facts}


def main() -> None:
    spark = get_spark()
    try:
        GoldPipeline(spark).run()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()