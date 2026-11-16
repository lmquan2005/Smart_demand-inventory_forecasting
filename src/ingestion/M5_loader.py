
from pathlib import Path
from typing import Any, Mapping

from ..common.spark_manager import get_spark


class M5Loader:
    """Load raw M5 data and simulated inventory data with Spark."""

    _FILES = {
        "sales": "sales_train_evaluation.csv",
        "calendar": "calendar.csv",
        "prices": "sell_prices.csv",
        "inventory": "inventory_simulation.csv",
    }

    _BRONZE_DIRS = {
        "sales": "bronze_sales",
        "calendar": "bronze_calendar",
        "prices": "bronze_sell_prices",
        "inventory": "bronze_inventory",
    }

    def __init__(self, config: Any):
        self.spark = get_spark()
        self.config = config

        self.raw_data_path = Path(
            self._config_value(
                "raw_data_path",
                Path(__file__).resolve().parents[2] / "dataset" / "raw",
            )
        )

        self.simulated_data_path = Path(
            self._config_value(
                "simulated_data_path",
                Path(__file__).resolve().parents[2] / "dataset" / "simulated",
            )
        )

        self.bronze_data_path = Path(
            self._config_value(
                "bronze_data_path",
                Path(__file__).resolve().parents[2]
                / "dataset"
                / "processed"
                / "bronze",
            )
        )

    def _config_value(self, key: str, default: Any) -> Any:
        if isinstance(self.config, Mapping):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    def _load_csv(self, path: Path) -> Any:
        return (
            self.spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(str(path))
        )

    def load_sales(self) -> Any:
        path = self._config_value(
            "sales_path",
            self.raw_data_path / self._FILES["sales"],
        )
        return self._load_csv(Path(path))

    def load_calendar(self) -> Any:
        path = self._config_value(
            "calendar_path",
            self.raw_data_path / self._FILES["calendar"],
        )
        return self._load_csv(Path(path))

    def load_prices(self) -> Any:
        path = self._config_value(
            "prices_path",
            self.raw_data_path / self._FILES["prices"],
        )
        return self._load_csv(Path(path))

    def load_inventory(self) -> Any:
        path = self._config_value(
            "inventory_path",
            self.simulated_data_path / self._FILES["inventory"],
        )
        return self._load_csv(Path(path))

    def _write_delta(self, name: str, dataframe: Any) -> None:
        path = self._config_value(
            f"bronze_{name}_path",
            self.bronze_data_path / self._BRONZE_DIRS[name],
        )

        (
            dataframe.write
            .format("delta")
            .mode("overwrite")
            .save(str(path))
        )

    def write_sales(self, sales: Any) -> None:
        self._write_delta("sales", sales)

    def write_calendar(self, calendar: Any) -> None:
        self._write_delta("calendar", calendar)

    def write_prices(self, prices: Any) -> None:
        self._write_delta("prices", prices)

    def write_inventory(self, inventory: Any) -> None:
        self._write_delta("inventory", inventory)

    def run(self) -> dict[str, Any]:
        dataframes = {
            "sales": self.load_sales(),
            "calendar": self.load_calendar(),
            "prices": self.load_prices(),
            "inventory": self.load_inventory(),
        }

        self.write_sales(dataframes["sales"])
        self.write_calendar(dataframes["calendar"])
        self.write_prices(dataframes["prices"])
        self.write_inventory(dataframes["inventory"])

        return dataframes
