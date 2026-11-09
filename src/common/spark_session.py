
from pathlib import Path

from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip


class SparkSessionFactory:
    """
    Factory responsible for creating a SparkSession
    integrated with Delta Lake.
    """

    def __init__(
        self,
        app_name="M5-Supply-Chain-Analytics",
        master="local[*]",
        warehouse_dir="data/spark-warehouse",
        shuffle_partitions=32
    ):
        self.app_name = app_name
        self.master = master
        self.warehouse_dir = str(
            Path(warehouse_dir).resolve()
        )
        self.shuffle_partitions = shuffle_partitions

    def create_spark_session(self) -> SparkSession:

        builder = (
            SparkSession.builder
            .appName(self.app_name)
            .master(self.master)

            # Enable Delta Lake integration
            .config(
                "spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension"
            )
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog"
            )

            # Spark SQL configuration
            .config(
                "spark.sql.shuffle.partitions",
                str(self.shuffle_partitions)
            )
            .config(
                "spark.sql.adaptive.enabled",
                "true"
            )

            # Timezone
            .config(
                "spark.sql.session.timeZone",
                "UTC"
            )

            # Spark warehouse
            .config(
                "spark.sql.warehouse.dir",
                self.warehouse_dir
            )

            # Local development
            .config(
                "spark.driver.memory",
                "4g"
            )
        )

        # Automatically configure Delta Lake dependencies
        builder = configure_spark_with_delta_pip(builder)

        spark = builder.getOrCreate()

        spark.sparkContext.setLogLevel("WARN")

        print("=" * 50)
        print("SPARK SESSION CREATED")
        print("=" * 50)
        print("Application:", self.app_name)
        print("Spark Version:", spark.version)
        print("Master:", spark.sparkContext.master)
        print("Warehouse:", self.warehouse_dir)
        print("Shuffle Partitions:", self.shuffle_partitions)
        print("=" * 50)

        return spark

    @staticmethod
    def stop_spark_session(spark: SparkSession):
        if spark is not None:
            spark.stop()
            print("SparkSession stopped.")
