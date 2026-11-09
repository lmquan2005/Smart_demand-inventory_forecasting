from src.common.spark_session import SparkSessionFactory


def get_spark():

    factory = SparkSessionFactory(
        app_name="M5-Supply-Chain",
        master="local[*]",
        warehouse_dir="data/spark-warehouse",
        shuffle_partitions=32
    )

    return factory.create_spark_session()
