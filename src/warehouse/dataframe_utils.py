from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def require_columns(dataframe: DataFrame, table_name: str, columns: list) -> None:
    missing_columns = sorted(set(columns) - set(dataframe.columns))
    if missing_columns:
        raise ValueError(
            f"{table_name} is missing required columns: {missing_columns}"
        )


def assert_no_nulls(
    dataframe: DataFrame,
    table_name: str,
    columns: list,
) -> None:
    condition = None
    for column in columns:
        column_is_null = F.col(column).isNull()
        condition = (
            column_is_null
            if condition is None
            else condition | column_is_null
        )

    if dataframe.filter(condition).limit(1).count():
        raise ValueError(
            f"{table_name} contains NULL values in key columns {columns}"
        )


def assert_unique(
    dataframe: DataFrame,
    table_name: str,
    columns: list,
) -> None:
    duplicate = (
        dataframe.groupBy(*columns)
        .count()
        .filter(F.col("count") > 1)
        .limit(1)
        .count()
    )
    if duplicate:
        raise ValueError(
            f"{table_name} contains duplicate rows for key {columns}"
        )
