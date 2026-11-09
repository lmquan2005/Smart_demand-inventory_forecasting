
import pytest

from src.common.spark_manager import get_spark


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for the entire test session."""

    session = get_spark()

    yield session

    session.stop()
