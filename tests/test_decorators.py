import pandas as pd
import pytest

from edvart.decorators import check_index_time_ascending
from edvart.example_datasets import dataset_pollution
from edvart.utils import reindex_to_datetime


@check_index_time_ascending
def dummy_function(df: pd.DataFrame) -> pd.DataFrame:
    return df


def test_no_datetime_index():
    data = dataset_pollution()
    with pytest.raises(ValueError):
        data_no_index = data.reset_index()
        dummy_function(data_no_index)


def test_not_ordered():
    # Randomly shuffle the data
    data_random_order = dataset_pollution().sample(frac=1)
    with pytest.raises(ValueError):
        dummy_function(data_random_order)

    data = data_random_order.sort_index()
    assert data.equals(dummy_function(data))


def test_period_index():
    data = pd.DataFrame(
        data=[
            [5, 4],
            [9, 3],
            [7, 9],
            [8, 10],
        ],
        columns=["time", "value"],
    )
    with pytest.raises(ValueError):
        dummy_function(data)
    data_period_index_not_sorted = reindex_to_datetime(
        data, datetime_column="time", unit="s", sort=False
    ).to_period(freq="s")
    with pytest.raises(ValueError):
        dummy_function(data)
    data_period_index_sorted = data_period_index_not_sorted.sort_index()
    assert data_period_index_sorted.equals(dummy_function(data_period_index_sorted))
