import numpy as np
import pandas as pd
import pytest

from edvart import data_types

from .pyarrow_utils import pyarrow_params


@pytest.mark.parametrize("pyarrow_dtypes", pyarrow_params)
@pytest.mark.parametrize(
    "data, expected",
    [
        (pd.Series([0.12, 3565.3, 234, 1, -49, 14, 5, 88, 12312]), data_types.DataType.NUMERIC),
        (
            pd.Series(["2014-01-01 12:05:02", "2014-01-02 13:05:02", "2014-12-03 14:05:02"]),
            data_types.DataType.DATE,
        ),
        (pd.Series(["A", "B", "C", "C", "A", "B"]), data_types.DataType.CATEGORICAL),
        (pd.Series([True, False, False, True, True]), data_types.DataType.BOOLEAN),
        (pd.Series([None, None, np.nan, float("nan")]), data_types.DataType.MISSING),
        (pd.Series(list(range(10))), data_types.DataType.UNIQUE),
        (pd.Series([1] + list(range(100))), data_types.DataType.NUMERIC),
        (pd.Series(dtype=pd.Float64Dtype), data_types.DataType.UNKNOWN),
        (pd.Series([True, False]), data_types.DataType.BOOLEAN),
    ],
)
def test_inference(data, expected, pyarrow_dtypes):
    if pyarrow_dtypes:
        data = data.convert_dtypes(dtype_backend="pyarrow")
    assert data_types.infer_data_type(data) == expected


@pytest.mark.parametrize("pyarrow_dtypes", pyarrow_params)
@pytest.mark.parametrize(
    "data, is_missing",
    [
        (pd.Series([None, None, np.nan, float("nan")]), True),
        (pd.Series([pd.NA]), True),
        (pd.Series([1, np.nan, None]), False),
        (pd.Series(["2023-01-01", None]), False),
    ],
)
def test_missing_series(data, is_missing, pyarrow_dtypes):
    if pyarrow_dtypes:
        data = data.convert_dtypes(dtype_backend="pyarrow")
    assert data_types.is_missing(data) == is_missing


@pytest.mark.parametrize("pyarrow_dtypes", pyarrow_params)
@pytest.mark.parametrize(
    "data, is_numeric",
    [
        (pd.Series([0.12, 3565.3, 234, 1, -49, 14, 5, 88, 12312]), True),
        (pd.Series([23, 45, 2, 1.2, -3, -66]), True),
        (pd.Series([23, 45, 2, 1, -3, -66, "NULL", "a string"]), False),
        (pd.Series([23, 45, 2, 1, -3, -66, "99", "-1207"]), False),
        (pd.Series([None, None, np.nan, float("nan")]), False),
    ],
)
def test_numeric_series(data, is_numeric, pyarrow_dtypes):
    if pyarrow_dtypes:
        data = data.convert_dtypes(dtype_backend="pyarrow")
    assert data_types.is_numeric(data) == is_numeric


@pytest.mark.parametrize("pyarrow_dtypes", pyarrow_params)
@pytest.mark.parametrize(
    "data, is_categorical",
    [
        (pd.Series(["A", "B", "C", "D"]), True),
        (pd.Series([1, 2, 3, 4, 4, 4, 1, 1, 1, 2, 2, 3, 4]), True),
        (
            pd.Series([1, 2, 31, 4, 52, 6, 87, 87.7, 9, 1, 3, 4, 1, 10, 123123, 9876, 1.2, 6.8]),
            False,
        ),
        (pd.Series([None, None, np.nan, float("nan")]), False),
        (pd.Series([pd.NA]), False),
    ],
)
def test_categorical_series(data, is_categorical, pyarrow_dtypes):
    if pyarrow_dtypes:
        data = data.convert_dtypes(dtype_backend="pyarrow")
    assert data_types.is_categorical(data) == is_categorical


@pytest.mark.parametrize("pyarrow_dtypes", pyarrow_params)
@pytest.mark.parametrize(
    "data, is_boolean",
    [
        (pd.Series([True, False, False, True, True]), True),
        (pd.Series([False, False, False]), True),
        (pd.Series([True, True, True]), True),
        (pd.Series([1, 0, 0, 1]), True),
        (pd.Series([0, 0, 0, 0]), True),
        (pd.Series([1, 1, 1, 1]), True),
        (pd.Series([True, False, False, True, True, "True"]), False),
        (pd.Series([2, 2, 2, 2]), False),
        (pd.Series([1, 0, 0, 1, 3]), False),
        (pd.Series(["a", "abc", "2"]), False),
        (pd.Series(["A", "B", "A", "A", "B"]), False),
        (pd.Series([-0.2, 1.6567, 3, 4, 5]), False),
        (pd.Series([None]), False),
    ],
)
def test_boolean_series(data, is_boolean, pyarrow_dtypes):
    if pyarrow_dtypes:
        data = data.convert_dtypes(dtype_backend="pyarrow")
    assert data_types.is_boolean(data) == is_boolean


@pytest.mark.parametrize("pyarrow_dtypes", pyarrow_params)
@pytest.mark.parametrize(
    "data, is_date",
    [
        (pd.Series(["2014-01-01 12:05:02", "2014-01-02 13:05:02", "2014-12-03 14:05:02"]), True),
        (pd.Series(["Mar 12 2018", "Dec 12 2018", "Jan 21 2020"]), True),
        (pd.Series(["2014-01-01", "2014-01-02", "2014-12-03T14:05:02", "nan"]), False),
        (pd.Series(["2014-01-01", "2014-01-02", "2014-12-03 14:05:02", 1, 2, 3]), False),
        (pd.Series([1, 2, 3, 4, 5]), False),
        (pd.Series([None, 2.0, 3, 4, 5]), False),
        (pd.Series([pd.Timestamp("20130101"), pd.Timestamp("20230102"), None]), True),
    ],
)
def test_date_series(data, is_date, pyarrow_dtypes):
    if pyarrow_dtypes:
        data = data.convert_dtypes(dtype_backend="pyarrow")
    assert data_types.is_date(data) == is_date
