import numpy as np
import pandas as pd

from edvart import data_types


def test_inference():
    assert (
        data_types.infer_data_type(pd.Series([0.12, 3565.3, 234, 1, -49, 14, 5, 88, 12312]))
        == data_types.DataType.NUMERIC
    ), "Should be numeric type"
    assert (
        data_types.infer_data_type(
            pd.Series(["2014-01-01 12:05:02", "2014-01-02 13:05:02", "2014-12-03 14:05:02"])
        )
        == data_types.DataType.DATE
    ), "Should be date type"
    assert (
        data_types.infer_data_type(pd.Series(["A", "B", "C", "C", "A", "B"]))
        == data_types.DataType.CATEGORICAL
    ), "Should be categorical type"
    assert (
        data_types.infer_data_type(pd.Series([True, False, False, True, True]))
        == data_types.DataType.BOOLEAN
    ), "Should be boolean type"
    assert data_types.infer_data_type(
        pd.Series([None, None, np.nan, float("nan")]) == data_types.DataType.MISSING
    ), "Should be missing"
    assert (
        data_types.infer_data_type(pd.Series(list(range(10)))) == data_types.DataType.UNIQUE
    ), "Should be unique"
    assert (
        data_types.infer_data_type(pd.Series([1] + list(range(100)))) == data_types.DataType.NUMERIC
    ), "Should be numeric"
    assert (
        data_types.infer_data_type(pd.Series(dtype=pd.Float64Dtype)) == data_types.DataType.UNKNOWN
    ), "Should be unknown"
    assert data_types.infer_data_type(
        pd.Series([True, False]) == data_types.DataType.BOOLEAN
    ), "Should be boolean"


def test_missing_series():
    assert data_types.is_missing(pd.Series([None, None, np.nan, float("nan")])), "Should be missing"
    assert data_types.is_missing(pd.Series([pd.NA])), "Should be missing"
    assert not data_types.is_missing(pd.Series([1, np.nan, None])), "Should not be missing"
    assert not data_types.is_missing(pd.Series(["2023-01-01", None])), "Should not be missing"


def test_numeric_series():
    assert data_types.is_numeric(
        pd.Series([0.12, 3565.3, 234, 1, -49, 14, 5, 88, 12312])
    ), "Should be numeric type"
    assert data_types.is_numeric(pd.Series([23, 45, 2, 1.2, -3, -66])), "Should be numeric type"
    assert not data_types.is_numeric(
        pd.Series([23, 45, 2, 1, -3, -66, "NULL", "a string"])
    ), "Should not be numeric type"
    assert not data_types.is_numeric(
        pd.Series([23, 45, 2, 1, -3, -66, "99", "-1207"])
    ), "Should not be numeric type"
    assert not data_types.is_numeric(
        pd.Series([None, None, np.nan, float("nan")])
    ), "Should not be numeric"


def test_categorical_series():
    assert data_types.is_categorical(pd.Series(["A", "B", "C", "D"])), "Should be categorical"
    assert data_types.is_categorical(
        pd.Series([1, 2, 3, 4, 4, 4, 1, 1, 1, 2, 2, 3, 4])
    ), "Should be categorical"
    assert not data_types.is_categorical(
        pd.Series([1, 2, 31, 4, 52, 6, 87, 87.7, 9, 1, 3, 4, 1, 10, 123123, 9876, 1.2, 6.8])
    ), "Should not be categorical"
    assert not data_types.is_categorical(
        pd.Series([None, None, np.nan, float("nan")])
    ), "Should not be categorical"
    assert not data_types.is_categorical(pd.Series([pd.NA])), "Should not be categorical"


def test_boolean_series():
    assert data_types.is_boolean(pd.Series([True, False, False, True, True])), "Should be boolean"
    assert data_types.is_boolean(pd.Series([False, False, False])), "Should be boolean"
    assert data_types.is_boolean(pd.Series([True, True, True])), "Should be boolean"
    assert data_types.is_boolean(pd.Series([1, 0, 0, 1])), "Should be boolean"
    assert data_types.is_boolean(pd.Series([0, 0, 0, 0])), "Should be boolean"
    assert data_types.is_boolean(pd.Series([1, 1, 1, 1])), "Should be boolean"
    assert not data_types.is_boolean(
        pd.Series([True, False, False, True, True, "True"])
    ), "Should not be boolean"
    assert not data_types.is_boolean(pd.Series([2, 2, 2, 2])), "Should not be boolean"
    assert not data_types.is_boolean(pd.Series([1, 0, 0, 1, 3])), "Should not be boolean"
    assert not data_types.is_boolean(pd.Series(["a", "abc", "2"])), "Should not be boolean"
    assert not data_types.is_boolean(pd.Series(["A", "B", "A", "A", "B"])), "Should not be boolean"
    assert not data_types.is_boolean(pd.Series([-0.2, 1.6567, 3, 4, 5])), "Should not be boolean"
    assert not data_types.is_boolean(pd.Series([None])), "Should not be boolean"


def test_date_series():
    assert data_types.is_date(
        pd.Series(["2014-01-01 12:05:02", "2014-01-02 13:05:02", "2014-12-03 14:05:02"])
    ), "Should be type date"
    assert data_types.is_date(
        pd.Series(["Mar 12 2018", "Dec 12 2018", "Jan 21 2020"])
    ), "Should be type date"
    assert not data_types.is_date(
        pd.Series(["2014-01-01", "2014-01-02", "2014-12-03T14:05:02", "nan"])
    ), "Should not be type date"
    assert not data_types.is_date(
        pd.Series(["2014-01-01", "2014-01-02", "2014-12-03 14:05:02", 1, 2, 3])
    ), "Should not be type date"
    assert not data_types.is_date(pd.Series([1, 2, 3, 4, 5])), "Should not be type date"
    assert not data_types.is_date(pd.Series([None, 2.0, 3, 4, 5])), "Should not be type date"
