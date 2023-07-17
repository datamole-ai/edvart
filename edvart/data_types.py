"""Module defines data types and helper function for recognizing them."""

# Standard imports
from enum import IntEnum
from typing import Union

# Third-party library imports
import numpy as np
import pandas as pd


class DataType(IntEnum):
    """Class describe possible data types."""

    NUMERIC = (1,)
    CATEGORICAL = (2,)
    BOOLEAN = (3,)
    DATE = (4,)
    UNKNOWN = 5

    def __str__(self):
        return self.name.lower()


def infer_data_type(series: pd.Series, string_representation: bool = False) -> Union[DataType, str]:
    """Infers the data type of the series passed in.

    Parameters
    ----------
    series : pd.Series
        Series from which to infer data type.
    string_representation : bool
        Whether to return the resulting data type as DataType enum value or string.

    Returns
    -------
    DataType : Union[DataType, str]
        Inferred custom edvart data type or its string representation.
    """
    ret = None

    if is_boolean(series):
        ret = DataType.BOOLEAN
    elif is_date(series):
        ret = DataType.DATE
    elif is_categorical(series):
        ret = DataType.CATEGORICAL
    elif is_numeric(series):
        ret = DataType.NUMERIC
    else:
        ret = DataType.UNKNOWN

    return str(ret) if string_representation else ret


def is_numeric(series: pd.Series) -> bool:
    """
    Heuristic to tell if a series contains numbers only.

    Parameters
    ----------
    series : pd.Series
        Series from which to infer data type.

    Returns
    -------
    bool
        Boolean indicating whether series contains only numbers.
    """
    # When an unkown dtype is encountered, `np.issubdtype(series.dtype, np.number)`
    # raises a TypeError. This happens for example if `series` is `pd.Categorical`
    # If the dtype is unknown, we treat it as non-numeric, therefore return False.
    try:
        return np.issubdtype(series.dtype, np.number)
    except TypeError:
        return False


def is_categorical(series: pd.Series, unique_value_count_threshold: int = 10) -> bool:
    """Heuristic to tell if a series is categorical.

    Parameters
    ---
    series : pd.Series
        Series from which to infer data type.
    unique_value_count_threshold : int
        The number of unique values of the series has to be less than or equal to this number for
        the series to satisfy one of the requirements to be a categorical series.

    Returns
    ---
    bool
        Boolean indicating if series is categorical.
    """
    return (
        not is_boolean(series)
        and not is_date(series)
        and (
            (
                series.nunique() <= unique_value_count_threshold
                and pd.api.types.is_integer_dtype(series)
            )
            or pd.api.types.is_string_dtype(series)
        )
    )


def is_boolean(series: pd.Series) -> bool:
    """Heuristic which tells if a series contains only boolean values.

    Parameters
    ----------
    series : pd.Series
        Series from which to infer data type.

    Returns
    -------
    bool
        Boolean indicating if series is boolean.
    """
    return pd.api.types.is_bool_dtype(series) or set(series.unique()) <= {1, 0, pd.NA}


def is_date(series: pd.Series) -> bool:
    """Heuristic which tells if a series is of type date.

    Parameters
    ----------
    series : pd.Series
        Series from which to infer data type.

    Returns
    -------
    bool
        Boolean indicating if series is of type datetime.
    """
    if isinstance(series.dtype, pd.PeriodDtype):
        return True
    no_numerics = np.all(~series.astype(str).str.isnumeric())
    converted_series = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    return converted_series.notna().all() and no_numerics and not is_numeric(series)
