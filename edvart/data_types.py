import warnings
from enum import IntEnum

import numpy as np
import pandas as pd

try:
    import pyarrow  # pylint: disable=unused-import
except ImportError:
    PYARROW_PANDAS_BACKEND_AVAILABLE = False
else:
    PYARROW_PANDAS_BACKEND_AVAILABLE = pd.__version__ >= "2.0"


class DataType(IntEnum):
    """Class describe possible data types."""

    NUMERIC = 1
    CATEGORICAL = 2
    BOOLEAN = 3
    DATE = 4
    UNKNOWN = 5
    MISSING = 6
    UNIQUE = 7

    def __str__(self):
        return self.name.lower()


# pylint: disable=too-many-return-statements
def infer_data_type(series: pd.Series) -> DataType:
    """Infers the data type of the series passed in.

    Parameters
    ----------
    series : pd.Series
        Series from which to infer data type.

    Returns
    -------
    DataType
        Inferred custom edvart data type.
    """
    if series.empty:
        return DataType.UNKNOWN
    if is_missing(series):
        return DataType.MISSING
    if is_boolean(series):
        return DataType.BOOLEAN
    if is_date(series):
        return DataType.DATE
    if is_unique(series):
        return DataType.UNIQUE
    if is_categorical(series):
        return DataType.CATEGORICAL
    if is_numeric(series):
        return DataType.NUMERIC

    return DataType.UNKNOWN


def is_unique(series: pd.Series) -> bool:
    """Heuristic to tell if a series is categorical with only unique values.

    Parameters
    ----------
    series : pd.Series
        Series from which to infer data type.

    Returns
    -------
    bool
        Boolean indicating whether series contains only unique values.
    """
    return is_categorical(series) and series.nunique() == len(series)


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
    if is_missing(series):
        return False
    return pd.api.types.is_numeric_dtype(series)


def is_missing(series: pd.Series) -> bool:
    """Function to tell if the series contains only missing values.

    Parameters
    ----------
    series : pd.Series
        Series from which to infer data type.

    Returns
    -------
    bool
        True if all values in the series are missing, False otherwise.
    """
    return series.isnull().all()


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
        not is_missing(series)
        and not is_boolean(series)
        and not is_date(series)
        and (
            (
                series.nunique() <= unique_value_count_threshold
                and pd.api.types.is_integer_dtype(series)
            )
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_object_dtype(series)
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
    return not is_missing(series) and (
        pd.api.types.is_bool_dtype(series) or set(series.unique()) <= {1, 0, pd.NA}
    )


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
    if is_missing(series) or is_numeric(series):
        return False
    contains_numerics = np.any(series.astype(str).str.isnumeric())
    if contains_numerics:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            converted_series = pd.to_datetime(series.dropna(), errors="coerce")
    except ValueError:
        return False
    return converted_series.notna().all()
