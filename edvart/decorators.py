import functools

import pandas as pd


def check_index_time_ascending(func):
    """
    Check whether the index of the DataFrame is sorted in ascending order.

    The DataFrame which is checked needs to be the first argument of the decorated function.

    Raises
    ------
    ValueError
        If the index is not a datetime or is not ascending.
    """

    @functools.wraps(func)
    def wrapper_check_index_ascending(df: pd.DataFrame, *args, **kwargs):
        if not (
            df.index.inferred_type.startswith("datetime") or df.index.inferred_type == "period"
        ):
            raise ValueError(
                "The index of the provided DataFrame is not a time index. Please reindex the data. "
                "See `edvart.utils.reindex_to_period` or `edvart.utils.reindex_to_datetime` "
            )
        if not df.index.is_monotonic_increasing:
            raise ValueError(
                "The index of the provided DataFrame is not in an ascending order. "
                "Please sort the index."
            )

        return func(df, *args, **kwargs)

    return wrapper_check_index_ascending
