import math
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from edvart import utils

from .pyarrow_utils import pyarrow_params


@pytest.mark.parametrize("pyarrow_dtypes", pyarrow_params)
def test_full_na_series(pyarrow_dtypes: bool):
    series = pd.Series([None, np.nan, None])
    if pyarrow_dtypes:
        series = series.convert_dtypes(dtype_backend="pyarrow")
    for func in (
        utils.quartile1,
        utils.median,
        utils.quartile3,
        utils.minimum,
        utils.maximum,
        utils.mean,
        utils.std,
        utils.kurtosis,
        utils.skewness,
        utils.mad,
        utils.median_absolute_deviation,
        utils.mode,
    ):
        with warnings.catch_warnings():
            print(func)
            warnings.simplefilter(action="error", category=RuntimeWarning)
            result = func(series)
            assert math.isnan(float(result))
    assert utils.num_unique_values(series) == 0


def test_env_var():
    test_var_name = "TEST_VAR"
    test_var_value = "test"
    with utils.env_var(test_var_name, test_var_value):
        assert os.environ[test_var_name] == test_var_value
    assert test_var_value not in os.environ
