import math
import warnings

import numpy as np
import pandas as pd

from edvart import utils


def test_full_na_series():
    series = pd.Series([None, np.nan, None])
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
    assert utils.is_numeric(series)
    assert utils.is_categorical(series)
    assert utils.num_unique_values(series) == 0
