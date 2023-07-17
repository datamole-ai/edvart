import warnings
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from edvart import Report


def test_column_selection():
    test_df = pd.DataFrame(
        data=np.random.random_sample((50, 20)), columns=[f"Col{i}" for i in range(20)]
    )
    report = Report(dataframe=test_df, use_default_sections=False)

    # Default column selection
    report.add_overview()
    assert report.sections[0].columns is None, "Default column selection should be None"

    # Omitting columns
    report.add_univariate_analysis(omit_columns=["Col15", "Col16"])
    univariate_analysis_columns = {f"Col{i}" for i in range(20) if i != 15 and i != 16}
    assert set(report.sections[1].columns) == univariate_analysis_columns, "Wrong column selection"

    # Specific column subset
    report.add_overview(use_columns=["Col5", "Col7", "Col13"])
    assert set(report.sections[2].columns) == {"Col5", "Col7", "Col13"}, "Wrong column selection"

    # use_columns and omit_columns at the same time
    use_columns = {"Col1", "Col2", "Col3", "Col4"}
    omit_columns = {"Col4", "Col5", "Col6", "Col7"}
    report.add_univariate_analysis(use_columns=use_columns, omit_columns=omit_columns)
    assert set(report.sections[3].columns) == use_columns - omit_columns, "Wrong column selection"


def test_show():
    test_df = pd.DataFrame(
        data=np.random.random_sample((50, 20)), columns=[f"Col{i}" for i in range(20)]
    )
    report = Report(dataframe=test_df, use_default_sections=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with redirect_stdout(None):
            report.show()
