import warnings
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import pytest

from edvart.report import DefaultReport, Report
from edvart.report_sections.bivariate_analysis import BivariateAnalysis
from edvart.report_sections.section_base import Verbosity
from edvart.report_sections.univariate_analysis import UnivariateAnalysis


@pytest.fixture
def test_df() -> pd.DataFrame:
    return pd.DataFrame(
        data=np.random.random_sample((50, 20)), columns=[f"Col{i}" for i in range(20)]
    )


def test_report(test_df: pd.DataFrame):
    report = Report(dataframe=test_df)
    assert len(report.sections) == 0, "Report should be empty"

    report.add_overview(verbosity=Verbosity.MEDIUM)
    assert len(report.sections) == 1, "Report should have one section"

    report.add_bivariate_analysis(verbosity=Verbosity.HIGH, columns=["Col1", "Col2", "Col3"])
    assert len(report.sections) == 2, "Report should have two sections"

    assert report.sections[0].name == "Overview", "Wrong section name"
    assert report.sections[0].verbosity == Verbosity.MEDIUM, "Wrong section verbosity"
    assert report.sections[0].columns is None, "Default column selection should be None"

    assert report.sections[1].columns == ["Col1", "Col2", "Col3"], "Wrong columns"


def test_add_section(test_df: pd.DataFrame):
    bivariate_analysis_section = BivariateAnalysis()
    univariate_analysis_section = UnivariateAnalysis()
    report = (
        Report(dataframe=test_df)
        .add_section(bivariate_analysis_section)
        .add_section(univariate_analysis_section)
    )

    assert report.sections == [bivariate_analysis_section, univariate_analysis_section]


def test_default_report(test_df: pd.DataFrame):
    report = DefaultReport(
        dataframe=test_df,
        verbosity_overview=Verbosity.MEDIUM,
        verbosity_univariate_analysis=Verbosity.HIGH,
        columns_bivariate_analysis=["Col1", "Col2", "Col3"],
    )
    assert len(report.sections) > 0, "Default report should not be empty"

    assert report.sections[0].verbosity == Verbosity.MEDIUM, "Wrong section verbosity"
    assert report.sections[0].columns is None, "Default column selection should be None"

    assert report.sections[1].verbosity == Verbosity.HIGH, "Wrong section verbosity"
    assert report.sections[1].columns is None, "Default column selection should be None"

    assert report.sections[2].verbosity == Verbosity.LOW, "Wrong section verbosity"
    assert report.sections[2].columns == ["Col1", "Col2", "Col3"], "Wrong columns"


def test_column_selection(test_df: pd.DataFrame):
    report = Report(dataframe=test_df)

    # Default column selection
    report.add_overview()
    assert report.sections[0].columns is None, "Default column selection should be None"

    # Omitting columns
    report.add_univariate_analysis(columns=set(test_df.columns) - {"Col15", "Col16"})
    univariate_analysis_columns = {f"Col{i}" for i in range(20) if i != 15 and i != 16}
    assert set(report.sections[1].columns) == univariate_analysis_columns, "Wrong column selection"

    # Specific column subset
    report.add_overview(columns=["Col5", "Col7", "Col13"])
    assert set(report.sections[2].columns) == {"Col5", "Col7", "Col13"}, "Wrong column selection"


def test_show(test_df: pd.DataFrame):
    report = Report(dataframe=test_df)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with redirect_stdout(None):
            report.show()
