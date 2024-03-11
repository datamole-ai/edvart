import warnings
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.io as pio
import pytest

import edvart
from edvart.report_sections import timeseries_analysis
from edvart.report_sections.code_string_formatting import get_code
from edvart.report_sections.section_base import Verbosity
from edvart.report_sections.timeseries_analysis import (
    BoxplotsOverTime,
    TimeseriesAnalysis,
    TimeseriesAnalysisSubsection,
)

from .execution_utils import check_section_executes

pio.renderers.default = "json"


@pytest.fixture
def test_df() -> pd.DataFrame:
    n_rows = 20
    columns = ["a", "b", "c"]
    return pd.DataFrame(
        index=[pd.Timestamp.now() + pd.Timedelta(minutes=i) for i in range(n_rows)],
        data=np.random.rand(n_rows, len(columns)),
        columns=columns,
    )


def test_default_config_verbosity():
    timeseries_section = TimeseriesAnalysis()
    assert timeseries_section.verbosity == Verbosity.LOW, "Verbosity should be Verbosity.LOW"
    for s in timeseries_section.subsections:
        assert s.verbosity == Verbosity.LOW, "Verbosity should be Verbosity.LOW"


def test_high_verbosities():
    with pytest.raises(ValueError):
        TimeseriesAnalysis(verbosity=4)
    with pytest.raises(ValueError):
        TimeseriesAnalysis(verbosity_time_series_line_plot=4)
    with pytest.raises(ValueError):
        TimeseriesAnalysis(verbosity_stationarity_tests=5)
    with pytest.raises(ValueError):
        TimeseriesAnalysis(verbosity_rolling_statistics=10)


def test_global_verbosity_overriding():
    timeseries_section = TimeseriesAnalysis(
        verbosity=Verbosity.LOW,
        verbosity_autocorrelation=Verbosity.HIGH,
        verbosity_stationarity_tests=Verbosity.MEDIUM,
        verbosity_rolling_statistics=Verbosity.HIGH,
        verbosity_time_series_line_plot=Verbosity.MEDIUM,
    )

    assert timeseries_section.verbosity == Verbosity.LOW
    for subsec in timeseries_section.subsections:
        if isinstance(subsec, timeseries_analysis.Autocorrelation):
            assert (
                subsec.verbosity == Verbosity.HIGH
            ), "Verbosity of autocorrelation should be Verbosity.HIGH"
        elif isinstance(subsec, timeseries_analysis.StationarityTests):
            assert (
                subsec.verbosity == Verbosity.MEDIUM
            ), "Verbosity of stationarity tests should be Verbosity.MEDIUM"
        elif isinstance(subsec, timeseries_analysis.RollingStatistics):
            assert (
                subsec.verbosity == Verbosity.HIGH
            ), "Verbosity of rolling stats should be Verbosity.HIGH"
        elif isinstance(subsec, timeseries_analysis.TimeSeriesLinePlot):
            assert (
                subsec.verbosity == Verbosity.MEDIUM
            ), "Verbosity of time series line plot should be 1"
        else:
            assert (
                subsec.verbosity == Verbosity.LOW
            ), "Verbosity of other sections should be Verbosity.LOW"


def test_verbosity_propagation():
    timeseries_section = TimeseriesAnalysis(verbosity=Verbosity.HIGH)
    assert (
        timeseries_section.verbosity == Verbosity.HIGH
    ), "Timeseries analysis global verbosity should be Verbosity.HIGH."

    for subsec in timeseries_section.subsections:
        assert (
            subsec.verbosity == Verbosity.HIGH
        ), f"{type(subsec)} verbosity should be Verbosity.HIGH."


def test_negative_verbosities():
    with pytest.raises(ValueError):
        TimeseriesAnalysis(verbosity=-2)
    with pytest.raises(ValueError):
        TimeseriesAnalysis(verbosity_rolling_statistics=-2)
    with pytest.raises(ValueError):
        TimeseriesAnalysis(verbosity_seasonal_decomposition=-1)
    with pytest.raises(ValueError):
        TimeseriesAnalysis(verbosity_boxplots_over_time=-3)


def test_section_adding():
    bivariate_section = TimeseriesAnalysis(
        subsections=[
            TimeseriesAnalysisSubsection.RollingStatistics,
            TimeseriesAnalysisSubsection.BoxplotsOverTime,
            TimeseriesAnalysisSubsection.StationarityTests,
            TimeseriesAnalysisSubsection.RollingStatistics,
            TimeseriesAnalysisSubsection.SeasonalDecomposition,
        ]
    )
    assert isinstance(
        bivariate_section.subsections[0], timeseries_analysis.RollingStatistics
    ), "Subsection should be RollingStatistics"
    assert isinstance(
        bivariate_section.subsections[1], timeseries_analysis.BoxplotsOverTime
    ), "Subsection should be BoxplotsOverTime"
    assert isinstance(
        bivariate_section.subsections[2], timeseries_analysis.StationarityTests
    ), "Subsection should be StationarityTests"
    assert isinstance(
        bivariate_section.subsections[3], timeseries_analysis.RollingStatistics
    ), "Subsection should be RollingStatistics"
    assert isinstance(
        bivariate_section.subsections[4], timeseries_analysis.SeasonalDecomposition
    ), "Subsection should be SeasonalDecomposition"


def test_ft_stft_excluded():
    ts = TimeseriesAnalysis()
    for subsec in ts.subsections:
        assert not isinstance(subsec, timeseries_analysis.FourierTransform)
        assert not isinstance(subsec, timeseries_analysis.ShortTimeFT)


def test_ft_included_stft_excluded():
    ts = TimeseriesAnalysis(sampling_rate=1)
    found_ft = False
    for subsec in ts.subsections:
        assert not isinstance(subsec, timeseries_analysis.ShortTimeFT)
        if isinstance(subsec, timeseries_analysis.FourierTransform):
            found_ft = True

    assert found_ft


def test_ft_stft_included():
    ts = TimeseriesAnalysis(sampling_rate=1, stft_window_size=1)
    found_ft = False
    found_stft = False
    for subsec in ts.subsections:
        if isinstance(subsec, timeseries_analysis.ShortTimeFT):
            found_stft = True
            continue
        if isinstance(subsec, timeseries_analysis.FourierTransform):
            found_ft = True

    assert found_ft
    assert found_stft


def test_ft_no_sampling_rate_error():
    with pytest.raises(ValueError):
        _ts = TimeseriesAnalysis(subsections=[TimeseriesAnalysisSubsection.FourierTransform])
    with pytest.raises(ValueError):
        _ts = TimeseriesAnalysis(
            subsections=[TimeseriesAnalysisSubsection.FourierTransform],
            stft_window_size=2,
        )
    with pytest.raises(ValueError):
        _ts = TimeseriesAnalysis(
            subsections=[TimeseriesAnalysisSubsection.ShortTimeFT],
        )
    with pytest.raises(ValueError):
        _ts = TimeseriesAnalysis(
            subsections=[TimeseriesAnalysisSubsection.ShortTimeFT],
            sampling_rate=1,
        )


def test_code_export_verbosity_low(test_df: pd.DataFrame):
    ts_section = TimeseriesAnalysis(verbosity=Verbosity.LOW)
    test_df = test_df
    # Export code
    exported_cells = []
    ts_section.add_cells(exported_cells, df=test_df)
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = ["show_timeseries_analysis(df=df)"]
    # Test code equivalence
    assert len(exported_code) == 1
    assert exported_code[0] == expected_code[0], "Exported code mismatch"

    check_section_executes(ts_section, test_df)


def test_code_export_verbosity_low_with_subsections(test_df: pd.DataFrame):
    ts_section = TimeseriesAnalysis(
        subsections=[
            TimeseriesAnalysisSubsection.RollingStatistics,
            TimeseriesAnalysisSubsection.StationarityTests,
        ],
        verbosity=Verbosity.LOW,
    )
    test_df = test_df
    # Export code
    exported_cells = []
    ts_section.add_cells(exported_cells, df=test_df)
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = [
        "show_timeseries_analysis(df=df, subsections=["
        "TimeseriesAnalysisSubsection.RollingStatistics, "
        "TimeseriesAnalysisSubsection.StationarityTests])"
    ]
    # Test code equivalence
    assert len(exported_code) == 1
    assert exported_code[0] == expected_code[0], "Exported code mismatch"

    check_section_executes(ts_section, test_df)


def test_code_export_verbosity_low_with_fft_stft(test_df: pd.DataFrame):
    ts_section = TimeseriesAnalysis(
        subsections=[
            TimeseriesAnalysisSubsection.FourierTransform,
            TimeseriesAnalysisSubsection.ShortTimeFT,
        ],
        verbosity=Verbosity.LOW,
        sampling_rate=1,
        stft_window_size=1,
    )
    # Export code
    exported_cells = []
    ts_section.add_cells(exported_cells, df=test_df)
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = [
        "show_timeseries_analysis(df=df, subsections=["
        "TimeseriesAnalysisSubsection.FourierTransform, "
        "TimeseriesAnalysisSubsection.ShortTimeFT], "
        "sampling_rate=1, stft_window_size=1)"
    ]
    # Test code equivalence
    assert len(exported_code) == 1
    assert exported_code[0] == expected_code[0], "Exported code mismatch"

    check_section_executes(ts_section, test_df)


def test_generated_code_verbosity_medium(test_df: pd.DataFrame):
    ts_section = TimeseriesAnalysis(verbosity=Verbosity.MEDIUM)

    exported_cells = []
    ts_section.add_cells(exported_cells, df=test_df)
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]

    expected_code = [
        "show_time_series_line_plot(df=df)",
        "show_rolling_statistics(df=df)",
        "show_boxplots_over_time(df=df)",
        "show_seasonal_decomposition(df=df, model='additive')",
        "show_stationarity_tests(df=df)",
        "plot_acf(df=df)",
        "plot_pacf(df=df)",
    ]

    assert len(expected_code) == len(exported_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"
    check_section_executes(ts_section, test_df)


def test_generated_code_verbosity_high(test_df: pd.DataFrame):
    ts_section = TimeseriesAnalysis(verbosity=Verbosity.HIGH, sampling_rate=1, stft_window_size=1)

    pairplot_cells = []
    ts_section.add_cells(pairplot_cells, df=test_df)
    exported_code = [cell["source"] for cell in pairplot_cells if cell["cell_type"] == "code"]

    expected_code = [
        "\n\n".join(
            (
                get_code(timeseries_analysis.time_series_line_plot.show_time_series_line_plot),
                get_code(timeseries_analysis.time_series_line_plot._time_series_line_plot_colored),
                "show_time_series_line_plot(df=df)",
            )
        ),
        "\n\n".join(
            (
                get_code(timeseries_analysis.rolling_statistics.show_rolling_statistics),
                "show_rolling_statistics(df=df)",
            )
        ),
        "\n\n".join(
            (
                get_code(timeseries_analysis.boxplots_over_time.default_grouping_functions),
                get_code(timeseries_analysis.boxplots_over_time.get_default_grouping_func),
                get_code(timeseries_analysis.boxplots_over_time.show_boxplots_over_time),
                "show_boxplots_over_time(df=df)",
            )
        ),
        "\n\n".join(
            (
                get_code(timeseries_analysis.seasonal_decomposition.show_seasonal_decomposition),
                "show_seasonal_decomposition(df=df, model='additive')",
            )
        ),
        "\n\n".join(
            (
                get_code(timeseries_analysis.stationarity_tests.default_stationarity_tests),
                get_code(timeseries_analysis.stationarity_tests.show_stationarity_tests),
                "show_stationarity_tests(df=df)",
            )
        ),
        get_code(timeseries_analysis.autocorrelation.plot_acf) + "\n\n" + "plot_acf(df=df)",
        "\n\n".join(
            (
                get_code(timeseries_analysis.autocorrelation.plot_pacf),
                "plot_pacf(df=df)",
            )
        ),
        "\n\n".join(
            (
                get_code(timeseries_analysis.fourier_transform.show_fourier_transform),
                "show_fourier_transform(df=df, sampling_rate=1)",
            )
        ),
        "\n\n".join(
            (
                get_code(timeseries_analysis.short_time_ft.show_short_time_ft),
                "show_short_time_ft(df=df, sampling_rate=1, window_size=1)",
            )
        ),
    ]

    assert len(expected_code) == len(exported_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"

    check_section_executes(ts_section, test_df)


def test_verbosity_low_different_subsection_verbosities(test_df: pd.DataFrame):
    ts_section = TimeseriesAnalysis(
        verbosity=Verbosity.LOW,
        subsections=[
            TimeseriesAnalysisSubsection.TimeSeriesLinePlot,
            TimeseriesAnalysisSubsection.FourierTransform,
            TimeseriesAnalysisSubsection.RollingStatistics,
            TimeseriesAnalysisSubsection.StationarityTests,
            TimeseriesAnalysisSubsection.BoxplotsOverTime,
            TimeseriesAnalysisSubsection.ShortTimeFT,
        ],
        sampling_rate=1,
        stft_window_size=2,
        verbosity_rolling_statistics=Verbosity.MEDIUM,
        verbosity_fourier_transform=Verbosity.MEDIUM,
        verbosity_short_time_ft=Verbosity.HIGH,
    )

    ts_cells = []
    ts_section.add_cells(ts_cells, df=test_df)
    exported_code = [cell["source"] for cell in ts_cells if cell["cell_type"] == "code"]

    expected_code = [
        "show_timeseries_analysis(df=df, "
        "subsections=[TimeseriesAnalysisSubsection.TimeSeriesLinePlot, "
        "TimeseriesAnalysisSubsection.StationarityTests, "
        "TimeseriesAnalysisSubsection.BoxplotsOverTime])",
        "show_fourier_transform(df=df, sampling_rate=1)",
        "show_rolling_statistics(df=df)",
        (
            get_code(timeseries_analysis.short_time_ft.show_short_time_ft)
            + "\n\n"
            + "show_short_time_ft(df=df, sampling_rate=1, window_size=2)"
        ),
    ]

    assert len(exported_code) == len(expected_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"


def test_boxplots_over_time_def(test_df: pd.DataFrame):
    def month_func(x: datetime) -> str:
        return str(x.month)

    boxplots_sub = BoxplotsOverTime(
        grouping_name="Month",
        grouping_function=month_func,
        grouping_function_imports=["from datetime import datetime"],
    )
    # Export code
    exported_cells = []
    boxplots_sub.add_cells(exported_cells, df=pd.DataFrame())
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]

    expected_code = (
        get_code(month_func) + "\n\n",
        "show_boxplots_over_time(df=df, grouping_function=month_func, grouping_name='Month')",
    )

    assert len(expected_code) == len(exported_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"

    check_section_executes(boxplots_sub, test_df)


def test_boxplots_over_time_lambda(test_df: pd.DataFrame):
    month_lambda = lambda x: x.month  # noqa: E731

    boxplots_sub = BoxplotsOverTime(grouping_name="Month", grouping_function=month_lambda)

    # Export code
    exported_cells = []
    boxplots_sub.add_cells(exported_cells, df=pd.DataFrame())
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]

    expected_code = [
        get_code(month_lambda) + "\n\n",
        "show_boxplots_over_time(df=df, grouping_function=month_lambda, grouping_name='Month')",
    ]

    assert len(expected_code) == len(exported_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"

    check_section_executes(boxplots_sub, test_df)


def test_imports_verbosity_low():
    ts_section = TimeseriesAnalysis(verbosity=Verbosity.LOW)

    exported_imports = ts_section.required_imports()
    expected_imports = [
        "from edvart.report_sections.timeseries_analysis.timeseries_analysis import show_timeseries_analysis"  # noqa: E501
    ]

    assert isinstance(exported_imports, list)
    assert len(expected_imports) == len(exported_imports)
    for expected_import, exported_import in zip(expected_imports, exported_imports):
        assert expected_import == exported_import, "Exported import mismatch"


def test_imports_verbosity_medium():
    ts_section = TimeseriesAnalysis(verbosity=Verbosity.MEDIUM)

    exported_imports = ts_section.required_imports()
    expected_imports = list(set().union(*[s.required_imports() for s in ts_section.subsections]))

    assert isinstance(exported_imports, list)
    assert len(expected_imports) == len(exported_imports)
    for expected_import, exported_import in zip(expected_imports, exported_imports):
        assert expected_import == exported_import, "Exported import mismatch"


def test_imports_verbosity_high():
    ts_section = TimeseriesAnalysis(verbosity=Verbosity.HIGH)

    exported_imports = ts_section.required_imports()
    expected_imports = list(set().union(*[s.required_imports() for s in ts_section.subsections]))

    assert isinstance(exported_imports, list)
    assert len(expected_imports) == len(exported_imports)
    for expected_import, exported_import in zip(expected_imports, exported_imports):
        assert expected_import == exported_import, "Exported import mismatch"


def test_imports_verbosity_low_different_subsection_verbosities():
    ts_section = TimeseriesAnalysis(
        verbosity=Verbosity.LOW,
        subsections=[
            TimeseriesAnalysisSubsection.TimeSeriesLinePlot,
            TimeseriesAnalysisSubsection.FourierTransform,
            TimeseriesAnalysisSubsection.RollingStatistics,
            TimeseriesAnalysisSubsection.StationarityTests,
            TimeseriesAnalysisSubsection.BoxplotsOverTime,
            TimeseriesAnalysisSubsection.ShortTimeFT,
        ],
        sampling_rate=1,
        stft_window_size=2,
        verbosity_rolling_statistics=Verbosity.MEDIUM,
        verbosity_short_time_ft=Verbosity.HIGH,
    )

    exported_imports = ts_section.required_imports()

    expected_imports = {
        "from edvart.report_sections.timeseries_analysis.timeseries_analysis import show_timeseries_analysis",  # noqa: E501
        "from edvart.report_sections.timeseries_analysis.timeseries_analysis import TimeseriesAnalysisSubsection",  # noqa: E501
    }
    for s in ts_section.subsections:
        if s.verbosity > Verbosity.LOW:
            expected_imports.update(s.required_imports())

    assert isinstance(exported_imports, list)
    assert set(exported_imports) == set(expected_imports)


def test_show():
    df = edvart.example_datasets.dataset_global_temp()
    ts_section = TimeseriesAnalysis()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with redirect_stdout(None):
            ts_section.show(df)
