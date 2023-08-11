import datetime
import warnings
from contextlib import redirect_stdout

import pytest

import edvart
from edvart.report_sections import timeseries_analysis
from edvart.report_sections.code_string_formatting import get_code
from edvart.report_sections.section_base import Verbosity
from edvart.report_sections.timeseries_analysis import BoxplotsOverTime, TimeseriesAnalysis


def test_default_config_verbosity():
    timeseries_section = TimeseriesAnalysis()
    assert timeseries_section.verbosity == Verbosity.LOW, "Verbosity should be Verbosity.LOW"
    for s in timeseries_section.subsections:
        assert s.verbosity == Verbosity.LOW, "Verbosity should be Verbosity.LOW"


def test_high_verobisities():
    with pytest.raises(ValueError):
        TimeseriesAnalysis(verbosity=4)
    with pytest.raises(ValueError):
        TimeseriesAnalysis(verbosity_time_analysis_plot=4)
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
        verbosity_time_analysis_plot=Verbosity.MEDIUM,
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
        elif isinstance(subsec, timeseries_analysis.TimeAnalysisPlot):
            assert (
                subsec.verbosity == Verbosity.MEDIUM
            ), "Verbosity of timeanalysis plot should be 1"
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
            TimeseriesAnalysis.TimeseriesAnalysisSubsection.RollingStatistics,
            TimeseriesAnalysis.TimeseriesAnalysisSubsection.BoxplotsOverTime,
            TimeseriesAnalysis.TimeseriesAnalysisSubsection.StationarityTests,
            TimeseriesAnalysis.TimeseriesAnalysisSubsection.RollingStatistics,
            TimeseriesAnalysis.TimeseriesAnalysisSubsection.SeasonalDecomposition,
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
        ts = TimeseriesAnalysis(
            subsections=[TimeseriesAnalysis.TimeseriesAnalysisSubsection.FourierTransform]
        )
    with pytest.raises(ValueError):
        ts = TimeseriesAnalysis(
            subsections=[TimeseriesAnalysis.TimeseriesAnalysisSubsection.FourierTransform],
            stft_window_size=2,
        )
    with pytest.raises(ValueError):
        ts = TimeseriesAnalysis(
            subsections=[TimeseriesAnalysis.TimeseriesAnalysisSubsection.ShortTimeFT],
        )
    with pytest.raises(ValueError):
        ts = TimeseriesAnalysis(
            subsections=[TimeseriesAnalysis.TimeseriesAnalysisSubsection.ShortTimeFT],
            sampling_rate=1,
        )


def test_code_export_verbosity_low():
    ts_section = TimeseriesAnalysis(verbosity=Verbosity.LOW)
    # Export code
    exported_cells = []
    ts_section.add_cells(exported_cells)
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = ["timeseries_analysis(df=df)"]
    # Test code equivalence
    assert len(exported_code) == 1
    assert exported_code[0] == expected_code[0], "Exported code mismatch"


def test_code_export_verbosity_low_with_subsections():
    ts_section = TimeseriesAnalysis(
        subsections=[
            TimeseriesAnalysis.TimeseriesAnalysisSubsection.RollingStatistics,
            TimeseriesAnalysis.TimeseriesAnalysisSubsection.StationarityTests,
        ],
        verbosity=Verbosity.LOW,
    )
    # Export code
    exported_cells = []
    ts_section.add_cells(exported_cells)
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = [
        "timeseries_analysis(df=df, subsections=["
        "TimeseriesAnalysis.TimeseriesAnalysisSubsection.RollingStatistics, "
        "TimeseriesAnalysis.TimeseriesAnalysisSubsection.StationarityTests])"
    ]
    # Test code equivalence
    assert len(exported_code) == 1
    assert exported_code[0] == expected_code[0], "Exported code mismatch"


def test_code_export_verbosity_low_with_fft_stft():
    ts_section = TimeseriesAnalysis(
        subsections=[
            TimeseriesAnalysis.TimeseriesAnalysisSubsection.FourierTransform,
            TimeseriesAnalysis.TimeseriesAnalysisSubsection.ShortTimeFT,
        ],
        verbosity=Verbosity.LOW,
        sampling_rate=1,
        stft_window_size=1,
    )
    # Export code
    exported_cells = []
    ts_section.add_cells(exported_cells)
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = [
        "timeseries_analysis(df=df, subsections=["
        "TimeseriesAnalysis.TimeseriesAnalysisSubsection.FourierTransform, "
        "TimeseriesAnalysis.TimeseriesAnalysisSubsection.ShortTimeFT], "
        "sampling_rate=1, stft_window_size=1)"
    ]
    # Test code equivalence
    assert len(exported_code) == 1
    assert exported_code[0] == expected_code[0], "Exported code mismatch"


def test_generated_code_verobsity_medium():
    ts_section = TimeseriesAnalysis(verbosity=Verbosity.MEDIUM)

    exported_cells = []
    ts_section.add_cells(exported_cells)
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]

    expected_code = [
        "time_analysis_plot(df=df)",
        "rolling_statistics(df=df)",
        "boxplots_over_time(df=df)",
        "seasonal_decomposition(df=df, model='additive')",
        "stationarity_tests(df=df)",
        "plot_acf(df=df)",
        "plot_pacf(df=df)",
    ]

    assert len(expected_code) == len(exported_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"


def test_generated_code_verobsity_high():
    ts_section = TimeseriesAnalysis(verbosity=Verbosity.HIGH, sampling_rate=1, stft_window_size=1)

    pairplot_cells = []
    ts_section.add_cells(pairplot_cells)
    exported_code = [cell["source"] for cell in pairplot_cells if cell["cell_type"] == "code"]

    expected_code = [
        "\n\n".join(
            (
                get_code(timeseries_analysis.TimeAnalysisPlot.time_analysis_plot).replace(
                    "TimeAnalysisPlot.", ""
                ),
                get_code(timeseries_analysis.TimeAnalysisPlot._time_analysis_colored_plot),
                "time_analysis_plot(df=df)",
            )
        ),
        "\n\n".join(
            (
                get_code(timeseries_analysis.RollingStatistics.rolling_statistics),
                "rolling_statistics(df=df)",
            )
        ),
        "\n\n".join(
            (
                get_code(timeseries_analysis.BoxplotsOverTime.default_grouping_functions),
                get_code(timeseries_analysis.BoxplotsOverTime.get_default_grouping_func).replace(
                    "BoxplotsOverTime.", ""
                ),
                get_code(timeseries_analysis.BoxplotsOverTime.boxplots_over_time).replace(
                    "BoxplotsOverTime.", ""
                ),
                "boxplots_over_time(df=df)",
            )
        ),
        "\n\n".join(
            (
                get_code(timeseries_analysis.SeasonalDecomposition.seasonal_decomposition),
                "seasonal_decomposition(df=df, model='additive')",
            )
        ),
        "\n\n".join(
            (
                get_code(timeseries_analysis.StationarityTests.default_stationarity_tests),
                get_code(timeseries_analysis.StationarityTests.stationarity_tests).replace(
                    "StationarityTests.", ""
                ),
                "stationarity_tests(df=df)",
            )
        ),
        get_code(timeseries_analysis.Autocorrelation.plot_acf) + "\n\n" + "plot_acf(df=df)",
        "\n\n".join(
            (
                get_code(timeseries_analysis.Autocorrelation.plot_pacf).replace(
                    "Autocorrelation.", ""
                ),
                "plot_pacf(df=df)",
            )
        ),
        "\n\n".join(
            (
                get_code(timeseries_analysis.FourierTransform.fourier_transform),
                "fourier_transform(df=df, sampling_rate=1)",
            )
        ),
        "\n\n".join(
            (
                get_code(timeseries_analysis.ShortTimeFT.short_time_ft),
                "short_time_ft(df=df, sampling_rate=1, window_size=1)",
            )
        ),
    ]

    assert len(expected_code) == len(exported_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"


def test_verbosity_low_different_subsection_verbosities():
    ts_section = TimeseriesAnalysis(
        verbosity=Verbosity.LOW,
        subsections=[
            TimeseriesAnalysis.TimeseriesAnalysisSubsection.TimeAnalysisPlot,
            TimeseriesAnalysis.TimeseriesAnalysisSubsection.FourierTransform,
            TimeseriesAnalysis.TimeseriesAnalysisSubsection.RollingStatistics,
            TimeseriesAnalysis.TimeseriesAnalysisSubsection.StationarityTests,
            TimeseriesAnalysis.TimeseriesAnalysisSubsection.BoxplotsOverTime,
            TimeseriesAnalysis.TimeseriesAnalysisSubsection.ShortTimeFT,
        ],
        sampling_rate=1,
        stft_window_size=2,
        verbosity_rolling_statistics=Verbosity.MEDIUM,
        verbosity_short_time_ft=Verbosity.HIGH,
    )

    ts_cells = []
    ts_section.add_cells(ts_cells)
    exported_code = [cell["source"] for cell in ts_cells if cell["cell_type"] == "code"]

    expected_code = [
        "timeseries_analysis(df=df, "
        "subsections=[TimeseriesAnalysis.TimeseriesAnalysisSubsection.TimeAnalysisPlot, "
        "TimeseriesAnalysis.TimeseriesAnalysisSubsection.FourierTransform, "
        "TimeseriesAnalysis.TimeseriesAnalysisSubsection.StationarityTests, "
        "TimeseriesAnalysis.TimeseriesAnalysisSubsection.BoxplotsOverTime], sampling_rate=1)",
        "rolling_statistics(df=df)",
        (
            get_code(timeseries_analysis.ShortTimeFT.short_time_ft)
            + "\n\n"
            + "short_time_ft(df=df, sampling_rate=1, window_size=2)"
        ),
    ]

    assert len(exported_code) == len(expected_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"


def test_boxplots_over_time_def():
    def month_func(x: datetime.datetime) -> str:
        return str(x.month)

    boxplots_sub = BoxplotsOverTime(grouping_name="Month", grouping_function=month_func)
    # Export code
    exported_cells = []
    boxplots_sub.add_cells(exported_cells)
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]

    expected_code = (
        get_code(month_func) + "\n\n",
        "boxplots_over_time(df=df, grouping_function=month_func, grouping_name='Month')",
    )

    assert len(expected_code) == len(exported_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"


def test_boxplots_over_time_lambda():
    month_lambda = lambda x: x.month

    boxplots_sub = BoxplotsOverTime(grouping_name="Month", grouping_function=month_lambda)

    # Export code
    exported_cells = []
    boxplots_sub.add_cells(exported_cells)
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]

    expected_code = [
        get_code(month_lambda) + "\n\n",
        "boxplots_over_time(df=df, grouping_function=month_lambda, grouping_name='Month')",
    ]

    assert len(expected_code) == len(exported_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"


def test_imports_verbosity_low():
    ts_section = TimeseriesAnalysis(verbosity=Verbosity.LOW)

    exported_imports = ts_section.required_imports()
    expected_imports = [
        "from edvart.report_sections.timeseries_analysis import TimeseriesAnalysis\n"
        "timeseries_analysis = TimeseriesAnalysis.timeseries_analysis"
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
            TimeseriesAnalysis.TimeseriesAnalysisSubsection.TimeAnalysisPlot,
            TimeseriesAnalysis.TimeseriesAnalysisSubsection.FourierTransform,
            TimeseriesAnalysis.TimeseriesAnalysisSubsection.RollingStatistics,
            TimeseriesAnalysis.TimeseriesAnalysisSubsection.StationarityTests,
            TimeseriesAnalysis.TimeseriesAnalysisSubsection.BoxplotsOverTime,
            TimeseriesAnalysis.TimeseriesAnalysisSubsection.ShortTimeFT,
        ],
        sampling_rate=1,
        stft_window_size=2,
        verbosity_rolling_statistics=Verbosity.MEDIUM,
        verbosity_short_time_ft=Verbosity.HIGH,
    )

    exported_imports = ts_section.required_imports()

    expected_imports = {
        "from edvart.report_sections.timeseries_analysis import TimeseriesAnalysis\n"
        "timeseries_analysis = TimeseriesAnalysis.timeseries_analysis"
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
