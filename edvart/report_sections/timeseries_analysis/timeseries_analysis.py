from enum import IntEnum
from typing import Any, Dict, List, Optional

import nbformat.v4 as nbfv4
import pandas as pd
from IPython.display import Markdown, display

from edvart.decorators import check_index_time_ascending
from edvart.report_sections.section_base import ReportSection, Verbosity
from edvart.report_sections.timeseries_analysis import (
    Autocorrelation,
    BoxplotsOverTime,
    FourierTransform,
    RollingStatistics,
    SeasonalDecomposition,
    ShortTimeFT,
    StationarityTests,
    TimeSeriesLinePlot,
)


# pylint: disable=invalid-name
class TimeseriesAnalysisSubsection(IntEnum):
    """Enum of all implemented timeseries analysis subsections."""

    TimeSeriesLinePlot = 0
    RollingStatistics = 1
    BoxplotsOverTime = 2
    SeasonalDecomposition = 3
    StationarityTests = 4
    Autocorrelation = 5
    FourierTransform = 6
    ShortTimeFT = 7

    def __str__(self):
        return self.name


class TimeseriesAnalysis(ReportSection):
    """Generates the Timeseries analysis section of the report.

    Contains an enum TimeseriesAnalysisSubsection of possible subsections.

    Parameters
    ----------
    subsections : List[TimeseriesAnalysisSubsection], optional
        List of subsections to include.
        All subsection in TimeseriesAnalysisSubsection are included by default, except for
        FourierTransform, which is only included if `sampling_rate` is set and ShortTimeFT, which
        is only included if `sampling_rate` and `stft_window_size` are both set.
    verbosity : Verbosity
        Generated code verbosity global to the Overview sections.
        If subsection verbosities are None, then they will be overridden by this parameter.
    columns : List[str], optional
        Columns to include in timeseries analysis. Each column is treated as a separate time series.
        All columns are used by default.
    verbosity_series_line_plot : Verbosity, optional
        Time series line plot subsection code verbosity.
    verbosity_rolling_statistics: Verbosity, optional
        Rolling statistics interactive plot subsection code verbosity.
    verbosity_boxplots_over_time: Verbosity, optional
        Boxplots grouped over time intervals subsection code verbosity.
    verbosity_seasonal_decomposition: Verbosity, optional
        Seasonal decomposition subsection code verbosity.
    verbosity_stationarity_tests: Verbosity, optional
        Stationarity tests subsection code verbosity.
    verbosity_autocorrelation: Verbosity, optional
        Autocorrelation and partial autocorrelation plot subsection code verbosity.
    verbosity_fourier_transform: Verbosity, optional
        Discrete Fourier transform plot subsection code verbosity.
    verbosity_short_time_ft: Verbosity, optional
        Short-time discrete Fourier transform plot subsection code verbosity.
    sampling_rate: int, optional
        Sampling rate of the time-series, i.e., how many samples form one period. For example,
        if your time-series contains hourly data and want to investigate daily frequencies, use 24.
        If not set, Fourier transform and Short-time Fourier transform will not be included.
    stft_window_size: int
        Window size for Short-time Fourier transform, which will not be included if this parameter
        is not set.
    """

    def __init__(
        self,
        subsections: Optional[List[TimeseriesAnalysisSubsection]] = None,
        verbosity: Verbosity = Verbosity.LOW,
        columns: Optional[List[str]] = None,
        verbosity_time_series_line_plot: Optional[Verbosity] = None,
        verbosity_rolling_statistics: Optional[Verbosity] = None,
        verbosity_boxplots_over_time: Optional[Verbosity] = None,
        verbosity_seasonal_decomposition: Optional[Verbosity] = None,
        verbosity_stationarity_tests: Optional[Verbosity] = None,
        verbosity_autocorrelation: Optional[Verbosity] = None,
        verbosity_fourier_transform: Optional[Verbosity] = None,
        verbosity_short_time_ft: Optional[Verbosity] = None,
        sampling_rate: Optional[int] = None,
        stft_window_size: Optional[int] = None,
    ):
        self.sampling_rate = sampling_rate
        self.stft_window_size = stft_window_size

        verbosity_time_series_line_plot = verbosity_time_series_line_plot or verbosity
        verbosity_rolling_statistics = verbosity_rolling_statistics or verbosity
        verbosity_boxplots_over_time = verbosity_boxplots_over_time or verbosity
        verbosity_seasonal_decomposition = (
            verbosity_seasonal_decomposition
            if verbosity_seasonal_decomposition is not None
            else verbosity
        )
        verbosity_stationarity_tests = verbosity_stationarity_tests or verbosity
        verbosity_autocorrelation = verbosity_autocorrelation or verbosity
        verbosity_fourier_transform = verbosity_fourier_transform or verbosity
        verbosity_short_time_ft = verbosity_short_time_ft or verbosity

        self.subsection_verbosities = {
            TimeseriesAnalysisSubsection.TimeSeriesLinePlot: verbosity_time_series_line_plot,
            TimeseriesAnalysisSubsection.RollingStatistics: verbosity_rolling_statistics,
            TimeseriesAnalysisSubsection.BoxplotsOverTime: verbosity_boxplots_over_time,
            TimeseriesAnalysisSubsection.SeasonalDecomposition: verbosity_seasonal_decomposition,
            TimeseriesAnalysisSubsection.StationarityTests: verbosity_stationarity_tests,
            TimeseriesAnalysisSubsection.Autocorrelation: verbosity_autocorrelation,
            TimeseriesAnalysisSubsection.FourierTransform: verbosity_fourier_transform,
            TimeseriesAnalysisSubsection.ShortTimeFT: verbosity_short_time_ft,
        }

        enum_to_implementation = {
            TimeseriesAnalysisSubsection.TimeSeriesLinePlot: TimeSeriesLinePlot(
                verbosity_time_series_line_plot, columns
            ),
            TimeseriesAnalysisSubsection.RollingStatistics: RollingStatistics(
                verbosity_rolling_statistics, columns
            ),
            TimeseriesAnalysisSubsection.BoxplotsOverTime: BoxplotsOverTime(
                verbosity_boxplots_over_time, columns
            ),
            TimeseriesAnalysisSubsection.SeasonalDecomposition: SeasonalDecomposition(
                verbosity_seasonal_decomposition, columns
            ),
            TimeseriesAnalysisSubsection.StationarityTests: StationarityTests(
                verbosity_stationarity_tests, columns
            ),
            TimeseriesAnalysisSubsection.Autocorrelation: Autocorrelation(
                verbosity_autocorrelation, columns
            ),
        }
        # Add FT and STFT only if required parameters specified
        if sampling_rate is not None:
            enum_to_implementation[
                TimeseriesAnalysisSubsection.FourierTransform
            ] = FourierTransform(sampling_rate, verbosity_fourier_transform, columns)
            if stft_window_size is not None:
                enum_to_implementation[TimeseriesAnalysisSubsection.ShortTimeFT] = ShortTimeFT(
                    sampling_rate, stft_window_size, verbosity_short_time_ft, columns
                )
            elif (
                subsections is not None and TimeseriesAnalysisSubsection.ShortTimeFT in subsections
            ):
                raise ValueError(
                    "Need to set an `stft_window_size` to plot Short-time Fourier transform."
                )
        elif subsections is not None:
            if TimeseriesAnalysisSubsection.FourierTransform in subsections:
                raise ValueError("Need to set a `sampling_rate` to plot Fourier transform.")
            if TimeseriesAnalysisSubsection.ShortTimeFT in subsections:
                raise ValueError(
                    "Need to set a `sampling_rate` to plot Short-time Fourier transform."
                )

        self.default_subsections_to_show = list(enum_to_implementation.keys())

        if subsections is None:
            self.subsections_to_show = self.default_subsections_to_show
        else:
            self.subsections_to_show = subsections

        self.subsections_to_show_with_low_verbosity = [
            sub
            for sub in self.subsections_to_show
            if self.subsection_verbosities[sub] == Verbosity.LOW
        ]

        subsections_implementations = [
            enum_to_implementation[sub] for sub in self.subsections_to_show
        ]

        super().__init__(subsections_implementations, verbosity, columns)

    @property
    def name(self) -> str:
        return "Timeseries analysis"

    def add_cells(self, cells: List[Dict[str, Any]], df: pd.DataFrame) -> None:
        """Add cells to the list of cells.

        Cells can be either code cells or markdown cells.

        Parameters
        ----------
        cells : List[Dict[str, Any]]
            List of generated notebook cells which are represented as dictionaries
        df: pd.DataFrame
            Data for which to add the cells
        """
        section_header = nbfv4.new_markdown_cell(self.get_title(section_level=1))
        cells.append(section_header)

        if self.verbosity == Verbosity.LOW:
            code = "show_timeseries_analysis(df=df"
            if self.subsections_to_show_with_low_verbosity != self.default_subsections_to_show:
                arg_subsections_names = [
                    f"TimeseriesAnalysisSubsection.{str(sub)}"
                    for sub in self.subsections_to_show_with_low_verbosity
                ]
                code += f", subsections={arg_subsections_names}".replace("'", "")
            stft_included = (
                TimeseriesAnalysisSubsection.ShortTimeFT
                in self.subsections_to_show_with_low_verbosity
            )
            include_sampling_rate = self.sampling_rate is not None and (
                stft_included
                or TimeseriesAnalysisSubsection.FourierTransform
                in self.subsections_to_show_with_low_verbosity
            )
            if include_sampling_rate:
                code += f", sampling_rate={self.sampling_rate}"
                if self.stft_window_size is not None and stft_included:
                    code += f", stft_window_size={self.stft_window_size}"

            if self.columns is not None:
                code += f", columns={self.columns}"
            code += ")"
            cells.append(nbfv4.new_code_cell(code))

            for sub in self.subsections:
                if sub.verbosity > Verbosity.LOW:
                    sub.add_cells(cells=cells, df=df)
        else:
            super().add_cells(cells, df=df)

    def required_imports(self) -> List[str]:
        """Returns a list of imports to be put at the top of a generated notebook.

        Returns
        -------
        List[str]
            List of import strings to be added at the top of the generated notebook,
            e.g. ["import pandas as pd", "import numpy as np"]
        """
        if self.verbosity == Verbosity.LOW:
            imports = {
                "from edvart.report_sections.timeseries_analysis.timeseries_analysis"
                " import show_timeseries_analysis"
            }
            if self.subsections_to_show_with_low_verbosity != self.default_subsections_to_show:
                imports.add(
                    "from edvart.report_sections.timeseries_analysis.timeseries_analysis"
                    " import TimeseriesAnalysisSubsection"
                )
            for sub in self.subsections:
                if sub.verbosity > Verbosity.LOW:
                    imports.update(sub.required_imports())

            return list(imports)
        return super().required_imports()

    def show(self, df: pd.DataFrame) -> None:
        """Generates cell output of this section in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output.
        """
        display(Markdown(self.get_title(section_level=1)))
        super().show(df)


@check_index_time_ascending
def show_timeseries_analysis(
    df: pd.DataFrame,
    subsections: Optional[List[TimeseriesAnalysisSubsection]] = None,
    columns: Optional[List[str]] = None,
    sampling_rate: Optional[int] = None,
    stft_window_size: Optional[int] = None,
) -> None:
    """Generate timeseries analysis for df.

    Parameters
    ----------
    df : pd.DataFrame
        Data to be analyzed.
    subsections : List[TimeseriesAnalysisSubsection], optional
        Subsections to include in the analysis. All subsections are included by default.
    columns : List[str], optional
        Subset of columns of df to consider in timeseries analysis.
        All columns are used by default.
    sampling_rate : int, optional
        Sampling rate of the time-series, i.e., how many samples form one period. For example,
        if your timeseries contains hourly data and you want to investigate daily frequencies,
        use 24.
        If not set, Fourier transform and Short-time Fourier transform will not be included.
    stft_window_size : int, optional
        Window size for Short-time Fourier transform. Short-time Fourier transform will not be
        included if this parameter is not set.

    Raises
    ------
    ValueError
        If the input data is not indexed by time in ascending order.
    """
    if columns is not None:
        df = df[columns]

    timeseries_analysis = TimeseriesAnalysis(
        subsections=subsections,
        verbosity=Verbosity.LOW,
        columns=columns,
        sampling_rate=sampling_rate,
        stft_window_size=stft_window_size,
    )

    for sub in timeseries_analysis.subsections:
        sub.show(df)
