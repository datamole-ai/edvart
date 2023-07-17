"""Time series analysis package."""

# Standard imports
from enum import IntEnum
from typing import Any, Dict, List, Optional

# Third-party library imports
import nbformat.v4 as nbfv4
import pandas as pd
from IPython.display import Markdown, display

# Internal library imports
from edvart.decorators import check_index_time_ascending
from edvart.report_sections.section_base import ReportSection
from edvart.report_sections.timeseries_analysis import (
    Autocorrelation,
    BoxplotsOverTime,
    FourierTransform,
    RollingStatistics,
    SeasonalDecomposition,
    ShortTimeFT,
    StationarityTests,
    TimeAnalysisPlot,
)


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
    verbosity : int
        Generated code verbosity global to the Overview sections, must be one of [0, 1, 2].

        0
            A single cell which generates the timeseries analysis section is exported.
        1
            Parameterizable function calls for each subsection are exported.
        2
            Similar to 1, but in addition function definitions are also exported.

        If subsection verbosities are None, then they will be overridden by this parameter.
    columns : List[str], optional
        Columns to include in timeseries analysis. Each column is treated as a separate time series.
        All columns are used by default.
    verbosity_time_analysis_plot : int, optional
        Time analysis interactive plot subsection code verbosity.
    verbosity_rolling_statistics: int, optional
        Rolling statistics interactive plot subsection code verbosity.
    verbosity_boxplots_over_time: int, optional
        Boxplots grouped over time intervals subsection code verbosity.
    verbosity_seasonal_decomposition: int, optional
        Seasonal decomposition subsection code verbosity.
    verbosity_stationarity_tests: int, optional
        Stationarity tests subsection code verbosity.
    verbosity_autocorrelation: int, optional
        Autocorrelation and partial autocorrelation plot subsection code verbosity.
    verbosity_fourier_transform: int, optional
        Discrete Fourier transform plot subsection code verbosity.
    verbosity_short_time_ft: int, optional
        Short-time discrete Fourier transform plot subsection code verbosity.
    sampling_rate: int, optional
        Sampling rate of the time-series, i.e., how many samples form one period. For example,
        if your time-series contains hourly data and want to investigate daily frequencies, use 24.
        If not set, Fourier transform and Short-time Fourier transform will not be included.
    stft_window_size: int
        Window size for Short-time Fourier transform, which will not be included if this parameter
        is not set.
    """

    # pylint: disable=invalid-name
    class TimeseriesAnalysisSubsection(IntEnum):
        """Enum of all implemented timeseries analysis subsections."""

        TimeAnalysisPlot = 0
        RollingStatistics = 1
        BoxplotsOverTime = 2
        SeasonalDecomposition = 3
        StationarityTests = 4
        Autocorrelation = 5
        FourierTransform = 6
        ShortTimeFT = 7

        def __str__(self):
            return self.name

    def __init__(
        self,
        subsections: Optional[List[TimeseriesAnalysisSubsection]] = None,
        verbosity: int = 0,
        columns: Optional[List[str]] = None,
        verbosity_time_analysis_plot: Optional[int] = None,
        verbosity_rolling_statistics: Optional[int] = None,
        verbosity_boxplots_over_time: Optional[int] = None,
        verbosity_seasonal_decomposition: Optional[int] = None,
        verbosity_stationarity_tests: Optional[int] = None,
        verbosity_autocorrelation: Optional[int] = None,
        verbosity_fourier_transform: Optional[int] = None,
        verbosity_short_time_ft: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        stft_window_size: Optional[int] = None,
    ):
        self.sampling_rate = sampling_rate
        self.stft_window_size = stft_window_size

        verbosity_time_analysis_plot = (
            verbosity_time_analysis_plot if verbosity_time_analysis_plot is not None else verbosity
        )
        verbosity_rolling_statistics = (
            verbosity_rolling_statistics if verbosity_rolling_statistics is not None else verbosity
        )
        verbosity_boxplots_over_time = (
            verbosity_boxplots_over_time if verbosity_boxplots_over_time is not None else verbosity
        )
        verbosity_seasonal_decomposition = (
            verbosity_seasonal_decomposition
            if verbosity_seasonal_decomposition is not None
            else verbosity
        )
        verbosity_stationarity_tests = (
            verbosity_stationarity_tests if verbosity_stationarity_tests is not None else verbosity
        )
        verbosity_autocorrelation = (
            verbosity_autocorrelation if verbosity_autocorrelation is not None else verbosity
        )
        verbosity_fourier_transform = (
            verbosity_fourier_transform if verbosity_fourier_transform is not None else verbosity
        )
        verbosity_short_time_ft = (
            verbosity_short_time_ft if verbosity_short_time_ft is not None else verbosity
        )

        subsec = TimeseriesAnalysis.TimeseriesAnalysisSubsection

        verbosities = {
            subsec.TimeAnalysisPlot: verbosity_time_analysis_plot,
            subsec.RollingStatistics: verbosity_rolling_statistics,
            subsec.BoxplotsOverTime: verbosity_boxplots_over_time,
            subsec.SeasonalDecomposition: verbosity_seasonal_decomposition,
            subsec.StationarityTests: verbosity_stationarity_tests,
            subsec.Autocorrelation: verbosity_autocorrelation,
            subsec.FourierTransform: verbosity_fourier_transform,
            subsec.ShortTimeFT: verbosity_short_time_ft,
        }

        enum_to_implementation = {
            subsec.TimeAnalysisPlot: TimeAnalysisPlot(verbosity_time_analysis_plot, columns),
            subsec.RollingStatistics: RollingStatistics(verbosity_rolling_statistics, columns),
            subsec.BoxplotsOverTime: BoxplotsOverTime(verbosity_boxplots_over_time, columns),
            subsec.SeasonalDecomposition: SeasonalDecomposition(
                verbosity_seasonal_decomposition, columns
            ),
            subsec.StationarityTests: StationarityTests(verbosity_stationarity_tests, columns),
            subsec.Autocorrelation: Autocorrelation(verbosity_autocorrelation, columns),
        }

        if sampling_rate is not None:
            enum_to_implementation[subsec.FourierTransform] = FourierTransform(
                sampling_rate, verbosity_fourier_transform, columns
            )
            if stft_window_size is not None:
                enum_to_implementation[subsec.ShortTimeFT] = ShortTimeFT(
                    sampling_rate, stft_window_size, verbosity_short_time_ft, columns
                )

        # By default use all subsections, FT and STFT only if required parameters specified
        if subsections is None:
            subsections_all = list(enum_to_implementation.keys())
        else:
            subsections_all = subsections

        # Store subsections with 0 verbosity
        self.subsections_0 = [sub for sub in subsections_all if verbosities[sub] == 0]

        if len(self.subsections_0) == len(subsections_all) and subsections is None:
            self.subsections_0 = None

        if subsections is None:
            subsections_implementations = list(enum_to_implementation.values())
        else:
            if sampling_rate is None:
                if subsec.FourierTransform in subsections:
                    raise ValueError("Need to set a `sampling_rate` to plot Fourier transform.")
                if subsec.ShortTimeFT in subsections:
                    raise ValueError(
                        "Need to set a `sampling_rate` to plot Short-time Fourier transform."
                    )
            if stft_window_size is None and subsec.ShortTimeFT in subsections:
                raise ValueError(
                    "Need to set an `stft_window_size` to plot Short-time Fourier transform."
                )
            subsections_implementations = [enum_to_implementation[sub] for sub in subsections]

        super().__init__(subsections_implementations, verbosity, columns)

    @property
    def name(self) -> str:
        return "Timeseries analysis"

    @staticmethod
    @check_index_time_ascending
    def timeseries_analysis(
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
            verbosity=0,
            columns=columns,
            sampling_rate=sampling_rate,
            stft_window_size=stft_window_size,
        )

        for sub in timeseries_analysis.subsections:
            sub.show(df)

    def add_cells(self, cells: List[Dict[str, Any]]) -> None:
        """Add cells to the list of cells.

        Cells can be either code cells or markdown cells.

        Parameters
        ----------
        cells : List[Dict[str, Any]]
            List of generated notebook cells which are represented as dictionaries
        """
        section_header = nbfv4.new_markdown_cell(self.get_title(section_level=1))
        cells.append(section_header)

        if self.verbosity == 0:
            subsec = TimeseriesAnalysis.TimeseriesAnalysisSubsection
            code = "timeseries_analysis(df=df"

            if self.subsections_0 is not None:
                arg_subsections_names = [
                    f"TimeseriesAnalysis.TimeseriesAnalysisSubsection.{str(sub)}"
                    for sub in self.subsections_0
                ]
                code += f", subsections={arg_subsections_names}".replace("'", "")

            stft_included_or_empty = (
                self.subsections_0 is None or subsec.ShortTimeFT in self.subsections_0
            )
            include_sampling_rate = self.sampling_rate is not None and (
                stft_included_or_empty or subsec.FourierTransform in self.subsections_0
            )
            if include_sampling_rate:
                code += f", sampling_rate={self.sampling_rate}"
                if self.stft_window_size is not None and stft_included_or_empty:
                    code += f", stft_window_size={self.stft_window_size}"

            if self.columns is not None:
                code += f", columns={self.columns}"
            code += ")"
            cells.append(nbfv4.new_code_cell(code))

            for sub in self.subsections:
                if sub.verbosity > 0:
                    sub.add_cells(cells)
        else:
            super().add_cells(cells)

    def required_imports(self) -> List[str]:
        """Returns a list of imports to be put at the top of a generated notebook.

        Returns
        -------
        List[str]
            List of import strings to be added at the top of the generated notebook,
            e.g. ["import pandas as pd", "import numpy as np"]
        """
        if self.verbosity == 0:
            imports = {
                "from edvart.report_sections.timeseries_analysis import TimeseriesAnalysis\n"
                "timeseries_analysis = TimeseriesAnalysis.timeseries_analysis"
            }
            for sub in self.subsections:
                if sub.verbosity > 0:
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
