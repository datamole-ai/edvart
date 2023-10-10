import warnings
from typing import Any, Dict, List, Optional

import nbformat.v4 as nbfv4
import pandas as pd
import plotly.graph_objects as go
from IPython.display import Markdown, display

from edvart.data_types import is_numeric
from edvart.decorators import check_index_time_ascending
from edvart.report_sections.code_string_formatting import get_code
from edvart.report_sections.section_base import Section, Verbosity


class RollingStatistics(Section):
    """Generates the rolling statistics interactive plot subsection.

    Parameters
    ----------
    verbosity : Verbosity (default = Verbosity.LOW)
        Verbosity of the generated code in the exported notebook.
    columns : List[str], optional
        List of columns to analyze. Only numeric column can be analyzed.
        All numeric columns are analyzed by default.
    window_size : int (default = 20)
        Size of the rolling window to use when computing rolling statistics.
    """

    def __init__(
        self,
        verbosity: Verbosity = Verbosity.LOW,
        columns: Optional[List[str]] = None,
        window_size: int = 20,
    ):
        self.window_size = window_size
        super().__init__(verbosity, columns)

    @property
    def name(self) -> str:
        return "Rolling statistics"

    def required_imports(self) -> List[str]:
        """Returns a list of imports to be put at the top of a generated notebook.

        Returns
        -------
        List[str]
            List of import strings to be added at the top of the generated notebook,
            e.g. ["import pandas as pd", "import numpy as np"].
        """
        if self.verbosity <= Verbosity.MEDIUM:
            return [
                "from edvart.report_sections.timeseries_analysis.rolling_statistics"
                " import show_rolling_statistics"
            ]
        return [
            "import warnings",
            "from IPython.display import display, Markdown",
            "import plotly.graph_objects as go",
            "from edvart.data_types import is_numeric",
        ]

    def add_cells(self, cells: List[Dict[str, Any]], df: pd.DataFrame) -> None:
        """Adds cells to the list of cells.

        Cells can be either code cells or markdown cells.

        Parameters
        ----------
        cells : List[Dict[str, Any]]
            List of generated notebook cells which are represented as dictionaries
        df: pd.DataFrame
            Data for which to add the cells.
        """
        section_header = nbfv4.new_markdown_cell(self.get_title(section_level=2))
        cells.append(section_header)

        default_call = "show_rolling_statistics(df=df"
        if self.columns is not None:
            default_call += f", columns={self.columns}"
        default_call += ")"

        if self.verbosity <= Verbosity.MEDIUM:
            code = default_call
        else:
            code = get_code(show_rolling_statistics) + "\n\n" + default_call

        cells.append(nbfv4.new_code_cell(code))

    def show(self, df: pd.DataFrame) -> None:
        """Generates rolling statistics interactive plot(s) in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output
        """
        display(Markdown(self.get_title(section_level=2)))
        show_rolling_statistics(df=df, columns=self.columns, window_size=self.window_size)


# pylint: disable=too-many-locals
@check_index_time_ascending
def show_rolling_statistics(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    window_size: int = 20,
    show_bands: bool = True,
    band_width: float = 1.0,
    show_std_dev: bool = True,
    color_mean: str = "#2040FF",
    color_band: str = "#90E0FF",
    color_std: str = "#CD5C5C",
) -> None:
    """Display rolling statistics interactive plot.

    Displays a separate plot for each column of df.

    Parameters
    ----------
    df : pd.DataFrame
        Data to analyze.
    columns : List[str], optional
        Columns to analyze. Only numeric columns can be analyzed.
        All numeric columns are used by default.
    window_size : int (default = 20)
        Size of the rolling window to use when computing rolling statistics.
    show_bands : bool (default = True)
        Whether to show lines delimiting the range
        [rolling_mean - band_width * rolling_std, rolling_mean + band_width * rolling_std]
    band_width : float (default = 1.)
        Multiple of standard deviation from mean to show bands at.
        Ignored if not showing bands.
    show_std_dev : bool (default = True)
        Whether to plot rolling standard deviation.
    color_mean : str (default = "#2040FF")
        Color of the line showing rolling mean.
    color_band : str (default = "#90E0FF")
        Color of the lines showing bands around rolling mean. Ignored if not showing bands.
    color_std : str (default = "#CD5C5C")
        Color of the line showing standard deviation. Ignored if not showing standard deviation.

    Raises
    ------
    ValueError
        If the input data is not indexed by time in ascending order.
    """
    if window_size >= len(df):
        window_size = len(df) - 1
        warnings.warn(f"Reducing window size to {window_size} due to insufficient samples.")
    if columns is None:
        columns = [col for col in df.columns if is_numeric(df[col])]
    else:
        for col in columns:
            if not is_numeric(df[col]):
                raise ValueError(f"Cannot plot rolling statistics for non-numeric column `{col}`")

    df_rolling = df[columns].rolling(window_size)
    df_rolling_mean = df_rolling.mean()[window_size - 1 :]
    df_rolling_std = df_rolling.std()[window_size - 1 :]
    index = df.index[window_size - 1 :]

    layout = dict(xaxis_rangeslider_visible=True)

    data = []
    for col in columns:
        data.append([])
        if show_std_dev:
            trace_std = go.Scatter(
                x=index,
                y=df_rolling_std[col],
                mode="lines",
                name="Rolling std. dev.",
                line={"color": color_std},
            )
            data[-1].append(trace_std)

        trace_mean = go.Scatter(
            x=index,
            y=df_rolling_mean[col],
            mode="lines",
            name="Rolling mean",
            line={"color": color_mean},
        )
        data[-1].append(trace_mean)

        if show_bands:
            # Plot upper band
            trace_mean_plus_std = go.Scatter(
                x=index,
                y=df_rolling_mean[col] + band_width * df_rolling_std[col],
                mode="lines",
                # pylint: disable=consider-using-f-string
                name=(
                    "Rolling mean + {} rolling std. dev.".format(
                        "" if band_width == 1 else str(band_width) + " * "
                    )
                ),
                line={"color": color_band},
            )
            # Plot lower band
            trace_mean_minus_std = go.Scatter(
                x=index,
                y=df_rolling_mean[col] - band_width * df_rolling_std[col],
                mode="lines",
                # pylint: disable=consider-using-f-string
                name=(
                    "Rolling mean - {} rolling std. dev.".format(
                        "" if band_width == 1 else str(band_width) + " * "
                    )
                ),
                line={"color": color_band},
            )
            data[-1].extend([trace_mean_plus_std, trace_mean_minus_std])

    for col_name, col_data in zip(columns, data):
        display(Markdown(f"---\n### {col_name}"))
        go.Figure(data=col_data, layout=layout).show()
