from typing import Any, Dict, List, Optional

import nbformat.v4 as nbfv4
import pandas as pd
import plotly.graph_objects as go
from IPython.display import Markdown, display

from edvart.data_types import is_categorical, is_numeric
from edvart.decorators import check_index_time_ascending
from edvart.report_sections.code_string_formatting import get_code
from edvart.report_sections.section_base import Section, Verbosity


class TimeSeriesLinePlot(Section):
    """Generates the time series line plot section.

    Parameters
    ----------
    verbosity : Verbosity (default = Verbosity.LOW)
        Verbosity of the code generated in the exported notebook.
    columns : List[str], optional
        List of columns to analyze. Only numeric column can be analyzed.
        All numeric columns are analyzed by default.
    separate_plots : bool (default = False)
        Whether to plot each column in a separate plot.
        All columns are plotted in a single plot by default.
    color_col : str, optional
        Which column to use for coloring of the lines. Each segment of the line will be colored
        according to value in this column in the given time point. If this parameter is set, each
        column will be plotted in a separate plot (separate_plots param is ignored).
    """

    def __init__(
        self,
        verbosity: Verbosity = Verbosity.LOW,
        columns: Optional[List[str]] = None,
        separate_plots: bool = False,
        color_col: Optional[str] = None,
    ):
        self.separate_plots = separate_plots
        self.color_col = color_col
        super().__init__(verbosity, columns)

    @property
    def name(self) -> str:
        return "Time series line plot"

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
                "from edvart.report_sections.timeseries_analysis.time_series_line_plot"
                " import show_time_series_line_plot"
            ]
        return [
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

        default_call = "show_time_series_line_plot(df=df"
        if self.columns is not None:
            default_call += f", columns={self.columns}"
        if self.color_col is not None:
            default_call += f", color_col='{self.color_col}'"
        if self.separate_plots:
            default_call += ", separate_plots=True"
        default_call += ")"

        if self.verbosity <= Verbosity.MEDIUM:
            code = default_call
        else:
            code = (
                get_code(show_time_series_line_plot)
                + "\n\n"
                + get_code(_time_series_line_plot_colored)
                + "\n\n"
                + default_call
            )

        cells.append(nbfv4.new_code_cell(code))

    def show(self, df: pd.DataFrame) -> None:
        """Generates time series line plot(s) in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output
        """
        display(Markdown(self.get_title(section_level=2)))
        show_time_series_line_plot(
            df=df,
            columns=self.columns,
            color_col=self.color_col,
            separate_plots=self.separate_plots,
        )


@check_index_time_ascending
def show_time_series_line_plot(
    df,
    columns: Optional[List[str]] = None,
    separate_plots: bool = False,
    color_col: Optional[str] = None,
) -> None:
    """Display time series line plot.

    Parameters
    ----------
    df: pd.DataFrame
        Data to analyze.
    columns: List[str], optional
        Columns to analyze. Only numeric columns can be analyzed.
        All numeric columns are used by default.
    separate_plots: bool (default = False)
        Whether to plot each column in a separate plot.
        All columns are plotted in a single plot by default.
    color_col: str, optional
        Name of column to use for coloring of the lines. Each segment of the line will be
        colored according to value in this column in the given time point. If this parameter is
        set, each column will be plotted in a separate plot (separate_plots param is ignored).

    Raises
    ------
    ValueError
        If the input data is not indexed by time in ascending order.
    """
    if color_col is not None:
        _time_series_line_plot_colored(df, columns=columns, color_col=color_col)
        return
    if columns is None:
        columns = [col for col in df.columns if is_numeric(df[col])]
    else:
        for col in columns:
            if not is_numeric(df[col]):
                raise ValueError(
                    f"Cannot plot TimeSeriesLinePlot plot for non-numeric column {col}"
                )

    data = [go.Scatter(x=df.index, y=df[col], name=col, mode="lines") for col in columns]

    layout = dict(xaxis_rangeslider_visible=True)
    if separate_plots:
        for trace in data:
            display(Markdown(f"### {trace.name}"))
            go.Figure(data=trace, layout=layout).show()
    else:
        go.Figure(data=data, layout=layout).show()


def _time_series_line_plot_colored(df, columns=None, color_col=None):
    if columns is None:
        columns = [col for col in df.columns if is_numeric(df[col])]
    else:
        for col in columns:
            if not is_numeric(df[col]):
                raise ValueError(
                    f"Cannot plot TimeSeriesLinePlot plot for non-numeric column {col}"
                )

    layout = dict(xaxis_rangeslider_visible=True)
    if not is_categorical(df[color_col]):
        raise ValueError(f"Cannot color by non-categorical column `{color_col}`")
    df_color_shifted = df[color_col].shift(-1)
    for col in columns:
        data = [
            go.Scatter(
                x=df.index,
                # GroupBy would normally be preferred, but we want a connected line
                # Therefore, we also plot a connecting line
                # to the next point where category changes
                y=df[col].mask((df[color_col] != category) & (df_color_shifted != category)),
                name=str(category),
                mode="lines",
                connectgaps=False,
            )
            for category in df[color_col].unique()
        ]
        display(Markdown(f"### {col}"))
        go.Figure(data=data, layout=layout).show()
