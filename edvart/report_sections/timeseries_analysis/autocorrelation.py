import functools
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import nbformat.v4 as nbfv4
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from statsmodels.graphics import tsaplots

from edvart.data_types import is_numeric
from edvart.decorators import check_index_time_ascending
from edvart.report_sections.code_string_formatting import get_code
from edvart.report_sections.section_base import Section, Verbosity


class Autocorrelation(Section):
    """Generates autocorrelation (ACF) and partial autocorrelation function (PACF) subsection.

    Parameters
    ----------
    verbosity : Verbosity (default = Verbosity.LOW)
        Verbosity of the generated code in the exported notebook.
    columns : List[str], optional
        List of columns to analyze. Only numeric column can be analyzed.
        All numeric columns are analyzed by default.
    """

    @property
    def name(self) -> str:
        return "Autocorrelation"

    def required_imports(self) -> List[str]:
        """Returns a list of imports to be put at the top of a generated notebook.

        Returns
        -------
        List[str]
            List of import strings to be added at the top of the generated notebook,
            e.g. ["import pandas as pd", "import numpy as np"].
        """
        if self.verbosity == Verbosity.LOW:
            return [
                "from edvart.report_sections.timeseries_analysis.autocorrelation"
                " import show_autocorrelation"
            ]
        if self.verbosity == Verbosity.MEDIUM:
            return [
                "from edvart.report_sections.timeseries_analysis.autocorrelation"
                " import plot_acf, plot_pacf"
            ]
        # Verbosity.HIGH
        return [
            "import functools",
            "import matplotlib.pyplot as plt",
            "import statsmodels.graphics.tsaplots as tsaplots",
            "from IPython.display import display, Markdown",
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
        if self.verbosity == Verbosity.LOW:
            section_header = nbfv4.new_markdown_cell(self.get_title(section_level=2))
            default_call = "show_autocorrelation(df=df"
            if self.columns is not None:
                default_call += f", columns={self.columns}"
            default_call += ")"
            cells.append(section_header)
            cells.append(nbfv4.new_code_cell(default_call))
            return

        section_header_acf = nbfv4.new_markdown_cell(self.get_title(section_level=2))
        default_call_acf = "plot_acf(df=df"
        if self.columns is not None:
            default_call_acf += f", columns={self.columns}"
        default_call_acf += ")"

        section_header_pacf = nbfv4.new_markdown_cell("## Partial autocorrelation")
        default_call_pacf = "plot_pacf(df=df"
        if self.columns is not None:
            default_call_pacf += f", columns={self.columns}"
        default_call_pacf += ")"

        if self.verbosity == Verbosity.MEDIUM:
            code_acf = default_call_acf
            code_pacf = default_call_pacf
        else:
            code_acf = get_code(plot_acf) + "\n\n" + default_call_acf
            code_pacf = get_code(plot_pacf) + "\n\n" + default_call_pacf

        cells.append(section_header_acf)
        cells.append(nbfv4.new_code_cell(code_acf))
        cells.append(section_header_pacf)
        cells.append(nbfv4.new_code_cell(code_pacf))

    def show(self, df: pd.DataFrame) -> None:
        """Generates ACF and PACF plots in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output
        """
        display(Markdown(self.get_title(section_level=2)))
        show_autocorrelation(df=df, columns=self.columns)


@check_index_time_ascending
def show_autocorrelation(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    lags: Optional[List[int]] = None,
    figsize: Tuple[float, float] = (15, 6),
) -> None:
    """Generate autocorrelation (ACF) and partial autocorrelation function (PACF) plots.

    ACF returns, for a given lag, correlation between
    a given timeseries and itself shifted by this lag.

    Partial autocorrelation function returns conditional autocorrelation given all smaller
    lag values up to the given lag.

    Parameters
    ----------
    df: pd.DataFrame
        Data to analyze.
    columns : List[str], optional
        List of columns to analyze. Only numeric column can be analyzed.
        All numeric columns are analyzed by default.
    lags: List[int], optional
        List of lag values to plot ACF for
    figsize : Tuple[float, float] (default = (15, 6))
        Size of figure of (partial) autocorrelation plot.

    Raises
    ------
    ValueError
        If the input data is not indexed by time in ascending order.
    """
    plot_acf(df, columns, lags, figsize, partial=False)
    display(Markdown("## Partial autocorrelation"))
    plot_pacf(df, columns, lags, figsize)


@check_index_time_ascending
def plot_acf(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    lags: Optional[List[int]] = None,
    figsize: Tuple[float, float] = (15, 6),
    partial: bool = False,
) -> None:
    """Plot ACF or PACF.

    Autocorrelation function (ACF) returns, for a given lag, correlation between
    the timeseries and itself shifted by this lag.

    Partial autocorrelation function (PACF) returns conditional autocorrelation given all
    smaller lag values up to the given lag.

    Parameters
    ----------
    df: pd.DataFrame
        Data to analyze.
    columns : List[str], optional
        List of columns to analyze. Only numeric column can be analyzed.
        All numeric columns are analyzed by default.
    lags: List[int], optional
        List of lag values to plot ACF for
    figsize : Tuple[float, float] (default = (15, 6))
        Size of figure of (partial) autocorrelation plot.
    partial : bool (default = False)
        If True, PACF will be plotted, otherwise, ACF will be plotted.
    """
    if columns is None:
        columns = [col for col in df.columns if is_numeric(df[col])]
    else:
        for col in columns:
            if not is_numeric(df[col]):
                raise ValueError(f"Cannot plot autocorrelation for non-numeric column `{col}`")
    plot_func = (
        functools.partial(tsaplots.plot_pacf, method="ywm") if partial else tsaplots.plot_acf
    )
    if partial and lags is None:
        nobs = len(df)
        # See https://github.com/statsmodels/statsmodels/blob/01b19d7d111b29c183f620ff0a949ef6391ff8ee/statsmodels/graphics/tsaplots.py#L20
        default_lags_limit = int(np.ceil(10 * np.log10(nobs))) + 1
        # Partial function can only be computed for lags up to nobs // 2 - 1
        lags_limit = min(default_lags_limit, nobs // 2)
        lags = list(range(lags_limit))
    for col in columns:
        display(Markdown(f"---\n### {col}"))
        fig = plot_func(df[col].dropna(), lags=lags)
        ax = fig.axes[0]
        ax.set_title("")
        ax.set_xlabel("Lag")
        ax.set_ylabel(("Partial " if partial else "") + "Autocorrelation")
        fig.set_size_inches(*figsize)
        plt.show()


@check_index_time_ascending
def plot_pacf(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    lags: Optional[List[int]] = None,
    figsize: Tuple[float, float] = (15, 6),
) -> None:
    """Plot PACF.

    Partial autocorrelation function (PACF) returns conditional autocorrelation given all
    smaller lag values up to the given lag.

    Parameters
    ----------
    df: pd.DataFrame
        Data to analyze.
    columns : List[str], optional
        List of columns to analyze. Only numeric column can be analyzed.
        All numeric columns are analyzed by default.
    lags: List[int], optional
        List of lag values to plot ACF for
    figsize : Tuple[float, float] (default = (15, 6))
        Size of figure of (partial) autocorrelation plot.
    """
    plot_acf(df, columns, lags, figsize, partial=True)
