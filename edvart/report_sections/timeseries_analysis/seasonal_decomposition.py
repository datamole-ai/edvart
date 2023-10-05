from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import nbformat.v4 as nbfv4
import pandas as pd
import statsmodels.api as sm
from IPython.display import Markdown, display

from edvart.data_types import is_numeric
from edvart.decorators import check_index_time_ascending
from edvart.report_sections.code_string_formatting import get_code
from edvart.report_sections.section_base import Section, Verbosity


class SeasonalDecomposition(Section):
    """Generates seasonal decomposition subsection.

    Each timeseries represented by one column is decomposed into
    trend, seasonal and residual (noise) components. This is a primitive decomposition.
    The seasonal component is first removed by applying a convolution filter to the data.
    The average of this smoothed series for each period is the returned seasonal component.

    Parameters
    ----------
    verbosity : Verbosity (default = Verbosity.LOW)
        Verbosity of the generated code in the exported notebook.
    columns : List[str], optional
        List of columns to analyze. Only numeric column can be analyzed.
        All numeric columns are analyzed by default.
    period : int, optional
        Period to use when modelling seasonal component. If None, period is inferred from
        frequency of df.index, provided `pd.infer_freq` is able to infer the frequency.
        Otherwise, this parameter has to be set manually.
    model : str (default = "additive")
        Can be either "multiplicative" or "additive".
        If "additive", series is modelled as series = trend + seasonal + noise
        If "multiplicative", series is modelled as series = trend * seasonal * noise
    """

    def __init__(
        self,
        verbosity: Verbosity = Verbosity.LOW,
        columns: Optional[List[str]] = None,
        period: Optional[int] = None,
        model: str = "additive",
    ):
        self.model = model
        self.period = period
        super().__init__(verbosity, columns)

    @property
    def name(self) -> str:
        return "Seasonal decomposition"

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
                "from edvart.report_sections.timeseries_analysis.seasonal_decomposition"
                " import show_seasonal_decomposition"
            ]
        return [
            "from IPython.display import display, Markdown",
            "import matplotlib.pyplot as plt",
            "import statsmodels.api as sm",
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
        default_call = "show_seasonal_decomposition(df=df"
        if self.columns is not None:
            default_call += f", columns={self.columns}"
        if self.period is not None:
            default_call += f", period={self.period}"
        default_call += f", model='{self.model}')"

        if self.verbosity <= Verbosity.MEDIUM:
            code = default_call
        else:
            code = get_code(show_seasonal_decomposition) + "\n\n" + default_call

        cells.append(nbfv4.new_code_cell(code))

    def show(self, df: pd.DataFrame) -> None:
        """Generates seasonal decomposition plot(s) in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output
        """
        display(Markdown(self.get_title(section_level=2)))
        show_seasonal_decomposition(
            df=df, columns=self.columns, model=self.model, period=self.period
        )


@check_index_time_ascending
def show_seasonal_decomposition(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    period: Optional[int] = None,
    model: str = "additive",
    figsize: Tuple[float, float] = (20, 10),
) -> None:
    """Generate the seasonal decomposition plot.

    Parameters
    ----------
    df : pd.DataFrame
        Data to analyze.
    columns : List[str], optional
        List of columns to analyze. Only numeric column can be analyzed.
        All numeric columns are analyzed by default.
    period : int, optional
        Period to use when modelling seasonal component. If None, period is inferred from
        frequency of df.index, provided `pd.infer_freq` is able to infer the frequency.
        Otherwise, this parameter has to be set manually.
    model : str (default = "additive")
        Can be either "multiplicative" or "additive".
        If "additive", series is modelled as series = trend + seasonal + noise
        If "multiplicative", series is modelled as series = trend * seasonal * noise
    figsize : Tuple[float, float] (default = (20, 10))
        Size of the whole figure for one column (i.e. includes plots of all components).

    Raises
    ------
    ValueError
        If the input data is not indexed by time in ascending order.
    """
    df = df.interpolate(method="time")
    if pd.infer_freq(df.index) is None and period is None:
        display(
            Markdown(
                "<div class='alert alert-block alert-warning'>"
                "Period could not be inferred, please set the `period` parameter"
                " to a suitable value. Not plotting seasonal decomposition."
                "</div>"
            )
        )
        return
    if columns is None:
        columns = [col for col in df.columns if is_numeric(df[col])]

    for col in columns:
        if not is_numeric(df[col]):
            raise ValueError(f"Cannot plot decomposition for non-numeric column {col}")
        display(Markdown(f"---\n### {col}"))
        decomposition = sm.tsa.seasonal_decompose(df[col], period=period, model=model)
        fig = decomposition.plot()
        fig.set_size_inches(*figsize)
        fig.axes[0].set_title(None)
        fig.axes[0].set_ylabel("Original")
        fig.axes[-1].set_ylabel("Residual")
        plt.show()
