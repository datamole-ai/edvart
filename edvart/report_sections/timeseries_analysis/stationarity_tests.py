import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import nbformat.v4 as nbfv4
import pandas as pd
import statsmodels.api as sm
from IPython.display import Markdown, display

from edvart.data_types import is_numeric
from edvart.decorators import check_index_time_ascending
from edvart.pandas_formatting import format_number
from edvart.report_sections.code_string_formatting import get_code
from edvart.report_sections.section_base import Section, Verbosity


class StationarityTests(Section):
    """Generates the stationarity tests subsection.

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
        return "Stationarity tests"

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
                "from edvart.report_sections.timeseries_analysis.stationarity_tests"
                " import show_stationarity_tests"
            ]
        return [
            "from IPython.display import display, Markdown",
            "import statsmodels.api as sm",
            "from functools import partial",
            "from edvart.data_types import is_numeric",
            "from edvart.pandas_formatting import format_number",
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
        default_call = "show_stationarity_tests(df=df"
        if self.columns is not None:
            default_call += f", columns={self.columns}"
        default_call += ")"

        if self.verbosity <= Verbosity.MEDIUM:
            code = default_call
        else:
            code = (
                get_code(default_stationarity_tests)
                + "\n\n"
                + get_code(show_stationarity_tests)
                + "\n\n"
                + default_call
            )

        cells.append(nbfv4.new_code_cell(code))

    def show(self, df: pd.DataFrame) -> None:
        """Generates seasonal decomposition plot(s) in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output
        """
        display(Markdown(self.get_title(section_level=2)))
        show_stationarity_tests(df=df, columns=self.columns)


def default_stationarity_tests() -> Dict[pd.Series, Callable[[pd.Series], "test_result"]]:
    """Return a dictionary of stationarity test and functions.

    Stationarity tests are:

        KPSS (constant)
            which has a null hypothesis that a given series is stationary around a
            constant value

        KPSS (trend)
            which has a null hypothesis that a given series is stationary around a
            constant-slope, i.e. a linear function

        Augmented Dickey-Fuller
            which has a null hypothesis of a unit root, i.e. non-stationarity.

    Returns
    -------
    Dict[str, Callable]
        A dictionary from test name to function.
    """
    return {
        "KPSS Test (constant)": partial(sm.tsa.stattools.kpss, regression="c", nlags="auto"),
        "KPSS Test (trend)": partial(sm.tsa.stattools.kpss, regression="ct", nlags="auto"),
        "Augmented Dickey-Fuller": sm.tsa.stattools.adfuller,
    }


@check_index_time_ascending
def show_stationarity_tests(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    kpss_const: bool = True,
    kpss_trend: bool = True,
    adfuller: bool = True,
) -> None:
    """Show stationarity for each numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        Data to test.
    columns : List[str], optional
        List of columns to test. Only numeric columns can be used.
        All numeric columns are used by default.
    kpss_const : bool (default = True)
        Whether to perform KPSS (constant) test.
    kpss_trend : bool (default = True)
        Whether to perform KPSS (trend) test.
    adfuller : bool (default = True)
        Whether to perform Augmented Dickey-Fuller test.

    Raises
    ------
    ValueError
        If the input data is not indexed by time in ascending order.
    """
    df = df.dropna()
    if columns is None:
        columns = [col for col in df.columns if is_numeric(df[col])]
    stat_tests = default_stationarity_tests()
    if not kpss_const:
        stat_tests.pop("KPSS Test (constant)", None)
    if not kpss_trend:
        stat_tests.pop("KPSS Test (trend)", None)
    if not adfuller:
        stat_tests.pop("Augmented Dickey-Fuller Test", None)

    for col in columns:
        if not is_numeric(df[col]):
            raise ValueError(f"Cannot test stationarity of non-numeric column `{col}`")
        test_values_df = pd.DataFrame()
        display(Markdown(f"---\n### {col}"))
        for name, func in stat_tests.items():
            with warnings.catch_warnings(record=True) as warns:
                test_vals = func(df[col])
            stat, pvalue = test_vals[:2]
            value_dict = {
                "Test statistic": format_number(stat, thousand_separator=" "),
                "P-value": ("<" if len(warns) > 0 else "")
                + format_number(pvalue, thousand_separator=" "),
            }
            value_series = pd.Series(value_dict)
            test_values_df[name] = value_series
        display(test_values_df.style)
