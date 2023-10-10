from datetime import datetime
from itertools import takewhile
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import nbformat.v4 as nbfv4
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

from edvart.data_types import is_numeric
from edvart.decorators import check_index_time_ascending
from edvart.report_sections.code_string_formatting import get_code
from edvart.report_sections.section_base import Section, Verbosity


class BoxplotsOverTime(Section):
    """Generates the boxplots over time intervals section.

    For each column, generates a series of boxes, each box representing distribution of values
    in the given column during a time interval.

    Parameters
    ----------
    verbosity : Verbosity (default = Verbosity.LOW)
        Verbosity of the generated code in the exported notebook.
    columns : List[str], optional
        List of columns to analyze. Only numeric column can be analyzed.
        All numeric columns are analyzed by default.
    grouping_function : Callable[[Any], str], optional
        Function to group the data into intervals. Cannot pass an anonymous function, i.e. the
        function must be assigned an identifier.
        To pass a lambda, simply assign it to a variable and pass the variable.
        If None is passed, a default grouping will be selected (see `default_nunique_max`).
    grouping_function_imports: List[str], optional
        Additional imports required for the grouping function.
    grouping_name : str, optional
        Name of grouping, will be displayed as title of the horizontal axis.
    default_nunique_max : int (default = 80)
        If no grouping function is passed, the most granular grouping which produces at most
        `default_nunique_max` unique values is selected from the following:
        Hour, Day, Week, Month, Quarter, Year, Decade.
        If a default grouping is selected,
        a corresponding name is displayed on the horizontal axis by default
    """

    def __init__(
        self,
        verbosity: Verbosity = Verbosity.LOW,
        columns: Optional[List[str]] = None,
        grouping_function: Callable[[Any], str] = None,
        grouping_function_imports: Optional[List[str]] = None,
        grouping_name: Optional[str] = None,
        default_nunique_max: int = 80,
    ):
        self.grouping_function = grouping_function
        self.grouping_function_imports = grouping_function_imports
        self.grouping_name = grouping_name
        self.default_nunique_max = default_nunique_max
        super().__init__(verbosity, columns)

    @property
    def name(self) -> str:
        return "Boxplots over time intervals"

    def required_imports(self) -> List[str]:
        """Returns a list of imports to be put at the top of a generated notebook.

        Returns
        -------
        List[str]
            List of import strings to be added at the top of the generated notebook,
            e.g. ["import pandas as pd", "import numpy as np"].
        """
        if self.verbosity <= Verbosity.MEDIUM:
            imports = [
                "from edvart.report_sections.timeseries_analysis.boxplots_over_time"
                " import show_boxplots_over_time"
            ]
        else:
            imports = [
                "from datetime import datetime",
                "import matplotlib.pyplot as plt",
                "import seaborn as sns",
                "from IPython.display import display, Markdown",
                "from edvart.data_types import is_numeric",
            ]
        if self.grouping_function_imports is not None:
            imports.extend(self.grouping_function_imports)
        return imports

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
        if self.grouping_function is not None:
            grouping_func_code = get_code(self.grouping_function) + "\n\n"
            if grouping_func_code.startswith("def"):
                grouping_func_name = self.grouping_function.__name__
            else:
                # function was created using an assignment to a variable
                # -> get_code returns <var_name> = <lambda or function> -> we want <var_name>
                grouping_func_name = "".join(
                    takewhile(
                        lambda char: char.isalnum() or char == "_",
                        list(grouping_func_code),
                    )
                )
            cells.append(nbfv4.new_code_cell(grouping_func_code))
        default_call = "show_boxplots_over_time(df=df"
        if self.columns is not None:
            default_call += f", columns={self.columns}"
        if self.grouping_function is not None:
            default_call += f", grouping_function={grouping_func_name}"
        if self.grouping_name is not None:
            default_call += f", grouping_name='{self.grouping_name}'"
        default_call += ")"

        if self.verbosity <= Verbosity.MEDIUM:
            code = default_call
        else:
            if self.grouping_function is None:
                code = (
                    get_code(default_grouping_functions)
                    + "\n\n"
                    + get_code(get_default_grouping_func)
                    + "\n\n"
                )
            else:
                code = ""

            code += get_code(show_boxplots_over_time) + "\n\n" + default_call

        cells.append(nbfv4.new_code_cell(code))

    def show(self, df: pd.DataFrame) -> None:
        """Generates boxplots grouped over time intervals in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output
        """
        display(Markdown(self.get_title(section_level=2)))
        show_boxplots_over_time(
            df=df,
            columns=self.columns,
            grouping_function=self.grouping_function,
            grouping_name=self.grouping_name,
            default_nunique_max=self.default_nunique_max,
        )


def default_grouping_functions() -> Dict[str, Callable[[datetime], str]]:
    """Return a dictionary of function names and functions.

    The function takes a pandas datetime and represents it as a rougher (in terms of time)
    string, which can be used for grouping.
    Available groupings are: Hour, Day, Week, Month, Quarter, Year, Decade

    Returns
    -------
    Dict[str, Callable[[datetime], str]]
        Dictionary from grouping function names to grouping functions.
    """
    return {
        "Hour": lambda x: f"{x.day}/{x.month}/{x.year} {x.hour}:00",
        "Day": lambda x: f"{x.day}/{x.month}/{x.year}",
        "Week": lambda x: f"W{x.week}, {x.year if x.dayofweek < x.dayofyear else x.year - 1}",
        "Month": lambda x: f"{x.month_name()[:3]} {x.year}",
        "Quarter": lambda x: f"Q{x.quarter} {x.year}",
        "Year": lambda x: f"{x.year}",
        "Decade": lambda x: f"{x.year // 10 * 10}s",
    }


def get_default_grouping_func(df: pd.DataFrame, nunique_max: int = 80) -> Tuple[str, Callable]:
    """Return the most granular function to group df.index into at most nunique_max intervals.

    Uses grouping functions from `default_grouping_functions`.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe, for index of which a suitable grouping
    nunique_max : int (default = 80)
        Maximum number of intervals to group to.

    Returns
    -------
    Tuple[str, Callable]
        Name of selected grouping function and the grouping function itself.
    """
    # find most granular grouping which does not produce too many values
    index_series = df.index.to_series()
    items = list(default_grouping_functions().items())
    assert items  # check that there are some values

    for name, func in items:
        if index_series.apply(func).nunique() <= nunique_max:
            return name, func
    # If no grouping is rough enough, use the roughest available
    return items[-1]


@check_index_time_ascending
def show_boxplots_over_time(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    grouping_function: Callable[[Any], str] = None,
    grouping_name: Optional[str] = None,
    default_nunique_max: int = 80,
    figsize: Tuple[float, float] = (20, 7),
    color: Any = None,
) -> None:
    """Generate boxplots over time intervals.

    For each column, generates a series of boxes, each box representing distribution of values
    in the given column during a time interval.

    Parameters
    ----------
    df : pd.DataFrame
        Data to analyze.
    columns : List[str], optional
        List of columns to analyze. Only numeric column can be analyzed.
        All numeric columns are analyzed by default.
    grouping_function : Callable[[Any], str], optional
        Function to group the data into intervals. Cannot pass an anonymous function, i.e. the
        function must be assigned an identifier.
        To pass a lambda, simply assign it to a variable and pass the variable.
        If None is passed, a default grouping will be selected (see `default_nunique_max`).
    grouping_name : str, optional
        Name of grouping, will be displayed as title of the horizontal axis.
    default_nunique_max : int, optional
        If no grouping function is passed, the most granular grouping which produces at most
        `default_nunique_max` unique values is selected from the following:
        Hour, Day, Week, Month, Quarter, Year, Decade.
        If a default grouping is selected,
        a corresponding name is displayed on the horizontal axis by default
    figsize : Tuple[float, float] (default = (20, 7))
        Size of boxplot series figure for each column.
    color : Any, optional
        Color or color map compatible with matplotlib/seaborn.
        By default a "rainbow" color map - color of individual boxes changes over time.

    Raises
    ------
    ValueError
        If the input data is not indexed by time in ascending order.
    """
    default_grouping_funcs = default_grouping_functions()
    if grouping_function is None:
        grouping_name, grouping_function = get_default_grouping_func(
            df, nunique_max=default_nunique_max
        )
    elif default_grouping_funcs.get(grouping_name) is not None:
        grouping_function = default_grouping_funcs[grouping_name]

    if columns is None:
        columns = [col for col in df.columns if is_numeric(df[col])]

    for column in columns:
        if not is_numeric(df[column]):
            raise ValueError(f"Cannot plot boxplot for non-numeric column `{column}`")
        display(Markdown(f"---\n### {column}"))
        ax = sns.boxplot(
            x=df.index.to_series().apply(grouping_function),
            y=df[column],
            color=color,
        )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        if grouping_name is not None:
            ax.set_xlabel(grouping_name)
        ax.figure.set_size_inches(*figsize)
        plt.show()
