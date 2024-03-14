import itertools
from enum import IntEnum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import nbformat.v4 as nbfv4
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

from edvart import utils
from edvart.data_types import is_boolean, is_categorical, is_numeric
from edvart.report_sections.code_string_formatting import get_code
from edvart.report_sections.section_base import ReportSection, Section, Verbosity


# pylint:disable=invalid-name
class BivariateAnalysisSubsection(IntEnum):
    """Enum of all implemented bivariate analysis subsections."""

    CorrelationPlot = 0
    PairPlot = 1
    ContingencyTable = 2

    def __str__(self):
        return self.name


class BivariateAnalysis(ReportSection):
    """Generates the Bivariate analysis section of the report.

    Contains an enum BivariateAnalysisSubsection of possible subsections.

    Parameters
    ----------
    subsections : List[BivariateAnalysisSubsection], optional
        List of subsections to include.
        All subsection in BivariateAnalysisSubsection are included by default.
    verbosity : Verbosity (default = Verbosity.LOW)
        Generated code verbosity global to the Bivariate analysis sections.
        If subsection verbosities are None, then they will be overridden by this parameter.
    columns : List[str], optional
        Columns on which to do bivariate analysis.
        If none of columns_x, columns_y and columns_pairs is specified bivariate analysis is
        performed on all pairs of `columns`.
        Ignored if `columns_x` and `columns_y` is specified. Ignored in contingency table
        if `columns_x` and `columns_y` or `columns_pairs` is specified.
        All columns are used by default.
    columns_x : List[str], optional
        If specified, correlations and pairplots are performed on the cartesian product of
        `columns_x` and `columns_y`.
        If `columns_x` is specified, then `columns_y` must also be specified.
    columns_y : List[str], optional
        If specified, correlations and pairplots are performed on the cartesian product of
        `columns_x` and `columns_y`.
        If `columns_y` is specified, then `columns_x` must also be specified.
    columns_pairs : List[str], optional
        List of columns pairs on which to perform bivariate analysis.
        Used primarily in contingency tables.
        If specified, `columns`, `columns_x` and `columns_y` are ignored in contingency tables.
        Ignored in pairplots and correlations unless `columns_pairs` is specified and none of
        `columns`, `columns_x`, `columns_y` is specified. In that case, the first elements of each
        pair are treated as `columns_x` and the second elements as `columns_y` in pairplots and
        correlations.
    verbosity_correlations : Verbosity, optional
        Correlation plots subsection code verbosity.
    verbosity_pairplot: Verbosity, optional
        Pairplot subsection code verbosity.
    verbosity_contingency_table: Verbosity, optional
        Contingency table subsection code verbosity.
    color_col : str, optional
        Name of column according to use for coloring of the bivariate analysis subsections.
        Coloring is currently supported in pairplot.

    Raises
    ------
    ValueError
        If exactly one of `columns_x`, `columns_y` is specified.
    """

    # By default use all subsections
    _DEFAULT_SUBSECTIONS_TO_SHOW = list(BivariateAnalysisSubsection)

    def __init__(
        self,
        subsections: Optional[List[BivariateAnalysisSubsection]] = None,
        verbosity: Verbosity = Verbosity.LOW,
        columns: Optional[List[str]] = None,
        columns_x: Optional[List[str]] = None,
        columns_y: Optional[List[str]] = None,
        columns_pairs: Optional[List[Tuple[str, str]]] = None,
        verbosity_correlations: Optional[Verbosity] = None,
        verbosity_pairplot: Optional[Verbosity] = None,
        verbosity_contingency_table: Optional[Verbosity] = None,
        color_col: Optional[str] = None,
    ):
        verbosity_correlations = verbosity_correlations or verbosity
        verbosity_pairplot = verbosity_pairplot or verbosity
        verbosity_contingency_table = verbosity_contingency_table or verbosity

        # Store subsection verbosities
        self.subsection_verbosities = {
            BivariateAnalysisSubsection.CorrelationPlot: verbosity_correlations,
            BivariateAnalysisSubsection.PairPlot: verbosity_pairplot,
            BivariateAnalysisSubsection.ContingencyTable: verbosity_contingency_table,
        }

        if subsections is None:
            self.subsections_to_show = self._DEFAULT_SUBSECTIONS_TO_SHOW
        else:
            self.subsections_to_show = subsections
        self.subsections_to_show_with_low_verbosity = [
            sub
            for sub in self.subsections_to_show
            if self.subsection_verbosities[sub] == Verbosity.LOW
        ]

        if (columns_x is None) != (columns_y is None):
            raise ValueError("Either both or neither of columns_x, columns_y must be specified.")
        # For analyses which do not take columns_pairs, prepare columns_x and columns_y in case
        # columns_pairs is the only parameter specified
        columns_x_no_pairs: Optional[List[str]]
        columns_y_no_pairs: Optional[List[str]]
        if columns is None and columns_x is None and columns_pairs is not None:
            columns_x_no_pairs = [pair[0] for pair in columns_pairs]
            columns_y_no_pairs = [pair[1] for pair in columns_pairs]
        else:
            columns_x_no_pairs = columns_x
            columns_y_no_pairs = columns_y
        enum_to_implementation = {
            BivariateAnalysisSubsection.CorrelationPlot: CorrelationPlot(
                verbosity_correlations, columns, columns_x_no_pairs, columns_y_no_pairs
            ),
            BivariateAnalysisSubsection.PairPlot: PairPlot(
                verbosity_pairplot,
                columns,
                columns_x_no_pairs,
                columns_y_no_pairs,
                color_col=color_col,
            ),
            BivariateAnalysisSubsection.ContingencyTable: ContingencyTable(
                verbosity_contingency_table, columns, columns_x, columns_y, columns_pairs
            ),
        }

        subsections_implementations = [
            enum_to_implementation[sub] for sub in self.subsections_to_show
        ]
        super().__init__(subsections_implementations, verbosity, columns)

        self.columns_x = columns_x
        self.columns_y = columns_y
        self.columns_pairs = columns_pairs
        self.color_col = color_col

    @property
    def name(self) -> str:
        return "Bivariate analysis"

    def add_cells(self, cells: List[Dict[str, Any]], df: pd.DataFrame) -> None:
        """Adds cells to the list of cells.

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
            code = "show_bivariate_analysis(df=df"
            if self.subsections_to_show_with_low_verbosity != self._DEFAULT_SUBSECTIONS_TO_SHOW:
                arg_subsections_names = [
                    f"BivariateAnalysisSubsection.{str(sub)}"
                    for sub in self.subsections_to_show_with_low_verbosity
                ]

                code += f", subsections={arg_subsections_names}".replace("'", "")
            if self.columns_x is not None:
                code += f", columns_x={self.columns_x}"
                code += f", columns_y={self.columns_y}"
            elif self.columns is not None:
                code += f", columns={self.columns}"
            if self.columns_pairs is not None:
                code += f", columns_pairs={self.columns_pairs}"
            if self.color_col is not None:
                code += f", color_col='{self.color_col}'"
            code += ")"
            cells.append(nbfv4.new_code_cell(code))
            for sub in self.subsections:
                if sub.verbosity > Verbosity.LOW:
                    sub.add_cells(cells=cells, df=df)
        else:
            super().add_cells(cells=cells, df=df)

    def required_imports(self) -> List[str]:
        """Returns a list of imports to be put at the top of a generated notebook.

        Returns
        -------
        List[str]
            List of import strings to be added at the top of the generated notebook,
            e.g. ['import pandas as pd', 'import numpy as np']
        """
        if self.verbosity != Verbosity.LOW:
            return super().required_imports()

        imports = {"from edvart.report_sections.bivariate_analysis import show_bivariate_analysis"}
        if self.subsections_to_show_with_low_verbosity != self._DEFAULT_SUBSECTIONS_TO_SHOW:
            imports.add(
                "from edvart.report_sections.bivariate_analysis import BivariateAnalysisSubsection"
            )
        for subsec in self.subsections:
            if subsec.verbosity > Verbosity.LOW:
                imports.update(subsec.required_imports())

        return list(imports)

    def show(self, df: pd.DataFrame) -> None:
        """Generates cell output of this section in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output.
        """
        display(Markdown(self.get_title(section_level=1)))
        super().show(df)


def show_bivariate_analysis(
    df: pd.DataFrame,
    subsections: Optional[List[BivariateAnalysisSubsection]] = None,
    columns: Optional[List[str]] = None,
    columns_x: Optional[List[str]] = None,
    columns_y: Optional[List[str]] = None,
    columns_pairs: Optional[List[Tuple[str, str]]] = None,
    color_col: Optional[str] = None,
) -> None:
    """Generates bivariate analysis for df.

    Parameters
    ----------
    df : pd.DataFrame
        Data to be analyzed
    subsections : List[BivariateAnalysisSubsection], optional
        Subsections to include in the analysis. All subsections are included by default.
    columns : List[str], optional
        Columns on which to do bivariate analysis.
        If none of columns_x, columns_y and columns_pairs is specified bivariate analysis is
        performed on all pairs of `columns`.
        Ignored if `columns_x` and `columns_y` is specified. Ignored in contingency table
        if `columns_x` and `columns_y` or `columns_pairs` is specified.
        All columns are used by default.
    columns_x : List[str], optional
        If specified, correlations and pairplots are performed on the cartesian product of
        `columns_x` and `columns_y`.
        If `columns_x` is specified, then `columns_y` must also be specified.
    columns_y : List[str], optional
        If specified, correlations and pairplots are performed on the cartesian product of
        `columns_x` and `columns_y`.
        If `columns_y` is specified, then `columns_x` must also be specified.
    columns_pairs : List[str], optional
        List of columns pairs on which to perform bivariate analysis.
        Used primarily in contingency tables.
        If specified, `columns`, `columns_x` and `columns_y` are ignored in contingency tables.
        Ignored in pairplots and correlations unless `columns_pairs` is specified and none of
        `columns`, `columns_x`, `columns_y` is specified. In that case, the first elements of
        each pair are treated as `columns_x` and the second elements as `columns_y` in pairplots
        and correlations.
    color_col : str, optional
        Name of column to use for coloring of the bivariate analysis subsections.
        Coloring is currently supported in pairplot.
    """
    bivariate_analysis = BivariateAnalysis(
        subsections=subsections,
        verbosity=Verbosity.LOW,
        columns=columns,
        columns_x=columns_x,
        columns_y=columns_y,
        columns_pairs=columns_pairs,
        color_col=color_col,
    )

    for sub in bivariate_analysis.subsections:
        sub.show(df)


class CorrelationPlot(Section):
    """Generates the Correlation plot subsection.

    Parameters
    ----------
    verbosity : Verbosity (default = Verbosity.LOW)
        Verbosity of the code generated in the exported notebook.
    columns : List[str], optional
        Columns on which to plot pair-wise correlation plot.
        If `columns_x` and `columns_y` are unspecified, then analysis is performed on all pairs
        of `columns`. Otherwise ignored.
        All columns are used by default.
    columns_x : List[str], optional
        If specified, correlation is plotted on the cartesian product
        of `columns_x` and `columns_y`.
        If `columns_x` is specified, then `columns_y` must also be specified.
    columns_y : List[str], optional
        If specified, correlation is plotted on the cartesian product
        of `columns_x` and `columns_y`.
        If `columns_y` is specified, then `columns_x` must also be specified.

    Raises
    ------
    ValueError
        If exactly one of `columns_x`, `columns_y` is specified.
    """

    def __init__(
        self,
        verbosity: Verbosity = Verbosity.LOW,
        columns: Optional[List[str]] = None,
        columns_x: Optional[List[str]] = None,
        columns_y: Optional[List[str]] = None,
    ):
        if (columns_x is None) != (columns_y is None):
            raise ValueError("Either both or neither of columns_x, columns_y must be specified.")
        self.columns_x = columns_x
        self.columns_y = columns_y
        super().__init__(verbosity, columns)

    @property
    def name(self) -> str:
        return "Correlation Plot"

    def required_imports(self) -> List[str]:
        """Returns a list of imports to be put at the top of a generated notebook.

        Returns
        -------
        List[str]
            List of import strings to be added at the top of the generated notebook,
            e.g. ['import pandas as pd', 'import numpy as np'].
        """
        if self.verbosity <= Verbosity.MEDIUM:
            return ["from edvart.report_sections.bivariate_analysis import plot_correlations"]
        return [
            "import numpy as np",
            "import seaborn as sns",
            "from IPython.display import display, Markdown",
            "import matplotlib.pyplot as plt",
            "%matplotlib inline",
            "from edvart import utils",
            "from edvart.data_types import is_numeric",
        ]

    def add_cells(self, cells: List[Dict[str, Any]], df: pd.DataFrame) -> None:
        """Adds cells to the list of cells. Cells can be either code cells or markdown cells.

        Parameters
        ----------
        cells : List[Dict[str, Any]]
            List of generated notebook cells which are represented as dictionaries
        df: pd.DataFrame
            Data for which to add the cells.
        """
        section_header = nbfv4.new_markdown_cell(self.get_title(section_level=2))
        cells.append(section_header)

        default_call = "plot_correlations(df=df"
        if self.columns_x is not None:
            default_call += f", columns_x={self.columns_x}"
            default_call += f", columns_y={self.columns_y}"
        elif self.columns is not None:
            default_call += f", columns={self.columns}"

        default_call += ")"

        if self.verbosity <= Verbosity.MEDIUM:
            code = default_call
        else:
            code = (
                get_code(default_correlations)
                + "\n\n"
                + get_code(_get_columns_x_y)
                + "\n\n"
                + get_code(plot_correlation)
                + "\n\n"
                + get_code(plot_correlations)
                + "\n\n"
                + default_call
            )

        cells.append(nbfv4.new_code_cell(code))

    def show(self, df: pd.DataFrame) -> None:
        """Generates correlation plots in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output
        """
        display(Markdown(self.get_title(section_level=2)))
        plot_correlations(
            df=df, columns=self.columns, columns_x=self.columns_x, columns_y=self.columns_y
        )


def default_correlations() -> Dict[str, Callable[[pd.DataFrame], pd.DataFrame]]:
    """
    Default dictionary of bivariate statistics that will be calculated for numerical columns.

    Returns
    -------
    dict
        A dictionary assigning correlation functions to correlation names.
        Dictionary signature: 'CorrelationName': corr_func
    """
    return {
        "pearson": utils.pearson,
        "spearman": utils.spearman,
        "kendall": utils.kendall,
    }


def _get_columns_x_y(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    columns_x: Optional[List[str]] = None,
    columns_y: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Helper function to get columns_x and columns_y from provided parameters and dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.
    columns : List[str], optional
        Specified columns parameter.
    columns_x : List[str], optional
        Specified columns_x parameter.
    columns_y : List[str], optional
        Specified columns_x parameter.


    Returns
    -------
    columns_x, columns_y: Tuple[List[str], List[str]]
        Resolved columns_x and columns_y.
    """
    if (columns_x is None) != (columns_y is None):
        raise ValueError("Either both or neither of columns_x, columns_y must be specified.")
    if columns_x is None:
        if columns is None:
            columns = list(df.columns)
        columns_x = columns_y = columns
    assert columns_y is not None
    columns_x = [col for col in columns_x if is_numeric(df[col])]
    columns_y = [col for col in columns_y if is_numeric(df[col])]

    return columns_x, columns_y


def plot_correlations(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    columns_x: Optional[List[str]] = None,
    columns_y: Optional[List[str]] = None,
    pearson: bool = True,
    spearman: bool = True,
    kendall: bool = True,
) -> None:
    """Plots multiple correlations.

    Parameters
    ----------
    df : pd.DataFrame
        Data based on which to plot correlations.
    columns : Optional[List[str]]
        List of columns of df to analyze. All numeric columns of df are used by default.
    pearson : bool (default = True)
        If True, Pearson correlation will be plotted.
    spearman : bool (default = True)
        If True, Spearman correlation will be plotted.
    kendall : bool (default = True)
        If True, Kendall correlation will be plotted.
    """
    correlations = default_correlations()

    # remove correlations which are not to be plotted
    if not pearson:
        correlations.pop("pearson", None)
    if not spearman:
        correlations.pop("spearman", None)
    if not kendall:
        correlations.pop("kendall", None)

    for corr_name, corr_func in correlations.items():
        plot_correlation(df, corr_name, corr_func, columns, columns_x, columns_y)


def plot_correlation(
    df: pd.DataFrame,
    corr_name: str,
    corr_func: Callable[[pd.DataFrame], pd.DataFrame],
    columns: Optional[List[str]] = None,
    columns_x: Optional[List[str]] = None,
    columns_y: Optional[List[str]] = None,
    size_factor: float = 0.7,
    font_size: float = 15,
    color_map: Optional[Any] = None,
) -> None:
    """Plots a correlation heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Data based on which to plot correlation.
    corr_name : str
        Name of correlation being plotted.
    corr_func : Callable[[np.ndarray, Optional[List[Tuple[str, str]]]], pd.DataFrame]
        Correlation function to be used.
    columns : Optional[List[str]]
        List of columns of df to analyze. All numeric columns of df are used by default.
    size_factor: float (default = 0.7)
        Size of each cell in the table.
    font_size : float (default = 15)
        Size of axis labels of the correlation plot.
    color_map : Any, optional
        Color map compatible with matplotlib/seaborn to use for the correlation plot.
        A divergent red-blue color map is used by default.

    Raises
    ------
    ValueError
        If exactly one of `columns_x`, `columns_y` is specified.
    """
    columns_x, columns_y = _get_columns_x_y(df, columns, columns_x, columns_y)
    if color_map is None:
        color_map = sns.diverging_palette(10, 220, sep=80, n=100)

    # show header
    display(Markdown(f"### {corr_name.capitalize()} Correlation"))

    # calculate correlation between columns
    corr = corr_func(df)
    corr = corr.loc[columns_x, columns_y]

    # plot correlation heatmap
    ax = sns.heatmap(
        corr, cmap=color_map, vmin=-1, vmax=1, square=True, xticklabels=True, yticklabels=True
    )

    # set axes
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=font_size, rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=font_size, rotation=0)
    ax.figure.set_size_inches(len(columns_x) * size_factor + 1, len(columns_y) * size_factor + 1)

    plt.show()


class PairPlot(Section):
    """Generates the Pairplot subsection.

    Parameters
    ----------
    verbosity : Verbosity (default = Verbosity.LOW)
        Verbosity of the code generated in the exported notebook.
    columns : List[str], optional
        Columns on which to plot the pairplot.
        If `columns_x` and `columns_y` are unspecified, then analysis is performed on all pairs
        of `columns`. Otherwise ignored.
        All columns are used by default.
    columns_x : List[str], optional
        If specified, correlation is plotted
        on the cartesian product of `columns_x` and `columns_y`.
        If `columns_x` is specified, then `columns_y` must also be specified.
    columns_y : List[str], optional
        If specified, correlation is plotted
        on the cartesian product of `columns_x` and `columns_y`.
        If `columns_y` is specified, then `columns_x` must also be specified.
    color_col : str, optional
        Name of column according to use for coloring of points and histogram in the pairplot.

    Raises
    ------
    ValueError
        If exactly one of `columns_x`, `columns_y` is specified.
    """

    def __init__(
        self,
        verbosity: Verbosity = Verbosity.LOW,
        columns: Optional[List[str]] = None,
        columns_x: Optional[List[str]] = None,
        columns_y: Optional[List[str]] = None,
        color_col: Optional[str] = None,
    ):
        if (columns_x is None) != (columns_y is None):
            raise ValueError("Either both or neither of columns_x, columns_y must be specified.")
        self.columns_x = columns_x
        self.columns_y = columns_y
        self.color_col = color_col
        super().__init__(verbosity, columns)

    @property
    def name(self) -> str:
        return "Pairplot"

    def required_imports(self) -> List[str]:
        """Returns a list of imports to be put at the top of a generated notebook.

        Returns
        -------
        List[str]
            List of import strings to be added at the top of the generated notebook,
            e.g. ['import pandas as pd', 'import numpy as np'].
        """
        if self.verbosity <= Verbosity.MEDIUM:
            return ["from edvart.report_sections.bivariate_analysis import plot_pairplot"]
        return [
            "from edvart.data_types import is_categorical, is_boolean",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
        ]

    def add_cells(self, cells: List[Dict[str, Any]], df: pd.DataFrame) -> None:
        """Adds cells to the list of cells. Cells can be either code cells or markdown cells.

        Parameters
        ----------
        cells : List[Dict[str, Any]]
            List of generated notebook cells which are represented as dictionaries
        df: pd.DataFrame
            Data for which to add the cells.
        """
        section_header = nbfv4.new_markdown_cell(self.get_title(section_level=2))
        cells.append(section_header)

        default_call = "plot_pairplot(df=df"
        if self.columns_x is not None:
            default_call += f", columns_x={self.columns_x}"
            default_call += f", columns_y={self.columns_y}"
        elif self.columns is not None:
            default_call += f", columns={self.columns})"
        if self.color_col is not None:
            default_call += f", color_col='{self.color_col}'"
        default_call += ")"

        if self.verbosity <= Verbosity.MEDIUM:
            code = default_call
        else:
            code = get_code(plot_pairplot) + "\n\n" + default_call

        cells.append(nbfv4.new_code_cell(code))

    def show(self, df: pd.DataFrame) -> None:
        """Generates pairplot in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output
        """
        display(Markdown(self.get_title(section_level=2)))
        plot_pairplot(
            df=df,
            columns=self.columns,
            columns_x=self.columns_x,
            columns_y=self.columns_y,
            color_col=self.color_col,
        )


def plot_pairplot(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    columns_x: Optional[List[str]] = None,
    columns_y: Optional[List[str]] = None,
    allow_categorical: bool = False,
    color_col: Optional[str] = None,
) -> None:
    """Plot a pairplot for each pair of columns.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame for which to plot pairplot.
    columns : Union[List[Tuple[str, str]] or List[str]], optional
        Which columns to plot pairplot for.
        All columns that are not categorical and are not boolean are used by default.
    columns_x : List[str], optional
        If specified, correlation is plotted
        on the cartesian product of `columns_x` and `columns_y`.
        If `columns_x` is specified, then `columns_y` must also be specified.
    columns_y : List[str], optional
        If specified, correlation is plotted
        on the cartesian product of `columns_x` and `columns_y`.
        If `columns_y` is specified, then `columns_x` must also be specified.
    allow_categorical : bool (default = False)
        Whether to allow plotting of categorical columns. If False (default), then even
        explicitly specified columns will be excluded. If True, categorical columns are still
        excluded by default, unless explicitly specified via columns/columns_x/columns_y.
    color_col : str, optional
        Name of column according to use for coloring of points and histogram in the pairplot.

    Raises
    ------
    ValueError
        If exactly one of `columns_x`, `columns_y` is specified.
    """

    def include_column(col: str) -> bool:
        return not is_categorical(df[col]) and not is_boolean(df[col])

    if (columns_x is None) != (columns_y is None):
        raise ValueError("Either both or neither of columns_x, columns_y must be specified.")
    if columns_x is None:
        if columns is None:
            columns = list(filter(include_column, df.columns))
        columns_x = columns
        columns_y = columns
    if not allow_categorical:
        assert columns_y is not None
        columns_x = list(filter(include_column, columns_x))
        columns_y = list(filter(include_column, columns_y))
    sns.pairplot(df, x_vars=columns_x, y_vars=columns_y, hue=color_col)
    plt.tight_layout()
    plt.show()


class ContingencyTable(Section):
    """Generates the pairwise contingency tables subsection.

    Parameters
    ----------
    verbosity : Verbosity (default = Verbosity.LOW)
        Verbosity of the code generated in the exported notebook.
    columns : List[str], optional
        Columns on which to show contingency tables.
        If `columns_x` and `columns_y` and `columns_pairs` are unspecified,
        then analysis performed on all pairs of `columns`. Otherwise ignored.
        Columns which contain only null values are always excluded. All columns are used by default.
    columns_x : List[str], optional
        If specified, contingency tables are plotted for each pair
        in the cartesian product of `columns_x` and `columns_y`.
        If `columns_x` is specified, then `columns_y` must also be specified.
        Columns which contain only null values are always excluded.
        Ignored if `columns_pairs` is specified.
    columns_y : List[str], optional
        If specified, contingency tables are plotted for each pair
        in the cartesian product of `columns_x` and `columns_y`.
        If `columns_y` is specified, then `columns_x` must also be specified.
        Columns which contain only null values are always excluded.
        Ignored if `columns_pairs` is specified.
    columns_pairs : List[Tuple[str, str]], optional
        If specified, contingency tables are plotted for exactly the specified pairs.
        Columns which contain only null values are always excluded, i.e. if at least one of the
        columns in a pair is excluded, then is excluded.
    """

    def __init__(
        self,
        verbosity: Verbosity = Verbosity.LOW,
        columns: Optional[List[str]] = None,
        columns_x: Optional[List[str]] = None,
        columns_y: Optional[List[str]] = None,
        columns_pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        super().__init__(verbosity, columns)
        self.columns_x = columns_x
        self.columns_y = columns_y
        self.columns_pairs = columns_pairs

    @property
    def name(self) -> str:
        return "Contingency table"

    def required_imports(self) -> List[str]:
        """Returns a list of imports to be put at the top of a generated notebook.

        Returns
        -------
        List[str]
            List of import strings to be added at the top of the generated notebook,
            e.g. ['import pandas as pd', 'import numpy as np'].
        """
        if self.verbosity <= Verbosity.MEDIUM:
            return ["from edvart.report_sections.bivariate_analysis import contingency_tables"]
        return [
            "import itertools",
            "from edvart import utils",
            "import seaborn as sns",
            "import matplotlib.pyplot as plt",
            "from typing import Literal",
        ]

    def add_cells(self, cells: List[Dict[str, Any]], df: pd.DataFrame) -> None:
        """Adds cells to the list of cells. Cells can be either code cells or markdown cells.

        Parameters
        ----------
        cells : List[Dict[str, Any]]
            List of generated notebook cells which are represented as dictionaries
        df: pd.DataFrame
            Data for which to add the cells.
        """
        section_header = nbfv4.new_markdown_cell(self.get_title(section_level=2))
        cells.append(section_header)

        default_call = "contingency_tables(df=df"
        if self.columns_pairs is not None:
            default_call += f", columns_pairs={self.columns_pairs}"
        elif self.columns_x is not None:
            default_call += f", columns_x={self.columns_x}"
            default_call += f", columns_y={self.columns_y}"
        elif self.columns is not None:
            default_call += f", columns={self.columns}"
        default_call += ")"

        if self.verbosity <= Verbosity.MEDIUM:
            code = default_call
        else:
            code = (
                get_code(contingency_tables)
                + "\n\n"
                + get_code(contingency_table)
                + "\n\n"
                + default_call
            )

        cells.append(nbfv4.new_code_cell(code))

    def show(self, df: pd.DataFrame) -> None:
        """Generates contingency table in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output
        """
        display(Markdown(self.get_title(section_level=2)))
        contingency_tables(
            df=df,
            columns=self.columns,
            columns_x=self.columns_x,
            columns_y=self.columns_y,
            columns_pairs=self.columns_pairs,
        )


def contingency_tables(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    columns_x: Optional[List[str]] = None,
    columns_y: Optional[List[str]] = None,
    columns_pairs: Optional[List[Tuple[str, str]]] = None,
    table_threshold: int = 30,
) -> None:
    """Display a contingency table for each pairs of columns.

    Parameters
    ----------
    df : pd.DataFrame
        Data based on which to create a contingency table.
    columns : List[str], optional
        Which columns to generate pair-wise contingency tables for.
        Columns with more than `table_threshold` unique values are excluded.
        Columns which contain only null values are excluded.
        To override the excluded columns, specify `columns_pairs`.
        Ignored if `columns_x` and `columns_y` or `columns_pairs` is specified.
    columns_x : List[str], optional
        If specified, contingency tables are plotted for each pair
        in the cartesian product of `columns_x` and `columns_y`.
        Columns with more than `table_threshold` unique values are excluded.
        Columns which contain only null values are excluded.
        If `columns_x` is specified, then `columns_y` must also be specified.
        Ignored if `columns_pairs` is specified.
    columns_y : List[str], optional
        If specified, contingency tables are plotted for each pair
        in the cartesian product of `columns_x` and `columns_y`.
        Columns with more than `table_threshold` unique values are excluded.
        Columns which contain only null values are excluded.
        If `columns_y` is specified, then `columns_x` must also be specified.
        Ignored if `columns_pairs` is specified.
    columns_pairs : List[Tuple[str, str]], optional
        If specified, contingency tables are plotted for exactly the specified pairs.
    table_threshold : int (default = 30)
        Maximum number of unique values for a column to be used in contingency table.
        If non-positive, no columns are excluded according to this criterion.
    """

    def include_column(col: str) -> bool:
        n_unique = df[col].nunique()
        return (
            (table_threshold < 0 or n_unique <= table_threshold)
            and (not df[col].isnull().all())
            and n_unique > 1
        )

    if (columns_x is None) != (columns_y is None):
        raise ValueError("Either both or neither of columns_x, columns_y must be specified.")
    if columns is None:
        columns = list(df.columns)
    columns = [col for col in df.columns if include_column(col)]

    if columns_pairs is None:
        if columns_x is None:
            columns_pairs = list(itertools.combinations(columns, 2))
        else:
            assert columns_x is not None
            assert columns_y is not None
            columns_pairs = [
                (col_x, col_y)
                for (col_x, col_y) in itertools.product(columns_x, columns_y)
                # Filter out pairs of columns which contain the same column since
                # they make no sense in a contingency table
                if col_x != col_y and include_column(col_x) and include_column(col_y)
            ]

    for column1, column2 in columns_pairs:
        contingency_table(df, column1, column2)


def contingency_table(
    df: pd.DataFrame,
    columns1: Union[str, List[str]],
    columns2: Union[str, List[str]],
    include_total: bool = True,
    hide_zeros: bool = True,
    scaling_func: Callable[[np.ndarray], np.ndarray] = np.cbrt,
    colormap: Any = "Blues",
    size_factor: Union[float, Literal["auto"]] = "auto",
    fontsize: float = 15,
) -> None:
    """
    Show a colored contingency table for the two specified columns or lists of columns.

    Parameters
    ----------
    df : pd.DataFrame
        Data to analyze.
    columns1 : Union[str, List[str]]
        Name of column or list of column names to use in the vertical axis of the table.
    columns2 : Union[str, List[str]]
        Name of column or list of column names to use in the horizontal axis of the table.
    include_total : bool (default = True)
        Whether to include marginal counts.
    hide_zeros : bool (default = True)
        Whether to hide zero values in the table for better readability.
    scaling_func : Callable[["array-like"], "array-like"]
        Function to scale the values for the purpose of coloring for smaller spread.
        Cube root is used by default.
    colormap : Any (default = "Blues")
        Colormap compatible with matplotlib/seaborn.
    size_factor : float or "auto"
        Size of each cell in the table.
        If "auto", the cell size is automatically adjusted so that the numbers fit in the cells.
    fontsize : float (default = 15)
        Size of the font for axis labels.
    """
    if isinstance(columns1, str):
        columns1 = [columns1]
    if isinstance(columns2, str):
        columns2 = [columns2]
    table = pd.crosstab(
        [df[col] for col in columns1],
        [df[col] for col in columns2],
        margins_name="Total",
        margins=include_total,
    )

    annot = table.replace(0, "") if hide_zeros else table

    ax = sns.heatmap(
        scaling_func(table.values),
        annot=annot,
        fmt="",
        cbar=False,
        cmap=colormap,
        linewidths=0.1,
        xticklabels=1,
        yticklabels=1,
        annot_kws={"fontsize": fontsize},
        square=True,
    )
    if size_factor == "auto":
        n_digits_max = 1 + np.floor(np.log10(table.max().max()))
        size_factor = (
            # Constants chosen empirically to make the numbers fit in the cells
            0.18 * max(4, n_digits_max)
        )
    ax.figure.set_size_inches(size_factor * len(table.columns), size_factor * len(table))
    # Set y axis
    ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fontsize)

    # Set x axis
    xticklabels = ax.get_xticklabels()
    x_label_rotation = (
        0 if max(len(xticklabel.get_text()) for xticklabel in xticklabels) < 5 else 30
    )  # rotate if any x tick label is longer than 5 characters
    ax.set_xticklabels(xticklabels, fontsize=fontsize, rotation=x_label_rotation)

    ax.xaxis.tick_top()
    ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
    ax.xaxis.set_label_position("top")

    # Visually separate the margins
    if include_total:
        ax.vlines(len(table.columns) - 1, ymin=0, ymax=len(table), color="grey")
        ax.hlines(len(table) - 1, xmin=0, xmax=len(table.columns), color="grey")

    plt.show()
