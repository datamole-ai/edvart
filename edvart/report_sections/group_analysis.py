from typing import Any, Callable, Dict, List, Optional, Union

import colorlover as cl
import nbformat.v4 as nbfv4
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import Markdown, display
from matplotlib import cm
from matplotlib.colors import Normalize, to_hex
from plotly.subplots import make_subplots

from edvart import utils
from edvart.data_types import DataType, infer_data_type
from edvart.report_sections.code_string_formatting import code_dedent, get_code
from edvart.report_sections.section_base import Section, Verbosity


class GroupAnalysis(Section):
    """Generate the group analysis section of the report.

    Parameters
    ----------
    groupby : Union[str, List[str]]
        Name of column or list of columns names to group by.
    verbosity : Verbosity (default = Verbosity.LOW)
        Generated code verbosity global to the Group analysis sections.
        If subsection verbosities are None, then they will be overridden by this parameter.
    columns : List[str], optional
        Columns on which to do group analysis. All columns are used by default.
    show_within_group_statistics : bool (default = True)
        Whether to show per-group statistics.
    show_group_missing_values : bool (default = True)
        Whether to show per-group missing values.
    show_group_distribution_plots : bool (default = True)
        Whether to show per-group distribution plots.

    Raises
    ------
    ValueError
        If groupby columns are not a subset of the columns of the input DataFrame df.
    """

    def __init__(
        self,
        groupby: Union[str, List[str]],
        verbosity: Verbosity = Verbosity.LOW,
        columns: Optional[List[str]] = None,
        show_within_group_statistics: bool = True,
        show_group_missing_values: bool = True,
        show_group_distribution_plots: bool = True,
    ):
        if isinstance(groupby, str):
            groupby = [groupby]
        self.columns = columns
        self.groupby = groupby
        self.show_statistics = show_within_group_statistics
        self.show_missing_vals = show_group_missing_values
        self.show_dist = show_group_distribution_plots

        super().__init__(verbosity, columns)

    @property
    def name(self) -> str:
        return "Group Analysis"

    def required_imports(self) -> List[str]:
        """Returns a list of imports to be put at the top of a generated notebook.

        Returns
        -------
        List[str]
            List of import strings to be added at the top of the generated notebook,
            e.g. ["import pandas as pd", "import numpy as np"]
        """
        if self.verbosity == Verbosity.LOW:
            return ["from edvart.report_sections.group_analysis import show_group_analysis"]
        if self.verbosity == Verbosity.MEDIUM:
            return [
                code_dedent(
                    """from edvart.report_sections.group_analysis import (
                        default_group_descriptive_stats,
                        default_group_quantile_stats,
                        show_group_analysis,
                        group_barplot,
                        group_missing_values,
                        overlaid_histograms,
                        within_group_descriptive_stats,
                        within_group_quantile_stats,
                        within_group_stats
                    )"""
                )
            ]
        # verbosity HIGH
        return [
            "import colorlover as cl",
            "import matplotlib.pyplot as plt",
            "import matplotlib.cm as cm",
            "from matplotlib.colors import Normalize, to_hex",
            "import numpy as np",
            "import pandas as pd",
            "import plotly.graph_objects as go",
            "from edvart.data_types import infer_data_type, DataType",
            "from edvart import utils",
            "from typing import List, Dict, Optional, Callable",
            "from plotly.subplots import make_subplots",
        ]

    def _add_function_defs(self, cells: List[Dict[str, Any]]):
        """Add function definition cell to the list of cells.

        Cells can be either code cells or markdown cells.

        Parameters
        ----------
        cells : List[Dict[str, Any]]
            List of generated notebook cells which are represented as dictionaries
        df: pd.DataFrame
            Data for which to add the cells.
        """
        code = (
            get_code(default_group_quantile_stats)
            + "\n\n"
            + get_code(default_group_descriptive_stats)
            + "\n\n"
            + get_code(within_group_descriptive_stats)
            + "\n\n"
            + get_code(within_group_quantile_stats)
            + "\n\n"
            + get_code(within_group_stats)
            + "\n\n"
            + get_code(group_barplot)
            + "\n\n"
            + get_code(overlaid_histograms)
            + "\n\n"
            + get_code(group_missing_values)
        )

        cells.append(nbfv4.new_code_cell(code))

    def _add_cells_numeric_col(self, cells: List[Dict[str, Any]], column_name: str):
        """Add code cells for a numeric column at verbosity MEDIUM or HIGH.

        Parameters
        ----------
        cells : List[Dict[str, Any]]
            List of generated notebook cells which are represented as dictionaries
        df: pd.DataFrame
            Data for which to add the cells.
        column_name : str
            Name of column for which to generate code.
        """
        code = ""
        if self.show_statistics:
            if self.verbosity == Verbosity.MEDIUM:
                code += (
                    f"within_group_stats(df=df, groupby={self.groupby}, column='{column_name}')\n"
                )
            else:
                code += code_dedent(
                    f"""
                            within_group_stats(
                                df=df,
                                groupby={self.groupby},
                                column='{column_name}',
                                stats=default_group_descriptive_stats()
                            )

                            within_group_stats(
                                df=df,
                                groupby={self.groupby},
                                column='{column_name}',
                                stats=default_group_quantile_stats()
                            )
                        """
                )

        if self.show_dist:
            code += f"overlaid_histograms(df=df, groupby={self.groupby}, column='{column_name}')"
        cells.append(nbfv4.new_code_cell(code))

    def add_cells(self, cells: List[Dict[str, Any]], df: pd.DataFrame) -> None:
        """Add cells to the list of cells.

        Cells can be either code cells or markdown cells.

        Parameters
        ----------
        cells : List[Dict[str, Any]]
            List of generated notebook cells which are represented as dictionaries
        df: pd.DataFrame
            Data for which to add the cells.
        """
        if not set(self.groupby) <= set(df.columns):
            raise ValueError("Grouping by a column which is not in columns of df.")
        section_header = nbfv4.new_markdown_cell(self.get_title(section_level=1))
        cells.append(section_header)

        if self.verbosity == Verbosity.LOW:
            if self.columns is None:
                code = f"show_group_analysis(df=df, groupby={self.groupby})"
            else:
                code = f"show_group_analysis(df=df, groupby={self.groupby}, columns={self.columns})"
            code_cell = nbfv4.new_code_cell(code)
            cells.append(code_cell)
            return

        if self.verbosity == Verbosity.HIGH:
            self._add_function_defs(cells)

        if self.show_missing_vals:
            cells.append(nbfv4.new_markdown_cell("## Missing values for each group"))
            if self.columns is None:
                code = f"group_missing_values(df=df, groupby={self.groupby})"
            else:
                code = (
                    f"group_missing_values(df=df, columns={self.columns}, groupby={self.groupby})"
                )
            cells.append(nbfv4.new_code_cell(code))

        columns = self.columns if self.columns is not None else df.columns

        if not self.show_statistics and not self.show_dist:
            return
        for col in columns:
            if col in self.groupby:
                continue
            cells.append(nbfv4.new_markdown_cell(f"## *{col}*"))
            datatype = infer_data_type(df[col])
            if datatype == DataType.NUMERIC:
                self._add_cells_numeric_col(cells, col)
            else:
                code = f"group_barplot(df=df, groupby={self.groupby}, column='{col}')"
                cells.append(nbfv4.new_code_cell(code))

    def show(self, df: pd.DataFrame) -> None:
        """Generates cell output of this section in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output.
        """
        columns = (
            self.columns
            if self.columns is not None
            else [col for col in df.columns if col not in self.groupby]
        )
        display(Markdown(self.get_title(section_level=1)))

        if self.show_missing_vals:
            display(Markdown("## Missing values for each group"))
            group_missing_values(df, self.groupby)
        if not self.show_dist and not self.show_statistics:
            return
        for col in columns:
            if col in self.groupby:
                continue
            display(Markdown(f"## *{col}*"))
            datatype = infer_data_type(df[col])
            if datatype == DataType.NUMERIC:
                if self.show_statistics:
                    within_group_stats(df, self.groupby, col)
                if self.show_dist:
                    overlaid_histograms(df, self.groupby, col)
            else:
                group_barplot(df, self.groupby, col)


def default_group_descriptive_stats() -> Dict[str, Callable[[pd.Series], float]]:
    """Descriptive statistic functions.

    Returns
    -------
    Dict[str, Callable[[pd.Series], float]]
        A dictionary of statistic function names and functions.
    """
    return {
        "# Unique values": utils.num_unique_values,
        "Sum": utils.sum_,
        "Mode": utils.mode,
        "Mean": utils.mean,
        "Std": utils.std,
        "Mean abs dev": utils.mad,
        "Median abs dev": utils.median_absolute_deviation,
        "Relative std": utils.coefficient_of_variation,
        "Kurtosis": utils.kurtosis,
        "Skewness": utils.skewness,
    }


def default_group_quantile_stats() -> Dict[str, Callable[[pd.Series], float]]:
    """Quantile statistic functions.

    Returns
    -------
    Dict[str, Callable[[pd.Series], float]]
        A dictionary of statistic function names and functions.
    """
    return {
        "Min": utils.minimum,
        "Q1": utils.quartile1,
        "Median": utils.median,
        "Q3": utils.quartile3,
        "Max": utils.maximum,
        "Range": utils.value_range,
        "IQR": utils.iqr,
    }


def within_group_descriptive_stats(
    df: pd.DataFrame, groupby: List[str], column: str, round_decimals: int = 2
):
    """Display within-group descriptive statistics for a column.

    Parameters
    ----------
    df : pd.DataFrame
        Data to display statistics for.
    groupby : List[str]
        List of column names to group data by.
    column : str
        Which column to display statistics for.
    round_decimals : int (default = 2)
        Number of decimals to round displayed results to.
    """
    within_group_stats(
        df=df,
        groupby=groupby,
        column=column,
        stats=default_group_descriptive_stats(),
        round_decimals=round_decimals,
    )


def within_group_quantile_stats(
    df: pd.DataFrame, groupby: List[str], column: str, round_decimals: int = 2
) -> None:
    """Display within-group quantile statistics for a column.

    Parameters
    ----------
    df : pd.DataFrame
        Data to display statistics for.
    groupby : List[str]
        List of column names to group data by.
    column : str
        Which column to display statistics for.
    round_decimals : int (default = 2)
        Number of decimals to round displayed results to.
    """
    within_group_stats(
        df=df,
        groupby=groupby,
        column=column,
        stats=default_group_quantile_stats(),
        round_decimals=round_decimals,
    )


def within_group_stats(
    df: pd.DataFrame,
    groupby: List[str],
    column: str,
    stats: Dict[str, Callable[[pd.Series], float]] = None,
    round_decimals: int = 2,
) -> None:
    """Display withing group statistics for a column of df grouped by one or other more columns.

    Parameters
    ----------
    df : pd.DataFrame
        Data to display statistics for.
    groupby : List[str]
        List of column names to group by.
    column : str
        Name of column to display statistics for.
    stats : Dict[str, Callable[[pd.Series], float]], optional
        A dictionary of statistic function names and functions.
        If None, default_group_quantile_stats() and default_group_descriptive_stats() will be
        used.
    round_decimals : int (default = 2)
        Number of decimals to round displayed results to.
    """
    if stats is None:
        within_group_descriptive_stats(df, groupby, column, round_decimals)
        within_group_quantile_stats(df, groupby, column, round_decimals)
        return

    df_grouped = df.groupby(groupby)[column]
    group_stats = []
    for name, func in stats.items():
        group_stats.append(df_grouped.apply(func).rename(name))
    stats_table = pd.concat(group_stats, axis=1)
    stats_table = stats_table.round(decimals=round_decimals)
    display(stats_table)


# pylint: disable=too-many-locals
def group_missing_values(
    df: pd.DataFrame,
    groupby: Union[str, List[str]],
    columns: Optional[List[str]] = None,
    round_decimals: int = 2,
    heatmap: bool = True,
    foreground_colormap: str = "bone",
    background_colormap: str = "OrRd",
    sort: bool = True,
    sort_by: Optional[List[str]] = None,
    ascending: bool = False,
) -> None:
    """Display per-group number and percentage of missing values in each column.

    Parameters
    ----------
    df : pd.DataFrame
        Data to display missing values for.
    groupby : str or List[str]
        Name of column or list of columns names to group by.
    columns : List[str], optional
        Subset of columns to analyze. All columns except those for grouping are used by default.
    round_decimals : int (default = 2)
        Number of decimals to round displayed results to.
    heatmap : bool (default = True)
        Whether to color missing value percentage cells according to the corresponding value.
    foreground_colormap : str (default = "bone")
        Color map of the foreground.
    background_colormap : str (default = "OrRd)
        Color map of the background.
    sort : bool (default = True)
        Whether to sort the results.
    sort_by : List[str], optional
        List of column names to sort the results by. Sort by all column by default.
    ascending : bool (default = False)
        If True, sort in ascending order, otherwise sort in descending order.

    Raises
    ------
    ValueError
        If groupby columns are not a subset of the columns of the input DataFrame df.
    """
    if isinstance(groupby, str):
        groupby = [groupby]
    if not set(groupby) <= set(df.columns):
        raise ValueError("Grouping by a column which is not in columns of df.")
    if columns is None:
        columns = [col for col in df.columns if col not in groupby]
    df_grouped = df.groupby(groupby)[columns]

    # Calculate number of samples in each group
    sizes = df_grouped.size().rename("Group Size")

    # Calculate missing values percentage of each column for each group
    missing = df_grouped.apply(lambda g: g.isna().sum(axis=0))
    missing = missing.divide(sizes, axis=0) * 100  # `/` divides by axis=-1
    missing = missing.fillna(value=0)
    missing = missing.round(decimals=round_decimals)

    if missing.sum().sum() == 0:
        print("There are no missing values")
        return

    # Concatenate group sizes and missing value percentages
    final_table = pd.concat([sizes, missing], axis=1)

    # Sort columns to better identify groups with missing data
    all_columns = [col for col in missing.columns if col not in groupby]
    if sort:
        if sort_by is None:
            sort_by = all_columns
        final_table = final_table.sort_values(by=sort_by, axis=0, ascending=ascending)

    # Drop columns with no missing data
    missing = final_table.sum(axis=0) != 0
    final_table = final_table.loc[:, missing]

    colored_columns = [col for col in final_table if col in all_columns]

    # Apply conditional formatting to each cell except group size column
    if heatmap:
        fg_cmap = cm.get_cmap(foreground_colormap)
        bg_cmap = cm.get_cmap(background_colormap)
        norm = Normalize(vmin=0, vmax=100)

        def color_cell(value):
            fg_hex = to_hex(fg_cmap(norm(value)))
            bg_hex = to_hex(bg_cmap(norm(value)))
            return f"""
                color: {fg_hex};
                background-color: {bg_hex};
            """

        render = final_table.style.applymap(
            func=color_cell, subset=pd.IndexSlice[:, colored_columns]
        ).format(formatter="{0:.2f} %", subset=pd.IndexSlice[:, colored_columns])
    else:
        render = final_table.style.format(
            formatter="{0:.2f} %", subset=pd.IndexSlice[:, colored_columns]
        )

    # Render table
    display(render)


def group_barplot(
    df: pd.DataFrame,
    groupby: List[str],
    column: str,
    group_count_threshold: int = 20,
    conditional_probability: bool = True,
    xaxis_tickangle: float = 0,
    alpha: float = 0.5,
):
    """Display a per-group barplot for a column.

    Parameters
    ----------
    df : pd.DataFrame
        Data to analyze.
    groupby : List[str]
        List of column names to group by.
    column : str
        Which column to analyze.
    group_count_threshold : int (default = 20)
        Maximum number of unique values in column to plot. If the number of unique values
        is higher, a warning will be issued and plot will not be shown.
    conditional_probability : bool (default = True)
        If True, conditional probability conditioned on group will be displayed,
        otherwise conditional frequency will be displayed.
    xaxis_tickangle : float (default = 0)
        Rotation angle of ticks on the x axis.
    alpha : float (default = 0.5)
        Opacity of bars in the plot.
    """
    num_cat = df[column].nunique()
    if num_cat > group_count_threshold:
        display(
            Markdown(
                f"Number of unique values ({num_cat}) greater than threshold."
                " Not plotting distribution"
            )
        )
        return

    pivot = df.pivot_table(index=groupby, columns=column, aggfunc="size", fill_value=0)

    if conditional_probability:
        pivot = pivot.divide(pivot.sum(axis=1), axis=0)  # `/` divides by axis=-1
        pivot = pivot.fillna(value=0)

    # Choose color palette
    colors = cl.scales["9"]["qual"]["Set1"]
    color_idx = 0

    fig = go.Figure()
    for idx, row in pivot.iterrows():
        if hasattr(idx, "__len__") and not isinstance(idx, str):
            group_name = "_".join([str(i) for i in idx])
        else:
            group_name = idx
        color = colors[color_idx % len(colors)]
        color_idx += 1
        fig.add_trace(
            go.Bar(
                x=pivot.columns,
                y=row,
                name=group_name,
                opacity=alpha,
                marker_color=color,
            )
        )

    groupby_repr = ", ".join(groupby)
    if conditional_probability:
        yaxis_title = f"P({column} | {groupby_repr})"
    else:
        yaxis_title = f"Freq({column} | {groupby_repr})"

    fig.update_layout(
        barmode="group",
        xaxis_tickangle=xaxis_tickangle,
        xaxis_title=column,
        yaxis_title=yaxis_title,
    )
    fig.show()


def overlaid_histograms(
    df: pd.DataFrame,
    groupby: List[str],
    column: str,
    bins: Optional[int] = None,
    density: bool = True,
    alpha: float = 0.5,
):
    """Show per-group distribution histograms in a single plot overlaid over each other.

    Parameters
    ----------
    df : pd.DataFrame
        Data to analyze.
    groupby : List[str]
        List of column names to group by.
    column : str
        Name of column to analyze.
    bins : int, optional
        Number of bins in the histogram. If None, number of bin will be inferred using
        Freedman-Diaconis bin number inference.
    density : bool (default = True)
        If True, histograms will be normalized to display density.
    alpha : float
        Opacity of individual histograms.
    """
    data_min = df[column].min()
    data_max = df[column].max()
    data_range = data_max - data_min
    if bins is None:
        # Freedman-Diaconis bin number inference if bins is None
        iqr = utils.iqr(df[column])
        bin_width = 2 * iqr / (len(df[column]) ** (1 / 3))
        bins = int(np.ceil(data_range / bin_width))
        if bins > 1000:
            # Use Sturges' rule if number of bins is too large
            bins = int(np.ceil(np.log2(bins) + 1))
            bin_width = data_range / bins
    else:
        bin_width = data_range / bins
    bin_config = go.histogram.XBins(
        start=data_min,
        end=data_max,
        size=bin_width,
    )
    # Choose color palette
    colors = cl.scales["9"]["qual"]["Set1"]
    color_idx = 0

    # Distribution plot
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.3, 0.7],
        vertical_spacing=0.02,
    )

    for name, group in df.groupby(
        by=(groupby[0] if isinstance(groupby, list) and len(groupby) == 1 else groupby)
    ):
        if hasattr(name, "__len__") and not isinstance(name, str):
            group_name = "_".join([str(i) for i in name])
        else:
            group_name = name
        color = colors[color_idx % len(colors)]
        color_idx += 1
        # Add to boxplot
        fig.add_trace(
            go.Box(
                x=group[column],
                name=group_name,
                legendgroup=group_name,
                showlegend=False,
                marker_color=color,
            ),
            row=1,
            col=1,
        )
        # Add to histogram
        fig.add_trace(
            go.Histogram(
                x=group[column],
                name=group_name,
                legendgroup=group_name,
                xbins=bin_config,
                histnorm="density" if density else "",
                marker_color=color,
                opacity=alpha,
            ),
            row=2,
            col=1,
        )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text=column, row=2, col=1)
    yaxis_title = "Density" if density else "Frequency"
    fig.update_yaxes(title_text=yaxis_title, row=2, col=1)
    fig.show()


def show_group_analysis(
    df: pd.DataFrame,
    groupby: Union[str, List[str]],
    columns: Optional[List[str]] = None,
    show_within_group_statistics: bool = True,
    show_group_missing_values: bool = True,
    show_distribution_plots: bool = True,
) -> None:
    """Generate group analysis for df.

    Parameters
    ----------
    df : pd.DataFrame
        Data to be analyzed.
    groupby : Union[str, List[str]]
        Name of column or list of columns names to group by.
    columns : List[str], optional
        Subset of columns to analyze. All columns except those used for grouping are used
        by default.
    show_within_group_statistics : bool (default = True)
        Whether to show per-group statistics.
    show_group_missing_values : bool (default = True)
        Whether to show per-group missing values.
    show_distribution_plots : bool (default = True)
        Whether to show per-group distribution plots.

    Raises
    ------
    ValueError
        If groupby columns are not a subset of the columns of the input DataFrame df.
    """
    if isinstance(groupby, str):
        groupby = [groupby]
    if not set(groupby) <= set(df.columns):
        raise ValueError("Grouping by a column which is not in columns of df.")
    if columns is None:
        columns = [col for col in df.columns if col not in groupby]

    if show_group_missing_values:
        display(Markdown("## Missing values for each group"))
        group_missing_values(df, groupby)
    if not show_distribution_plots and not show_within_group_statistics:
        return
    for col in columns:
        display(Markdown("---"))
        display(Markdown(f"### *{col}*"))
        datatype = infer_data_type(df[col])
        if datatype == DataType.NUMERIC:
            if show_within_group_statistics:
                within_group_stats(df, groupby, col)
            if show_distribution_plots:
                overlaid_histograms(df, groupby, col)
        else:
            group_barplot(df, groupby, col)
