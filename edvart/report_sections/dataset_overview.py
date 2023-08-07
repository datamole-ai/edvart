from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import nbformat.v4 as nbfv4
import pandas as pd
from IPython.display import Markdown, display

from edvart.data_types import (
    DataType,
    infer_data_type,
    is_boolean,
    is_categorical,
    is_date,
    is_numeric,
)
from edvart.pandas_formatting import hide_index, render_dictionary, series_to_frame
from edvart.report_sections.code_string_formatting import get_code, total_dedent
from edvart.report_sections.section_base import ReportSection, Section, Verbosity


class Overview(ReportSection):
    """Generates the Overview section of the report.

    Contains an enum OverviewSubsection of possible subsections.

    Parameters
    ----------
    subsections : List[OverviewSubsection], optional
        List of subsections to inlcude into the Overview section.
        All subsections in OverviewSubsection are used by default.
    verbosity : Verbosity
        Generated code verbosity global to the Overview sections
        If subsection verbosities are None, then they will be overridden by this parameter.
    columns : List[str], optional
        Columns on which to do overview analysis. All columns are used by default.
    verbosity_quick_info : Verbosity, optional
        Quick info subsection code verbosity.
    verbosity_data_types : Verbosity, optional
        Data types subsection code verbosity.
    verbosity_data_preview : Verbosity, optional
        Data preview subsection code verbosity.
    verbosity_missing_values : Verbosity, optional
        Missing values subsection code verbosity.
    verbosity_rows_with_missing_value : Verbosity, optional
        Rows with missing value subsection code verbosity.
    verbosity_constant_occurence : Verbosity, optional
        Constant values subsection code verbosity.
    verbosity_duplicate_rows : Verbosity, optional
        Duplicate rows subsection code verbosity.
    """

    # pylint: disable=invalid-name
    class OverviewSubsection(IntEnum):
        """
        Enum of possible subsections of the Overview section.
        """

        QuickInfo = 1
        DataTypes = 2
        DataPreview = 3
        MissingValues = 4
        RowsWithMissingValue = 5
        ConstantOccurence = 6
        DuplicateRows = 7

        def __str__(self):
            return self.name

    def __init__(
        self,
        subsections: Optional[List[OverviewSubsection]] = None,
        verbosity: Verbosity = Verbosity.LOW,
        columns: Optional[List[str]] = None,
        verbosity_quick_info: Optional[Verbosity] = None,
        verbosity_data_types: Optional[Verbosity] = None,
        verbosity_data_preview: Optional[Verbosity] = None,
        verbosity_missing_values: Optional[Verbosity] = None,
        verbosity_rows_with_missing_value: Optional[Verbosity] = None,
        verbosity_constant_occurence: Optional[Verbosity] = None,
        verbosity_duplicate_rows: Optional[Verbosity] = None,
    ):
        # Propagate global verbosity to subsection verbosities
        verbosity_quick_info = (
            verbosity_quick_info if verbosity_quick_info is not None else verbosity
        )
        verbosity_data_types = (
            verbosity_data_types if verbosity_data_types is not None else verbosity
        )
        verbosity_data_preview = (
            verbosity_data_preview if verbosity_data_preview is not None else verbosity
        )
        verbosity_missing_values = (
            verbosity_missing_values if verbosity_missing_values is not None else verbosity
        )
        if verbosity_rows_with_missing_value is None:
            verbosity_rows_with_missing_value = verbosity
        verbosity_constant_occurence = (
            verbosity_constant_occurence if verbosity_constant_occurence is not None else verbosity
        )
        verbosity_duplicate_rows = (
            verbosity_duplicate_rows if verbosity_duplicate_rows is not None else verbosity
        )

        subsec = Overview.OverviewSubsection

        # Store subsection verbosities
        verbosities = {
            subsec.QuickInfo: verbosity_quick_info,
            subsec.DataTypes: verbosity_data_types,
            subsec.DataPreview: verbosity_data_preview,
            subsec.MissingValues: verbosity_missing_values,
            subsec.RowsWithMissingValue: verbosity_rows_with_missing_value,
            subsec.ConstantOccurence: verbosity_constant_occurence,
            subsec.DuplicateRows: verbosity_duplicate_rows,
        }

        # By default use all subsections
        if subsections is None:
            subsections_all = list(Overview.OverviewSubsection)
        else:
            subsections_all = subsections

        # Store subsections with 0 verbosity
        self.subsections_0 = [sub for sub in subsections_all if verbosities[sub] == 0]

        if len(self.subsections_0) == len(subsections_all) and subsections is None:
            self.subsections_0 = None

        # Construct objects that implement subsections
        enum_to_implementation = {
            subsec.QuickInfo: QuickInfo(verbosity_quick_info, columns),
            subsec.DataTypes: DataTypes(verbosity_data_types, columns),
            subsec.DataPreview: DataPreview(verbosity_data_preview, columns),
            subsec.MissingValues: MissingValues(verbosity_missing_values, columns),
            subsec.RowsWithMissingValue: RowsWithMissingValue(
                verbosity_rows_with_missing_value, columns
            ),
            subsec.ConstantOccurence: ConstantOccurence(verbosity_constant_occurence, columns),
            subsec.DuplicateRows: DuplicateRows(verbosity_duplicate_rows, columns),
        }

        # Store subsection selection
        if subsections is None:
            subsections_implementations = [
                enum_to_implementation[sub] for sub in Overview.OverviewSubsection
            ]
        else:
            subsections_implementations = [enum_to_implementation[sub] for sub in subsections]
        super().__init__(subsections_implementations, verbosity, columns)

    @property
    def name(self) -> str:
        return "Overview"

    @staticmethod
    def overview_analysis(
        df: pd.DataFrame,
        subsections: Optional[List[OverviewSubsection]] = None,
        columns: Optional[List[str]] = None,
    ) -> None:
        """Generates overview analysis for df.

        Parameters
        ----------
        df : pd.DataFrame
            Data to be analyzed
        subsections : List[OverviewSubsection], optional
            Subsections to include into the overview
        columns : List[str], optional
            Subset of columns of df to consider in overview, by default all columns are used.
        """
        if columns is not None:
            df = df[columns]
        overview = Overview(subsections=subsections, verbosity=Verbosity.LOW, columns=columns)
        for sub in overview.subsections:
            sub.show(df)

    def required_imports(self) -> List[str]:
        """Returns a list of imports to be put at the top of a generated notebook.

        Returns
        -------
        List[str]
            List of import strings to be added at the top of the generated notebook,
            e.g. ['import pandas as pd', 'import numpy as np']
        """
        if self.verbosity == Verbosity.LOW:
            imports = {
                "from edvart.report_sections.dataset_overview import Overview\n"
                "overview_analysis = Overview.overview_analysis"
            }
            for subsec in self.subsections:
                if subsec.verbosity > Verbosity.LOW:
                    imports.update(subsec.required_imports())

            return list(imports)
        return super().required_imports()

    def add_cells(self, cells: List[Dict[str, Any]]) -> None:
        """Adds cells to the list of cells.

        Cells can be either code cells or markdown cells.

        Parameters
        ----------
        cells : List[Dict[str, Any]]
            List of generated notebook cells which are represented as dictionaries.
        """
        section_header = nbfv4.new_markdown_cell(self.get_title(section_level=1))
        cells.append(section_header)

        if self.verbosity == Verbosity.LOW:
            code = "overview_analysis(df=df"
            if self.subsections_0 is not None:
                arg_subsections_names = [
                    f"Overview.OverviewSubsection.{str(sub)}" for sub in self.subsections_0
                ]
                code += f", subsections={arg_subsections_names}".replace("'", "")
            if self.columns is not None:
                code += f", columns={self.columns}"
            code += ")"
            cells.append(nbfv4.new_code_cell(code))
            for subsec in self.subsections:
                if subsec.verbosity > Verbosity.LOW:
                    subsec.add_cells(cells)
        else:
            super().add_cells(cells)

    def show(self, df: pd.DataFrame) -> None:
        """Generates cell output of this section in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output.
        """
        display(Markdown(self.get_title(section_level=1)))
        super().show(df)


class QuickInfo(Section):
    """Generates the Quick info subsection.

    Parameters
    ----------
    verbosity : Verbosity
        Verbosity of the code generated in the exported notebook.
    columns : List[str], optional
        List of columns to consider in quick info.
        If None, all columns are used.
    """

    @property
    def name(self) -> str:
        return "Quick Info"

    @staticmethod
    def quick_info(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        additional_rows: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Renders a quick info table about df in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data which to analyze.
        columns : List[str], optional
            List of columns of df to analyze. All columns of df are used by default.
        additional_rows : Dict[str, Any], optional
            Additional custom rows to add to the table.
        """
        if columns is not None:
            df = df[columns]
        missing_cells = df.isna().sum().sum()
        missing_cells_percent = 100 * missing_cells / (df.shape[0] * df.shape[1])

        zeros = (df == 0).sum().sum()
        zeros_percent = 100 * zeros / (df.shape[0] * df.shape[1])

        duplicate_rows = df.duplicated().sum()
        duplicate_rows_percent = 100 * duplicate_rows / len(df)

        df_info_rows = {
            "Rows": df.shape[0],
            "Columns": df.shape[1],
            "Missing cells": f"{missing_cells} ({missing_cells_percent:,.02f} %)",
            "Zeros": f"{zeros} ({zeros_percent:.02f} %)",
            "Duplicate rows": f"{duplicate_rows} ({duplicate_rows_percent:,.02f} %)",
        }

        if additional_rows is not None:
            df_info_rows = {**df_info_rows, **additional_rows}

        render_dictionary(df_info_rows)

    def required_imports(self) -> List[str]:
        """Returns a list of imports to be put at the top of a generated notebook.

        Returns
        -------
        List[str]
            List of import strings to be added at the top of the generated notebook,
            e.g. ['import pandas as pd', 'import numpy as np'].
        """
        if self.verbosity <= Verbosity.MEDIUM:
            return [
                total_dedent(
                    """
                    from edvart.report_sections.dataset_overview import QuickInfo
                    quick_info = QuickInfo.quick_info
                    """
                )
            ]
        return []

    def add_cells(self, cells: List[Dict[str, Any]]) -> None:
        """Adds cells to the list of cells.

        Parameters
        ----------
        cells : List[Dict[str, Any]]
            List of generated notebook cells which are represented as dictionaries.
        """
        section_header = nbfv4.new_markdown_cell(self.get_title(section_level=2))
        cells.append(section_header)

        if self.columns is None:
            default_call = "quick_info(df=df)"
        else:
            default_call = f"quick_info(df=df, columns={self.columns})"

        if self.verbosity <= Verbosity.MEDIUM:
            code = default_call
        elif self.verbosity == Verbosity.HIGH:
            code = (
                get_code(render_dictionary)
                + 2 * "\n"
                + get_code(QuickInfo.quick_info)
                + 2 * "\n"
                + default_call
            )

        cells.append(nbfv4.new_code_cell(code))

    def show(self, df: pd.DataFrame) -> None:
        """Renders the quick info table in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data which to use for the table.
        """
        display(Markdown(self.get_title(section_level=2)))
        QuickInfo.quick_info(df, columns=self.columns)


class DataTypes(Section):
    """Generates data types inference subsection.

    Parameters
    ----------
    verbosity : Verbosity
        Verbosity of the code generated in the exported notebook.
    columns : List[str], optional
        List of columns for which to infer data type.
        If None, all columns are used.
    """

    @property
    def name(self) -> str:
        return "Data Types"

    @staticmethod
    def data_types(df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
        """Renders a table with inferred data types in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame:
            Data for which to infer data types
        columns : List[str], optional
            List of columns for which to infer data type. All columns of df are used by default.
        """
        if columns is not None:
            df = df[columns]
        dtypes = df.apply(
            func=infer_data_type,
            axis=0,
            result_type="expand",
            string_representation=True,
        )

        # Convert result to frame for viewing
        dtypes_frame = series_to_frame(
            series=dtypes, index_name="Column Name", column_name="edvart Data Type"
        )

        display(hide_index(dtypes_frame))

    def required_imports(self) -> List[str]:
        """Returns a list of imports to be put at the top of a generated notebook.

        Returns
        -------
        List[str]
            List of import strings to be added at the top of the generated notebook,
            e.g. ['import pandas as pd', 'import numpy as np'].
        """
        base_imports = ["from edvart.pandas_formatting import hide_index"]
        if self.verbosity <= Verbosity.MEDIUM:
            return base_imports + [
                total_dedent(
                    """
                    from edvart.report_sections.dataset_overview import DataTypes
                    data_types = DataTypes.data_types
                    """
                )
            ]
        return base_imports + [
            "from enum import IntEnum",
            "import numpy as np",
            "from IPython.display import display",
        ]

    def add_cells(self, cells: List[Dict[str, Any]]) -> None:
        """Adds data type inference cells to the list of cells.

        Parameters
        ----------
        cells : List[Dict[str, Any]]
            List of generated notebook cells which are represented as dictionaries.
        """
        section_header = nbfv4.new_markdown_cell(self.get_title(section_level=2))
        cells.append(section_header)

        if self.columns is None:
            default_call = "data_types(df=df)"
        else:
            default_call = f"data_types(df=df, columns={self.columns})"

        if self.verbosity <= Verbosity.MEDIUM:
            code = default_call
        elif self.verbosity == Verbosity.HIGH:
            code = (
                get_code(series_to_frame)
                + 2 * "\n"
                + get_code(DataType)
                + 2 * "\n"
                + get_code(is_numeric)
                + 2 * "\n"
                + get_code(is_categorical)
                + 2 * "\n"
                + get_code(is_boolean)
                + 2 * "\n"
                + get_code(is_date)
                + 2 * "\n"
                + get_code(infer_data_type)
                + 2 * "\n"
                + get_code(DataTypes.data_types)
                + 2 * "\n"
                + default_call
            )

        cells.append(nbfv4.new_code_cell(code))

    def show(self, df: pd.DataFrame) -> None:
        """Renders a table with inferred data types in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame:
            Data for which to infer data types.
        """
        display(Markdown(self.get_title(section_level=2)))
        DataTypes.data_types(df=df, columns=self.columns)


class DataPreview(Section):
    """Generates data preview (head, tail, sample) subsection.

    Parameters
    ----------
    verbosity : Verbosity
        Verbosity of the code generated in the exported notebook.
    columns : List[str], optional
        List of columns to preview.
        If None, all columns are used.
    """

    @property
    def name(self) -> str:
        return "Data Preview"

    @staticmethod
    def data_preview(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        n_head: int = 5,
        n_tail: int = 5,
        n_sample: int = 5,
    ) -> None:
        """Renders data preview tables in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data which to preview.
        columns : List[str], optional
            Columns of df to preview. All columns of df are used by default.
        n_head : int
            Number of first n rows of df to render, if None no preview is rendered.
        n_tail : int
            Number of last n rows of df to render, if None no preview is rendered.
        n_sample : int
            Size of random sample of df to render, if None no preview is rendered.
        """
        if columns is not None:
            df = df[columns]
        if n_head is not None:
            display(Markdown("### First rows"))
            display(df.head(n_head))
        if n_tail is not None:
            display(Markdown("### Last rows"))
            display(df.tail(n_tail))
        if n_sample is not None:
            display(Markdown("### Sample"))
            display(df.sample(min(n_sample, len(df))))

    def required_imports(self) -> List[str]:
        """Returns a list of imports to be put at the top of a generated notebook.

        Returns
        -------
        List[str]
            List of import strings to be added at the top of the generated notebook,
            e.g. ['import pandas as pd', 'import numpy as np'].
        """
        if self.verbosity <= Verbosity.MEDIUM:
            return [
                total_dedent(
                    """
                    from edvart.report_sections.dataset_overview import DataPreview
                    data_preview = DataPreview.data_preview
                    """
                )
            ]
        return [
            "from IPython.display import display",
            "from IPython.display import Markdown",
        ]

    def add_cells(self, cells: List[Dict[str, Any]]) -> None:
        """Adds dataframe preview cells to the list of cells.

        Parameters
        ----------
        cells : List[Dict[str, Any]]
            List of generated notebook cells which are represented as dictionaries.
        """
        section_header = nbfv4.new_markdown_cell(self.get_title(section_level=2))
        cells.append(section_header)

        if self.columns is None:
            default_call = "data_preview(df=df)"
        else:
            default_call = f"data_preview(df=df, columns={self.columns})"

        if self.verbosity <= Verbosity.MEDIUM:
            code = default_call
        elif self.verbosity == Verbosity.HIGH:
            code = get_code(DataPreview.data_preview) + 2 * "\n" + default_call

        cells.append(nbfv4.new_code_cell(code))

    def show(self, df: pd.DataFrame) -> None:
        """Renders data preview tables in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data which to preview
        """
        display(Markdown(self.get_title(section_level=2)))
        DataPreview.data_preview(df=df, columns=self.columns)


class MissingValues(Section):
    """Generates missing values percentages table for each column of the dataframe.

    Parameters
    ----------
    verbosity : Verbosity
        Verbosity of the code generated in the exported notebook.
    columns : List[str], optional
        List of columns for which to count missing values. If None, all columns are used.
    """

    @property
    def name(self) -> str:
        return "Missing Values"

    @staticmethod
    def missing_values(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        bar_plot: bool = True,
        bar_plot_figsize: Tuple[int, int] = (15, 6),
        bar_plot_title: str = "Missing Values Percentage of Each Column",
        bar_plot_ylim: float = 0,
        bar_plot_color: str = "#FFA07A",
        **bar_plot_args: Dict[str, Any],
    ) -> None:
        """Displays a table of missing values percentages for each column of df and a bar plot
        of the percentages.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe for which to calculate missing values.
        columns : Optional[List[str]], optional
            Subset of columns for which to calculate missing values percentage.
            If None, all columns of df are used.
        bar_plot : bool
            Whether to also display a bar plot visualizing missing values percentages for each
            column.
        bar_plot_figsize : Tuple[int, int]
            Width and height of the bar plot.
        bar_plot_title : str
            Title of the bar plot.
        bar_plot_ylim : float
            Bar plot y axis bottom limit.
        bar_plot_color : str
            Color of bars in the bar plot in hex format.
        bar_plot_args : Dict[str, Any]
            Additional kwargs passed to pandas.Series.bar.
        """
        if columns is not None:
            df = df[columns]

        # Count null values
        null_count = df.isna().sum()
        null_percentage = 100 * null_count / len(df)

        if null_count.sum() == 0:
            print("There are no missing values")
            return

        # Convert series to frames
        null_count_frame = series_to_frame(
            series=null_count, index_name="Column Name", column_name="Null Count"
        )
        null_percentage_frame = series_to_frame(
            series=null_percentage, index_name="Column Name", column_name="Null %"
        )
        # Merge null count and percentage into one frame
        null_stats_frame = null_count_frame.merge(
            null_percentage_frame, on="Column Name"
        ).sort_values("Null Count", ascending=False)

        display(
            hide_index(null_stats_frame)
            .bar(color="#FFA07A", subset=["Null %"], vmax=100)
            .format({"Null %": "{:.03f}"})
        )

        # Display bar plot of missing value percentages
        if bar_plot:
            (
                null_percentage_frame.sort_values("Null %", ascending=False)
                .plot.bar(
                    x="Column Name",
                    figsize=bar_plot_figsize,
                    title=bar_plot_title,
                    ylim=bar_plot_ylim,
                    color=bar_plot_color,
                    **bar_plot_args,
                )
                .set_ylabel("Missing Values [%]")
            )
            plt.show()

    def required_imports(self) -> List[str]:
        """Returns a list of imports to be put at the top of a generated notebook.

        Returns
        -------
        List[str]
            List of import strings to be added at the top of the generated notebook,
            e.g. ['import pandas as pd', 'import numpy as np']
        """
        base_imports = ["from edvart.pandas_formatting import hide_index"]
        if self.verbosity <= Verbosity.MEDIUM:
            return base_imports + [
                total_dedent(
                    """
                    from edvart.report_sections.dataset_overview import MissingValues
                    missing_values = MissingValues.missing_values
                    """
                )
            ]
        return base_imports + [
            "from IPython.display import display",
            "import matplotlib.pyplot as plt",
        ]

    def add_cells(self, cells: List[Dict[str, Any]]) -> None:
        """Adds code cells which calculate missing values percentage table to the list of cells.

        Parameters
        ----------
        cells : List[Dict[str, Any]]
            List of generated notebook cells which are represented as dictionaries.
        """
        section_header = nbfv4.new_markdown_cell(self.get_title(section_level=2))
        cells.append(section_header)

        if self.columns is None:
            default_call = "missing_values(df=df)"
        else:
            default_call = f"missing_values(df=df, columns={self.columns})"

        if self.verbosity <= Verbosity.MEDIUM:
            code = default_call
        elif self.verbosity == Verbosity.HIGH:
            code = (
                get_code(series_to_frame)
                + 2 * "\n"
                + get_code(MissingValues.missing_values)
                + 2 * "\n"
                + default_call
            )

        cells.append(nbfv4.new_code_cell(code))

    def show(self, df: pd.DataFrame) -> None:
        """Generates missing values percentages table in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output
        """
        display(Markdown(self.get_title(section_level=2)))
        MissingValues.missing_values(df=df, columns=self.columns)


class ConstantOccurence(Section):
    """Generates table with occurence of a constant in each column.

    Parameters
    ----------
    verbosity : Verbosity
        Verbosity of the code generated in the exported notebook.
    columns : List[str], optional
        List of columns to count constant occurence in. If None, all columns are used.
    """

    @property
    def name(self) -> str:
        return "Constant Occurence"

    @staticmethod
    def constant_occurence(
        df: pd.DataFrame, columns: Optional[List[str]] = None, constant: Any = 0
    ) -> None:
        """Displays a table with occurence of a constant in each column.

        By default, check for 0 occurence.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe for which to calculate constant values occurence.
        columns : Optional[List[str]], optional
            Subset of columns for which to calculate constant values occurence.
            If None, all columns of df are used.
        constant : Any
            Constant for which to check occurence in df, by default 0.
        """
        if columns is not None:
            df = df[columns]

        # Count constant counts
        constant_count = (df == constant).sum()
        constant_percentage = 100 * constant_count / len(df)

        # Convert series to frames
        constant_count_frame = series_to_frame(
            series=constant_count,
            index_name="Column Name",
            column_name=f'"{constant}" Count',
        )

        constant_percentage_frame = series_to_frame(
            series=constant_percentage,
            index_name="Column Name",
            column_name=f'"{constant}" %',
        )

        # Merge absolute and relative counts
        constant_stats_frame = constant_count_frame.merge(
            constant_percentage_frame, on="Column Name"
        ).sort_values(f'"{constant}" %', ascending=False)

        # Display table
        display(
            hide_index(constant_stats_frame)
            .bar(color="#FFA07A", subset=[f'"{constant}" %'], vmax=100)
            .format({f'"{constant}" %': "{:.03f}"})
        )

    def required_imports(self) -> List[str]:
        """Returns a list of imports to be put at the top of a generated notebook.

        Returns
        -------
        List[str]
            List of import strings to be added at the top of the generated notebook,
            e.g. ['import pandas as pd', 'import numpy as np']
        """
        base_imports = ["from edvart.pandas_formatting import hide_index"]
        if self.verbosity <= Verbosity.MEDIUM:
            return base_imports + [
                total_dedent(
                    """
                    from edvart.report_sections.dataset_overview import ConstantOccurence
                    constant_occurence = ConstantOccurence.constant_occurence
                    """
                )
            ]
        return base_imports + ["from IPython.display import display"]

    def add_cells(self, cells: List[Dict[str, Any]]) -> None:
        """Adds code cells which calculate constant occurence table to the list of cells.

        Parameters
        ----------
        cells : List[Dict[str, Any]]
            List of generated notebook cells which are represented as dictionaries.
        """
        section_header = nbfv4.new_markdown_cell(self.get_title(section_level=2))
        cells.append(section_header)

        if self.columns is None:
            default_call = "constant_occurence(df=df)"
        else:
            default_call = f"constant_occurence(df=df, columns={self.columns})"

        if self.verbosity <= Verbosity.MEDIUM:
            code = default_call
        elif self.verbosity == Verbosity.HIGH:
            code = (
                get_code(series_to_frame)
                + 2 * "\n"
                + get_code(ConstantOccurence.constant_occurence)
                + 2 * "\n"
                + default_call
            )

        cells.append(nbfv4.new_code_cell(code))

    def show(self, df: pd.DataFrame) -> None:
        """Generates constant occurence table in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output
        """
        display(Markdown(self.get_title(section_level=2)))
        ConstantOccurence.constant_occurence(df=df, columns=self.columns)


class RowsWithMissingValue(Section):
    """Counts number of rows with at least one value missing.

    Parameters
    ----------
    verbosity : Verbosity
        Verbosity of the code generated in the exported notebook.
    columns : List[str], optional
        List of columns to consider when counting. If None, all columns are used.
    """

    @property
    def name(self) -> str:
        return "Rows With Missing Value"

    @staticmethod
    def missing_value_row_count(df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
        """Displays a table with missing value row count and percentage.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe for which to counnt missing value rows.
        columns : Optional[List[str]], optional
            List of columns to consider when counting. If None, all columns are used.
        """
        if columns is not None:
            df = df[columns]

        # Count rows with at least one value missing
        num_rows_missing_value = df.isna().any(axis=1).sum()

        if num_rows_missing_value == 0:
            print("There are no missing values")
            return

        # Relative count
        percentage_rows_missing_value = 100 * num_rows_missing_value / len(df)

        missing_value_rows_info = {
            "Missing value column subset": "all columns" if columns is None else columns,
            "Missing value row count": num_rows_missing_value,
            "Missing value row percentage": f"{percentage_rows_missing_value:.02f} %",
        }

        # Display table
        render_dictionary(missing_value_rows_info)

    def required_imports(self) -> List[str]:
        """Returns a list of imports to be put at the top of a generated notebook.

        Returns
        -------
        List[str]
            List of import strings to be added at the top of the generated notebook,
            e.g. ['import pandas as pd', 'import numpy as np']
        """
        if self.verbosity <= Verbosity.MEDIUM:
            return [
                total_dedent(
                    """
                    from edvart.report_sections.dataset_overview import RowsWithMissingValue
                    missing_value_row_count = RowsWithMissingValue.missing_value_row_count
                    """
                )
            ]
        return ["from IPython.display import display"]

    def add_cells(self, cells: List[Dict[str, Any]]) -> None:
        """Adds code cells which count the number of rows with missing value to the list of cells.

        Parameters
        ----------
        cells : List[Dict[str, Any]]
            List of generated notebook cells which are represented as dictionaries.
        """
        section_header = nbfv4.new_markdown_cell(self.get_title(section_level=2))
        cells.append(section_header)

        if self.columns is None:
            default_call = "missing_value_row_count(df=df)"
        else:
            default_call = f"missing_value_row_count(df=df, columns={self.columns})"

        if self.verbosity <= Verbosity.MEDIUM:
            code = default_call
        elif self.verbosity == Verbosity.HIGH:
            code = (
                get_code(render_dictionary)
                + 2 * "\n"
                + get_code(RowsWithMissingValue.missing_value_row_count)
                + 2 * "\n"
                + default_call
            )

        cells.append(nbfv4.new_code_cell(code))

    def show(self, df: pd.DataFrame) -> None:
        """Generates a table with missing value row count and percentage in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output
        """
        display(Markdown(self.get_title(section_level=2)))
        RowsWithMissingValue.missing_value_row_count(df=df, columns=self.columns)


class DuplicateRows(Section):
    """Counts number of duplicated rows.

    Parameters
    ----------
    verbosity : Verbosity
        Verbosity of the code generated in the exported notebook.
    columns : List[str], optional
        List of columns to consider when counting. If None, all columns are used.
    """

    @property
    def name(self) -> str:
        return "Duplicate Rows"

    @staticmethod
    def duplicate_row_count(df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
        """Displays a table with duplicated row count and percentage.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe for which to counnt missing value rows.
        columns : Optional[List[str]], optional
            List of columns to consider when counting. If None, all columns are used.
        """
        if columns is not None:
            df = df[columns]

        # Count duplicated rows
        num_duplicated_rows = df.duplicated().sum()

        if num_duplicated_rows == 0:
            print("There are no duplicated rows")
            return

        # Relative count
        percentage_duplicated_rows = 100 * num_duplicated_rows / len(df)

        duplicate_rows_info = {
            "Duplicate rows column subset": "all columns" if columns is None else columns,
            "Duplicate row count": num_duplicated_rows,
            "Duplicate row percentage": f"{percentage_duplicated_rows:.02f} %",
        }

        # Display table
        render_dictionary(duplicate_rows_info)

    def required_imports(self) -> List[str]:
        """Returns a list of imports to be put at the top of a generated notebook.

        Returns
        -------
        List[str]
            List of import strings to be added at the top of the generated notebook,
            e.g. ['import pandas as pd', 'import numpy as np']
        """
        if self.verbosity <= Verbosity.MEDIUM:
            return [
                total_dedent(
                    """
                    from edvart.report_sections.dataset_overview import DuplicateRows
                    duplicate_row_count = DuplicateRows.duplicate_row_count
                    """
                )
            ]
        return ["from IPython.display import display"]

    def add_cells(self, cells: List[Dict[str, Any]]) -> None:
        """Adds code cells which count the number of duplicated rows to the list of cells.

        Parameters
        ----------
        cells : List[Dict[str, Any]]
            List of generated notebook cells which are represented as dictionaries.
        """
        section_header = nbfv4.new_markdown_cell(self.get_title(section_level=2))
        cells.append(section_header)

        if self.columns is None:
            default_call = "duplicate_row_count(df=df)"
        else:
            default_call = f"duplicate_row_count(df=df, columns={self.columns})"

        if self.verbosity <= Verbosity.MEDIUM:
            code = default_call
        elif self.verbosity == Verbosity.HIGH:
            code = (
                get_code(render_dictionary)
                + 2 * "\n"
                + get_code(DuplicateRows.duplicate_row_count)
                + 2 * "\n"
                + default_call
            )

        cells.append(nbfv4.new_code_cell(code))

    def show(self, df: pd.DataFrame) -> None:
        """Displays a table with duplicated row count and percentage in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output
        """
        display(Markdown(self.get_title(section_level=2)))
        DuplicateRows.duplicate_row_count(df=df, columns=self.columns)
