"""Report package."""

# Standard imports
import base64
import logging
import os
import pickle
from abc import ABC
from typing import List, Optional, Tuple, Union

# External library imports
import nbconvert
import nbformat as nbf
import nbformat.v4 as nbf4
import pandas as pd

# Internal library imports
from edvart.data_types import is_date
from edvart.report_sections.bivariate_analysis import BivariateAnalysis
from edvart.report_sections.code_string_formatting import code_dedent
from edvart.report_sections.dataset_overview import Overview
from edvart.report_sections.group_analysis import GroupAnalysis
from edvart.report_sections.multivariate_analysis import MultivariateAnalysis
from edvart.report_sections.table_of_contents import TableOfContents
from edvart.report_sections.timeseries_analysis import TimeseriesAnalysis
from edvart.report_sections.univariate_analysis import UnivariateAnalysis


class ReportBase(ABC):
    """
    Abstract base class for reports.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Data from which to generate the report.
    use_default_sections : bool (default = True)
        Whether add the report's default sections to the report.
    verbosity : int (default = 0)
        The default verbosity for the exported code of the entire report, has to be one of
        [0, 1, 2], by default 0.

    Raises
    ------
    ValueError
        If verbosity is not one of [0, 1, 2].
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        use_default_sections: bool = True,
        verbosity: int = 0,
    ):
        self._class_logger = logging.getLogger(__name__).getChild(self.__class__.__name__)
        self.df = dataframe
        self.use_default_sections = use_default_sections
        self.sections = []
        # Check for global verbosity validity
        if verbosity not in [0, 1, 2]:
            raise ValueError(f"Verbosity has to be one of [0, 1, 2], not {verbosity}.")
        self.verbosity = verbosity

    def show(self) -> None:
        """Renders the report in the calling notebook."""
        for section in self.sections:
            if isinstance(section, TableOfContents):
                section.show(self.sections)
            else:
                section.show(self.df)

    def _select_columns(
        self,
        use_columns: Optional[List[str]] = None,
        omit_columns: Optional[List[str]] = None,
    ) -> Optional[List[str]]:
        """Returns list columns from self.df.

        Includes columns from use_columns and excludes the ones in omit_columns.

        Parameters
        ----------
        use_columns : List[str], optional
            Columns of self.df to include in the resulting list.
        omit_columns : List[str], optional
            Columns of self.df to exclude from the resulting list.

        Returns
        -------
        List[str], optional
            List of columns, None if use_columns and omit_columns are None
        """
        if use_columns is None and omit_columns is None:
            return None
        if omit_columns is None:
            return use_columns
        if use_columns is None:
            use_columns = list(self.df.columns)

        return [col for col in use_columns if col not in omit_columns]

    def export_notebook(self, notebook_filepath: str) -> None:
        """Exports the report as an .ipynb file.

        Parameters
        ----------
        notebook_filepath : str
            Filepath of the exported notebook.
        """
        nb = self._generate_notebook()

        # Save notebook to file
        with open(notebook_filepath, "w") as notebook_file:
            nbf.write(nb, notebook_file)

    def _generate_notebook(
        self,
        dataset_name: str = "[INSERT DATASET NAME]",
        dataset_description: str = "[INSERT DATASET DESCRIPTION]",
        load_df: str = "df = ...",
        extra_imports: Optional[List[str]] = None,
        show_load_data: bool = True,
    ) -> nbf.NotebookNode:
        """Generate a notebook object for the report.

        Parameters
        ----------
        dataset_name : str (default = "[INSERT DATASET NAME]")
            Name of dataset to be used in the title of the report.
        dataset_description : str (default = "[INSERT DATASET DESCRIPTION]")
            Descritpion of dataset to be used below the title of the report.
        load_df : str (default = 'df = ...')
            Code string for loading a dataset to variable `df`.
        extra_imports : List[str], optional
            Any additional imports to be included in imports section
            (other than imports required by report sections).
        show_load_data : bool (default = True)
            Whether to display a title for section where data is loaded.

        Returns
        -------
        nbf.NotebookNode
            Generated notebook object.
        """
        # Create notebook structure
        nb = nbf4.new_notebook()

        # Add notebook title cell in markdown
        nb["cells"].append(
            nbf4.new_markdown_cell(f"# {dataset_name} Report\n{dataset_description}")
        )

        # Add imports cell
        imports_set = {
            "import pandas as pd",
            "import os",
            "from typing import Any, Callable, Dict, List, Optional, Tuple, Union",
            "import plotly.offline as py",
            "py.init_notebook_mode()",
            "import plotly.io as pio",
            "pio.renderers.default = 'plotly_mimetype+notebook'",
        }
        if extra_imports is not None:
            imports_set.update(extra_imports)
        for section in self.sections:
            imports_set.update(section.required_imports())
        imports = sorted(list(imports_set))

        if len(imports) > 0:
            nb["cells"].append(nbf4.new_code_cell("\n".join(imports)))

        # Add load data cell
        if show_load_data:
            nb["cells"].append(nbf4.new_markdown_cell("## Load Data\n---"))
        nb["cells"].append(nbf4.new_code_cell(load_df))

        # Generate code for each report section
        for section in self.sections:
            if isinstance(section, TableOfContents):
                section.add_cells(self.sections, nb["cells"])
            else:
                section.add_cells(nb["cells"])

        return nb

    def _export_html(
        self,
        nb: nbf.NotebookNode,
        html_filepath: str,
        template_name: Optional[str] = None,
        template_filepath: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        """Export a notebook object to HTML.

        Parameters
        ----------
        nb : nbf.NotebookNode
            Notebook object to be exported to HTML.
        html_filepath : str
            File path to save exported HTML to.
        template_name: str, optional
            Name of the template to use for exporting the notebook.

            The template must be found in a Jupyter path
            (see
            https://nbconvert.readthedocs.io/en/latest/customizing.html#where-are-nbconvert-templates-installed
            ).
            The default location is `$HOME/.local/share/jupyter/nbconvert/templates`
        template_filepath : str, optional
            Path to template file to use for exporting the notebook to HTML.
            Overrides the parameter `template_name` if specified.
        timeout: int (default = 120)
            Maximum number of seconds to wait for a cell to finish execution.
        """
        # Execute notebook to produce output of cells
        html_exp_kwargs = dict(
            preprocessors=[nbconvert.preprocessors.ExecutePreprocessor(timeout=timeout)]
        )
        if template_name is not None:
            html_exp_kwargs["template_name"] = template_name
        elif template_filepath is not None:
            html_exp_kwargs["template_file"] = template_filepath
        else:
            html_exp_kwargs["exclude_input"] = True

        html_exporter = nbconvert.HTMLExporter(**html_exp_kwargs)

        # Workaround for a warning from `nbconvert` regarding debugging
        # and frozen modules. We are not debugging, so we can safely ignore it.
        disable_validation_env_var_name = "PYDEVD_DISABLE_FILE_VALIDATION"
        env_original = os.environ.copy()
        os.environ[disable_validation_env_var_name] = "1"

        html = html_exporter.from_notebook_node(nb)[0]
        os.environ = env_original

        # Save HTML to file
        with open(html_filepath, "w") as html_file:
            print(html, file=html_file)

    def export_html(
        self,
        html_filepath: str,
        template_name: Optional[str] = None,
        template_filepath: Optional[str] = None,
        dataset_name: str = "[INSERT DATASET NAME]",
        dataset_description: str = "[INSERT DATASET DESCRIPTION]",
        timeout: int = 120,
    ) -> None:
        """Generate HTML report for an already-loaded DataFrame.

        Parameters
        ----------
        html_filepath: str
            File path to save exported HTML report to.
        template_name: str, optional
            Path to template file to use for exporting the notebook to HTML.

            The template must be found in a Jupyter path
            (see
            https://nbconvert.readthedocs.io/en/latest/customizing.html#where-are-nbconvert-templates-installed
            ).
            The default location is `$HOME/.local/share/jupyter/nbconvert/templates`
        template_filepath: str, optional
            Template to use when exporting the HTML report.
        dataset_name : str (default = "[INSERT DATASET NAME]")
            Name of dataset to be used in the title of the report.
        dataset_description : str (default = "[INSERT DATASET DESCRIPTION]")
            Description of dataset to be used below the title of the report.
        timeout: int (default = 120)
            Maximum number of seconds to wait for a cell to finish execution.
        """
        self._class_logger.info(
            "Exporting report to HTML requires generating and executing a notebook, "
            "which may take some time."
        )

        # Pickle self (the whole report) and encode it into a base64 string
        buffer = pickle.dumps(self, fix_imports=False)
        buffer_base64 = base64.b85encode(buffer)

        # Prepare code to inject into notebook,
        # which decodes data from base64 (injected as string literal)
        # and unpickles the the whole report object from the decoded binary data
        unpickle_report = code_dedent(
            f"""
                data = {buffer_base64}
                report = pickle.loads(base64.b85decode(data), fix_imports=False)
            """
        )

        # Create notebook object
        nb = nbf4.new_notebook()

        # Add title and description
        nb["cells"].append(
            nbf4.new_markdown_cell(f"# {dataset_name} Report\n{dataset_description}")
        )

        # Add code cells to generated notebook
        nb["cells"].append(nbf4.new_code_cell("import pickle\nimport base64"))
        nb["cells"].append(nbf4.new_code_cell(unpickle_report))
        nb["cells"].append(nbf4.new_code_cell("report.show()"))

        self._export_html(
            nb=nb,
            html_filepath=html_filepath,
            template_filepath=template_filepath,
            template_name=template_name,
            timeout=timeout,
        )

    def _dev_export_notebook(self, notebook_filepath: str) -> None:
        """Same as export_notebook, but conveniently adds dataset into exported notebook for
        development purposes.
        """
        nb = self._generate_notebook(
            extra_imports=["import edvart"],
            load_df="df = edvart.example_datasets.dataset_titanic()",
        )

        # Save notebook to file
        with open(notebook_filepath, "w") as notebook_file:
            nbf.write(nb, notebook_file)

    def add_overview(
        self,
        use_columns: Optional[List[str]] = None,
        omit_columns: Optional[List[str]] = None,
        subsections: Optional[List[Overview.OverviewSubsection]] = None,
        verbosity: Optional[int] = None,
        verbosity_quick_info: Optional[int] = None,
        verbosity_data_types: Optional[int] = None,
        verbosity_data_preview: Optional[int] = None,
        verbosity_missing_values: Optional[int] = None,
        verbosity_rows_with_missing_value: Optional[int] = None,
        verbosity_constant_occurrence: Optional[int] = None,
        verbosity_duplicate_rows: Optional[int] = None,
    ) -> "ReportBase":
        """
        Adds a dataset overview section to the report.

        Parameters
        ----------
        use_columns : List[str], optional
            Columns which to include in the overview section.
            If None, all columns are used.
        omit_columns : List[str], optional
            Columns to exclude from the overview section.
            If None, use_columns dictates column selection.
        subsections : List[Overview.OverviewSubsection], optional
            List of sub-sections to include into the Overview section.
            If None, all subsections are added.
        verbosity : int, optional
            Generated code verbosity global to the Overview sections, must be on of [0, 1, 2].

            0
                A single cell which generates the overview section is exported.
            1
                Parameterizable function calls for each subsection of the overview section are
                exported.
            2
                Similar to 1, but in addition function definitions are also exported.

            If subsection verbosities are None, then they will be overridden by this parameter.
        verbosity_quick_info : int, optional
            Quick info sub-section code verbosity.
        verbosity_data_types : int, optional
            Data types sub-section code verbosity.
        verbosity_data_preview : int, optional
            Data preview sub-section code verbosity.
        verbosity_missing_values : int, optional
            Missing values sub-section code verbosity.
        verbosity_rows_with_missing_value : int, optional
            Rows with missing value sub-section code verbosity.
        verbosity_constant_occurrence : int, optional
            Constant values occurrence sub-section code verbosity.
        verbosity_duplicate_rows : int, optional
            Duplicate rows sub-section code verbosity.

        """
        # Construct and store overview configuration
        self.sections.append(
            Overview(
                subsections=subsections,
                verbosity=verbosity if verbosity is not None else self.verbosity,
                columns=self._select_columns(use_columns, omit_columns),
                verbosity_quick_info=verbosity_quick_info,
                verbosity_data_types=verbosity_data_types,
                verbosity_data_preview=verbosity_data_preview,
                verbosity_missing_values=verbosity_missing_values,
                verbosity_rows_with_missing_value=verbosity_rows_with_missing_value,
                verbosity_constant_occurence=verbosity_constant_occurrence,
                verbosity_duplicate_rows=verbosity_duplicate_rows,
            )
        )
        return self

    def add_univariate_analysis(
        self,
        use_columns: Optional[List[str]] = None,
        omit_columns: Optional[List[str]] = None,
        verbosity: Optional[int] = None,
    ) -> "ReportBase":
        """Adds univariate section to the report.

        Parameters
        ----------
        use_columns : List[str], optional
            Columns which to analyze.
            If None, all columns are used.
        omit_columns : List[str], optional
            Columns to exclude from analysis.
            If None, use_columns dictates column selection.
        verbosity : int
            The verbosity of the code generated in the exported notebook,
            must be one of [0, 1, 2].

            0
                A single function call generates the entire univariate analysis section.
            1
                Function calls to parameterizable functions are generated for each column separately
                in separate cells.
            2
                Similar to 1, but in addition, function definitions are generated, column
                data type inference and default statistics become customizable.

        """
        self.sections.append(
            UnivariateAnalysis(
                df=self.df,
                verbosity=verbosity if verbosity is not None else self.verbosity,
                columns=self._select_columns(use_columns, omit_columns),
            )
        )
        return self

    def add_bivariate_analysis(
        self,
        use_columns: Optional[List[str]] = None,
        omit_columns: Optional[List[str]] = None,
        columns_x: Optional[List[str]] = None,
        columns_y: Optional[List[str]] = None,
        columns_pairs: Optional[List[Tuple[str, str]]] = None,
        subsections: Optional[List[BivariateAnalysis.BivariateAnalysisSubsection]] = None,
        verbosity: Optional[int] = None,
        verbosity_correlations: Optional[int] = None,
        verbosity_pairplot: Optional[int] = None,
        verbosity_contingency_table: Optional[int] = None,
        color_col: Optional[str] = None,
    ) -> "ReportBase":
        """Adds bivariate analysis section to the report.

        Parameters
        ----------
        use_columns : List[str], optional
            Columns which to analyze.
            If None, all columns are used.
        omit_columns : List[str], optional
            Columns to exclude from analysis.
            If None, use_columns dictates column selection.
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
            `columns`, `columns_x`, `columns_y` is specified. In that case, the first elements
            of each pair are treated as `columns_x` and the second elements as `columns_y` in
            pairplots and correlations.
        subsections : List[BivariateAnalysis.BivariateAnalysisSubsection], optional
            List of sub-sections to include into the BivariateAnalysis section.
            If None, all subsections are added.
        verbosity : int, optional
            The verbosity of the code generated in the exported notebook,
            must be one of [0, 1, 2].

            0
                A single function call generates the entire bivariate analysis section.
            1
                Function calls to parameterizable functions are generated for each column separately
                in separate cells.
            2
                Similar to 1, but in addition, function definitions are generated, column
                data type inference and default statistics become customizable.

        verbosity_correlations : int, optional
            Correlation plots subsection code verbosity.
        verbosity_pairplot : int, optional
            Pairplot subsection code verbosity.
        verbosity_contingency_table : int, optional
            Contingency table code verbosity.
        color_col : str, optional
            Name of column according to use for coloring of the multivariate analysis subsections.
            Coloring is currently supported in pairplot.
        """
        self.sections.append(
            BivariateAnalysis(
                subsections=subsections,
                verbosity=verbosity if verbosity is not None else self.verbosity,
                columns=self._select_columns(use_columns, omit_columns),
                columns_x=columns_x,
                columns_y=columns_y,
                columns_pairs=columns_pairs,
                verbosity_correlations=verbosity_correlations,
                verbosity_pairplot=verbosity_pairplot,
                verbosity_contingency_table=verbosity_contingency_table,
                color_col=color_col,
            )
        )

        return self

    def add_multivariate_analysis(
        self,
        use_columns: Optional[List[str]] = None,
        omit_columns: Optional[List[str]] = None,
        subsections: Optional[List[MultivariateAnalysis.MultivariateAnalysisSubsection]] = None,
        verbosity: Optional[int] = None,
        verbosity_pca: Optional[int] = None,
        verbosity_umap: Optional[int] = None,
        verbosity_parallel_coordinates: Optional[int] = None,
        verbosity_parallel_categories: Optional[int] = None,
        color_col: Optional[str] = None,
    ) -> "ReportBase":
        """Add multivariate analysis section to the report.

        Parameters
        ----------
        use_columns : List[str], optional
            Columns which to analyze.
            If None, all columns are used.
        omit_columns : List[str], optional
            Columns to exclude from analysis.
            If None, use_columns dictates column selection.
        subsections : List[MultivariateAnalysis.MultivariateAnalysisSubsection], optional
            List of sub-sections to include into the BivariateAnalysis section.
            If None, all subsections are added.
        verbosity : int, optional
            verbosity : int
            The verbosity of the code generated in the exported notebook,
            must be one of [0, 1, 2].

            0
                A single function call generates the entire univariate analysis section.
            1
                Function calls to parameterizable functions are generated for each column separately
                in separate cells.
            2
                Similar to 1, but in addition, function definitions are generated, column
                data type inference and default statistics become customizable.

        verbosity_pca : int, optional
            Principal component analysis subsection code verbosity.
        verbosity_umap : int, optional
            UMAP subsection code verbosity.
        verbosity_parallel_coordinates: int, optional
            Parallel coordinates subsection code verbosity.
        verbosity_parallel_categories: int, optional
            Parallel categories subsection code verbosity.
        color_col : str, optional
            Name of column to use for coloring of the multivariate analysis subsections.
            The exact method of coloring depends on each particular subsection.
        """
        self.sections.append(
            MultivariateAnalysis(
                subsections=subsections,
                df=self.df,
                verbosity=verbosity if verbosity is not None else self.verbosity,
                columns=self._select_columns(use_columns, omit_columns),
                verbosity_pca=verbosity_pca,
                verbosity_umap=verbosity_umap,
                verbosity_parallel_coordinates=verbosity_parallel_coordinates,
                verbosity_parallel_categories=verbosity_parallel_categories,
                color_col=color_col,
            )
        )

        return self

    def add_group_analysis(
        self,
        groupby: Union[str, List[str]],
        use_columns: Optional[List[str]] = None,
        omit_columns: Optional[List[str]] = None,
        verbosity: Optional[int] = None,
        show_within_group_statistics: bool = True,
        show_group_missing_values: bool = True,
        show_group_distribution_plots: bool = True,
    ) -> "ReportBase":
        """Add group analysis section to the report.

        Parameters
        ----------
        groupby: Union[str, List[str]]
            Column or list of columns to group by in group analysis.
        use_columns : List[str], optional
            Columns which to analyze.
            If None, all columns are used.
        omit_columns : List[str], optional
            Columns to exclude from analysis.
            If None, use_columns dictates column selection.
        verbosity : int, optional
            verbosity : int
            The verbosity of the code generated in the exported notebook,
            must be one of [0, 1, 2].

            0
                A single function call generates the entire univariate analysis section.
            1
                Function calls to parameterizable functions are generated for each column separately
                in separate cells.
            2
                Similar to 1, but in addition, function definitions are generated, column
                data type inference and default statistics become customizable.
        show_within_group_statistics : bool (default = True)
            Whether to show per-group statistics.
        show_group_missing_values : bool (default = True)
            Whether to show per-group missing values.
        show_group_distribution_plots : bool (default = True)
            Whether to show per-group distribution plots.
        """
        self.sections.append(
            GroupAnalysis(
                df=self.df,
                groupby=groupby,
                verbosity=verbosity if verbosity is not None else self.verbosity,
                columns=self._select_columns(use_columns, omit_columns),
                show_within_group_statistics=show_within_group_statistics,
                show_group_missing_values=show_group_missing_values,
                show_group_distribution_plots=show_group_distribution_plots,
            )
        )

        return self

    def add_table_of_contents(self, include_subsections: bool = True) -> "ReportBase":
        """Adds table of contents section to the report.

        Parameters
        ----------
        include_subsections: bool
            A boolean controlling whether the subsections should be included in the table of
            contents. However, they won't be included in an exported notebook created by report's
            export_notebook function.
        """
        self.sections.append(TableOfContents(include_subsections))
        return self


class Report(ReportBase):
    """This class instantiates an object that the edvart user should mainly use for report
    configuration and export.

    This class is intended for creating general-purpose reports.
    For creating a report for time-series data, please see `TimeseriesReport`.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Data from which to generate the report.
    use_default_sections : bool (default = True)
        If True, all default sections of the report are added, otherwise you have to add
        the sections manually using add_<section name>() methods.
        Default sections for this report are: overview, univariate analysis, bivariate analysis and
        multivariate analysis.
    verbosity : int (default = 0)
        The default verbosity for the exported code of the entire report, has to be one of
        [0, 1, 2], by default 0.
    verbosity_overview : int, optional
        Verbosity of the overview section
    verbosity_univariate_analysis : int, optional
        Verbosity of the univariate analysis section
    verbosity_bivariate_analysis : int, optiona
        Verbosity of the bivariate analysis section.
    verbosity_multivariate_analysis: int, optional
        Verbosity of the multivariate analysis section
    columns_overview : List[str], optional
        Subset of columns to use in overview section
    columns_univariate_analysis : List[str], optional
        Subset of columns to use in univariate analysis section
    columns_bivariate_analysis: List[str], optional
        Subset of columns to use in bivariate analysis section
    columns_multivariate_analysis: List[str], optional
        Subset of columns to use in multivariate analysis section
    columns_group_analysis: List[str], optional
        Subset of columns to use in group analysis section
    groupby: Union[str, List[str]], optional
        Column or list of columns to group by in group analysis. If None, group analysis will not be
        included by default. It can still be added later using `add_group_analysis`.
        If a single column is specified, it will be used to color points in multivariate analysis.
        Default: None.

    """

    # pylint:disable=too-many-locals
    def __init__(
        self,
        dataframe: pd.DataFrame,
        use_default_sections: bool = True,
        verbosity: int = 0,
        verbosity_overview: Optional[int] = None,
        verbosity_univariate_analysis: Optional[int] = None,
        verbosity_bivariate_analysis: Optional[int] = None,
        verbosity_multivariate_analysis: Optional[int] = None,
        verbosity_group_analysis: Optional[int] = None,
        columns_overview: Optional[List[str]] = None,
        columns_univariate_analysis: Optional[List[str]] = None,
        columns_bivariate_analysis: Optional[List[str]] = None,
        columns_multivariate_analysis: Optional[List[str]] = None,
        columns_group_analysis: Optional[List[str]] = None,
        groupby: Union[str, List[str]] = None,
    ):
        super().__init__(dataframe, use_default_sections, verbosity)

        # If section verbosities are not set, default to the global verbosity
        if verbosity_overview is None:
            verbosity_overview = verbosity
        if verbosity_univariate_analysis is None:
            verbosity_univariate_analysis = verbosity
        if verbosity_bivariate_analysis is None:
            verbosity_bivariate_analysis = verbosity
        if verbosity_multivariate_analysis is None:
            verbosity_multivariate_analysis = verbosity
        # Add default sections if user doesn't build the report manually
        if use_default_sections:
            self.add_table_of_contents(include_subsections=True)
            self.add_overview(verbosity=verbosity_overview, use_columns=columns_overview)
            self.add_univariate_analysis(
                verbosity=verbosity_univariate_analysis,
                use_columns=columns_univariate_analysis,
            )
            if isinstance(groupby, str):
                color_col = groupby
            elif hasattr(groupby, "__len__") and len(groupby) == 1:
                color_col = groupby[0]
            else:
                color_col = None
            self.add_bivariate_analysis(
                verbosity=verbosity_bivariate_analysis,
                use_columns=columns_bivariate_analysis,
                color_col=color_col,
            )
            self.add_multivariate_analysis(
                verbosity=verbosity_multivariate_analysis,
                use_columns=columns_multivariate_analysis,
                color_col=color_col,
            )
            if groupby is not None:
                self.add_group_analysis(
                    groupby=groupby,
                    use_columns=columns_group_analysis,
                    verbosity=verbosity_group_analysis,
                )


class TimeseriesReport(ReportBase):
    """This class instantiates an object that the edvart user should mainly use for report
    configuration and export, specifically for timeseries data.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Data from which to generate the report. Data needs to be indexed by time: pd.DateTimeIndex
        or pd.PeriodIndex.
        The data is assumed to be sorted according to the time index in ascending order.
    use_default_sections : bool, optional
        If True, all default sections of the report are added, otherwise you have to add
        the sections manually using add_<section name>() methods.
        Default sections for this report are overview, univariate analysis and timeseries analysis.
    verbosity : int (default = 0)
        The default verbosity for the exported code of the entire report, has to be one of
        [0, 1, 2], by default 0.
    verbosity_overview : int, optional
        Verbosity of the overview section
    verbosity_univariate_analysis : int, optional
        Verbosity of the univariate analysis section
    verbosity_timeseries_analysis : int, optional
        Verbosity of the timeseries analysis section
    columns_overview : List[str], optional
        Subset of columns to use in overview section
    columns_univariate_analysis : List[str], optional
        Subset of columns to use in univariate analysis section
    columns_timeseries_analysis : List[str], optional
        Subset of columns to use in timeseries analysis section
    sampling_rate : int, optional
        Sampling rate for Fourier transform and Short-time Fourier transform subsections. Determines
        frequency unit for analysis of frequencies, for example with monthly data and sampling rate
        12, yearly frequncy spectrum is produced.
        If not set, these two sections will not be included.
    stft_window_size : int, optional
        Windows size for short-time Fourier transform subsection. If not set, STFT will be exluded.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        use_default_sections: bool = True,
        verbosity: int = 0,
        verbosity_overview: Optional[int] = None,
        verbosity_univariate_analysis: Optional[int] = None,
        verbosity_timeseries_analysis: Optional[int] = None,
        columns_overview: Optional[List[str]] = None,
        columns_univariate_analysis: Optional[List[str]] = None,
        columns_timeseries_analysis: Optional[List[str]] = None,
        sampling_rate: Optional[int] = None,
        stft_window_size: Optional[int] = None,
    ):
        if not is_date(dataframe.index):
            raise ValueError(
                "Input dataframe needs to be indexed by time."
                "Please reindex your data to be indexed by either a DatetimeIndex or a PeriodIndex."
                "See `edvart.utils.reindex_to_datetime`, `edvart.utils.reindex_to_period`."
            )
        dataframe = dataframe.copy()
        if isinstance(dataframe.index.dtype, pd.PeriodDtype):
            dataframe.index = pd.PeriodIndex(dataframe.index)
        else:
            dataframe.index = pd.DatetimeIndex(dataframe.index)
        super().__init__(dataframe, use_default_sections, verbosity)

        if verbosity_overview is None:
            verbosity_overview = verbosity
        if verbosity_univariate_analysis is None:
            verbosity_univariate_analysis = verbosity
        if verbosity_timeseries_analysis is None:
            verbosity_timeseries_analysis = verbosity
        if use_default_sections:
            self.add_table_of_contents(include_subsections=True)
            self.add_overview(verbosity=verbosity_overview, use_columns=columns_overview)
            self.add_univariate_analysis(
                verbosity=verbosity_univariate_analysis,
                use_columns=columns_univariate_analysis,
            )
            self.add_timeseries_analysis(
                verbosity=verbosity_timeseries_analysis,
                use_columns=columns_timeseries_analysis,
                sampling_rate=sampling_rate,
                stft_window_size=stft_window_size,
            )

    def add_timeseries_analysis(
        self,
        use_columns: Optional[List[str]] = None,
        omit_columns: Optional[List[str]] = None,
        subsections: Optional[List[TimeseriesAnalysis.TimeseriesAnalysisSubsection]] = None,
        verbosity: Optional[int] = None,
        verbosity_time_analysis_plot: Optional[int] = None,
        verbosity_rolling_statistics: Optional[int] = None,
        verbosity_boxplots_over_time: Optional[int] = None,
        verbosity_seasonal_decomposition: Optional[int] = None,
        verbosity_autocorrelation: Optional[int] = None,
        verbosity_stationarity_tests: Optional[int] = None,
        verbosity_fourier_transform: Optional[int] = None,
        verbosity_short_time_ft: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        stft_window_size: Optional[int] = None,
    ) -> "TimeseriesReport":
        """Add timeseries analysis section to the report.

        Parameters
        ----------
        use_columns : List[str], optional
            Columns which to analyze.
            If None, all columns are used.
        omit_columns : List[str], optional
            Columns to exclude from analysis.
            If None, use_columns dictates column selection.
        subsections : List[TimeseriesAnalysis.TimeseriesAnalysisSubsection], optional
            List of sub-sections to include into the BivariateAnalysis section.
            If None, all subsections are added.
        verbosity : int, optional
            The verbosity of the code generated in the exported notebook,
            must be one of [0, 1, 2].

            0
                A single function call generates the entire bivariate analysis section.
            1
                Function calls to parameterizable functions are generated for each column separately
                in separate cells.
            2
                Similar to 1, but in addition, function definitions are generated, column
                data type inference and default statistics become customizable.

        verbosity_time_analysis_plot : int, optional
            Time analysis interactive plot subsection code verbosity.
        verbosity_rolling_statistics : int, optional
            Rolling statistics interactive plot subsection code verbosity.
        verbosity_boxplots_over_time : int, optional
            Boxplots grouped over time intervals plot subsection code verbosity.
        verbosity_seasonal_decomposition : int, optional
            Decomposition into trend, seasonal and residual components code verbosity.
        verbosity_autocorrelation : int, optional
            Autocorrelation and partial autocorrelation vs. lag code verbosity.
        verbosity_stationarity_tests : int, optional
            Stationarity tests code verbosity.
        verbosity_fourier_transform: int, optional
            Fourier transform and short-time Fourier transform code verbosity.
        verbosity_short_time_ft: int, optional
            Short-time Fourier transform transform spectrogram code verbosity.
        sampling_rate: int, optional
            Sampling rate for Fourier transform and Short-time Fourier transform subsections.
            Needs to be set in order for these two subs to be included.
        stft_window_size : int, optional
            Window size for Short-time Fourier transform. Needs to be set in order for the STFT
            subsection to be included.
        """
        self.sections.append(
            TimeseriesAnalysis(
                subsections=subsections,
                verbosity=verbosity if verbosity is not None else self.verbosity,
                columns=self._select_columns(use_columns, omit_columns),
                verbosity_time_analysis_plot=verbosity_time_analysis_plot,
                verbosity_rolling_statistics=verbosity_rolling_statistics,
                verbosity_boxplots_over_time=verbosity_boxplots_over_time,
                verbosity_seasonal_decomposition=verbosity_seasonal_decomposition,
                verbosity_autocorrelation=verbosity_autocorrelation,
                verbosity_stationarity_tests=verbosity_stationarity_tests,
                verbosity_fourier_transform=verbosity_fourier_transform,
                verbosity_short_time_ft=verbosity_short_time_ft,
                sampling_rate=sampling_rate,
                stft_window_size=stft_window_size,
            )
        )

        return self
