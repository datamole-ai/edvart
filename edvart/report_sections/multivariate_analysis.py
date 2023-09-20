"""Multivariate analysis package."""

from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import nbformat.v4 as nbfv4
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn.decomposition
from IPython.display import Markdown, display
from sklearn.preprocessing import StandardScaler

from edvart.data_types import is_boolean, is_categorical, is_numeric
from edvart.plots import scatter_plot_2d
from edvart.report_sections.code_string_formatting import get_code, total_dedent
from edvart.report_sections.section_base import ReportSection, Section, Verbosity
from edvart.utils import discrete_colorscale

try:
    from edvart.report_sections.umap import UMAP
except ImportError:
    UMAP_AVAILABLE = False
else:
    UMAP_AVAILABLE = True


class MultivariateAnalysis(ReportSection):
    """Generates the Multivariate analysis section of the report.

    Contains an enum MultivariateAnalysisSubsection of possible subsections.

    Parameters
    ----------
    df : pd.DataFrame
        Data for which to perform analysis.
    subsections : List[MultivariateAnalysisSubsection], optional
        List of subsections to include.
        All subsection in MultivariateAnalysisSubsection are included by default.
    verbosity : Verbosity
        Generated code verbosity global to the Multivariate sections.
        If subsection verbosities are None, then they will be overridden by this parameter.
    columns : List[str], optional
        Columns on which to do multivariate analysis.
        All columns of df will be used by default.
    verbosity_pca : Verbosity, optional
        Principal component analysis subsection code verbosity.
    verbosity_umap : Verbosity, optional
        UMAP subsection code verbosity.
    verbosity_parallel_coordinates : Verbosity, optional
        Parallel coordinates subsection code verbosity.
    verbosity_parallel_categories : Verbosity, optional
        Parallel categories subsection code verbosity.
    color_col : str, optional
        Name of the column according to which to color points in the sections.
        Both numerical and categorical columns are supported.
    """

    # pylint:disable=invalid-name
    class MultivariateAnalysisSubsection(IntEnum):
        """Enum of all implemented multivariate analysis subsections."""

        PCA = 0
        if UMAP_AVAILABLE:
            UMAP = 1
        ParallelCoordinates = 2
        ParallelCategories = 3

        def __str__(self):
            return self.name

    # By default use all subsections
    _DEFAULT_SUBSECTIONS_TO_SHOW = list(MultivariateAnalysisSubsection)

    def __init__(
        self,
        df: pd.DataFrame,
        subsections: Optional[List[MultivariateAnalysisSubsection]] = None,
        verbosity: Verbosity = Verbosity.LOW,
        columns: Optional[List[str]] = None,
        verbosity_pca: Optional[Verbosity] = None,
        verbosity_umap: Optional[Verbosity] = None,
        verbosity_parallel_coordinates: Optional[Verbosity] = None,
        verbosity_parallel_categories: Optional[Verbosity] = None,
        color_col: Optional[str] = None,
    ):
        verbosity_pca = verbosity_pca or verbosity
        verbosity_umap = verbosity_umap or verbosity
        verbosity_parallel_coordinates = (
            verbosity_parallel_coordinates
            if verbosity_parallel_coordinates is not None
            else verbosity
        )
        verbosity_parallel_categories = (
            verbosity_parallel_categories
            if verbosity_parallel_categories is not None
            else verbosity
        )

        subsec = MultivariateAnalysis.MultivariateAnalysisSubsection
        self.subsection_verbosities = {
            subsec.PCA: verbosity_pca,
            subsec.ParallelCoordinates: verbosity_parallel_coordinates,
            subsec.ParallelCategories: verbosity_parallel_categories,
        }
        if UMAP_AVAILABLE:
            self.subsection_verbosities[subsec.UMAP] = verbosity_umap

        if subsections is None:
            self.subsections_to_show = self._DEFAULT_SUBSECTIONS_TO_SHOW
        else:
            self.subsections_to_show = subsections

        enum_to_implementation = {
            subsec.PCA: PCA(df, verbosity_pca, columns, color_col=color_col),
            subsec.ParallelCoordinates: ParallelCoordinates(
                df, verbosity_parallel_coordinates, columns, color_col=color_col
            ),
            subsec.ParallelCategories: ParallelCategories(
                df, verbosity_parallel_categories, columns, color_col=color_col
            ),
        }
        if UMAP_AVAILABLE:
            enum_to_implementation[subsec.UMAP] = UMAP(
                df, verbosity_umap, columns, color_col=color_col
            )

        subsections_implementations = [
            enum_to_implementation[sub] for sub in self.subsections_to_show
        ]

        self.color_col = color_col
        super().__init__(subsections_implementations, verbosity, columns)

    @property
    def name(self) -> str:
        return "Multivariate analysis"

    @staticmethod
    def multivariate_analysis(
        df: pd.DataFrame,
        subsections: Optional[List[MultivariateAnalysisSubsection]] = None,
        columns: Optional[List[str]] = None,
        color_col: Optional[str] = None,
    ) -> None:
        """Generates multivariate analysis for df.

        Parameters
        ----------
        df : pd.DataFrame
            Data to be analyzed
        subsections : List[MultivariateAnalysisSubsection], optional
            Subsections to include in the analysis. All subsections are included by default.
        columns : List[str], optional
            Subset of columns of df to consider in multivariate analysis.
            All numeric columns are used by default.
        color_col : str, optional
            Name of the column according to which to color points in the sections.
            Both numberic and categorical columns are supported.
        """
        if columns is not None:
            df = df[columns]

        multivariate_analysis = MultivariateAnalysis(
            df=df,
            subsections=subsections,
            verbosity=Verbosity.LOW,
            columns=columns,
            color_col=color_col,
        )

        for sub in multivariate_analysis.subsections:
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
                "from edvart.report_sections.multivariate_analysis import MultivariateAnalysis\n"
                "multivariate_analysis = MultivariateAnalysis.multivariate_analysis"
            }
            for subsec in self.subsections:
                if subsec.verbosity > Verbosity.LOW:
                    imports.update(subsec.required_imports())

            return list(imports)
        return super().required_imports()

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
            code = "multivariate_analysis(df=df"
            subsections_to_show_with_low_verbosity = [
                sub
                for sub in self.subsections_to_show
                if self.subsection_verbosities[sub] == Verbosity.LOW
            ]
            if subsections_to_show_with_low_verbosity != self._DEFAULT_SUBSECTIONS_TO_SHOW:
                arg_subsections_names = [
                    f"MultivariateAnalysis.MultivariateAnalysisSubsection.{str(sub)}"
                    for sub in subsections_to_show_with_low_verbosity
                ]
                code += f", subsections={arg_subsections_names}".replace("'", "")
            if self.columns is not None:
                code += f", columns={self.columns}"
            if self.color_col is not None:
                code += f", color_col='{self.color_col}'"
            code += ")"
            cells.append(nbfv4.new_code_cell(code))
            for sub in self.subsections:
                if sub.verbosity > Verbosity.LOW:
                    sub.add_cells(cells=cells, df=df)
        else:
            super().add_cells(cells=cells, df=df)

    def show(self, df: pd.DataFrame) -> None:
        """Generates cell output of this section in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output.
        """
        display(Markdown(self.get_title(section_level=1)))
        super().show(df)


class PCA(Section):
    """Generates the Principal component analysis subsection.

    Parameters
    ----------
    df : pd.DataFrame
        Data on which to perform PCA.
    verbosity : Verbosity (default = Verbosity.LOW)
        Verbosity of the code generated in the exported notebook.
    columns : List[str], optional
        Columns on which to perform PCA. Only numeric columns can be used.
        All numeric columns of df are used by default.
    color_col : str, optional
        Name of column according to values of which to color points in the first vs second component
        plot. Can be both numeric and categorical. By default, all points have the same color.
    standardize : bool (default = True)
        Whether to standardize the data to zero mean and unit variance before applying PCA.
    interactive : bool (default = True)
        Whether to plot the first vs second principal component as an interactive plot.
        The interactive plot also shows labels for each point on hover.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        verbosity: Verbosity = Verbosity.LOW,
        columns: Optional[List[str]] = None,
        color_col: Optional[str] = None,
        standardize: bool = True,
        interactive: bool = True,
    ):
        # By default use only numeric columns
        if columns is None:
            columns = [col for col in df.columns if is_numeric(df[col])]
            # If all columns are numeric we don't want to list them all in the generated call
            # Setting columns to None will result in the columns argumented not being included
            # instead of showing a potentially long list of all columns
            # in the generated call, therefore the default (all columns) will be used
            if len(columns) == len(df.columns):
                columns = None
        else:
            for col in columns:
                if not is_numeric(df[col]):
                    raise ValueError(
                        f"Cannot use non-numeric column {col} of dtype {df[col].dtype} in PCA"
                    )

        self.color_col = color_col
        self.interactive = interactive
        self.standardize = standardize
        super().__init__(verbosity, columns)

    @property
    def name(self) -> str:
        return "Principal Component Analysis"

    @staticmethod
    def pca_first_vs_second(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        color_col: Optional[str] = None,
        interactive: bool = True,
        standardize: bool = True,
        figsize: Tuple[float, float] = (12, 12),
        opacity: float = 0.8,
    ) -> None:
        """Plot a 2D scatter of first vs second PCA components.

        Parameters
        ----------
        df : pd.DataFrame
            Data to perform PCA on.
        columns : List[str], optional
            Which columns to perform PCA on. All columns will be used by default.
        color_col : str, optional
            Name of column according to values of which to color points in the plot.
            Can be both numeric and categorical. By default, all points have the same color.
        interactive : bool (default = True)
            Whether to show an interactive plot.
        standardize : bool (default = True)
            Whether to standardize the data to zero mean and unit variance before applying PCA.
        figsize : Tuple[float, float] (default = (12, 12))
            Size of the plot.
        opacity : float (default = 0.8)
            Opacity of the points in the plot. Higher means more opaque (less transparent).
        """
        df = df.dropna(subset=columns)

        pca = sklearn.decomposition.PCA(n_components=2)

        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(df[columns])
            pca_components = pca.fit_transform(data_scaled)
        else:
            pca_components = pca.fit_transform(df[columns])

        scatter_plot_2d(
            df=df,
            x=pca_components[:, 0],
            y=pca_components[:, 1],
            color_col=color_col,
            interactive=interactive,
            figsize=figsize,
            opacity=opacity,
            xlabel="First principal component",
            ylabel="Second principal component",
            show_xticks=True,
            show_yticks=True,
            show_zerolines=True,
            equal_scale_axes=True,
        )

        print(f"Explained variance ratio: {pca.explained_variance_ratio_[:2].sum() * 100 :.2f}%")

    @staticmethod
    def pca_explained_variance(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        standardize: bool = True,
        show_grid: bool = True,
        figsize: Tuple[float, float] = (10, 7),
    ) -> None:
        """Plot a plot of variance explained by each principal component.

        Parameters
        ----------
        df : pd.DataFrame
            Data on which to perform PCA.
        columns : List[str], optional
            Which columns to perform PCA on. All columns will be used by default.
        standardize : bool (default = True)
            Whether to standardize the data zero mean and unit variance before applying PCA.
        show_grid : bool (default = True)
            Whether to show a grid in the plot.
        figsize : Tuple[float, float] (default = (10, 7))
            Size of the plot.
        """
        df = df.dropna(subset=columns)

        pca = sklearn.decomposition.PCA()

        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(df[columns])
            pca.fit(data_scaled)
        else:
            pca.fit(df[columns])

        fig = plt.figure(figsize=figsize)
        plt.plot(pca.explained_variance_ratio_, figure=fig)
        plt.plot(np.cumsum(pca.explained_variance_ratio_), figure=fig)

        plt.legend(["Individual component", "Cumulative"])
        plt.xlabel("Principal component #")
        plt.ylabel("Explained variance ratio")
        plt.xticks(
            ticks=range(len(pca.explained_variance_ratio_)),
            labels=range(1, (len(pca.explained_variance_ratio_) + 1)),
        )
        if show_grid:
            plt.grid()
        plt.show()

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
                    from edvart.report_sections.multivariate_analysis import PCA
                    pca_first_vs_second = PCA.pca_first_vs_second
                    pca_explained_variance = PCA.pca_explained_variance
                    """
                )
            ]
        return [
            "from edvart.plots import scatter_plot_2d",
            "import matplotlib.pyplot as plt",
            "import numpy as np",
            "import sklearn.decomposition",
            "from sklearn.preprocessing import StandardScaler",
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

        first_vs_second_header = nbfv4.new_markdown_cell("### First vs second principal component")
        cells.append(first_vs_second_header)
        explained_variance_header = nbfv4.new_markdown_cell("### Explained variance ratio")

        first_vs_second_call = "pca_first_vs_second(df=df"
        if not self.interactive:
            first_vs_second_call += ", interactive=False"
        explained_variance_call = "pca_explained_variance(df=df"
        if self.columns is not None:
            first_vs_second_call += f", columns={self.columns}"
            explained_variance_call += f", columns={self.columns}"
        if not self.standardize:
            explained_variance_call += ", standardize=False"
            first_vs_second_call += ", standardize=False"
        if not self.interactive:
            first_vs_second_call += ", interactive=False"
        if self.color_col is not None:
            first_vs_second_call += f", color_col='{self.color_col}'"
        first_vs_second_call += ")"
        explained_variance_call += ")"

        if self.verbosity <= Verbosity.MEDIUM:
            cells.append(nbfv4.new_code_cell(first_vs_second_call))
            cells.append(explained_variance_header)
            cells.append(nbfv4.new_code_cell(explained_variance_call))
        else:
            first_vs_second_code = get_code(PCA.pca_first_vs_second) + "\n\n" + first_vs_second_call
            cells.append(nbfv4.new_code_cell(first_vs_second_code))

            cells.append(explained_variance_header)
            explained_variance_code = (
                get_code(PCA.pca_explained_variance) + "\n\n" + explained_variance_call
            )
            cells.append(nbfv4.new_code_cell(explained_variance_code))

    def show(self, df: pd.DataFrame) -> None:
        """Generates the PCA section in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output
        """
        display(Markdown(self.get_title(section_level=2)))
        display(Markdown("### First vs second principal component"))
        PCA.pca_first_vs_second(
            df=df,
            columns=self.columns,
            color_col=self.color_col,
            interactive=self.interactive,
            standardize=self.standardize,
        )
        display(Markdown("### Explained variance ratio"))
        PCA.pca_explained_variance(df=df, columns=self.columns, standardize=self.standardize)


class ParallelCoordinates(Section):
    """Generates the Parallel coordinates subsection.

    Parameters
    ----------
    df : pd.DataFrame
        Data for which to generate the parallel coordinates plot.
    verbosity : Verbosity (default = Verbosity.LOW)
        Verbosity of the code generated in the exported notebook.
    columns : List[str], optional
        Columns for which to generate parallel coordinates. All columns which are either numeric or
        categorical with at most `nunique_max` unique values are used by default.
    nunique_max : int (default = 20)
        Maximum number of unqiue values for non-numeric columns to be plotted.
        Ignored if set to a negative number.
    color_col : str, optional
        Name of column determining color of the coordinate lines.
        Both numeric and categorical columns are supported.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        verbosity: Verbosity = Verbosity.LOW,
        columns: Optional[List[str]] = None,
        nunique_max: int = 20,
        color_col: Optional[str] = None,
    ):
        self.color_col = color_col
        if columns is None:
            columns = [
                col
                for col in df.columns
                if is_categorical(df[col], unique_value_count_threshold=nunique_max)
                or is_boolean(df[col])
                or is_numeric(df[col])
            ]
            # If all columns are numeric we don't want to list them all in the generated call
            # Setting columns to None will result in the columns argument not being included
            # instead of showing a potentially long list of all columns
            # in the generated call, therefore the default (all columns) will be used
            if len(columns) == len(df.columns):
                columns = None

        super().__init__(verbosity, columns)

    @property
    def name(self) -> str:
        return "Parallel coordinates"

    @staticmethod
    def parallel_coordinates(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        hide_columns: Optional[List[str]] = None,
        drop_na: bool = False,
        color_col: Optional[str] = None,
        show_colorscale: bool = True,
    ) -> None:
        """Generate the parallel coordinates interactive plot.

        Parameters
        ----------
        df : pd.DataFrame
            Data for which to generate the parallel coordinates plot.
        columns : List[str], optional
            Columns for which to generate the plot. All columns are used by default.
        hide_columns : List[str], optional
            Columns to exclude from plotting.
        drop_na : bool (default = False)
            Whether to drop NaNs in data.
        color_col : str, optional
            Which column to use for coloring of lines. Can be both numeric and categorical.
        show_colorscale : bool (default = True)
            Whether to show a color scale on the right side of the plot.
        """
        if columns is None:
            columns = df.columns
        if hide_columns is not None:
            columns = [col for col in columns if col not in hide_columns]
        if drop_na:
            df = df.dropna()
        if color_col is not None:
            is_categorical_color = not is_numeric(df[color_col]) or is_boolean(df[color_col])

            if is_categorical_color:
                categories = df[color_col].unique()
                colorscale = list(discrete_colorscale(len(categories)))
                # encode categories into numbers
                color_series = pd.Series(pd.Categorical(df[color_col]).codes)
            else:
                color_series = df[color_col]
                colorscale = "Bluered_r"

            line = {
                "color": color_series,
                "colorscale": colorscale,
                "showscale": show_colorscale,
                "colorbar": {"title": color_col, "lenmode": "pixels", "len": 300},
            }

            if is_categorical_color:
                line["colorbar"].update(
                    {
                        "tickvals": color_series.unique(),
                        "ticktext": categories,
                        "lenmode": "pixels",
                        "len": min(40 * len(categories), 300),
                    }
                )
        else:
            line = None

        numeric_columns = [col for col in columns if is_numeric(df[col])]
        categorical_columns = [col for col in columns if not is_numeric(df[col])]

        # Add numeric columns to dimensions
        dimensions = [{"label": col_name, "values": df[col_name]} for col_name in numeric_columns]
        # Add categorical columns to dimensions
        for col_name in categorical_columns:
            categories = df[col_name].unique()
            values = pd.Series(pd.Categorical(df[col_name]).codes)
            dimensions.append(
                {
                    "label": col_name,
                    "values": values,
                    "tickvals": values.unique(),
                    "ticktext": categories,
                }
            )

        fig = go.Figure(go.Parcoords(line=line, dimensions=dimensions))
        fig.show()

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
                    from edvart.report_sections.multivariate_analysis import ParallelCoordinates
                    parallel_coordinates = ParallelCoordinates.parallel_coordinates
                    """
                )
            ]
        return [
            "from edvart.utils import discrete_colorscale",
            "import plotly",
            "import plotly.graph_objects as go",
            "plotly.offline.init_notebook_mode()",
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

        default_call = "parallel_coordinates(df=df"
        if self.columns is not None:
            default_call += f", columns={self.columns}"
        if self.color_col is not None:
            default_call += f", color_col='{self.color_col}'"
        default_call += ")"

        if self.verbosity <= Verbosity.MEDIUM:
            code = default_call
        else:
            code = (
                get_code(discrete_colorscale)
                + "\n\n"
                + get_code(ParallelCoordinates.parallel_coordinates)
                + "\n\n"
                + default_call
            )

        cells.append(nbfv4.new_code_cell(code))

    def show(self, df: pd.DataFrame) -> None:
        """Generates the Parallel coordinates section in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output
        """
        display(Markdown(self.get_title(section_level=2)))
        ParallelCoordinates.parallel_coordinates(
            df=df, columns=self.columns, color_col=self.color_col
        )


class ParallelCategories(Section):
    """
    Generates the Parallel categories subsection.

    Parameters
    ----------
    df : pd.DataFrame
        Data for which to generate the parallel coordinates plot.
    verbosity : Verbosity (default = Verbosity.LOW)
        Verbosity of the code generated in the exported notebook.
    columns : List[str], optional
        Columns for which to generate parallel coordinates.
        All categorical columns with at most `nunique_max` unique values are used by default.
    nunique_max : int (default = 20)
        Maximum number of unqiue values for non-numeric columns to be plotted.
        Ignored if set to a negative number.
    color_col : str, optional
        Name of column determining colors within categories.
        Both numeric and categorical columns are supported.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        verbosity: Verbosity = Verbosity.LOW,
        columns: Optional[List[str]] = None,
        nunique_max: int = 20,
        color_col: Optional[str] = None,
    ):
        self.color_col = color_col
        if columns is None:
            columns = [
                col
                for col in df.columns
                if is_categorical(df[col], unique_value_count_threshold=nunique_max)
                or is_boolean(df[col])
            ]

            # If all columns are numeric we don't want to list them all in the generated call
            # Setting columns to None will result in the columns argumented not being included
            # instead of showing a potentially long list of all columns
            # in the generated call, therefore the default (all columns) will be used
            if len(columns) == len(df.columns):
                columns = None

        super().__init__(verbosity, columns)

    @property
    def name(self) -> str:
        return "Parallel categories"

    @staticmethod
    def parallel_categories(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        hide_columns: Optional[List[str]] = None,
        drop_na: bool = False,
        color_col: Optional[str] = None,
    ) -> None:
        """Generate the parallel coordinates interactive plot.

        Parameters
        ----------
        df : pd.DataFrame
            Data for which to generate the parallel coordinates plot.
        columns : List[str], optional
            Columns for which to generate the plot.
        hide_columns : List[str], optional
            Columns to exclude from plotting.
        drop_na : bool (default = False)
            Whether to drop NaNs in data.
        color_col : str, optional
            Which column to use for coloring of lines. Can be both numeric and categorical.
        """
        if columns is None:
            columns = df.columns
        if hide_columns is not None:
            columns = [col for col in columns if col not in hide_columns]
        if drop_na:
            df = df.dropna()
        if color_col is not None:
            categorical_color = not is_numeric(df[color_col]) or is_boolean(df[color_col])
            if categorical_color:
                categories = df[color_col].unique()
                colorscale = list(discrete_colorscale(len(categories)))
                # encode categories into numbers
                color_series = pd.Series(pd.Categorical(df[color_col]).codes)
            else:
                color_series = df[color_col]
                colorscale = "Bluered_r"

            line = {
                "color": color_series,
                "colorscale": colorscale,
                "colorbar": {"title": color_col},
            }

            if categorical_color:
                line["colorbar"].update(
                    {
                        "tickvals": color_series.unique(),
                        "ticktext": categories,
                        "lenmode": "pixels",
                        "len": min(40 * len(categories), 300),
                    }
                )
        else:
            line = None

        dimensions = [
            go.parcats.Dimension(values=df[col_name], label=col_name) for col_name in columns
        ]

        fig = go.Figure(go.Parcats(dimensions=dimensions, line=line))
        fig.show()

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
                    from edvart.report_sections.multivariate_analysis import ParallelCategories
                    parallel_categories = ParallelCategories.parallel_categories
                    """
                )
            ]
        return [
            "from edvart.utils import discrete_colorscale",
            "import plotly",
            "import plotly.graph_objects as go",
            "plotly.offline.init_notebook_mode()",
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

        default_call = "parallel_categories(df=df"
        if self.columns is not None:
            default_call += f", columns={self.columns}"
        if self.color_col is not None:
            default_call += f", color_col='{self.color_col}'"
        default_call += ")"

        if self.verbosity <= Verbosity.MEDIUM:
            code = default_call
        else:
            code = (
                get_code(discrete_colorscale)
                + "\n\n"
                + get_code(ParallelCategories.parallel_categories)
                + "\n\n"
                + default_call
            )

        cells.append(nbfv4.new_code_cell(code))

    def show(self, df: pd.DataFrame) -> None:
        """Generates the Parallel coordinates section in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output
        """
        display(Markdown(self.get_title(section_level=2)))
        ParallelCategories.parallel_categories(
            df=df, columns=self.columns, color_col=self.color_col
        )
