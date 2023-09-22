import warnings
from typing import Any, Dict, List, Optional, Tuple

import nbformat.v4 as nbfv4
import pandas as pd
from IPython.display import Markdown, display

from edvart.plots import scatter_plot_2d
from edvart.report_sections.code_string_formatting import code_dedent, get_code
from edvart.report_sections.section_base import Section, Verbosity
from edvart.utils import select_numeric_columns

try:
    with warnings.catch_warnings():
        # Ignore warning from umap 0.5.0 that ParametricUMAP is unavailable
        # due to missing TensorFlow
        warnings.simplefilter("ignore", UserWarning)
        import umap
except ImportError as exc:
    raise ImportError("To use UMAP, install edvart with optional dependency 'umap'.") from exc


class UMAP(Section):
    """Plot a 2-dimensional UMAP embedding scatter plot.

    Parameters
    ----------
    verbosity : Verbosity (default = Verbosity.LOW)
        Verbosity of the code generated in the exported notebook.
    columns : List[str], optional
        Columns to use in computing in the UMAP embedding. Only numeric columns can be used.
        All numeric columns are used by default.
    color_col : str, optional
        Name of column to color points on the plot by. Can be both numeric and categorical.
        By default, all points have the same color.
    interactive : bool (default = True)
        Whether to plot an interactive plot. The interactive plot also shows labels for each
        point on hover.
    n_neighbors : int (default = 15)
        UMAP embedding parameter controlling the balance between focusing on the
        local structure vs global structure. A low value means more focus on local structure.
    min_dist : int (default = 0.1)
        UMAP embedding parameter controlling how tightly points in the embedding are clumped
        together. A low value results in tighter clumping, which can show clusters or other
        similar structures, while a higher value encourages preservation of topological
        structure present in the input data.
    metric : str (default = "euclidean")
        UMAP embedding parameter controlling how distance is computed in the ambient space of
        the input data. Many different metrics are available, see UMAP documentation for a
        complete list.
    """

    def __init__(
        self,
        verbosity: Verbosity = Verbosity.LOW,
        columns: Optional[List[str]] = None,
        color_col: Optional[str] = None,
        interactive: bool = True,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
    ):
        self.color_col = color_col
        self.interactive = interactive
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        super().__init__(verbosity, columns)

    @property
    def name(self) -> str:
        return "UMAP"

    def required_imports(self) -> List[str]:
        """Returns a list of imports to be put at the top of a generated notebook.

        Returns
        -------
        List[str]
            List of import strings to be added at the top of the generated notebook,
            e.g. ['import pandas as pd', 'import numpy as np'].
        """
        if self.verbosity <= Verbosity.MEDIUM:
            return ["from edvart.report_sections.umap import plot_umap"]
        return [
            "from edvart.plots import scatter_plot_2d",
            "from edvart import utils",
            "import umap",
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

        default_call = code_dedent(
            f"""
            plot_umap(
                df=df,
                n_neighbors={self.n_neighbors},
                min_dist={self.min_dist},
                metric='{self.metric}',
            """.rstrip()
        )
        if self.columns is not None:
            default_call += f"\n    columns={self.columns},"
        if self.color_col is not None:
            default_call += f"\n    color_col='{self.color_col}',"
        if not self.interactive:
            default_call += "\n    interactive=False,"
        default_call += "\n)"

        if self.verbosity <= Verbosity.MEDIUM:
            code = default_call
        else:
            code = (
                get_code(select_numeric_columns)
                + "\n\n"
                + get_code(plot_umap)
                + "\n\n"
                + default_call
            )

        cells.append(nbfv4.new_code_cell(code))

    def show(self, df: pd.DataFrame) -> None:
        """Generates the UMAP plot section in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output
        """
        display(Markdown(self.get_title(section_level=2)))
        plot_umap(
            df=df,
            columns=self.columns,
            color_col=self.color_col,
            interactive=self.interactive,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
        )


def plot_umap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    color_col: Optional[str] = None,
    interactive: bool = True,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
    figsize: Tuple[float, float] = (12, 12),
    opacity: float = 0.8,
    show_message: bool = True,
) -> None:
    """Plot a 2-dimensional UMAP embedding scatter plot.

    Parameters
    ----------
    df : pd.DataFrame
        Data to analyze.
    columns : List[str], optional
        Columns to use in computing in the UMAP embedding. Only numeric columns can be used.
        All numeric columns are used by default.
    color_col : str, optional
        Name of column to color points on the plot by. Can be both numeric and categorical.
        By default, all points have the same color.
    interactive : bool (default = True)
        Whether to plot an interactive plot. The interactive plot also shows labels for each
        point on hover.
    n_neighbors : int (default = 15)
        UMAP embedding parameter controlling the balance between focusing on the
        local structure vs global structure. A low value means more focus on local structure.
    min_dist : int (default = 0.1)
        UMAP embedding parameter controlling how tightly points in the embedding are clumped
        together. A low value results in tighter clumping, which can show clusters or other
        similar structures, while a higher value encourages preservation of topological
        structure present in the input data.
    metric : str (default = "euclidean")
        UMAP embedding parameter controlling how distance is computed in the ambient space of
        the input data. Many different metrics are available, see UMAP documentation for a
        complete list.
    random_state : int (default = 42)
        Random state for reproducibility of results, since UMAP is stochastic. If None, a random
        seed is used.
    figsize : Tuple[float, float] (default = (12, 12))
        Size of the resulting plot in inches.
    opacity : float (default = 0.8)
        Opacity of the points drawn in the scatter plot.
    show_message : bool (default = True)
        Whether to show a message informing the user to tune the embedding parameters.
    """
    columns = select_numeric_columns(df, columns)
    df = df.dropna(subset=columns)
    embedder = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state
    )
    embedded = embedder.fit_transform(df[columns])

    scatter_plot_2d(
        df=df,
        x=embedded[:, 0],
        y=embedded[:, 1],
        color_col=color_col,
        interactive=interactive,
        figsize=figsize,
        opacity=opacity,
    )

    if show_message:
        print("UMAP requires proper setting of hyperparameters. ")
        print(
            "If results are unsatisfactory, consider trying different values of parameters"
            " `n_neighbors`, `min_dist` and `metric`."
        )
