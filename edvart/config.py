from dataclasses import dataclass
from typing import List, Optional, Tuple

from edvart import Verbosity
from edvart.report_sections.bivariate_analysis import BivariateAnalysis


@dataclass
class BivariateAnalysisConfig:
    use_columns: Optional[List[str]] = None,
    omit_columns: Optional[List[str]] = None,
    columns_x: Optional[List[str]] = None,
    columns_y: Optional[List[str]] = None,
    columns_pairs: Optional[List[Tuple[str, str]]] = None,
    subsections: Optional[List[BivariateAnalysis.BivariateAnalysisSubsection]] = None,
    verbosity: Optional[Verbosity] = None,
    verbosity_correlations: Optional[Verbosity] = None,
    verbosity_pairplot: Optional[Verbosity] = None,
    verbosity_contingency_table: Optional[Verbosity] = None,
    color_col: Optional[str] = None,
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
        verbosity : Verbosity, optional
            The verbosity of the code generated in the exported notebook.
        verbosity_correlations : Verbosity, optional
            Correlation plots subsection code verbosity.
        verbosity_pairplot : Verbosity, optional
            Pairplot subsection code verbosity.
        verbosity_contingency_table : Verbosity, optional
            Contingency table code verbosity.
        color_col : str, optional
            Name of column according to use for coloring of the multivariate analysis subsections.
            Coloring is currently supported in pairplot.
    """
