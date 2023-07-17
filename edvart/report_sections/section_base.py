import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd


class Section(ABC):
    """Base class for report sections and subsections.

    Parameters
    -----------
    verbosity : int
        The verbosity of the code generated in the exported notebook,
        must be one of [0, 1, 2].
    columns : List[str], optional
        List of columns that are considered in the analysis of the section,
        all columns are used by default
    """

    def __init__(self, verbosity: int = 0, columns: Optional[List[str]] = None):
        if verbosity not in [0, 1, 2]:
            raise ValueError(f"Verbosity must be one of [0, 1, 2], not {verbosity}")
        self.verbosity = verbosity
        self.columns = columns
        self._section_id: str = str(uuid.uuid4())

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the section.

        Returns
        -------
        str
            Name of the section.
        """

    @property
    def uid(self) -> str:
        """Identifier of the section used for generating table of contents.

        Should be unique across sections.

        Returns
        -------
        str
            Unique identifier of the section
        """
        return self._section_id

    def get_title(self, section_level: int) -> str:
        """Gets the title of the section in markdown format.

        Includes a hyperlink id tag that is used by the table of contents.

        Parameters
        ----------
        section_level: int
            The level of the section. Adds # according to it. Highest level sections should have it
            set to 1.

        Returns
        -------
        str
            Title of the section in markdown format.
        """
        title = f"{'#' * section_level} {self.name}<a id='{self.uid}'>"
        if section_level == 1:
            title += "\n---"
        return title

    @abstractmethod
    def required_imports(self) -> List[str]:
        """Returns a list of imports to be put at the top of a generated notebook.

        Returns
        -------
        List[str]
            List of import strings to be added at the top of the generated notebook,
            e.g. ['import pandas as pd', 'import numpy as np']
        """

    @abstractmethod
    def add_cells(self, cells: List[Dict[str, Any]]) -> None:
        """Adds cells to the list of cells.

        Cells can be either code cells or markdown cells.

        Parameters
        ----------
        cells : List[Dict[str, Any]]
            List of generated notebook cells which are represented as dictionaries.
            The dictionaries can be generated with nbformat.v4.new_code_cell() and/or
            nbformat.v4.new_markdown_cell().
        """

    @abstractmethod
    def show(self, df: pd.DataFrame) -> None:
        """Generates cell output in the calling notebook using IPython.display.display().

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output
        """


class ReportSection(Section):
    """Base class for top level report sections.

    Contains subsections which are also of subtype Section and implement the report generation.

    Parameters
    ----------
    subsections : List[Section]
        List of subsections that should be contained in this top level section
    verbosity : int
        The verbosity of the code generated in the exported notebook,
        must be one of [0, 1, 2].
    columns : List[str], optional
        List of columns that are considered in the analysis of the section,
        all columns are used by default
    """

    def __init__(
        self,
        subsections: List[Section],
        verbosity: int = 0,
        columns: Optional[List[str]] = None,
    ):
        super().__init__(verbosity, columns)
        self.subsections = subsections

    def required_imports(self) -> List[str]:
        """Returns a list of imports to be put at the top of a generated notebook.

        Returns
        -------
        List[str]
            List of import strings to be added at the top of the generated notebook,
            e.g. ['import pandas as pd', 'import numpy as np'].
        """
        imports_set = set()
        for subsec in self.subsections:
            imports_set.update(subsec.required_imports())

        return list(imports_set)

    def add_cells(self, cells: List[Dict[str, Any]]) -> None:
        """Adds cells to the list of cells.

        Cells can be either code cells or markdown cells.

        Parameters
        ----------
        cells : List[Dict[str, Any]]
            List of generated notebook cells which are represented as dictionaries.
        """
        for subsec in self.subsections:
            subsec.add_cells(cells)

    def show(self, df: pd.DataFrame) -> None:
        """Generates cell output of this section in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output.
        """
        for subsec in self.subsections:
            subsec.show(df)
