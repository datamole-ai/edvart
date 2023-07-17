"""Table of contents analysis package."""
# Standard imports
from typing import Any, Dict, List

# Third-party library imports
import nbformat.v4 as nbfv4
from IPython.display import Markdown, display

# Internal library imports
from edvart.report_sections.section_base import ReportSection, Section


class TableOfContents(Section):
    """Generates the Table of Contents section of the report.

    Parameters
    ----------
    include_subsections: bool
        A boolean controlling whether the subsections should be included in the table of contents.
        However, they won't be included in an exported notebook created by report's
        export_notebook function.
    """

    def __init__(self, include_subsections: bool):
        self._include_subsections = include_subsections
        super().__init__(verbosity=0, columns=None)

    def required_imports(self) -> List[str]:
        return []

    @property
    def name(self) -> str:
        return "Table of Contents"

    @staticmethod
    def _get_section_link(section: Section, section_level: int) -> str:
        """Gets the section link in the markdown format.

        Parameters
        ----------
        section: Section
            Section for which the link should be generated.

        section_level: int
            The level of the section. Highest level sections should have it set to 1.

        Returns
        -------
        str
            The section link in the markdown format.
        """
        beginning_whitespace = "\t" * (section_level - 1)
        return f"{beginning_whitespace} * [{section.name}](#{section.uid})\n"

    def _add_section_lines(
        self, section: Section, section_level: int, lines: List[str], include_subsections: bool
    ) -> None:
        """Generates table of contents' section's cell output in the calling notebook.

        Parameters
        ----------
        section: Section
            Section for which table of contents' lines are added to the provided list.
        section_level: int
            The level of the section. Highest level sections should have it set to 1.
        lines: List[str]
            Lines that should be added to the table of content.
        include_subsections: bool
            A boolean controlling whether the subsections' lines should be added as well.
        """
        lines.append(TableOfContents._get_section_link(section, section_level))

        if isinstance(section, ReportSection) and include_subsections:
            for subsection in section.subsections:
                self._add_section_lines(subsection, section_level + 1, lines, True)

    # pylint: disable=arguments-differ
    def add_cells(self, sections: List[Section], cells: List[Dict[str, Any]]) -> None:
        """Adds table of contents cells to the list of cells. The subsections won't be included.

        Parameters
        ----------
        sections: List[Section]
            List of sections that should be included in the table of contents.
        cells : List[Dict[str, Any]]
            List of generated notebook cells which are represented as dictionaries.
        """
        section_header = nbfv4.new_markdown_cell(self.get_title(section_level=1))
        cells.append(section_header)

        lines: List[str] = []
        # Add links to all main sections (not including subsections) besides the first (table of
        # content) section
        for section in sections:
            if not isinstance(section, TableOfContents):
                lines.append(TableOfContents._get_section_link(section, 1))
        cells.append(nbfv4.new_markdown_cell("\n".join(lines)))

    # pylint: disable=arguments-renamed
    def show(self, sections: List[Section]) -> None:
        """Generates table of contents' cell output in the calling notebook.

        Parameters
        ----------
        sections: List[Section]
            List of sections that should be included in the table of contents.
        """
        display(Markdown(self.get_title(section_level=1)))

        lines = []
        # Add links to all sections including their subsections besides the first (table of content)
        # section
        for section in sections:
            if not isinstance(section, TableOfContents):
                self._add_section_lines(section, 1, lines, self._include_subsections)
        display(Markdown("\n".join(lines)))
