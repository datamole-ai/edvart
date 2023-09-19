import warnings
from contextlib import redirect_stdout

import pandas as pd
import pytest

from edvart.pandas_formatting import add_html_heading, dict_to_html, format_number, subcells_html
from edvart.report_sections import univariate_analysis
from edvart.report_sections.code_string_formatting import code_dedent, get_code
from edvart.report_sections.section_base import Verbosity


def test_invalid_verbosity():
    test_df = pd.DataFrame(data=[[1, 2], [2, 3], [3, 4]], columns=["A", "B"])
    with pytest.raises(ValueError):
        univariate_analysis.UnivariateAnalysis(df=test_df, verbosity=-1)
    with pytest.raises(ValueError):
        univariate_analysis.UnivariateAnalysis(df=test_df, verbosity=4)
    with pytest.raises(ValueError):
        univariate_analysis.UnivariateAnalysis(df=test_df, verbosity=100)
    with pytest.raises(ValueError):
        univariate_analysis.UnivariateAnalysis(df=test_df, verbosity="1")


def test_code_export_verbosity_low():
    test_df = pd.DataFrame(data=[[1.9, "a"], [2.1, "b"], [3.3, "c"]], columns=["A", "B"])
    # Construct univariate analysis section
    univariate_section = univariate_analysis.UnivariateAnalysis(df=test_df, verbosity=Verbosity.LOW)
    # Export code
    exported_cells = []
    univariate_section.add_cells(exported_cells, df=test_df)
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = ["univariate_analysis(df=df)"]
    # Test code equivalence
    assert exported_code[0] == expected_code[0], "Exported code mismatch"


def test_code_export_verbosity_medium():
    test_df = pd.DataFrame(data=[[1.9, "a"], [2.1, "b"], [3.3, "c"]], columns=["A", "B"])
    # Construct univariate analysis section
    univariate_section = univariate_analysis.UnivariateAnalysis(
        df=test_df, verbosity=Verbosity.MEDIUM
    )
    # Export code
    exported_cells = []
    univariate_section.add_cells(exported_cells, df=test_df)
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = [
        "numeric_statistics(df['A'])\n" "histogram(df['A'])",
        "top_most_frequent(df['B'])\n" "bar_plot(df['B'])",
    ]
    # Test code equivalence
    for i in range(len(exported_code)):
        assert exported_code[i] == expected_code[i], "Exported code mismatch"


def test_code_export_verbosity_high():
    test_df = pd.DataFrame(data=[[1.9, "a"], [2.1, "b"], [3.3, "c"]], columns=["A", "B"])
    # Construct univariate analysis section
    univariate_section = univariate_analysis.UnivariateAnalysis(
        df=test_df, verbosity=Verbosity.HIGH
    )
    # Export code
    exported_cells = []
    univariate_section.add_cells(exported_cells, df=test_df)
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = [
        "\n\n".join(
            (
                "# Default statistics dictionaries",
                get_code(univariate_analysis.UnivariateAnalysis.default_descriptive_statistics),
                get_code(univariate_analysis.UnivariateAnalysis.default_quantile_statistics),
            )
        ),
        "\n\n".join(
            (
                get_code(univariate_analysis.UnivariateAnalysis.top_most_frequent),
                get_code(univariate_analysis.UnivariateAnalysis.bar_plot),
                get_code(univariate_analysis.UnivariateAnalysis.numeric_statistics).replace(
                    "UnivariateAnalysis.", ""
                ),
                get_code(univariate_analysis.UnivariateAnalysis.histogram),
            )
        ),
        "\n\n".join(
            (
                get_code(format_number),
                get_code(dict_to_html),
                get_code(add_html_heading),
                get_code(subcells_html),
            )
        ),
        code_dedent(
            """
            numeric_statistics(
                df['A'],
                descriptive_stats=default_descriptive_statistics(),
                quantile_stats=default_quantile_statistics()
            )
            histogram(df['A'])"""
        ),
        "top_most_frequent(df['B'])\n" "bar_plot(df['B'])",
    ]
    # Test code equivalence
    for i in range(len(exported_code)):
        assert exported_code[i] == expected_code[i], "Exported code mismatch"


def test_show():
    test_df = pd.DataFrame(data=[[1.9, "a"], [2.1, "b"], [3.3, "c"]], columns=["A", "B"])
    univariate_section = univariate_analysis.UnivariateAnalysis(df=test_df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with redirect_stdout(None):
            univariate_section.show(test_df)
