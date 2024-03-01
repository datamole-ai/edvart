import warnings
from contextlib import redirect_stdout

import pandas as pd
import pytest

from edvart.pandas_formatting import add_html_heading, dict_to_html, format_number, subcells_html
from edvart.report_sections import univariate_analysis
from edvart.report_sections.code_string_formatting import code_dedent, get_code
from edvart.report_sections.section_base import Verbosity

from .execution_utils import check_section_executes
from .pyarrow_utils import pyarrow_params


@pytest.fixture(params=pyarrow_params)
def test_df(request) -> pd.DataFrame:
    test_df = pd.DataFrame(data=[[1.9, "a"], [2.1, "b"], [3.3, "c"]], columns=["A", "B"])
    if request.param:
        test_df = test_df.convert_dtypes(dtype_backend="pyarrow")

    return test_df


def test_invalid_verbosity():
    with pytest.raises(ValueError):
        univariate_analysis.UnivariateAnalysis(verbosity=-1)
    with pytest.raises(ValueError):
        univariate_analysis.UnivariateAnalysis(verbosity=4)
    with pytest.raises(ValueError):
        univariate_analysis.UnivariateAnalysis(verbosity=100)
    with pytest.raises(ValueError):
        univariate_analysis.UnivariateAnalysis(verbosity="1")


def test_code_export_verbosity_low(test_df: pd.DataFrame):
    # Construct univariate analysis section
    univariate_section = univariate_analysis.UnivariateAnalysis(verbosity=Verbosity.LOW)
    # Export code
    exported_cells = []
    univariate_section.add_cells(exported_cells, df=test_df)
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = ["show_univariate_analysis(df=df)"]
    # Test code equivalence
    assert exported_code[0] == expected_code[0], "Exported code mismatch"

    check_section_executes(univariate_section, test_df)


def test_code_export_verbosity_medium(test_df: pd.DataFrame):
    # Construct univariate analysis section
    univariate_section = univariate_analysis.UnivariateAnalysis(verbosity=Verbosity.MEDIUM)
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

    check_section_executes(univariate_section, test_df)


def test_code_export_verbosity_high(test_df: pd.DataFrame):
    # Construct univariate analysis section
    univariate_section = univariate_analysis.UnivariateAnalysis(verbosity=Verbosity.HIGH)
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
                get_code(univariate_analysis.default_descriptive_statistics),
                get_code(univariate_analysis.default_quantile_statistics),
            )
        ),
        "\n\n".join(
            (
                get_code(univariate_analysis.top_most_frequent),
                get_code(univariate_analysis.bar_plot),
                get_code(univariate_analysis.numeric_statistics),
                get_code(univariate_analysis.histogram),
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

    check_section_executes(univariate_section, test_df)


def test_show(test_df: pd.DataFrame):
    univariate_section = univariate_analysis.UnivariateAnalysis()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with redirect_stdout(None):
            univariate_section.show(test_df)
