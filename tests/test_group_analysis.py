import warnings
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import plotly.io
import pytest

from edvart.report_sections.code_string_formatting import code_dedent, get_code, total_dedent
from edvart.report_sections.group_analysis import (
    GroupAnalysis,
    default_group_descriptive_stats,
    default_group_quantile_stats,
    group_barplot,
    group_missing_values,
    overlaid_histograms,
    show_group_analysis,
    within_group_descriptive_stats,
    within_group_quantile_stats,
    within_group_stats,
)
from edvart.report_sections.section_base import Verbosity

# Workaround to prevent multiple browser tabs opening with figures
plotly.io.renderers.default = "json"


def get_test_df():
    return pd.DataFrame(
        data=[
            ["P" if np.random.uniform() < 0.4 else "N", 1.5 * i, "X" if i % 2 == 0 else "Y"]
            for i in range(60)
        ],
        columns=["A", "B", "C"],
    )


def test_default_config_verbosity():
    group_section = GroupAnalysis(groupby=[])
    assert group_section.verbosity == Verbosity.LOW, "Verbosity should be Verbosity.LOW"


def test_invalid_verbosities():
    with pytest.raises(ValueError):
        GroupAnalysis(groupby=[], verbosity=4)
    with pytest.raises(ValueError):
        GroupAnalysis(groupby=[], verbosity=-1)


def test_groupby_nonexistent_col():
    with pytest.raises(ValueError):
        show_group_analysis(df=get_test_df(), groupby=["non-existent"])
    with pytest.raises(ValueError):
        group_missing_values(df=get_test_df(), groupby=["non-existent"])


def test_static_methods():
    df = get_test_df()
    with redirect_stdout(None):
        show_group_analysis(df=df, groupby="C")
        show_group_analysis(df=df, groupby=["C"], columns=["A"])
        show_group_analysis(df=df, groupby=["C"], columns=["A", "B"])
        show_group_analysis(df=df, groupby="C", columns=["A", "B", "C"])
        show_group_analysis(df=df, groupby="C", columns=["C"])

        group_barplot(df, groupby=["A"], column="B")
        group_barplot(df, groupby=["A"], column="A")
        group_barplot(df, groupby=["A", "C"], column="B")
        group_barplot(df, groupby=["A"], column="C")
        group_barplot(df, groupby=["A"], column="C")

        group_missing_values(df, groupby=["C"])
        group_missing_values(df, groupby=["C"], columns=["A", "B"])
        group_missing_values(df, groupby=["C"], columns=["A", "B", "C"])
        group_missing_values(df, groupby=["C"], columns=["C"])

        overlaid_histograms(df, groupby=["A"], column="B")
        overlaid_histograms(df, groupby=["A", "C"], column="B")
        overlaid_histograms(df, groupby=["A", "C"], column="B")
        overlaid_histograms(df, groupby=["B"], column="B")


def test_code_export_verbosity_low():
    df = get_test_df()
    group_section = GroupAnalysis(groupby="B", verbosity=Verbosity.LOW)

    # Export code
    exported_cells = []
    group_section.add_cells(exported_cells, df=df)
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = ["show_group_analysis(df=df, groupby=['B'])"]
    # Test code equivalence
    assert len(exported_code) == 1
    assert exported_code[0] == expected_code[0], "Exported code mismatch"


def test_code_export_verbosity_medium():
    df = get_test_df()
    group_section = GroupAnalysis(groupby="A", verbosity=Verbosity.MEDIUM)

    # Export code
    exported_cells = []
    group_section.add_cells(exported_cells, df=df)
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = [
        "group_missing_values(df=df, groupby=['A'])",
        total_dedent(
            """
                within_group_stats(df=df, groupby=['A'], column='B')
                overlaid_histograms(df=df, groupby=['A'], column='B')
            """
        ),
        "group_barplot(df=df, groupby=['A'], column='C')",
    ]

    assert len(expected_code) == len(exported_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"


def test_code_export_verbosity_high():
    df = get_test_df()
    group_section = GroupAnalysis(groupby="A", verbosity=Verbosity.HIGH)

    # Export code
    exported_cells = []
    group_section.add_cells(exported_cells, df=df)
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = [
        "\n\n".join(
            (
                get_code(default_group_quantile_stats),
                get_code(default_group_descriptive_stats),
                get_code(within_group_descriptive_stats),
                get_code(within_group_quantile_stats),
                get_code(within_group_stats),
                get_code(group_barplot),
                get_code(overlaid_histograms),
                get_code(group_missing_values),
            )
        )
    ]

    expected_code += [
        "group_missing_values(df=df, groupby=['A'])",
        # [:-1] to remove trailing newline
        code_dedent(
            """
                within_group_stats(
                    df=df,
                    groupby=['A'],
                    column='B',
                    stats=default_group_descriptive_stats()
                )

                within_group_stats(
                    df=df,
                    groupby=['A'],
                    column='B',
                    stats=default_group_quantile_stats()
                )
                overlaid_histograms(df=df, groupby=['A'], column='B')
            """
        )[:-1],
        "group_barplot(df=df, groupby=['A'], column='C')",
    ]

    assert len(expected_code) == len(exported_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"


def test_columns_parameter():
    df = get_test_df()
    ga = GroupAnalysis(groupby="A", columns=["B"])
    assert ga.groupby == ["A"]
    assert ga.columns == ["B"]

    ga = GroupAnalysis(groupby="A")
    assert ga.groupby == ["A"]
    assert ga.columns is None
    ga.show(df)
    ga.add_cells([], df=df)
    assert ga.groupby == ["A"]
    assert ga.columns is None


def test_column_list_not_modified():
    df = get_test_df()
    columns = ["C"]
    GroupAnalysis(groupby=["A"], columns=columns)
    assert columns == ["C"], "Column list modified"


def test_show():
    df = get_test_df()
    group_section = GroupAnalysis(groupby="A")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with redirect_stdout(None):
            group_section.show(df)
