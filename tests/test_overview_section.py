import warnings
from contextlib import redirect_stdout

import pandas as pd
import pytest

from edvart.data_types import (
    DataType,
    infer_data_type,
    is_boolean,
    is_categorical,
    is_date,
    is_missing,
    is_numeric,
    is_unique,
)
from edvart.pandas_formatting import render_dictionary, series_to_frame
from edvart.report_sections import dataset_overview
from edvart.report_sections.code_string_formatting import get_code
from edvart.report_sections.dataset_overview import Overview, OverviewSubsection
from edvart.report_sections.section_base import Verbosity

from .execution_utils import check_section_executes


@pytest.fixture
def test_df() -> pd.DataFrame:
    test_df = pd.DataFrame(data=[[1.1, "a"], [2.2, "b"], [3.3, "c"]], columns=["A", "B"])

    return test_df


def test_default_verbosity():
    overview_section = Overview()
    assert overview_section.verbosity == Verbosity.LOW, "Verbosity should be Verbosity.LOW"
    for s in overview_section.subsections:
        assert s.verbosity == Verbosity.LOW, "Verbosity should be Verbosity.LOW"


def test_global_section_verbosity():
    overview_section = Overview(verbosity=Verbosity.MEDIUM)
    assert overview_section.verbosity == Verbosity.MEDIUM, "Verbosity should be Verbosity.MEDIUM"
    for s in overview_section.subsections:
        assert s.verbosity == Verbosity.MEDIUM, "Verbosity should be Verbosity.MEDIUM"


def test_subsection_verbosity_overriding():
    overview_section = Overview(verbosity=Verbosity.LOW, verbosity_quick_info=Verbosity.MEDIUM)
    assert overview_section.verbosity == Verbosity.LOW, "Verbosity should be Verbosity.LOW"
    for s in overview_section.subsections:
        if isinstance(s, dataset_overview.QuickInfo):
            assert s.verbosity == Verbosity.MEDIUM, "Verbosity should be Verbosity.MEDIUM"
        else:
            assert s.verbosity == Verbosity.LOW, "Verbosity should be Verbosity.LOW"

    overview_section = Overview(
        verbosity=Verbosity.LOW,
        verbosity_quick_info=Verbosity.MEDIUM,
        verbosity_constant_occurrence=Verbosity.LOW,
        verbosity_data_preview=Verbosity.MEDIUM,
        verbosity_data_types=Verbosity.HIGH,
        verbosity_rows_with_missing_value=Verbosity.MEDIUM,
        verbosity_duplicate_rows=Verbosity.MEDIUM,
    )
    assert overview_section.verbosity == Verbosity.LOW, "Verbosity should be Verbosity.LOW"
    for s in overview_section.subsections:
        if isinstance(s, dataset_overview.QuickInfo):
            assert s.verbosity == Verbosity.MEDIUM, "Verbosity should be Verbosity.MEDIUM"
        elif isinstance(s, dataset_overview.ConstantOccurrence):
            assert s.verbosity == Verbosity.LOW, "Verbosity should be Verbosity.LOW"
        elif isinstance(s, dataset_overview.DataPreview):
            assert s.verbosity == Verbosity.MEDIUM, "Verbosity should be Verbosity.MEDIUM"
        elif isinstance(s, dataset_overview.MissingValues):
            assert s.verbosity == Verbosity.LOW, "Verbosity should be Verbosity.LOW"
        elif isinstance(s, dataset_overview.DataTypes):
            assert s.verbosity == Verbosity.HIGH, "Verbosity should be 2"
        elif isinstance(s, dataset_overview.RowsWithMissingValue):
            assert s.verbosity == Verbosity.MEDIUM, "Verbosity should be Verbosity.MEDIUM"
        elif isinstance(s, dataset_overview.DuplicateRows):
            assert s.verbosity == Verbosity.MEDIUM, "Verbosity should be Verbosity.MEDIUM"
        else:
            pytest.fail("Invalid overview subsection type")


def test_negative_verbosities():
    with pytest.raises(ValueError):
        Overview(verbosity=-1)
    with pytest.raises(ValueError):
        Overview(verbosity_quick_info=-3)


def test_high_verbosities():
    with pytest.raises(ValueError):
        Overview(verbosity=10)
    with pytest.raises(ValueError):
        Overview(verbosity_data_types=4)
    with pytest.raises(ValueError):
        Overview(verbosity_quick_info=4)
    with pytest.raises(ValueError):
        Overview(verbosity_missing_values=5)


def test_section_adding():
    overview_section = Overview(
        subsections=[
            OverviewSubsection.QuickInfo,
            OverviewSubsection.QuickInfo,
            OverviewSubsection.MissingValues,
            OverviewSubsection.DuplicateRows,
            OverviewSubsection.DuplicateRows,
        ]
    )
    assert isinstance(
        overview_section.subsections[0], dataset_overview.QuickInfo
    ), "Subsection should be QuickInfo"
    assert isinstance(
        overview_section.subsections[1], dataset_overview.QuickInfo
    ), "Subsection should be QuickInfo"
    assert isinstance(
        overview_section.subsections[2], dataset_overview.MissingValues
    ), "Subsection should be MissingValues"
    assert isinstance(
        overview_section.subsections[3], dataset_overview.DuplicateRows
    ), "Subsection should be DuplicateRows"
    assert isinstance(
        overview_section.subsections[4], dataset_overview.DuplicateRows
    ), "Subsection should be DuplicateRows"


def test_code_export_verbosity_low(test_df: pd.DataFrame):
    overview_section = Overview(verbosity=Verbosity.LOW)
    # Export code
    exported_cells = []
    overview_section.add_cells(exported_cells, df=pd.DataFrame())
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = ["show_overview(df=df)"]
    # Test code equivalence
    assert exported_code[0] == expected_code[0], "Exported code mismatch"

    check_section_executes(overview_section, df=test_df)


def test_code_export_verbosity_low_with_subsections(test_df: pd.DataFrame):
    overview_section = Overview(
        subsections=[
            OverviewSubsection.QuickInfo,
            OverviewSubsection.MissingValues,
        ],
        verbosity=Verbosity.LOW,
    )
    # Export code
    exported_cells = []
    overview_section.add_cells(exported_cells, df=pd.DataFrame())
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = [
        "show_overview(df=df, subsections=[OverviewSubsection.QuickInfo, "
        "OverviewSubsection.MissingValues])"
    ]
    # Test code equivalence
    assert exported_code[0] == expected_code[0], "Exported code mismatch"

    check_section_executes(overview_section, df=test_df)


def test_code_export_verbosity_medium(test_df: pd.DataFrame):
    # Construct overview section
    overview_section = Overview(
        subsections=[
            OverviewSubsection.QuickInfo,
            OverviewSubsection.DataTypes,
            OverviewSubsection.DataPreview,
            OverviewSubsection.MissingValues,
            OverviewSubsection.RowsWithMissingValue,
            OverviewSubsection.ConstantOccurrence,
            OverviewSubsection.DuplicateRows,
        ],
        verbosity=Verbosity.MEDIUM,
    )
    # Export code
    exported_cells = []
    overview_section.add_cells(exported_cells, df=pd.DataFrame())
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = [
        "quick_info(df=df)",
        "data_types(df=df)",
        "data_preview(df=df)",
        "missing_values(df=df)",
        "missing_value_row_count(df=df)",
        "constant_occurrence(df=df)",
        "duplicate_row_count(df=df)",
    ]
    # Test code equivalence
    for i in range(len(exported_code)):
        assert exported_code[i] == expected_code[i], "Exported code mismatch"

    check_section_executes(overview_section, df=test_df)


def test_code_export_verbosity_high(test_df: pd.DataFrame):
    # Construct overview section
    overview_section = Overview(
        subsections=[
            OverviewSubsection.QuickInfo,
            OverviewSubsection.DataTypes,
            OverviewSubsection.DataPreview,
            OverviewSubsection.MissingValues,
            OverviewSubsection.RowsWithMissingValue,
            OverviewSubsection.ConstantOccurrence,
            OverviewSubsection.DuplicateRows,
        ],
        verbosity=Verbosity.HIGH,
    )
    # Export code
    exported_cells = []
    overview_section.add_cells(exported_cells, df=pd.DataFrame())
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = [
        "\n\n".join(
            (
                get_code(render_dictionary),
                get_code(dataset_overview.quick_info),
                "quick_info(df=df)",
            )
        ),
        "\n\n".join(
            (
                get_code(series_to_frame),
                get_code(DataType),
                get_code(is_unique),
                get_code(is_numeric),
                get_code(is_missing),
                get_code(is_categorical),
                get_code(is_boolean),
                get_code(is_date),
                get_code(infer_data_type),
                get_code(dataset_overview.data_types),
                "data_types(df=df)",
            )
        ),
        get_code(dataset_overview.data_preview) + "\n\n" + "data_preview(df=df)",
        "\n\n".join(
            (
                get_code(series_to_frame),
                get_code(dataset_overview.missing_values),
                "missing_values(df=df)",
            )
        ),
        "\n\n".join(
            (
                get_code(render_dictionary),
                get_code(dataset_overview.missing_value_row_count),
                "missing_value_row_count(df=df)",
            )
        ),
        "\n\n".join(
            (
                get_code(series_to_frame),
                get_code(dataset_overview.constant_occurrence),
                "constant_occurrence(df=df)",
            )
        ),
        "\n\n".join(
            (
                get_code(render_dictionary),
                get_code(dataset_overview.duplicate_row_count),
                "duplicate_row_count(df=df)",
            )
        ),
    ]
    # Test code equivalence
    for i in range(len(exported_code)):
        assert exported_code[i] == expected_code[i], "Exported code mismatch"

    check_section_executes(overview_section, df=test_df)


def test_verbosity_low_different_subsection_verbosities(test_df: pd.DataFrame):
    overview_section = Overview(
        verbosity=Verbosity.LOW,
        verbosity_quick_info=Verbosity.MEDIUM,
        verbosity_duplicate_rows=Verbosity.HIGH,
    )

    overview_cells = []
    overview_section.add_cells(overview_cells, df=pd.DataFrame())
    exported_code = [cell["source"] for cell in overview_cells if cell["cell_type"] == "code"]

    expected_code = [
        "show_overview(df=df, "
        "subsections=[OverviewSubsection.DataTypes, "
        "OverviewSubsection.DataPreview, "
        "OverviewSubsection.MissingValues, "
        "OverviewSubsection.RowsWithMissingValue, "
        "OverviewSubsection.ConstantOccurrence])",
        "quick_info(df=df)",
        (
            get_code(render_dictionary)
            + 2 * "\n"
            + get_code(dataset_overview.duplicate_row_count)
            + 2 * "\n"
            + "duplicate_row_count(df=df)"
        ),
    ]

    assert len(exported_code) == len(expected_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"

    check_section_executes(overview_section, df=test_df)


def test_imports_verbosity_low():
    overview_section = Overview(verbosity=Verbosity.LOW)

    exported_imports = overview_section.required_imports()
    expected_imports = ["from edvart.report_sections.dataset_overview import show_overview"]

    assert isinstance(exported_imports, list)
    assert len(expected_imports) == len(exported_imports)
    for expected_import, exported_import in zip(expected_imports, exported_imports):
        assert expected_import == exported_import, "Exported import mismatch"


def test_imports_verbosity_medium():
    multivariate_section = Overview(verbosity=Verbosity.MEDIUM)

    exported_imports = multivariate_section.required_imports()
    expected_imports = list(
        set().union(*[s.required_imports() for s in multivariate_section.subsections])
    )

    assert isinstance(exported_imports, list)
    assert len(expected_imports) == len(exported_imports)
    for expected_import, exported_import in zip(expected_imports, exported_imports):
        assert expected_import == exported_import, "Exported import mismatch"


def test_imports_verbosity_high():
    multivariate_section = Overview(verbosity=Verbosity.HIGH)

    exported_imports = multivariate_section.required_imports()
    expected_imports = list(
        set().union(*[s.required_imports() for s in multivariate_section.subsections])
    )

    assert isinstance(exported_imports, list)
    assert len(expected_imports) == len(exported_imports)
    for expected_import, exported_import in zip(expected_imports, exported_imports):
        assert expected_import == exported_import, "Exported import mismatch"


def test_imports_verbosity_low_different_subsection_verbosities():
    overview_section = Overview(
        verbosity=Verbosity.LOW,
        verbosity_quick_info=Verbosity.MEDIUM,
        verbosity_duplicate_rows=Verbosity.HIGH,
    )

    exported_imports = overview_section.required_imports()

    expected_imports = {
        "from edvart.report_sections.dataset_overview import show_overview",
        "from edvart.report_sections.dataset_overview import OverviewSubsection",
    }
    for s in overview_section.subsections:
        if s.verbosity > Verbosity.LOW:
            expected_imports.update(s.required_imports())

    assert isinstance(exported_imports, list)
    assert set(exported_imports) == set(expected_imports)


def test_show(test_df: pd.DataFrame):
    overview_section = Overview()
    df = test_df
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with redirect_stdout(None):
            overview_section.show(df)
