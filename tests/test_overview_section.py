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
    is_numeric,
)
from edvart.pandas_formatting import render_dictionary, series_to_frame
from edvart.report_sections import dataset_overview
from edvart.report_sections.code_string_formatting import get_code
from edvart.report_sections.dataset_overview import Overview


def get_test_df() -> pd.DataFrame:
    test_df = pd.DataFrame(data=[[1.1, "a"], [2.2, "b"], [3.3, "c"]], columns=["A", "B"])

    return test_df


def test_default_verbosity():
    overview_section = Overview()
    assert overview_section.verbosity == 0, "Verbosity should be 0"
    for s in overview_section.subsections:
        assert s.verbosity == 0, "Verbosity should be 0"


def test_global_section_verbosity():
    overview_section = Overview(verbosity=1)
    assert overview_section.verbosity == 1, "Verbosity should be 1"
    for s in overview_section.subsections:
        assert s.verbosity == 1, "Verbosity should be 1"


def test_subsection_verbosity_overriding():
    overview_section = Overview(verbosity=0, verbosity_quick_info=1)
    assert overview_section.verbosity == 0, "Verbosity should be 0"
    for s in overview_section.subsections:
        if isinstance(s, dataset_overview.QuickInfo):
            assert s.verbosity == 1, "Verbosity should be 1"
        else:
            assert s.verbosity == 0, "Verbosity should be 0"

    overview_section = Overview(
        verbosity=0,
        verbosity_quick_info=1,
        verbosity_constant_occurence=0,
        verbosity_data_preview=1,
        verbosity_data_types=2,
        verbosity_rows_with_missing_value=1,
        verbosity_duplicate_rows=1,
    )
    assert overview_section.verbosity == 0, "Verbosity should be 0"
    for s in overview_section.subsections:
        if isinstance(s, dataset_overview.QuickInfo):
            assert s.verbosity == 1, "Verbosity should be 1"
        elif isinstance(s, dataset_overview.ConstantOccurence):
            assert s.verbosity == 0, "Verbosity should be 0"
        elif isinstance(s, dataset_overview.DataPreview):
            assert s.verbosity == 1, "Verbosity should be 1"
        elif isinstance(s, dataset_overview.MissingValues):
            assert s.verbosity == 0, "Verbosity should be 0"
        elif isinstance(s, dataset_overview.DataTypes):
            assert s.verbosity == 2, "Verbosity should be 2"
        elif isinstance(s, dataset_overview.RowsWithMissingValue):
            assert s.verbosity == 1, "Verbosity should be 1"
        elif isinstance(s, dataset_overview.DuplicateRows):
            assert s.verbosity == 1, "Verbosity should be 1"
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
        Overview(verbosity_quick_info=3)
    with pytest.raises(ValueError):
        Overview(verbosity_missing_values=5)


def test_section_adding():
    overview_section = Overview(
        subsections=[
            Overview.OverviewSubsection.QuickInfo,
            Overview.OverviewSubsection.QuickInfo,
            Overview.OverviewSubsection.MissingValues,
            Overview.OverviewSubsection.DuplicateRows,
            Overview.OverviewSubsection.DuplicateRows,
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


def test_code_export_verbosity_0():
    overview_section = Overview(verbosity=0)
    # Export code
    exported_cells = []
    overview_section.add_cells(exported_cells)
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = ["overview_analysis(df=df)"]
    # Test code equivalence
    assert exported_code[0] == expected_code[0], "Exported code mismatch"


def test_code_export_verbosity_0_with_subsections():
    overview_section = Overview(
        subsections=[
            Overview.OverviewSubsection.QuickInfo,
            Overview.OverviewSubsection.MissingValues,
        ],
        verbosity=0,
    )
    # Export code
    exported_cells = []
    overview_section.add_cells(exported_cells)
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = [
        "overview_analysis(df=df, subsections=[Overview.OverviewSubsection.QuickInfo, "
        "Overview.OverviewSubsection.MissingValues])"
    ]
    # Test code equivalence
    assert exported_code[0] == expected_code[0], "Exported code mismatch"


def test_code_export_verbosity_1():
    # Construct overview section
    overview_section = Overview(
        subsections=[
            Overview.OverviewSubsection.QuickInfo,
            Overview.OverviewSubsection.DataTypes,
            Overview.OverviewSubsection.DataPreview,
            Overview.OverviewSubsection.MissingValues,
            Overview.OverviewSubsection.RowsWithMissingValue,
            Overview.OverviewSubsection.ConstantOccurence,
            Overview.OverviewSubsection.DuplicateRows,
        ],
        verbosity=1,
    )
    # Export code
    exported_cells = []
    overview_section.add_cells(exported_cells)
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = [
        "quick_info(df=df)",
        "data_types(df=df)",
        "data_preview(df=df)",
        "missing_values(df=df)",
        "missing_value_row_count(df=df)",
        "constant_occurence(df=df)",
        "duplicate_row_count(df=df)",
    ]
    # Test code equivalence
    for i in range(len(exported_code)):
        assert exported_code[i] == expected_code[i], "Exported code mismatch"


def test_code_export_verbosity_2():
    # Construct overview section
    overview_section = Overview(
        subsections=[
            Overview.OverviewSubsection.QuickInfo,
            Overview.OverviewSubsection.DataTypes,
            Overview.OverviewSubsection.DataPreview,
            Overview.OverviewSubsection.MissingValues,
            Overview.OverviewSubsection.RowsWithMissingValue,
            Overview.OverviewSubsection.ConstantOccurence,
            Overview.OverviewSubsection.DuplicateRows,
        ],
        verbosity=2,
    )
    # Export code
    exported_cells = []
    overview_section.add_cells(exported_cells)
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = [
        "\n\n".join(
            (
                get_code(render_dictionary),
                get_code(dataset_overview.QuickInfo.quick_info),
                "quick_info(df=df)",
            )
        ),
        "\n\n".join(
            (
                get_code(series_to_frame),
                get_code(DataType),
                get_code(is_numeric),
                get_code(is_categorical),
                get_code(is_boolean),
                get_code(is_date),
                get_code(infer_data_type),
                get_code(dataset_overview.DataTypes.data_types),
                "data_types(df=df)",
            )
        ),
        get_code(dataset_overview.DataPreview.data_preview) + "\n\n" + "data_preview(df=df)",
        "\n\n".join(
            (
                get_code(series_to_frame),
                get_code(dataset_overview.MissingValues.missing_values),
                "missing_values(df=df)",
            )
        ),
        "\n\n".join(
            (
                get_code(render_dictionary),
                get_code(dataset_overview.RowsWithMissingValue.missing_value_row_count),
                "missing_value_row_count(df=df)",
            )
        ),
        "\n\n".join(
            (
                get_code(series_to_frame),
                get_code(dataset_overview.ConstantOccurence.constant_occurence),
                "constant_occurence(df=df)",
            )
        ),
        "\n\n".join(
            (
                get_code(render_dictionary),
                get_code(dataset_overview.DuplicateRows.duplicate_row_count),
                "duplicate_row_count(df=df)",
            )
        ),
    ]
    # Test code equivalence
    for i in range(len(exported_code)):
        assert exported_code[i] == expected_code[i], "Exported code mismatch"


def test_verbosity_0_different_subsection_verbosities():
    overview_section = Overview(
        verbosity=0,
        verbosity_quick_info=1,
        verbosity_duplicate_rows=2,
    )

    overview_cells = []
    overview_section.add_cells(overview_cells)
    exported_code = [cell["source"] for cell in overview_cells if cell["cell_type"] == "code"]

    expected_code = [
        "overview_analysis(df=df, "
        "subsections=[Overview.OverviewSubsection.DataTypes, "
        "Overview.OverviewSubsection.DataPreview, "
        "Overview.OverviewSubsection.MissingValues, "
        "Overview.OverviewSubsection.RowsWithMissingValue, "
        "Overview.OverviewSubsection.ConstantOccurence])",
        "quick_info(df=df)",
        (
            get_code(render_dictionary)
            + 2 * "\n"
            + get_code(dataset_overview.DuplicateRows.duplicate_row_count)
            + 2 * "\n"
            + "duplicate_row_count(df=df)"
        ),
    ]

    assert len(exported_code) == len(expected_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"


def test_imports_verbosity_0():
    overview_section = Overview(verbosity=0)

    exported_imports = overview_section.required_imports()
    expected_imports = [
        "from edvart.report_sections.dataset_overview import Overview\n"
        "overview_analysis = Overview.overview_analysis"
    ]

    assert isinstance(exported_imports, list)
    assert len(expected_imports) == len(exported_imports)
    for expected_import, exported_import in zip(expected_imports, exported_imports):
        assert expected_import == exported_import, "Exported import mismatch"


def test_imports_verbosity_1():
    multivariate_section = Overview(verbosity=1)

    exported_imports = multivariate_section.required_imports()
    expected_imports = list(
        set().union(*[s.required_imports() for s in multivariate_section.subsections])
    )

    assert isinstance(exported_imports, list)
    assert len(expected_imports) == len(exported_imports)
    for expected_import, exported_import in zip(expected_imports, exported_imports):
        assert expected_import == exported_import, "Exported import mismatch"


def test_imports_verbosity_2():
    multivariate_section = Overview(verbosity=2)

    exported_imports = multivariate_section.required_imports()
    expected_imports = list(
        set().union(*[s.required_imports() for s in multivariate_section.subsections])
    )

    assert isinstance(exported_imports, list)
    assert len(expected_imports) == len(exported_imports)
    for expected_import, exported_import in zip(expected_imports, exported_imports):
        assert expected_import == exported_import, "Exported import mismatch"


def test_imports_verbosity_0_different_subsection_verbosities():
    overview_section = Overview(
        verbosity=0,
        verbosity_quick_info=1,
        verbosity_duplicate_rows=2,
    )

    exported_imports = overview_section.required_imports()

    expected_imports = {
        "from edvart.report_sections.dataset_overview import Overview\n"
        "overview_analysis = Overview.overview_analysis"
    }
    for s in overview_section.subsections:
        if s.verbosity > 0:
            expected_imports.update(s.required_imports())

    assert isinstance(exported_imports, list)
    assert set(exported_imports) == set(expected_imports)


def test_show():
    overview_section = Overview()
    df = get_test_df()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with redirect_stdout(None):
            overview_section.show(df)
