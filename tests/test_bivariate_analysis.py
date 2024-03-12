import warnings
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import pytest

from edvart.report_sections import bivariate_analysis
from edvart.report_sections.bivariate_analysis import BivariateAnalysis, BivariateAnalysisSubsection
from edvart.report_sections.code_string_formatting import get_code
from edvart.report_sections.section_base import Verbosity

from .execution_utils import check_section_executes
from .pyarrow_utils import pyarrow_params


@pytest.fixture(params=pyarrow_params)
def test_df(request) -> pd.DataFrame:
    test_df = pd.DataFrame(data=[[1.1, "a"], [2.2, "b"], [3.3, "c"]], columns=["A", "B"])
    if request.param:
        test_df = test_df.convert_dtypes(dtype_backend="pyarrow")

    return test_df


def test_default_config_verbosity():
    bivariate_section = bivariate_analysis.BivariateAnalysis()
    assert bivariate_section.verbosity == Verbosity.LOW, "Verbosity should be Verbosity.LOW"
    for s in bivariate_section.subsections:
        assert s.verbosity == Verbosity.LOW, "Verbosity should be Verbosity.LOW"


def test_high_verbosities():
    with pytest.raises(ValueError):
        bivariate_analysis.BivariateAnalysis(verbosity=4)
    with pytest.raises(ValueError):
        bivariate_analysis.BivariateAnalysis(verbosity_contingency_table=4)
    with pytest.raises(ValueError):
        bivariate_analysis.BivariateAnalysis(verbosity_pairplot=5)
    with pytest.raises(ValueError):
        bivariate_analysis.BivariateAnalysis(verbosity_correlations=10)


def test_global_verbosity_overriding():
    bivariate_section = bivariate_analysis.BivariateAnalysis(
        verbosity=Verbosity.LOW,
        verbosity_pairplot=Verbosity.MEDIUM,
        verbosity_correlations=Verbosity.HIGH,
        verbosity_contingency_table=Verbosity.MEDIUM,
    )

    assert bivariate_section.verbosity == Verbosity.LOW
    for subsec in bivariate_section.subsections:
        if isinstance(subsec, bivariate_analysis.PairPlot):
            assert (
                subsec.verbosity == Verbosity.MEDIUM
            ), "Verbosity of pairplot should be Verbosity.MEDIUM"
        elif isinstance(subsec, bivariate_analysis.CorrelationPlot):
            assert (
                subsec.verbosity == Verbosity.HIGH
            ), "Verbosity of correlation plot should be Verbosity.HIGH"
        elif isinstance(subsec, bivariate_analysis.ContingencyTable):
            assert (
                subsec.verbosity == Verbosity.MEDIUM
            ), "Verbosity of contingency table should be Verbosity.MEDIUM"
        else:
            pytest.fail("Unexpected subsection type.")


def test_verbosity_propagation():
    bivariate_section = bivariate_analysis.BivariateAnalysis(verbosity=Verbosity.HIGH)
    assert (
        bivariate_section.verbosity == Verbosity.HIGH
    ), "Bivariate analysis global verbosity should be Verbosity.HIGH."

    for subsec in bivariate_section.subsections:
        if isinstance(subsec, bivariate_analysis.PairPlot):
            assert subsec.verbosity == Verbosity.HIGH, "PairPlot verbosity should be Verbosity.HIGH"
        elif isinstance(subsec, bivariate_analysis.ContingencyTable):
            assert (
                subsec.verbosity == Verbosity.HIGH
            ), "ContingencyTable verbosity should be Verbosity.HIGH."
        elif isinstance(subsec, bivariate_analysis.CorrelationPlot):
            assert (
                subsec.verbosity == Verbosity.HIGH
            ), "Correlation plot verbosity should be Verbosity.HIGH."
        else:
            pytest.fail("Unexpected subsection type")


def test_negative_verbosities():
    with pytest.raises(ValueError):
        bivariate_analysis.BivariateAnalysis(verbosity=-2)
    with pytest.raises(ValueError):
        bivariate_analysis.BivariateAnalysis(verbosity_correlations=-2)
    with pytest.raises(ValueError):
        bivariate_analysis.BivariateAnalysis(verbosity_pairplot=-1)
    with pytest.raises(ValueError):
        bivariate_analysis.BivariateAnalysis(verbosity_contingency_table=-3)


def test_section_adding():
    bivariate_section = bivariate_analysis.BivariateAnalysis(
        subsections=[
            BivariateAnalysisSubsection.PairPlot,
            BivariateAnalysisSubsection.CorrelationPlot,
            BivariateAnalysisSubsection.PairPlot,
            BivariateAnalysisSubsection.PairPlot,
            BivariateAnalysisSubsection.ContingencyTable,
        ]
    )
    assert isinstance(
        bivariate_section.subsections[0], bivariate_analysis.PairPlot
    ), "Subsection should be Pairplot"
    assert isinstance(
        bivariate_section.subsections[1], bivariate_analysis.CorrelationPlot
    ), "Subsection should be CorrelationPlot"
    assert isinstance(
        bivariate_section.subsections[2], bivariate_analysis.PairPlot
    ), "Subsection should be Pairplot"
    assert isinstance(
        bivariate_section.subsections[3], bivariate_analysis.PairPlot
    ), "Subsection should be Pairplot"
    assert isinstance(
        bivariate_section.subsections[4], bivariate_analysis.ContingencyTable
    ), "Subsection should be ContingencyTable"


def test_code_export_verbosity_low(test_df: pd.DataFrame):
    bivariate_section = bivariate_analysis.BivariateAnalysis(verbosity=Verbosity.LOW)
    # Export code
    exported_cells = []
    bivariate_section.add_cells(exported_cells, df=pd.DataFrame())
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = ["show_bivariate_analysis(df=df)"]
    # Test code equivalence
    assert len(exported_code) == 1
    assert exported_code[0] == expected_code[0], "Exported code mismatch"

    check_section_executes(bivariate_section, df=test_df)


def test_code_export_verbosity_low_with_subsections(test_df: pd.DataFrame):
    bivariate_section = bivariate_analysis.BivariateAnalysis(
        subsections=[
            BivariateAnalysisSubsection.ContingencyTable,
            BivariateAnalysisSubsection.PairPlot,
        ],
        verbosity=Verbosity.LOW,
    )
    # Export code
    exported_cells = []
    bivariate_section.add_cells(exported_cells, df=pd.DataFrame())
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = [
        "show_bivariate_analysis(df=df, subsections=["
        "BivariateAnalysisSubsection.ContingencyTable, "
        "BivariateAnalysisSubsection.PairPlot])"
    ]
    # Test code equivalence
    assert len(exported_code) == 1
    assert exported_code[0] == expected_code[0], "Exported code mismatch"

    check_section_executes(bivariate_section, df=test_df)


def test_generated_code_verbosity_low_columns():
    columns = [f"col{i}" for i in range(5)]
    columns_x = [f"col_x{i}" for i in range(6)]
    columns_y = [f"col_y{i}" for i in range(4)]
    columns_pairs = [(f"first{i}", f"second{i}") for i in range(8)]
    columns_all = (
        columns
        + columns_x
        + columns_y
        + [col_pair[0] for col_pair in columns_pairs]
        + [col_pair[1] for col_pair in columns_pairs]
    )
    test_df = pd.DataFrame(
        data=np.random.rand(4, len(columns_all)),
        columns=columns_all,
    )

    bivariate_section = bivariate_analysis.BivariateAnalysis(
        columns=columns,
        columns_x=columns_x,
        columns_y=columns_y,
        columns_pairs=columns_pairs,
        verbosity=Verbosity.LOW,
        color_col="col3",
    )
    # Export code
    exported_cells = []
    bivariate_section.add_cells(exported_cells, df=pd.DataFrame())
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = [
        f"show_bivariate_analysis(df=df, columns_x={columns_x}, "
        f"columns_y={columns_y}, columns_pairs={columns_pairs}, color_col='col3')"
    ]
    # Test code equivalence
    assert len(exported_code) == 1
    assert exported_code[0] == expected_code[0], "Exported code mismatch"

    check_section_executes(bivariate_section, df=test_df)


def test_generated_code_verbosity_medium(test_df: pd.DataFrame):
    bivariate_section = bivariate_analysis.BivariateAnalysis(
        verbosity=Verbosity.MEDIUM,
        subsections=[
            BivariateAnalysisSubsection.PairPlot,
            BivariateAnalysisSubsection.CorrelationPlot,
            BivariateAnalysisSubsection.ContingencyTable,
        ],
    )

    exported_cells = []
    bivariate_section.add_cells(exported_cells, df=pd.DataFrame())
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]

    expected_code = [
        "plot_pairplot(df=df)",
        "plot_correlations(df=df)",
        "contingency_tables(df=df)",
    ]

    assert len(expected_code) == len(exported_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"

    check_section_executes(bivariate_section, df=test_df)


def test_generated_code_verbosity_medium_columns_x_y():
    columns_x = ["a", "b"]
    columns_y = ["c", "d"]
    bivariate_section = bivariate_analysis.BivariateAnalysis(
        verbosity=Verbosity.MEDIUM,
        columns_x=columns_x,
        columns_y=columns_y,
        subsections=[
            BivariateAnalysisSubsection.PairPlot,
            BivariateAnalysisSubsection.CorrelationPlot,
            BivariateAnalysisSubsection.ContingencyTable,
        ],
        color_col="b",
    )
    test_df = pd.DataFrame(
        columns=columns_x + columns_y,
        data=np.random.rand(10, 4),
    )

    exported_cells = []
    bivariate_section.add_cells(exported_cells, df=pd.DataFrame())
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]

    expected_code = [
        f"plot_pairplot(df=df, columns_x={columns_x}, columns_y={columns_y}, color_col='b')",
        f"plot_correlations(df=df, columns_x={columns_x}, columns_y={columns_y})",
        f"contingency_tables(df=df, columns_x={columns_x}, columns_y={columns_y})",
    ]

    assert len(expected_code) == len(exported_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"

    check_section_executes(bivariate_section, df=test_df)


def test_generated_code_verbosity_medium_columns_pairs():
    columns_pairs = [("a", "b"), ("c", "d")]
    columns_x_correct = ["a", "c"]
    columns_y_correct = ["b", "d"]
    bivariate_section = bivariate_analysis.BivariateAnalysis(
        verbosity=Verbosity.MEDIUM,
        columns_pairs=columns_pairs,
        subsections=[
            BivariateAnalysisSubsection.PairPlot,
            BivariateAnalysisSubsection.CorrelationPlot,
            BivariateAnalysisSubsection.ContingencyTable,
        ],
    )
    test_df = pd.DataFrame(
        columns=columns_x_correct + columns_y_correct,
        data=np.random.rand(10, 4),
    )

    exported_cells = []
    bivariate_section.add_cells(exported_cells, df=pd.DataFrame())
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]

    expected_code = [
        f"plot_pairplot(df=df, columns_x={columns_x_correct}, columns_y={columns_y_correct})",
        f"plot_correlations(df=df, columns_x={columns_x_correct}, columns_y={columns_y_correct})",
        f"contingency_tables(df=df, columns_pairs={columns_pairs})",
    ]

    assert len(expected_code) == len(exported_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"

    check_section_executes(bivariate_section, df=test_df)


def test_generated_code_verbosity_high(test_df: pd.DataFrame):
    bivariate_section = bivariate_analysis.BivariateAnalysis(
        verbosity=Verbosity.HIGH,
        subsections=[
            BivariateAnalysisSubsection.PairPlot,
            BivariateAnalysisSubsection.CorrelationPlot,
            BivariateAnalysisSubsection.ContingencyTable,
        ],
    )

    pairplot_cells = []
    bivariate_section.add_cells(pairplot_cells, df=pd.DataFrame())
    exported_code = [cell["source"] for cell in pairplot_cells if cell["cell_type"] == "code"]

    expected_code = [
        "\n\n".join((get_code(bivariate_analysis.plot_pairplot), "plot_pairplot(df=df)")),
        "\n\n".join(
            (
                get_code(bivariate_analysis.default_correlations),
                get_code(bivariate_analysis._get_columns_x_y),
                get_code(bivariate_analysis.plot_correlation),
                get_code(bivariate_analysis.plot_correlations),
                "plot_correlations(df=df)",
            )
        ),
        "\n\n".join(
            (
                get_code(bivariate_analysis.contingency_tables),
                get_code(bivariate_analysis.contingency_table),
                "contingency_tables(df=df)",
            )
        ),
    ]

    assert len(expected_code) == len(exported_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"

    check_section_executes(bivariate_section, df=test_df)


def test_verbosity_low_different_subsection_verbosities(test_df: pd.DataFrame):
    bivariate_section = BivariateAnalysis(
        verbosity=Verbosity.LOW,
        subsections=[
            BivariateAnalysisSubsection.PairPlot,
            BivariateAnalysisSubsection.ContingencyTable,
            BivariateAnalysisSubsection.PairPlot,
            BivariateAnalysisSubsection.CorrelationPlot,
        ],
        verbosity_pairplot=Verbosity.HIGH,
        verbosity_correlations=Verbosity.MEDIUM,
    )

    bivariate_cells = []
    bivariate_section.add_cells(bivariate_cells, df=pd.DataFrame())
    exported_code = [cell["source"] for cell in bivariate_cells if cell["cell_type"] == "code"]

    expected_code = [
        "show_bivariate_analysis(df=df, "
        "subsections=[BivariateAnalysisSubsection.ContingencyTable])",
        get_code(bivariate_analysis.plot_pairplot) + "\n\n" + "plot_pairplot(df=df)",
        get_code(bivariate_analysis.plot_pairplot) + "\n\n" + "plot_pairplot(df=df)",
        "plot_correlations(df=df)",
    ]

    assert len(expected_code) == len(exported_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"

    check_section_executes(bivariate_section, df=test_df)


def test_imports_verbosity_low():
    bivariate_section = BivariateAnalysis(verbosity=Verbosity.LOW)

    exported_imports = bivariate_section.required_imports()
    expected_imports = [
        "from edvart.report_sections.bivariate_analysis import show_bivariate_analysis"
    ]

    assert isinstance(exported_imports, list)
    assert len(expected_imports) == len(exported_imports)
    for expected_import, exported_import in zip(expected_imports, exported_imports):
        assert expected_import == exported_import, "Exported import mismatch"


def test_imports_verbosity_medium():
    bivariate_section = BivariateAnalysis(verbosity=Verbosity.MEDIUM)

    exported_imports = bivariate_section.required_imports()
    expected_imports = list(
        set().union(*[s.required_imports() for s in bivariate_section.subsections])
    )

    assert isinstance(exported_imports, list)
    assert len(expected_imports) == len(exported_imports)
    for expected_import, exported_import in zip(expected_imports, exported_imports):
        assert expected_import == exported_import, "Exported import mismatch"


def test_imports_verbosity_high():
    bivariate_section = BivariateAnalysis(verbosity=Verbosity.HIGH)

    exported_imports = bivariate_section.required_imports()
    expected_imports = list(
        set().union(*[s.required_imports() for s in bivariate_section.subsections])
    )

    assert isinstance(exported_imports, list)
    assert len(expected_imports) == len(exported_imports)
    for expected_import, exported_import in zip(expected_imports, exported_imports):
        assert expected_import == exported_import, "Exported import mismatch"


def test_imports_verbosity_low_different_subsection_verbosities():
    bivariate_section = BivariateAnalysis(
        verbosity=Verbosity.LOW,
        subsections=[
            BivariateAnalysisSubsection.PairPlot,
            BivariateAnalysisSubsection.ContingencyTable,
            BivariateAnalysisSubsection.PairPlot,
            BivariateAnalysisSubsection.CorrelationPlot,
        ],
        verbosity_pairplot=Verbosity.HIGH,
        verbosity_correlations=Verbosity.MEDIUM,
    )

    exported_imports = bivariate_section.required_imports()

    expected_imports = {
        "from edvart.report_sections.bivariate_analysis import show_bivariate_analysis",
        "from edvart.report_sections.bivariate_analysis import BivariateAnalysisSubsection",
    }
    for s in bivariate_section.subsections:
        if s.verbosity > Verbosity.LOW:
            expected_imports.update(s.required_imports())

    assert isinstance(exported_imports, list)
    assert set(exported_imports) == set(expected_imports)


def test_show(test_df: pd.DataFrame):
    bivariate_section = BivariateAnalysis()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with redirect_stdout(None):
            bivariate_section.show(test_df)
