import warnings
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import plotly.io as pio
import pytest

from edvart import utils
from edvart.report_sections import multivariate_analysis
from edvart.report_sections.code_string_formatting import code_dedent, get_code
from edvart.report_sections.multivariate_analysis import (
    UMAP_AVAILABLE,
    MultivariateAnalysis,
    MultivariateAnalysisSubsection,
)
from edvart.report_sections.section_base import Verbosity
from edvart.utils import select_numeric_columns

if UMAP_AVAILABLE:
    from edvart.report_sections import umap

from .execution_utils import check_section_executes
from .pyarrow_utils import pyarrow_params

pio.renderers.default = "json"


@pytest.fixture(params=pyarrow_params)
def test_df(request) -> pd.DataFrame:
    test_df = pd.DataFrame(
        data=[
            [1.1, "a", 3.7, 3.9],
            [2.2, "b", 4.2, 5.1],
            [3.3, "c", 7.9, 5.8],
            [4.4, "d", 7.6, 5.2],
            [5.5, "e", 7.1, 3.7],
        ],
        columns=["A", "B", "C", "D"],
    )
    if request.param:
        test_df = test_df.convert_dtypes(dtype_backend="pyarrow")

    return test_df


def test_default_config_verbosity():
    multivariate_section = MultivariateAnalysis()
    assert multivariate_section.verbosity == Verbosity.LOW, "Verbosity should be Verbosity.LOW"
    for s in multivariate_section.subsections:
        assert s.verbosity == Verbosity.LOW, "Verbosity should be Verbosity.LOW"


def test_high_verbosities():
    with pytest.raises(ValueError):
        MultivariateAnalysis(verbosity=4)
    with pytest.raises(ValueError):
        MultivariateAnalysis(verbosity_pca=5)


def test_global_verbosity_overriding():
    multivariate_section = MultivariateAnalysis(
        verbosity=Verbosity.LOW,
        verbosity_pca=Verbosity.HIGH,
        verbosity_umap=Verbosity.MEDIUM,
        verbosity_parallel_categories=Verbosity.MEDIUM,
        verbosity_parallel_coordinates=Verbosity.HIGH,
    )

    assert multivariate_section.verbosity == Verbosity.LOW
    for subsec in multivariate_section.subsections:
        if isinstance(subsec, multivariate_analysis.PCA):
            assert subsec.verbosity == Verbosity.HIGH, "Verbosity of PCA should be Verbosity.HIGH"
        if UMAP_AVAILABLE:
            if isinstance(subsec, multivariate_analysis.UMAP):
                assert (
                    subsec.verbosity == Verbosity.MEDIUM
                ), "Verbosity of UMAP should be Verbosity.MEDIUM"
        if isinstance(subsec, multivariate_analysis.ParallelCategories):
            assert (
                subsec.verbosity == Verbosity.MEDIUM
            ), "Verbosity of Par cats should be Verbosity.MEDIUM"
        if isinstance(subsec, multivariate_analysis.ParallelCoordinates):
            assert (
                subsec.verbosity == Verbosity.HIGH
            ), "Verbosity of Par coords should be Verbosity.HIGH"


def test_verbosity_propagation():
    multivariate_section = MultivariateAnalysis(verbosity=Verbosity.HIGH)
    assert (
        multivariate_section.verbosity == Verbosity.HIGH
    ), "Multivariate analysis global verbosity should be Verbosity.HIGH."

    for subsec in multivariate_section.subsections:
        assert (
            subsec.verbosity == Verbosity.HIGH
        ), f"Subsection {type(subsec)} verbosity should be Verbosity.HIGH"


def test_negative_verbosities():
    with pytest.raises(ValueError):
        MultivariateAnalysis(verbosity=-2)
    with pytest.raises(ValueError):
        multivariate_analysis.MultivariateAnalysis(verbosity_pca=-1)


def test_section_adding():
    subsections = [
        MultivariateAnalysisSubsection.PCA,
        MultivariateAnalysisSubsection.PCA,
        MultivariateAnalysisSubsection.ParallelCoordinates,
        MultivariateAnalysisSubsection.ParallelCategories,
    ]
    if UMAP_AVAILABLE:
        subsections.append(MultivariateAnalysisSubsection.UMAP)
    multivariate_section = MultivariateAnalysis(subsections=subsections)
    if UMAP_AVAILABLE:
        assert len(multivariate_section.subsections) == 5
    else:
        assert len(multivariate_section.subsections) == 4
    assert isinstance(
        multivariate_section.subsections[0], multivariate_analysis.PCA
    ), "Subsection should be PCA"
    assert isinstance(
        multivariate_section.subsections[1], multivariate_analysis.PCA
    ), "Subsection should be PCA"
    assert isinstance(
        multivariate_section.subsections[2], multivariate_analysis.ParallelCoordinates
    ), "Subsection should be Parallel coordinates"
    assert isinstance(
        multivariate_section.subsections[3], multivariate_analysis.ParallelCategories
    ), "Subsection should be Parallel categories"
    if UMAP_AVAILABLE:
        assert isinstance(
            multivariate_section.subsections[4], multivariate_analysis.UMAP
        ), "Subsection should be UMAP"


def test_code_export_verbosity_low(test_df: pd.DataFrame):
    df = test_df
    multivariate_section = MultivariateAnalysis(verbosity=Verbosity.LOW)
    # Export code
    exported_cells = []
    multivariate_section.add_cells(exported_cells, df=df)
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = ["show_multivariate_analysis(df=df)"]
    # Test code equivalence
    assert len(exported_code) == 1
    assert exported_code[0] == expected_code[0], "Exported code mismatch"

    check_section_executes(multivariate_section, df)


def test_code_export_verbosity_low_with_subsections(test_df: pd.DataFrame):
    subsec = MultivariateAnalysisSubsection
    subsections = [subsec.ParallelCategories, subsec.PCA, subsec.ParallelCoordinates, subsec.PCA]
    if UMAP_AVAILABLE:
        subsections.append(subsec.UMAP)
    df = test_df
    multivariate_section = multivariate_analysis.MultivariateAnalysis(
        subsections=subsections, verbosity=Verbosity.LOW
    )

    # Export code
    exported_cells = []
    multivariate_section.add_cells(exported_cells, df=df)
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    if UMAP_AVAILABLE:
        # Define expected code
        expected_code = [
            "show_multivariate_analysis(df=df, subsections=["
            "MultivariateAnalysisSubsection.ParallelCategories, "
            "MultivariateAnalysisSubsection.PCA, "
            "MultivariateAnalysisSubsection.ParallelCoordinates, "
            "MultivariateAnalysisSubsection.PCA, "
            "MultivariateAnalysisSubsection.UMAP])"
        ]
    else:
        expected_code = [
            "show_multivariate_analysis(df=df, subsections=["
            "MultivariateAnalysisSubsection.ParallelCategories, "
            "MultivariateAnalysisSubsection.PCA, "
            "MultivariateAnalysisSubsection.ParallelCoordinates, "
            "MultivariateAnalysisSubsection.PCA])"
        ]
    # Test code equivalence
    assert len(exported_code) == 1
    assert exported_code[0] == expected_code[0], "Exported code mismatch"

    check_section_executes(multivariate_section, df)


def test_code_export_verbosity_medium_all_cols_valid():
    all_numeric_df = pd.DataFrame(
        data=[[1.1, 1, -2], [2.2, 2, -5.3], [3.3, 3, 4]], columns=["col1", "col2", "col3"]
    )

    multivariate_section = MultivariateAnalysis(
        subsections=[
            MultivariateAnalysisSubsection.PCA,
            MultivariateAnalysisSubsection.ParallelCategories,
        ],
        verbosity=Verbosity.MEDIUM,
    )

    exported_cells = []
    multivariate_section.add_cells(exported_cells, df=all_numeric_df)
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]

    expected_code = [
        "pca_first_vs_second(df=df)",
        "pca_explained_variance(df=df)",
        "parallel_categories(df=df)",
    ]

    assert len(exported_code) == len(expected_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"

    check_section_executes(multivariate_section, all_numeric_df)


def test_generated_code_verbosity_1(test_df: pd.DataFrame):
    multivariate_section = MultivariateAnalysis(verbosity=Verbosity.MEDIUM)

    exported_cells = []
    multivariate_section.add_cells(exported_cells, df=test_df)
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    if UMAP_AVAILABLE:
        expected_code = [
            "pca_first_vs_second(df=df)",
            "pca_explained_variance(df=df)",
            code_dedent(
                """
                plot_umap(
                    df=df,
                    n_neighbors=15,
                    min_dist=0.1,
                    metric='euclidean',
                )"""
            ),
            "parallel_coordinates(df=df)",
            "parallel_categories(df=df)",
        ]
    else:
        expected_code = [
            "pca_first_vs_second(df=df)",
            "pca_explained_variance(df=df)",
            "parallel_coordinates(df=df)",
            "parallel_categories(df=df)",
        ]

    assert len(exported_code) == len(expected_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"

    check_section_executes(multivariate_section, test_df)


def test_generated_code_verbosity_2(test_df: pd.DataFrame):
    multivariate_section = MultivariateAnalysis(verbosity=Verbosity.HIGH)

    multivariate_cells = []
    multivariate_section.add_cells(multivariate_cells, df=test_df)
    exported_code = [cell["source"] for cell in multivariate_cells if cell["cell_type"] == "code"]
    expected_code = [
        get_code(select_numeric_columns),
        "\n\n".join(
            (
                get_code(multivariate_analysis.pca_first_vs_second),
                "pca_first_vs_second(df=df)",
            )
        ),
        "\n\n".join(
            (
                get_code(multivariate_analysis.pca_explained_variance),
                "pca_explained_variance(df=df)",
            )
        ),
        "\n\n".join(
            (
                get_code(utils.hsl_wheel_colorscale),
                get_code(utils.make_discrete_colorscale),
                get_code(utils.get_default_discrete_colorscale),
                get_code(multivariate_analysis.parallel_coordinates),
                "parallel_coordinates(df=df)",
            )
        ),
        "\n\n".join(
            (
                get_code(utils.hsl_wheel_colorscale),
                get_code(utils.make_discrete_colorscale),
                get_code(utils.get_default_discrete_colorscale),
                get_code(multivariate_analysis.parallel_categories),
                "parallel_categories(df=df)",
            )
        ),
    ]
    if UMAP_AVAILABLE:
        expected_code.insert(
            3,
            (
                get_code(select_numeric_columns)
                + "\n\n"
                + get_code(umap.plot_umap)
                + "\n\n"
                + code_dedent(
                    """
                    plot_umap(
                        df=df,
                        n_neighbors=15,
                        min_dist=0.1,
                        metric='euclidean',
                    )"""
                )
            ),
        )

    assert len(exported_code) == len(expected_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"

    check_section_executes(multivariate_section, test_df)


@pytest.mark.parametrize("pyarrow_dtypes", pyarrow_params)
def test_verbosity_medium_non_categorical_col(pyarrow_dtypes: bool):
    random_array = np.random.randint(low=1, high=40, size=(100, 3))
    random_df = pd.DataFrame(data=random_array, columns=["integral", "floating", "cat"])
    random_df = random_df.astype({"integral": int, "floating": float, "cat": "category"})
    if pyarrow_dtypes:
        random_df = random_df.convert_dtypes(dtype_backend="pyarrow")
    subsec = MultivariateAnalysisSubsection
    multivariate_section = multivariate_analysis.MultivariateAnalysis(
        subsections=[subsec.ParallelCategories], verbosity=Verbosity.MEDIUM
    )

    multivariate_cells = []
    multivariate_section.add_cells(multivariate_cells, df=random_df)
    exported_code = [cell["source"] for cell in multivariate_cells if cell["cell_type"] == "code"]

    expected_code = ["parallel_categories(df=df)"]

    assert len(exported_code) == len(expected_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"

    check_section_executes(multivariate_section, random_df)


def test_verbosity_low_different_subsection_verbosities(test_df: pd.DataFrame):
    subsections = [
        MultivariateAnalysisSubsection.PCA,
        MultivariateAnalysisSubsection.PCA,
        MultivariateAnalysisSubsection.ParallelCategories,
        MultivariateAnalysisSubsection.ParallelCoordinates,
    ]
    if UMAP_AVAILABLE:
        subsections.insert(2, MultivariateAnalysisSubsection.UMAP)
    multivariate_section = MultivariateAnalysis(
        verbosity=Verbosity.LOW,
        subsections=subsections,
        verbosity_parallel_categories=Verbosity.MEDIUM,
        verbosity_parallel_coordinates=Verbosity.HIGH,
    )

    multivariate_cells = []
    multivariate_section.add_cells(multivariate_cells, df=test_df)
    exported_code = [cell["source"] for cell in multivariate_cells if cell["cell_type"] == "code"]
    expected_subsections = [
        "MultivariateAnalysisSubsection.PCA",
        "MultivariateAnalysisSubsection.PCA",
    ]
    if UMAP_AVAILABLE:
        expected_subsections.append("MultivariateAnalysisSubsection.UMAP")
    expected_subsections_str = ", ".join(expected_subsections)
    expected_code = [
        "show_multivariate_analysis(df=df, " f"subsections=[{expected_subsections_str}])",
        "parallel_categories(df=df)",
        "\n\n".join(
            (
                get_code(utils.hsl_wheel_colorscale),
                get_code(utils.make_discrete_colorscale),
                get_code(utils.get_default_discrete_colorscale),
                get_code(multivariate_analysis.parallel_coordinates),
                "parallel_coordinates(df=df)",
            )
        ),
    ]

    assert len(exported_code) == len(expected_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"

    check_section_executes(multivariate_section, test_df)


def test_imports_verbosity_low():
    multivariate_section = MultivariateAnalysis(verbosity=Verbosity.LOW)

    exported_imports = multivariate_section.required_imports()
    expected_imports = [
        "from edvart.report_sections.multivariate_analysis import show_multivariate_analysis"
    ]

    assert isinstance(exported_imports, list)
    assert len(expected_imports) == len(exported_imports)
    for expected_import, exported_import in zip(expected_imports, exported_imports):
        assert expected_import == exported_import, "Exported import mismatch"


def test_imports_verbosity_medium():
    multivariate_section = MultivariateAnalysis(verbosity=Verbosity.MEDIUM)

    exported_imports = multivariate_section.required_imports()
    expected_imports = list(
        set().union(*[s.required_imports() for s in multivariate_section.subsections])
    )

    assert isinstance(exported_imports, list)
    assert len(expected_imports) == len(exported_imports)
    for expected_import, exported_import in zip(expected_imports, exported_imports):
        assert expected_import == exported_import, "Exported import mismatch"


def test_imports_verbosity_high():
    multivariate_section = MultivariateAnalysis(verbosity=Verbosity.HIGH)

    exported_imports = multivariate_section.required_imports()
    expected_imports = list(
        set().union(*[s.required_imports() for s in multivariate_section.subsections])
    )

    assert isinstance(exported_imports, list)
    assert len(expected_imports) == len(exported_imports)
    for expected_import, exported_import in zip(expected_imports, exported_imports):
        assert expected_import == exported_import, "Exported import mismatch"


def test_imports_verbosity_low_different_subsection_verbosities():
    subsections = [
        MultivariateAnalysisSubsection.PCA,
        MultivariateAnalysisSubsection.PCA,
        MultivariateAnalysisSubsection.ParallelCategories,
        MultivariateAnalysisSubsection.ParallelCoordinates,
    ]
    if UMAP_AVAILABLE:
        subsections.insert(3, MultivariateAnalysisSubsection.UMAP)
    multivariate_section = MultivariateAnalysis(
        verbosity=Verbosity.LOW,
        subsections=subsections,
        verbosity_parallel_categories=Verbosity.MEDIUM,
        verbosity_parallel_coordinates=Verbosity.HIGH,
    )

    exported_imports = multivariate_section.required_imports()

    expected_imports = {
        "from edvart.report_sections.multivariate_analysis import show_multivariate_analysis",
        "from edvart.report_sections.multivariate_analysis import MultivariateAnalysisSubsection",
    }
    for s in multivariate_section.subsections:
        if s.verbosity > Verbosity.LOW:
            expected_imports.update(s.required_imports())

    assert isinstance(exported_imports, list)
    assert set(exported_imports) == set(expected_imports)


def test_show(test_df: pd.DataFrame):
    multivariate_section = MultivariateAnalysis()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with redirect_stdout(None):
            multivariate_section.show(test_df)
