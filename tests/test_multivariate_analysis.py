import warnings
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import plotly.io as pio
import pytest

pio.renderers.default = "json"

from edvart import utils
from edvart.report_sections import multivariate_analysis
from edvart.report_sections.code_string_formatting import code_dedent, get_code
from edvart.report_sections.multivariate_analysis import (
    UMAP_AVAILABLE,
    MultivariateAnalysis,
)


def get_test_df() -> pd.DataFrame:
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

    return test_df


def test_default_config_verbosity():
    multivariate_section = MultivariateAnalysis(get_test_df())
    assert multivariate_section.verbosity == 0, "Verbosity should be 0"
    for s in multivariate_section.subsections:
        assert s.verbosity == 0, "Verbosity should be 0"


def test_high_verobisities():
    with pytest.raises(ValueError):
        MultivariateAnalysis(df=get_test_df(), verbosity=3)
    with pytest.raises(ValueError):
        MultivariateAnalysis(df=get_test_df(), verbosity_pca=5)


def test_global_verbosity_overriding():
    multivariate_section = MultivariateAnalysis(
        get_test_df(),
        verbosity=0,
        verbosity_pca=2,
        verbosity_umap=1,
        verbosity_parallel_categories=1,
        verbosity_parallel_coordinates=2,
    )

    assert multivariate_section.verbosity == 0
    for subsec in multivariate_section.subsections:
        if isinstance(subsec, multivariate_analysis.PCA):
            assert subsec.verbosity == 2, "Verbosity of PCA should be 2"
        if UMAP_AVAILABLE:
            if isinstance(subsec, multivariate_analysis.UMAP):
                assert subsec.verbosity == 1, "Verbosity of UMAP should be 1"
        if isinstance(subsec, multivariate_analysis.ParallelCategories):
            assert subsec.verbosity == 1, "Verbosity of Par cats should be 1"
        if isinstance(subsec, multivariate_analysis.ParallelCoordinates):
            assert subsec.verbosity == 2, "Verbosity of Par coords should be 2"


def test_verbosity_propagation():
    multivariate_section = MultivariateAnalysis(get_test_df(), verbosity=2)
    assert (
        multivariate_section.verbosity == 2
    ), "Multivariate analysis global verbosity should be 2."

    for subsec in multivariate_section.subsections:
        assert subsec.verbosity == 2, f"Subsection {type(subsec)} verbosity should be 2"


def test_negative_verbosities():
    test_df = get_test_df()
    with pytest.raises(ValueError):
        MultivariateAnalysis(test_df, verbosity=-2)
    with pytest.raises(ValueError):
        multivariate_analysis.MultivariateAnalysis(test_df, verbosity_pca=-1)


def test_section_adding():
    subsections = [
        MultivariateAnalysis.MultivariateAnalysisSubsection.PCA,
        MultivariateAnalysis.MultivariateAnalysisSubsection.PCA,
        MultivariateAnalysis.MultivariateAnalysisSubsection.ParallelCoordinates,
        MultivariateAnalysis.MultivariateAnalysisSubsection.ParallelCategories,
    ]
    if UMAP_AVAILABLE:
        subsections.append(MultivariateAnalysis.MultivariateAnalysisSubsection.UMAP)
    multivariate_section = MultivariateAnalysis(df=get_test_df(), subsections=subsections)
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


def test_code_export_verbosity_0():
    multivariate_section = multivariate_analysis.MultivariateAnalysis(df=get_test_df(), verbosity=0)
    # Export code
    exported_cells = []
    multivariate_section.add_cells(exported_cells)
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    # Define expected code
    expected_code = ["multivariate_analysis(df=df)"]
    # Test code equivalence
    assert len(exported_code) == 1
    assert exported_code[0] == expected_code[0], "Exported code mismatch"


def test_code_export_verbosity_0_with_subsections():
    subsec = multivariate_analysis.MultivariateAnalysis.MultivariateAnalysisSubsection
    subsections = [subsec.ParallelCategories, subsec.PCA, subsec.ParallelCoordinates, subsec.PCA]
    if UMAP_AVAILABLE:
        subsections.append(subsec.UMAP)
    multivariate_section = multivariate_analysis.MultivariateAnalysis(
        df=get_test_df(), subsections=subsections, verbosity=0
    )

    # Export code
    exported_cells = []
    multivariate_section.add_cells(exported_cells)
    # Remove markdown and other cells and get code strings
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    if UMAP_AVAILABLE:
        # Define expected code
        expected_code = [
            "multivariate_analysis(df=df, subsections=["
            "MultivariateAnalysis.MultivariateAnalysisSubsection.ParallelCategories, "
            "MultivariateAnalysis.MultivariateAnalysisSubsection.PCA, "
            "MultivariateAnalysis.MultivariateAnalysisSubsection.ParallelCoordinates, "
            "MultivariateAnalysis.MultivariateAnalysisSubsection.PCA, "
            "MultivariateAnalysis.MultivariateAnalysisSubsection.UMAP])"
        ]
    else:
        expected_code = [
            "multivariate_analysis(df=df, subsections=["
            "MultivariateAnalysis.MultivariateAnalysisSubsection.ParallelCategories, "
            "MultivariateAnalysis.MultivariateAnalysisSubsection.PCA, "
            "MultivariateAnalysis.MultivariateAnalysisSubsection.ParallelCoordinates, "
            "MultivariateAnalysis.MultivariateAnalysisSubsection.PCA])"
        ]
    # Test code equivalence
    assert len(exported_code) == 1
    assert exported_code[0] == expected_code[0], "Exported code mismatch"


def test_code_export_verbosity_1_all_cols_valid():
    all_numeric_df = pd.DataFrame(
        data=[[1.1, 1, -2], [2.2, 2, -5.3], [3.3, 3, 4]], columns=["col1", "col2", "col3"]
    )

    subsec = multivariate_analysis.MultivariateAnalysis.MultivariateAnalysisSubsection
    multivariate_section = multivariate_analysis.MultivariateAnalysis(
        df=all_numeric_df, subsections=[subsec.PCA, subsec.ParallelCategories], verbosity=1
    )

    exported_cells = []
    multivariate_section.add_cells(exported_cells)
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]

    expected_code = [
        "pca_first_vs_second(df=df)",
        "pca_explained_variance(df=df)",
        "parallel_categories(df=df)",
    ]

    assert len(exported_code) == len(expected_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"


def test_generated_code_verobsity_1():
    multivariate_section = multivariate_analysis.MultivariateAnalysis(df=get_test_df(), verbosity=1)

    exported_cells = []
    multivariate_section.add_cells(exported_cells)
    exported_code = [cell["source"] for cell in exported_cells if cell["cell_type"] == "code"]
    if UMAP_AVAILABLE:
        expected_code = [
            "pca_first_vs_second(df=df, columns=['A', 'C', 'D'])",
            "pca_explained_variance(df=df, columns=['A', 'C', 'D'])",
            code_dedent(
                """
                plot_umap(
                    df=df,
                    n_neighbors=15,
                    min_dist=0.1,
                    metric='euclidean',
                    columns=['A', 'C', 'D'],
                )"""
            ),
            "parallel_coordinates(df=df)",
            "parallel_categories(df=df)",
        ]
    else:
        expected_code = [
            "pca_first_vs_second(df=df, columns=['A', 'C', 'D'])",
            "pca_explained_variance(df=df, columns=['A', 'C', 'D'])",
            "parallel_coordinates(df=df)",
            "parallel_categories(df=df)",
        ]

    assert len(exported_code) == len(expected_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"


def test_generated_code_verobsity_2():
    multivariate_section = multivariate_analysis.MultivariateAnalysis(df=get_test_df(), verbosity=2)

    multivariate_cells = []
    multivariate_section.add_cells(multivariate_cells)
    exported_code = [cell["source"] for cell in multivariate_cells if cell["cell_type"] == "code"]
    expected_code = [
        "\n\n".join(
            (
                get_code(multivariate_analysis.PCA.pca_first_vs_second),
                "pca_first_vs_second(df=df, columns=['A', 'C', 'D'])",
            )
        ),
        "\n\n".join(
            (
                get_code(multivariate_analysis.PCA.pca_explained_variance),
                "pca_explained_variance(df=df, columns=['A', 'C', 'D'])",
            )
        ),
        "\n\n".join(
            (
                get_code(utils.discrete_colorscale),
                get_code(multivariate_analysis.ParallelCoordinates.parallel_coordinates),
                "parallel_coordinates(df=df)",
            )
        ),
        "\n\n".join(
            (
                get_code(utils.discrete_colorscale),
                get_code(multivariate_analysis.ParallelCategories.parallel_categories),
                "parallel_categories(df=df)",
            )
        ),
    ]
    if UMAP_AVAILABLE:
        expected_code.insert(
            2,
            (
                get_code(multivariate_analysis.UMAP.plot_umap)
                + "\n\n"
                + code_dedent(
                    """
                    plot_umap(
                        df=df,
                        n_neighbors=15,
                        min_dist=0.1,
                        metric='euclidean',
                        columns=['A', 'C', 'D'],
                    )"""
                )
            ),
        )

    assert len(exported_code) == len(expected_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"


def test_verbosity_1_non_categorical_col():
    random_array = np.random.randint(low=1, high=40, size=(100, 3))
    random_df = pd.DataFrame(data=random_array, columns=["integral", "floating", "cat"])
    random_df = random_df.astype({"integral": int, "floating": float, "cat": "category"})
    subsec = multivariate_analysis.MultivariateAnalysis.MultivariateAnalysisSubsection
    multivariate_section = multivariate_analysis.MultivariateAnalysis(
        df=random_df, subsections=[subsec.ParallelCategories], verbosity=1
    )

    multivariate_cells = []
    multivariate_section.add_cells(multivariate_cells)
    exported_code = [cell["source"] for cell in multivariate_cells if cell["cell_type"] == "code"]

    expected_code = ["parallel_categories(df=df, columns=[])"]

    assert len(exported_code) == len(expected_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"


def test_verbosity_0_different_subsection_verbosities():
    subsections = [
        MultivariateAnalysis.MultivariateAnalysisSubsection.PCA,
        MultivariateAnalysis.MultivariateAnalysisSubsection.PCA,
        MultivariateAnalysis.MultivariateAnalysisSubsection.ParallelCategories,
        MultivariateAnalysis.MultivariateAnalysisSubsection.ParallelCoordinates,
    ]
    if UMAP_AVAILABLE:
        subsections.insert(2, MultivariateAnalysis.MultivariateAnalysisSubsection.UMAP)
    multivariate_section = MultivariateAnalysis(
        df=get_test_df(),
        verbosity=0,
        subsections=subsections,
        verbosity_parallel_categories=1,
        verbosity_parallel_coordinates=2,
    )

    multivariate_cells = []
    multivariate_section.add_cells(multivariate_cells)
    exported_code = [cell["source"] for cell in multivariate_cells if cell["cell_type"] == "code"]
    expected_subsections = [
        "MultivariateAnalysis.MultivariateAnalysisSubsection.PCA",
        "MultivariateAnalysis.MultivariateAnalysisSubsection.PCA",
    ]
    if UMAP_AVAILABLE:
        expected_subsections.append("MultivariateAnalysis.MultivariateAnalysisSubsection.UMAP")
    expected_subsections_str = ", ".join(expected_subsections)
    expected_code = [
        "multivariate_analysis(df=df, " f"subsections=[{expected_subsections_str}])",
        "parallel_categories(df=df)",
        "\n\n".join(
            (
                get_code(utils.discrete_colorscale),
                get_code(multivariate_analysis.ParallelCoordinates.parallel_coordinates),
                "parallel_coordinates(df=df)",
            )
        ),
    ]

    assert len(exported_code) == len(expected_code)
    for expected_line, exported_line in zip(expected_code, exported_code):
        assert expected_line == exported_line, "Exported code mismatch"


def test_imports_verbosity_0():
    multivariate_section = MultivariateAnalysis(df=get_test_df(), verbosity=0)

    exported_imports = multivariate_section.required_imports()
    expected_imports = [
        "from edvart.report_sections.multivariate_analysis import MultivariateAnalysis\n"
        "multivariate_analysis = MultivariateAnalysis.multivariate_analysis"
    ]

    assert isinstance(exported_imports, list)
    assert len(expected_imports) == len(exported_imports)
    for expected_import, exported_import in zip(expected_imports, exported_imports):
        assert expected_import == exported_import, "Exported import mismatch"


def test_imports_verbosity_1():
    multivariate_section = MultivariateAnalysis(df=get_test_df(), verbosity=1)

    exported_imports = multivariate_section.required_imports()
    expected_imports = list(
        set().union(*[s.required_imports() for s in multivariate_section.subsections])
    )

    assert isinstance(exported_imports, list)
    assert len(expected_imports) == len(exported_imports)
    for expected_import, exported_import in zip(expected_imports, exported_imports):
        assert expected_import == exported_import, "Exported import mismatch"


def test_imports_verbosity_2():
    multivariate_section = MultivariateAnalysis(df=get_test_df(), verbosity=2)

    exported_imports = multivariate_section.required_imports()
    expected_imports = list(
        set().union(*[s.required_imports() for s in multivariate_section.subsections])
    )

    assert isinstance(exported_imports, list)
    assert len(expected_imports) == len(exported_imports)
    for expected_import, exported_import in zip(expected_imports, exported_imports):
        assert expected_import == exported_import, "Exported import mismatch"


def test_imports_verbosity_0_different_subsection_verbosities():
    subsections = [
        MultivariateAnalysis.MultivariateAnalysisSubsection.PCA,
        MultivariateAnalysis.MultivariateAnalysisSubsection.PCA,
        MultivariateAnalysis.MultivariateAnalysisSubsection.ParallelCategories,
        MultivariateAnalysis.MultivariateAnalysisSubsection.ParallelCoordinates,
    ]
    if UMAP_AVAILABLE:
        subsections.insert(3, MultivariateAnalysis.MultivariateAnalysisSubsection.UMAP)
    multivariate_section = MultivariateAnalysis(
        df=get_test_df(),
        verbosity=0,
        subsections=subsections,
        verbosity_parallel_categories=1,
        verbosity_parallel_coordinates=2,
    )

    exported_imports = multivariate_section.required_imports()

    expected_imports = {
        "from edvart.report_sections.multivariate_analysis import MultivariateAnalysis\n"
        "multivariate_analysis = MultivariateAnalysis.multivariate_analysis"
    }
    for s in multivariate_section.subsections:
        if s.verbosity > 0:
            expected_imports.update(s.required_imports())

    assert isinstance(exported_imports, list)
    assert set(exported_imports) == set(expected_imports)


def test_show():
    df = get_test_df()
    multivariate_section = MultivariateAnalysis(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with redirect_stdout(None):
            multivariate_section.show(df)
