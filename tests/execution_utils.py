import base64
import pickle

import nbconvert
import nbformat
import pandas as pd

from edvart.report import ReportBase
from edvart.report_sections.code_string_formatting import code_dedent
from edvart.report_sections.section_base import Section


def check_section_executes(section: Section, df: pd.DataFrame) -> None:
    nb = nbformat.v4.new_notebook()
    section_code_cells = []
    section.add_cells(section_code_cells, df)

    buffer = pickle.dumps(df, fix_imports=False)
    buffer_base64 = base64.b85encode(buffer)

    unpickle_df = code_dedent(
        f"""
        import pickle
        import base64

        data = {buffer_base64}
        df = pickle.loads(base64.b85decode(data), fix_imports=False)"""
    )

    all_imports = [
        *ReportBase._DEFAULT_IMPORTS,
        *section.required_imports(),
    ]

    nb["cells"] = [
        nbformat.v4.new_code_cell("\n".join(all_imports)),
        nbformat.v4.new_code_cell(unpickle_df),
        *section_code_cells,
    ]
    preprocessor = nbconvert.preprocessors.ExecutePreprocessor(timeout=60)
    preprocessor.preprocess(nb)
