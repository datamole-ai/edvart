from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import nbformat.v4 as nbfv4
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from edvart.data_types import is_numeric
from edvart.decorators import check_index_time_ascending
from edvart.report_sections.code_string_formatting import get_code
from edvart.report_sections.section_base import Section, Verbosity


class FourierTransform(Section):
    """Generates the Discrete Fourier Transform spectrum plot subsection.

    Parameters
    ----------
    sampling_rate : int
        The time series will be considered as samples from a lower-frequency at this rate, i.e.
        frequencies in multiples of (1 / sampling rate) will be analyzed.
    verbosity : Verbosity (default = Verbosity.LOW)
        Verbosity of the generated code in the exported notebook.
    columns : List[str], optional
        List of columns to analyze. Only numeric column can be analyzed.
        All numeric columns are analyzed by default.
    """

    def __init__(
        self,
        sampling_rate: int,
        verbosity: Verbosity = Verbosity.LOW,
        columns: Optional[List[str]] = None,
    ):
        if sampling_rate <= 0:
            raise ValueError(f"sampling_rate has to be a positive integer, not {sampling_rate}")
        super().__init__(verbosity, columns)
        self.sampling_rate = sampling_rate

    @property
    def name(self) -> str:
        return "Fourier Transform"

    def add_cells(self, cells: List[Dict[str, Any]], df: pd.DataFrame) -> None:
        """Adds cells to the list of cells.

        Cells can be either code cells or markdown cells.

        Parameters
        ----------
        cells : List[Dict[str, Any]]
            List of generated notebook cells which are represented as dictionaries
        df: pd.DataFrame
            Data for which to add the cells.
        """
        section_header = nbfv4.new_markdown_cell(self.get_title(section_level=2))
        cells.append(section_header)
        default_call = f"show_fourier_transform(df=df, sampling_rate={self.sampling_rate}"
        if self.columns is not None:
            default_call += f", columns={self.columns}"
        default_call += ")"

        if self.verbosity <= Verbosity.MEDIUM:
            code = default_call
        else:
            code = get_code(show_fourier_transform) + "\n\n" + default_call

        cells.append(nbfv4.new_code_cell(code))

    def show(self, df: pd.DataFrame) -> None:
        """Generates Fourier transform spectrum plot(s) in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output
        """
        display(Markdown(self.get_title(section_level=2)))
        show_fourier_transform(df=df, sampling_rate=self.sampling_rate, columns=self.columns)

    def required_imports(self) -> List[str]:
        """Returns a list of imports to be put at the top of a generated notebook.

        Returns
        -------
        List[str]
            List of import strings to be added at the top of the generated notebook,
            e.g. ["import pandas as pd", "import numpy as np"].
        """
        if self.verbosity <= Verbosity.MEDIUM:
            return [
                """from edvart.report_sections.timeseries_analysis.fourier_transform import (
                    FourierTransform, fourier_transform
                )"""
            ]
        return [
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "from edvart.data_types import is_numeric",
        ]


@check_index_time_ascending
def show_fourier_transform(
    df: pd.DataFrame,
    sampling_rate: int,
    columns: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (15, 6),
    log: bool = False,
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
) -> None:
    """Generate Discrete Fourier Transform frequency vs amplitude plot.

    Parameters
    ----------
    df : pd.DataFrame
        Data to analyze.
    sampling_rate : int
        The time series will be considered as samples from a lower-frequency at this rate, i.e.
        frequencies in multiples of (1 / sampling rate) will be analyzed.
    columns : List[str], optional
        List of columns to analyze. Only numeric column can be analyzed.
        All numeric columns are analyzed by default.
    figsize : Tuple[float, float] (default = (15, 6))
        Size of frequency-amplitude plot.
    log : bool (default = False)
        Whether to plot amplitude in logarithmic scale -- in decibel.
    freq_min : float, optional
        Lowest frequency to show in the plot. All computed frequencies are shown by default.
    freq_max : float, optional
        Highest frequency to show in the plot. All computed frequencies are shown by default.

    Raises
    ------
    ValueError
        If the input data is not indexed by time in ascending order.
    """
    if sampling_rate <= 0:
        raise ValueError(f"sampling_rate has to be a positive integer, not {sampling_rate}")
    if columns is None:
        columns = [col for col in df.columns if is_numeric(df[col])]
    else:
        for col in columns:
            if not is_numeric(df[col]):
                raise ValueError(f"Cannot perform Fourier transform for non-numeric column `{col}`")
    index_freq = pd.infer_freq(df.index) or ""
    for col in columns:
        # FFT requires samples at regular intervals
        df_col = df[col].interpolate(method="time")
        df_col_centered = df_col - df_col.mean()
        fft_result = np.fft.fft(df_col_centered)

        amplitude = np.abs(fft_result) * 2 / len(df)
        fft_freq = np.fft.fftfreq(len(amplitude), 1.0 / sampling_rate)
        idx_pos_freq = fft_freq > 0
        fft_freq, amplitude = fft_freq[idx_pos_freq], amplitude[idx_pos_freq]

        y = 10 * np.log10(amplitude) if log else amplitude
        _fig, ax = plt.subplots(figsize=figsize)
        ax.stem(fft_freq, y, use_line_collection=True, markerfmt="")
        ax.set_xlabel(f"Frequency [1 / {sampling_rate}{index_freq}]")
        ax.set_ylabel("Amplitude" + (" [dB]" if log else ""))
        ax.set_xlim(freq_min, freq_max)
        display(Markdown(f"---\n### {col}"))
        plt.show()
