from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import nbformat.v4 as nbfv4
import numpy as np
import numpy.typing as npt
import pandas as pd
from IPython.display import Markdown, display
from scipy import signal

from edvart.data_types import is_numeric
from edvart.decorators import check_index_time_ascending
from edvart.report_sections.code_string_formatting import get_code, total_dedent
from edvart.report_sections.section_base import Section, Verbosity


class ShortTimeFT(Section):
    """Generates Short-time discrete Fourier transform spectrogram plot subsection.

    Parameters
    ----------
    sampling_rate : int
        The time series will be considered as samples from a lower-frequency at this rate, i.e.
        frequencies in multiples of (1 / sampling rate) will be analyzed.
    window_size : int
        Size of window to perform DFT on to obtain Short-time Fourier transform.
    verbosity : Verbosity (default = Verbosity.LOW)
        Verbosity of the generated code in the exported notebook.
    columns : List[str], optional
        List of columns to analyze. Only numeric column can be analyzed.
        All numeric columns are analyzed by default.
    """

    def __init__(
        self,
        sampling_rate: int,
        window_size: int,
        verbosity: Verbosity = Verbosity.LOW,
        columns: Optional[List[str]] = None,
    ):
        if sampling_rate <= 0:
            raise ValueError(f"sampling_rate has to be a positive integer, not {sampling_rate}")
        if window_size <= 0:
            raise ValueError(f"window_size has to a positive integer, not {window_size}")
        super().__init__(verbosity, columns)
        self.sampling_rate = sampling_rate
        self.window_size = window_size

    @property
    def name(self) -> str:
        return "Short-time Fourier Transform"

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
                total_dedent(
                    """
                    from edvart.report_sections.timeseries_analysis import ShortTimeFT
                    short_time_ft = ShortTimeFT.short_time_ft
                    """
                )
            ]
        return [
            "import numpy.typing as npt",
            "from IPython.display import display, Markdown",
            "from edvart.data_types import is_numeric",
            "import matplotlib.pyplot as plt",
            "from scipy import signal",
        ]

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
        default_call = (
            f"show_short_time_ft(df=df, sampling_rate={self.sampling_rate}"
            f", window_size={self.window_size}"
        )
        if self.columns is not None:
            default_call += f", columns={self.columns}"
        default_call += ")"

        if self.verbosity <= Verbosity.MEDIUM:
            code = default_call
        else:
            code = get_code(show_short_time_ft) + "\n\n" + default_call

        cells.append(nbfv4.new_code_cell(code))

    def show(self, df: pd.DataFrame) -> None:
        """Generates Short-time Fourier transform spectrogram in the calling notebook.

        Parameters
        ----------
        df : pd.DataFrame
            Data based on which to generate the cell output
        """
        display(Markdown(self.get_title(section_level=2)))
        show_short_time_ft(
            df=df,
            sampling_rate=self.sampling_rate,
            window_size=self.window_size,
            columns=self.columns,
        )


@check_index_time_ascending
def show_short_time_ft(
    df: pd.DataFrame,
    sampling_rate: int,
    window_size: int,
    columns: Optional[List[str]] = None,
    overlap: Optional[int] = None,
    log: bool = True,
    window: Union[str, Tuple, npt.ArrayLike] = "hamming",
    scaling: str = "spectrum",
    figsize: Tuple[float, float] = (20, 7),
    colormap: Any = "viridis",
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
) -> None:
    """Generates Short-time discrete Fourier transform spectrogram plot.

    Parameters
    ----------
    df : pd.DataFrame
        Data to analyze.
    sampling_rate : int
        The time series will be considered as samples from a lower-frequency at this rate, i.e.
        frequencies in multiples of (1 / sampling rate) will be analyzed.
    window_size : int
        Size of window to perform DFT on to obtain Short-time Fourier transform.
    columns : List[str], optional
        List of columns to analyze. Only numeric column can be analyzed.
        All numeric columns are analyzed by default.
    overlap : int, optional
        How many samples adjacent windows overlap by. Default `window_size // 8`.
    log : bool (default = True)
        Whether to color plot according by linear-scale amplitude or log-scale (in decibel).
    window : str (default = "hann")
        Type of weighting of individual samples in a window. If string or tuple, it is passed
        to `scipy.signal.get_window`. If array-like, each term is weight for the corresponding
        sample within the windows
    scaling : str (default = "density")
        Selects between computing the power spectral density ("density") with units of V**2/Hz
        and computing the power spectrum ("spectrum") with units of V**2,
        if input values are measured in V and sampling_rate is measured in Hz.
    figsize : Tuple[float, float] (default = (20, 7))
        Size of generated spectral plot figure.
    colormap : Any
        Any seaborn-compatible colormap.
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
        raise ValueError(f"Sampling rate has to be a positive integer, not {sampling_rate}")
    if window_size <= 0:
        raise ValueError(f"window_size has to a positive integer, not {window_size}")
    if columns is None:
        columns = [col for col in df.columns if is_numeric(df[col])]
    else:
        for col in columns:
            if not is_numeric(df[col]):
                raise ValueError(f"Cannot perform STFT for non-numeric column {col}")
    index_freq = pd.infer_freq(df.index.to_series()) or ""
    for col in columns:
        display(Markdown(f"---\n### {col}"))
        freqs, times, sx = signal.spectrogram(  # pylint: disable=invalid-name
            # interpolate to get samples at regular time intervals
            df[col].interpolate(method="time"),
            fs=sampling_rate,
            window=window,
            nperseg=window_size,
            # Overlap defaults to window_size // 8
            noverlap=overlap,
            scaling=scaling,
        )

        # Add small positive value to avoid 0 in log
        y = 10 * np.log10(sx + 1e-12) if log else sx

        _fig, ax = plt.subplots(figsize=figsize)
        ax.pcolormesh(times, freqs, y, cmap=colormap)

        ax.set_ylabel(f"Frequency [1/({sampling_rate}{index_freq})]")
        ax.set_xlabel("Time")
        ax.set_ylim(freq_min, freq_max)
        # Show times from index in xticks
        ax.set_xticklabels(
            df.index[list(map(lambda time: int(time * sampling_rate), ax.get_xticks()[:-1]))]
        )
        plt.show()
