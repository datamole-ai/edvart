from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from edvart.data_types import is_boolean, is_categorical, is_numeric

# Multiplier which makes plotly interactive plots (size in pixels) and
# matplotlib plots (size in inches) about the same size
_INCHES_TO_PIXELS = 64


def _is_color_col_categorical(column: pd.Series) -> bool:
    return is_categorical(column) or is_boolean(column) or not is_numeric(column)


# pylint: disable=too-many-locals, too-many-branches
def scatter_plot_2d(
    df: pd.DataFrame,
    x: Union[str, pd.Series, np.ndarray],
    y: Union[str, pd.Series, np.ndarray],
    color_col: Optional[str] = None,
    interactive: bool = True,
    figsize: Tuple[float, float] = (12, 12),
    opacity: float = 0.8,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    show_xticks: bool = False,
    show_yticks: bool = False,
    show_zerolines: bool = False,
    equal_scale_axes: bool = False,
) -> None:
    """Display a 2D scatter plot of x and y, with optional coloring of points by values in a column.

    Parameters
    ----------
    df : pd.DataFrame
        Data to plot.
    x : Union[str, pd.Series, np.ndarray]
        Name of column in df or flat array or series of x coordinates of plotted points.
    y : Union[str, pd.Series, np.ndarray]
        Name of column in df or flat array or series of y coordinates of plotted points.
    color_col : str, optional
        Name of column in df to color points in the plot by. Can be both numeric or categorical.
        By default, all points have the same color.
    interactive : bool (default = True)
        Whether to plot an interactive plot. The interactive plot also shows labels for each
        point on hover.
    figsize : Tuple[float, float] (default = (12, 12))
        Size of the resulting plot in inches.
    opacity : float (default = 0.8)
        Opacity of the points drawn in the scatter plot.
    xlabel: str, optional
        Label for the x axis. No label is displayed by default.
    ylabel: str, optional
        Label for the y axis. No label is displayed by default.
    show_xticks : bool (default = False)
        Whether to display ticks on the x axis.
    show_yticks : bool (default = False)
        Whether to display ticks on the y axis.
    show_zerolines : bool (default = False)
        Whether to display zero lines.
    equal_scale_axes : bool (default = False)
        Whether to make the x and y axes have the same scale.
    """
    if isinstance(x, str):
        x = df[x]
    if isinstance(y, str):
        y = df[y]
    plotting_func = _scatter_plot_2d_interactive if interactive else _scatter_plot_2d_noninteractive
    plotting_func(
        df=df,
        x=x,
        y=y,
        color_col=color_col,
        figsize=figsize,
        opacity=opacity,
        xlabel=xlabel,
        ylabel=ylabel,
        show_xticks=show_xticks,
        show_yticks=show_yticks,
        show_zerolines=show_zerolines,
        equal_scale_axes=equal_scale_axes,
    )


def _scatter_plot_2d_noninteractive(
    df: pd.DataFrame,
    x: Union[str, pd.Series, np.ndarray],
    y: Union[str, pd.Series, np.ndarray],
    color_col: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 12),
    opacity: float = 0.8,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    show_xticks: bool = False,
    show_yticks: bool = False,
    show_zerolines: bool = False,
    equal_scale_axes: bool = False,
) -> None:
    _fig, ax = plt.subplots(figsize=figsize)
    if color_col is not None:
        is_color_categorical = _is_color_col_categorical(df[color_col])
        if is_color_categorical:
            color_categorical = pd.Categorical(df[color_col])
            color_codes = color_categorical.codes
        else:
            color_codes = df[color_col]
        scatter = ax.scatter(x, y, c=color_codes, alpha=opacity)

        if is_color_categorical:
            legend_elements = scatter.legend_elements()
            ax.legend(legend_elements[0], color_categorical.categories, title=color_col)
        else:
            cbar = plt.colorbar(scatter)
            cbar.ax.set_ylabel(color_col)
    else:
        ax.scatter(x, y, alpha=opacity)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if not show_xticks:
        ax.set_xticks([])
    if not show_yticks:
        ax.set_yticks([])
    if not show_zerolines:
        ax.grid(False)
    if equal_scale_axes:
        ax.set_aspect("equal", "datalim")
    plt.show()


def _scatter_plot_2d_interactive(
    df: pd.DataFrame,
    x: Union[str, pd.Series, np.ndarray],
    y: Union[str, pd.Series, np.ndarray],
    color_col: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 12),
    opacity: float = 0.8,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    show_xticks: bool = False,
    show_yticks: bool = False,
    show_zerolines: bool = False,
    equal_scale_axes: bool = False,
) -> None:
    layout = go.Layout(
        width=figsize[0] * _INCHES_TO_PIXELS,
        height=figsize[1] * _INCHES_TO_PIXELS,
        xaxis=go.layout.XAxis(
            showgrid=False, showticklabels=show_xticks, zeroline=show_zerolines, title=xlabel
        ),
        yaxis=go.layout.YAxis(
            showgrid=False, showticklabels=show_yticks, zeroline=show_zerolines, title=ylabel
        ),
        legend=go.layout.Legend(title=f"<b>{color_col}</b>"),
    )
    if equal_scale_axes:
        layout.yaxis.scaleanchor = "x"
        layout.yaxis.scaleratio = 1
    fig = go.Figure(layout=layout)
    if color_col is not None:
        is_color_categorical = _is_color_col_categorical(df[color_col])
        if is_color_categorical:
            df = df.copy()
            x_name, y_name = "__edvart_scatter_x", "__edvart_scatter_y"
            df[x_name] = x
            df[y_name] = y
            for group_name, group in df.groupby(color_col):
                fig.add_trace(
                    go.Scatter(
                        x=group[x_name],
                        y=group[y_name],
                        mode="markers",
                        marker=dict(opacity=opacity),
                        name=group_name,
                        text=[
                            "</br>".join(
                                f"{col_name}: {df.loc[row, col_name]}"
                                for col_name in group.columns.drop([x_name, y_name])
                            )
                            for row in group.index
                        ],
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(
                        color=df[color_col], opacity=opacity, colorbar=dict(title=color_col)
                    ),
                    text=[
                        "</br>".join(
                            f"{col_name}: {df.loc[row, col_name]}" for col_name in df.columns
                        )
                        for row in df.index
                    ],
                ),
            )
    else:  # color_col is None
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(opacity=opacity),
                text=[
                    "</br>".join(f"{col_name}: {df.loc[row, col_name]}" for col_name in df.columns)
                    for row in df.index
                ],
            ),
        )

    fig.show()
