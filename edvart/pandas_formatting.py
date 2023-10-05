from typing import Any, Dict, List, Union

import pandas as pd
from IPython.display import display
from pandas.io.formats.style import Styler


def series_to_frame(series: pd.Series, index_name: str, column_name: str) -> pd.DataFrame:
    """Converts a pandas.Series to a pandas.DataFrame by putting the series index into a separate
    column.

    Parameters
    ---
    series : pd.Series
        Input series
    index_name : str
        Name of the new column into which the series index will be put
    column_name : str
        Name of the series values column

    Returns
    ---
    pd.DataFrame
        Dataframe with two columns index_name and column_name with values of series.index and
        series.values respectively
    """
    return series.rename_axis(index=index_name).to_frame(name=column_name).reset_index()


def render_dictionary(dictionary: Dict[str, Any]) -> None:
    """
    Converts a dictionary to a dataframe and renders that dataframe in the report notebook.

    Parameters
    ---
    dictionary: Dict['str', Any]
        Dictionary to be rendered
    """
    dictionary = {key: str(value) for key, value in dictionary.items()}

    display(pd.DataFrame.from_dict(dictionary, orient="index", columns=[""]))


def dict_to_html(dictionary: Dict[str, Any]) -> str:
    """Converts a dictionary to a dataframe in HTML string form.

    Parameters
    ----------
    dictionary : Dict['str', Any]
        DictDictionary to be converted
    """
    dictionary = {key: str(value) for key, value in dictionary.items()}

    return pd.DataFrame.from_dict(dictionary, orient="index", columns=[""]).to_html()


def add_html_heading(html: str, heading: str, heading_level: int = 2) -> str:
    """
    Adds a heading to an HTML string with the specified text and heading level.

    Parameters
    ---
    html : str
        HTML string to which to add heading
    heading : str
        Text of the heading
    heading_level : int
        Level of the heading

    Returns
    ---
    str
        HTML string with heading added
    """
    return f"<h{heading_level}> {heading} </h{heading_level}>" + html


def subcells_html(elements: List[List[str]]) -> str:
    """
    Returns HTML table in string format according to the elements matrix.

    Parameters
    ---
    elements : List[List[str]]
        Elements which should be rendered in table cells, outer list represents rows,
        inner list represents columns, elements themselves should be HTML strings

    Returns
    ---
    str
        Table in HTML string ready to be rendered for example by IPython.display.display_html
    """
    default_cell_css = """
        "
        text-align: left;
        vertical-align: text-top;
        align-content: center;
        "
    """

    i = 0
    html = '<table width="100%">'
    for row in elements:
        html += "<tr>"
        for element in row:
            if i == 0:
                html += f"<td style={default_cell_css}> {element} </td>"
                i = 1
            else:
                html += f"<td style={default_cell_css}> {element} </td>"
                i = 0
        html += "</tr>"
    html += "</table>"

    return html


def format_number(
    number: Union[int, float], decimal_places: int = 2, thousand_separator: str = ""
) -> str:
    """
    Formats a number by truncating decimal places (if it is a float) and optionally adds thousand
    separators.

    Parameters
    ---
    number : Union[int, float]
        Number to be converted to string
    decimal_places : int
        Number of decimal places in case of float
    thousand_separator : str
        Character or string with which thousands should be separated

    Returns
    ---
    str
        Formatted number in a string representation
    """
    if float(number).is_integer():
        formatted = f"{number:,}"
    else:
        formatted = f"{number:,.0{decimal_places}f}"
    return formatted.replace(",", thousand_separator)


def hide_index(df: pd.DataFrame) -> Styler:
    """
    Hides the index of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where the index should be hidden.

    Returns
    -------
    Styler
        Styler object with the index hidden.
    """
    return df.style.hide(axis="index")
