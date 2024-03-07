from inspect import getsource
from itertools import dropwhile
from textwrap import dedent
from typing import Any


def total_dedent(input_string: str) -> str:
    """Removes all white space from the beginning of each line.

    Parameters
    ----------
    input_string : str
        Input string with lines.

    Returns
    -------
    str
        input_string with no whitespace at the beginning of each line.
    """
    input_string = input_string.strip()
    lstripped_lines = [line.strip() for line in input_string.split("\n")]
    return "\n".join(lstripped_lines)


def code_dedent(input_string: str) -> str:
    """Removes all white spaces from each line that is common for all lines.

    Parameters
    ----------
    input_string : str
        Input string with lines.

    Returns
    -------
    str
        input_string with common leading whitespace removed from each line.
    """

    return dedent(input_string.strip("\n"))


def dedecorate(input_string: str) -> str:
    """Removes all decorators from the beginning of a function source.

    Parameters
    ----------
    input_string : str
        Input function source.

    Returns
    -------
    str
        input_string with beginning lines starting with '@' removed.
    """
    lines = input_string.splitlines()
    filtered_lines = dropwhile(lambda line_: line_.lstrip().startswith("@"), lines)

    return "\n".join(filtered_lines)


def get_code(code_object: Any) -> str:
    """Gets the source code of code object and formats it.

    Parameters
    ----------
    code_object : Any
        Object from which to extract code (function, method)

    Returns
    -------
    str
        Formatted code
    """
    code = getsource(code_object)
    code = dedecorate(code)
    # Dedent code
    code = code_dedent(code)

    return code
