"""Module for loading example datasets."""
import os

import pandas as pd

_DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../example-datasets"))


def dataset_auto() -> pd.DataFrame:
    """
    Returns a dataset containing information on the technical specifications of cars.
    The dataset contains 398 rows and 9 columns. There are no missing values.

    Source: https://www.kaggle.com/datasets/uciml/autompg-dataset
    Unmodified.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(os.path.join(_DATASETS_DIR, "auto.csv"))


def dataset_meteorite_landings() -> pd.DataFrame:
    """
    Returns a dataset that includes the location, mass, composition, and fall year
    for over 45,000 meteorites that have struck our planet.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(os.path.join(_DATASETS_DIR, "Meteorite_Landings.csv"))


def dataset_titanic() -> pd.DataFrame:
    """
    Returns a dataset that contains data for 887 of the real Titanic passengers.
    Each row represents one person. The columns describe different attributes about the person
    including whether they survived (S), their age (A), their passenger-class (C), their sex (G)
    and the fare they paid (X).

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(os.path.join(_DATASETS_DIR, "titanic.csv"))


def dataset_global_temp() -> pd.DataFrame:
    """
    Returns a time-series dataset containing monthly deviations from mean global average temperature
    from 1880 until 2016 according to two methodologies: GCAG and GISTEMP.
    """
    return pd.read_csv(
        os.path.join(_DATASETS_DIR, "global_temp.csv"),
        index_col="Date",
        parse_dates=["Date"],
    )


def dataset_pollution() -> pd.DataFrame:
    """
    Returns a time-series dataset containing hourly weather and pollution data from 2010 until 2014
    from Beijing, China. There are 43800 rows and 8 columns.
    """
    return pd.read_csv(
        os.path.join(_DATASETS_DIR, "pollution.csv"),
        index_col=["date"],
        parse_dates=["date"],
    )
