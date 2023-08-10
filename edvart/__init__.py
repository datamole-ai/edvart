"""EDVART package."""

import logging
from importlib.metadata import PackageNotFoundError, version

from edvart import example_datasets
from edvart.report import DefaultReport, DefaultTimeseriesReport, Report, TimeseriesReport
from edvart.report_sections.dataset_overview import Overview
from edvart.report_sections.section_base import Verbosity

logging.basicConfig(level=logging.INFO)

try:
    __version__ = version("edvart")
except PackageNotFoundError:
    __version__ = "dev"
