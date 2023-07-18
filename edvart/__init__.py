"""EDVART package."""

import logging
from importlib.metadata import PackageNotFoundError, version

from edvart import example_datasets
from edvart.report import Report
from edvart.report import Report as create_report
from edvart.report import TimeseriesReport
from edvart.report_sections.dataset_overview import Overview

logging.basicConfig(level=logging.INFO)

try:
    __version__ = version("edvart")
except PackageNotFoundError:
    __version__ = "dev"
