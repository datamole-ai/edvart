"""EDVART package."""

import logging

import pkg_resources

from edvart import example_datasets
from edvart.report import Report
from edvart.report import Report as create_report
from edvart.report import TimeseriesReport
from edvart.report_sections.dataset_overview import Overview

logging.basicConfig(level=logging.INFO)

try:
    __version__ = pkg_resources.get_distribution("edvart").version
except pkg_resources.DistributionNotFound:
    __version__ = "dev"
