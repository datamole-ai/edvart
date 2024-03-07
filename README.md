# Edvart

<p align="center">
    <a href="https://pypi.org/project/edvart">
        <img src="https://img.shields.io/pypi/pyversions/edvart.svg?color=%2334D058" alt="Supported Python versions">
    </a>
    <a href="https://pypi.org/project/edvart" target="_blank">
        <img src="https://img.shields.io/pypi/v/edvart?color=%2334D058&label=pypi%20package" alt="Package version">
    </a>
    <a href="https://pypi.org/project/edvart">
        <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/edvart.svg?label=PyPI%20downloads">
    </a>
    <a href="https://github.com/astral-sh/ruff">
        <img alt="Ruff", src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json">
    </a>
</p>

Edvart is an open-source Python library designed to simplify and streamline
your exploratory data analysis (EDA) process.

## Key Features
* **One-line Reports**: Generate a comprehensive set of pandas DataFrame
visualizations using a single Python statement.
Edvart supports:
    - Data overview,
    - Univariate analysis,
    - Bivariate analysis,
    - Multivariate analysis,
    - Grouped analysis,
    - Time series analysis.
* **Customizable Reports**: Produce, iterate, and style detailed reports
    in Jupyter notebooks and HTML formats.
* **Flexible API**: From high-level simplicity in a single line of code
    to detailed control, choose the API level that fits your needs.
* **Interactive Visualizations**: Many of the visualizations are interactive
    and can be used to explore the data in detail.

## One-line Report

![Edvart report demo](images/edvart-demo.gif)

## Installation

Edvart is available on PyPI and can be installed using pip:

```bash
pip install edvart
```

## Usage


See the notebook
[examples/report-example.ipynb](https://nbviewer.org/github/datamole-ai/edvart/blob/main/examples/report-example.ipynb)
for an example report on a tabular dataset or
[examples/time-series-report-example.ipynb](https://nbviewer.org/github/datamole-ai/edvart/blob/main/examples/time-series-report-example.ipynb)
for an example report on a time-series dataset.

See the [Usage section](https://datamole-ai.github.io/edvart/usage.html) of the documentation
for more information.

### Creating a Default Report

```python
import edvart

# Load a dataset to a pandas DataFrame
dataset = edvart.example_datasets.dataset_titanic()
# Create a default report
report = edvart.DefaultReport(dataset)
# Show the report in the current Jupyter notebook
report.show()
# Export the report to an HTML file
report.export_html("report.html")
# Export the code generating the report to a Jupyter notebook
report.export_notebook("report.ipynb")
```

## User Documentation

The user documentation is available at https://datamole-ai.github.io/edvart/.

## License

Edvart is licensed under the [MIT
license](https://opensource.org/license/mit/). See the LICENSE file for more
details.

## Contact
Edvart has a [Gitter room](https://app.gitter.im/#/room/#edvart:gitter.im)
for development-related and general discussions.

## How to Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md).
