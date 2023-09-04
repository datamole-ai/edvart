# Edvart

Edvart is an open-source Python library designed to simplify and streamline
your exploratory data analysis (EDA) process.

## Key features
* **One-line reports**: Generate a comprehensive set of visualizations
in a single statement from pandas DataFrames.
Edvart supports:
    - Data overview,
    - Univariate analysis,
    - Bivariate analysis,
    - Multivariate analysis,
    - Grouped analysis
    - Time series analysis.
* **Customizable Reports**: Produce, iterate, and style detailed reports
    in Jupyter notebooks and HTML formats.
* **Flexible API**: From high-level simplicity in a single line of code
    to detailed control, choose the API level that fits your needs.
* **Interactive Visualizations**: Many of the visualizations are interactive
    and can be used to explore the data in detail.

## Installation

Edvart is available on PyPI and can be installed using pip:

```bash
pip install edvart
```

## Usage

### Creating a default report

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

See the notebook [api-example.ipynb](api-example.ipynb) for usage examples.

## User documentation

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
