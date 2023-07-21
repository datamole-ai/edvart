# Edvart

Exploratory Data Analysis (EDA) is a very initial task a data scientist
or data analyst does when he reaches new data.
EDA refers to the critical process of performing
initial investigations on data to discover patterns, to spot
anomalies, to test hypothesis and to check assumptions with the help
of summary statistics and graphical representations.

Effective data visualization and reporting tool (edvart for short) is a tool that
generates a report in the form of a Jupyter notebook that contains various
analyses of the data passed in.

## Installation
Edvart is available on PyPI and can be installed using pip:
```bash
pip install edvart
```

## Usage
See the notebook `api-example.md` for usage examples.

## User documentation

The user documentation is available at https://datamole-ai.github.io/edvart/.

## License
edvart is licensed under the [MIT license](https://opensource.org/license/mit/). See the LICENSE file for more details.

## How to contribute
See [CONTRIBUTING.md](CONTRIBUTING.md).

## Markdown notebooks
Jupyter notebooks are stored in markdown format in the repository. To convert a Markdown notebook to a Jupyter notebook, use `jupytext`. `jupytext` is included in the development dependencies of this project.
For example convert `api-example.md` to `ipynb` Jupyter notebook format:
```bash
jupytext --to ipynb api-example.md
```

To convert an `ipynb` notebook to markdown:
```bash
jupytext --to md api-example.ipynb
```
