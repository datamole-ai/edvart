# Edvart

Exploratory Data Analysis (EDA) is the initial task a data scientist or data
analyst undertakes when they obtain new data. EDA refers to the critical
process of conducting preliminary investigations on data to uncover patterns,
spot anomalies, test hypotheses, and verify assumptions with the help of
summary statistics and graphical representations.

The Effective Data Visualization and Reporting Tool (Edvart for short) is a
tool that generates a report in the form of a Jupyter notebook, containing
various analyses of the input data.

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

Edvart is licensed under the [MIT
license](https://opensource.org/license/mit/). See the LICENSE file for more
details.

## How to Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Markdown Notebooks

Jupyter notebooks are stored in markdown format in the repository. To convert a
Markdown notebook to a Jupyter notebook, use
[jupytext](https://github.com/mwouts/jupytext). `jupytext` is included in the
development dependencies of this project. For example convert `api-example.md`
to `ipynb` Jupyter notebook format:

```bash
jupytext --to ipynb api-example.md
```

To convert an `ipynb` notebook to markdown:

```bash
jupytext --to md api-example.ipynb
```
