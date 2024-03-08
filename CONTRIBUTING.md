# Contributing to EDVART

## Suggesting changes
1. Create an [issue](https://github.com/datamole-ai/edvart/issues) describing the change you want to make.

## General workflow

### Environment setup
EDVART uses [Poetry](https://python-poetry.org/) for managing dependencies.
Follow the instructions on the Poetry website to install it.
We recommend [pyenv](https://github.com/pyenv/pyenv)
([installer](https://github.com/pyenv/pyenv-installer)) for managing Python versions.
```bash
# Install Python 3.11
pyenv install 3.11

# Use pyenv's Python 3.11 for the current folder
pyenv local 3.11

# Create virtual environment (install all optional dependencies)
poetry install --extras all

# Activate Poetry virtual environment in the current shell
poetry shell
```

You can also use `poetry run` to run commands in the virtual environment without activating it in the current shell (via `poetry shell`).

### Implement a new type of analysis
Start by implementing a prototype of the analysis in a Jupyter notebook.
See the [example prototype](prototypes/example-data-types.ipynb) for an example prototype.

You can test your implementation by loading the example datasets using `edvart.example_datasets.dataset_*`
and running your analysis on the dataset.

### Implement an API for the analysis

The basic idea is that the code of your analysis will get exported by the user to a new notebook with varying levels of detail configurable by the user.

Your analysis will be its own section in the exported report notebook.
To implement the API, create a new Python module `edvart/report_sections/your_analysis.py` containing a class that implements your analysis.
Your class should be a subclass of `edvart.report_sections.section_base.Section` and implement all of the abstract methods.
See the documentation of `edvart.report_sections.section_base.Section` for details on the methods you need to implement.

For a reference section implementation, see `edvart/report_sections/univariate_analysis.py`.
If your analysis section contains subsections and you want the user to be able to configure which subsections are used,
use the `edvart/report_sections/section_base.ReportSection` base class instead.
See `edvart/report_sections/dataset_overview.py` for reference.

Once you have implemented the class, expose the functionality to the user by integrating
your class into the class that the user will use directly, namely `edvart.Report` or `edvart.TimeseriesReport`.

Add a method to the `ReportBase`-based class that adds an instance of your section analysis class
to the list of sections of the report `self.sections` and prefix it with `add_`.
If you want your section to be added to the report by default, call the newly created method in `__init__`
of `DefaultReport` or `DefaultTimeseriesReport`.

### Test the newly implemented API
Create unit tests by creating a Python script in the folder `tests` prefixed with `test_`.
The script should contain functions also prefixed with `test_` that make assertions.
Test whether your class exports the correct code for the given verbosity. See the `tests` folder for reference.

### Remove the prototype
The PR implementing the visualization to the API should also remove the prototype.
If there are multiple PRs implementing the same visualization, the prototype should be removed in the last PR.

### Modify documentation

If you add a new section, add the section description into `docs/sections.rst`

## Pull Requests & Git

* Split your work into separate and atomic pull requests. Put any
  non-obvious reasoning behind any change to the pull request description.
  Separate “preparatory” changes and modifications from new features &
  improvements.
* The pull requests are squashed when merged. The PR title is used as the commit title.
  The PR description is used as the commit description.
* Use conventional commit messages in the PR title and description.
  See [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
  Usage of conventional commit PR titles and descriptions is enforced by the CI pipeline.
* Prefer adding new commits over amending existing ones during the review process.
  The latter makes it harder to review changes and track down modifications.


## Code style

* The line length is limited to 100 characters in Python code,
except if it would make the code less readable.
* `ruff` is used for formatting and linting Python code.
The following commands can be used to properly format the code and check
for linting errors with automatic fixing:
```bash
poetry run ruff format .
poetry run ruff check . --fix
```
The following command can be used to check if the code is properly
formatted and check for linting errors:
```bash
poetry run ruff format --check .
poetry run ruff check .
```

All of the above code style requirements are enforced by the CI pipeline.
