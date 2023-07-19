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
Start by implementing a prototype of the analysis in the `prototype.md` notebook.

For instructions on converting between markdown notebook format and Jupyter notebook format, see [README](README.md#markdown-notebooks).

You can test your implementation by loading the example datasets using `edvart.example_datasets.dataset_*` and running your analysis on the dataset.

### Implement an API for the analysis
The basic idea is that the code of your analysis will get exported by the user to a new notebook with varying levels of detail configurable by the user.

Your analysis will be its own section in the exported report notebook.
To implement the API, create a new Python script `edvart/report_sections/your_analysis.py` containing a class that implements your analysis.
Your class should be a subclass of `edvart.report_sections.section_base.Section` and implement all of the abstract methods.
See the documentation of `edvart.report_sections.section_base.Section` for details on the methods you need to implement.

For a reference section implementation, see `edvart/report_sections/univariate_analysis.py`.
If your analysis section contains subsections and you want the user to be able to leave some of them out,
use the `edvart/report_sections/section_base.ReportSection` base class instead,
see `edvart/report_sections/dataset_overview.py` for reference.

Once you have implemented the class, expose the functionality to the user by integrating
your class into the class that the user will use directly, namely `edvart.Report` or `edvart.TimeseriesReport`.

Add a method to the `ReportBase`-based class that adds an instance of your section analysis class
to the list of sections of the report `self.sections` and prefix it with `add_`.
If you want your section to be added to the report by default, call the newly created method in `Report.__init__`.

### Test the newly implemented API
Create unit tests by creating a Python script in the folder `tests` prefixed with `test_`.
The script should contain functions also prefixed with `test_` that make assertions.
Test whether your class exports the correct code for the given verbosity. See the `tests` folder for reference.

### Modify documentation
If you add a new section, add the section description into `docs/advanced.rst`

## Pull Requests & Git

* Try to split your work into separate and atomic pull requests. Put any
  non-obvious reasoning behind any change to the pull request description.
  Separate “preparatory” changes and modifications from new features &
  improvements.
* Try to keep the history clean and simple. Use `git rebase` to clean up commits before submitting a pull request.
* Use conventional commit messages. See [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
  Usage of conventional commit messages is enforced by the CI pipeline.


## Code style
* The line length is limited to 100 characters in Python code, except if it would make the code less readable.
* `black` is used for formatting Python code. The following command can be used to properly format the code:
```bash
poetry run black --line-length 100 edvart/ tests/
```
The following command can be used to check if the code is properly formatted:
```bash
poetry run black --check --line-length 100 edvart/ tests/
````

* `isort` is used for sorting imports.
The following command can be used to properly sort imports:
```bash
poetry run isort --profile black edvart/ tests/
```
The following command can be used to check if the imports are properly sorted:
```bash
poetry run isort --check --line-length 100 --profile black edvart/ tests/
```

* `pylint` is used to lint Python code. Tests are not required to be linted.
The following command can be used to lint the code:
```bash
poetry run pylint --rcfile=".pylintrc" edvart
```

All of the above code style requirements are enforced by the CI pipeline.
