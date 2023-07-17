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


## User documentation

TODO: Add link

## How to contribute


### License
edvart is licensed under the [MIT license](https://opensource.org/license/mit/). See the LICENSE file for more details.

### Environment setup
If you've never contributed to edvart before, get started by following these steps:

Clone the repository
```
https://github.com/datamole-ai/edvart/
cd edvart
```
We use [Poetry](https://python-poetry.org/) for managing dependencies.
Set up your poetry environment and Python binary. We recommend `pyenv` (https://github.com/pyenv/pyenv) for managing Python versions.
```
pip install poetry pyenv

# Install Python 3.11
pyenv install 3.11

# Use pyenv's Python 3.11 for the current folder
pyenv local 3.11

# Create virtual environment (install all optional dependencies)
poetry install --extras all

# Activate virtual environment
poetry shell
```

### General workflow
1. Suggest a type of analysis
2. Implement a prototype analysis in the prototype notebook
    * The `prototype.md` notebook serves as a preview of what the final report will look like
    * The implementation gets reviewed by another edvart contributor and approved to be integrated into the edvart package by implementing an API for the analysis
3. Implement a user friendly API of the analysis
    * A certain structure of the API is enforced and is elaborated on below
4. Test the newly implemented API
    * Create unit tests

#### Suggest a type of analysis
Based on your particular needs/ideas, suggest a new type of analysis that could be generalized so that others with a similar data format could use it in their projects by running edvart.

#### Implement a prototype analysis in the prototype notebook
Once you have an idea of what analysis to implement, create your own branch from the master branch and in it implement the analysis in the `prototype.md` notebook.
You can test your implementation by loading the example datasets using `edvart.example_datasets.dataset_*` and running your analysis on the dataset.

Make sure to add necessary package dependencies required by your analysis to `pyproject.toml` and freeze the package versions using
```
poetry lock
```
then version both `pyproject.toml` and `poetry.lock`.

Let another contributor review your prototype with a merge request.

#### Implement a user friendly API of the analysis
The basic idea is that the code of your analysis will get exported by the user to a new notebook with varying levels of detail configurable by the user.

Your analysis will be its own section in the exported report notebook. To implement the API, create a new Python script `edvart/report_sections/your_analysis.py` which contains a class that implements your analysis and the code exporting. Your class should be a subclass of `edvart/report_sections/section_base.Section` and implement all of the abstract methods:

* `__init__` initializes your object and accepts `verbosity` and `columns` possibly among other arguments
    - `verbosity` is an integer representing the detail level of the exported code and is either of `0`, `1`, or `2`
        * `0` exports a single function call that generates the entire section of your analysis
        * `1` exports a function call for each of the subsection that your analysis has
        * `2` exports the full code of your analysis
        * The functions getting exported can of course have additional arguments that configure your analysis, that's how you control how the user interacts with it.
  - `columns` is a list of names of columns which will be used in your analysis (so that different analyses can be applied to different columns)
* `required_imports` returns a list of lines of code that import the packages required by your analysis which will get added to a cell at the top of the exported notebook
* `add_cells(cells)` adds cells to the list of cells `cells`. This is how the exported notebook gets built.
    - To create markdown cell pass a string to `nbformat.v4.new_markdown_cell()`, to create a code cell pass a string to `nbformat.v4.new_code_cell()`, finally append the objects returned by these functions to `cells`
    - Keep in mind that the code created should conform to `verbosity`
* `show` renders your analysis inplace in the calling notebook

For a reference section implementation, see `edvart/report_sections/univariate_analysis.py`. If your analysis section contains subsections and you want the user to be able to leave some of them out, inherit from `edvart/report_sections/section_base.ReportSection` instead, see `edvart/report_sections/dataset_overview.py` for reference.

Once you have the class implemented, expose the functionality to the user by integrating your class into the class that the user will use directly, namely `edvart/report.Report`. Add a method to the `Report` class that adds an instance of your section analysis class to the list of sections of the report `self.sections` and prefix it with `add_`. If you want your section to be added to the report by default, call the newly created method in `Report.__init__`.

#### Test the newly implemented API
Create unit tests by creating a Python script in the folder `tests` prefixed with `test_`. The script should contain functions also prefixed with `test_` that make assertions. Test whether your class exports the correct code for the given verbosity. See the `tests` folder for reference.

#### Modify documentation
If you add a new section. Add the section into `docs/advanced.rst`

#### Update changelog
Add your functionality into [Unreleased] section in CHANGELOG.md
