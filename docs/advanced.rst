.. _advanced_usage:

Advanced usage of EDVART
===========================================

This section describes several concepts behind edvart
and how you can modify your report before exporting it.

Report class
------------

The most important class of the package :py:class:`~edvart.report.Report`.
The report consists of sections, which can be added via methods of the `Report` class.
The report is empty by default.
The class :py:class:`~edvart.report.DefaultReport` is a subclass of `Report`,
which contains a default set of sections.

With created instance of `Report` you can:

1. Show the report directly in your jupyter notebook using :py:meth:`~edvart.report.Report.show` method.
2. Export a new notebook using :py:meth:`~edvart.report.Report.export_notebook` method and edit it by yourself.
3. Export the output to a HTML report. You can also use a `.tpl` template to style the report.

Exporting to HTML
-----------------
Apart from directly exporting a `Report`, you may also wish to export a generated notebook to HTML.
To export a notebook, you may use a tool called `jupyter nbconvert` (https://nbconvert.readthedocs.io/en/latest/).
For example, to export a notebook called `notebook.ipynb` using the `lab` template, you may use the following command:

.. code-block:: bash

   poetry run jupyter nbconvert --to html notebook.ipynb --template lab



TimeseriesReport class
----------------------

This class is a special version of the :py:class:`~edvart.report.Report` class which is specifically meant to be used for analysis of time series.

The main differences are a different set of default sections including :py:class:`~edvart.report_sections.TimeseriesAnalysis`,
which cannot be added to the normal `Report` and the assumption that analyzed data is time-indexed.

Helper functions :py:func:`~edvart.utils.reindex_to_period` or :py:func:`~edvart.utils.reindex_to_datetime`
can be used to index a DataFrame by a `pd.PeriodIndex` or a `pd.DatetimeIndex` respectively.

Each column is treated as a separate timeseries.

.. code-block:: python

   df = pd.DataFrame(
      data=[
            ['2018Q1', 120000, 11000],
            ['2018Q2', 150000, 13000],
            ['2018Q3', 100000, 12000],
            ['2018Q4', 110000, 11000],
            ['2019Q1', 120000, 13000],
            ['2019Q2', 110000, 12000],
            ['2019Q3', 120000, 14000],
            ['2019Q4', 90000, 12000],
            ['2020Q1', 130000, 12000],
      ],
      columns=['Quarter', 'Revenue', 'Profit'],
   )

   # Reindex using helper function to have 'Quarter' as index
   df = edvart.utils.reindex_to_datetime(df, datetime_column='Quarter')
   report_ts = edvart.TimeseriesReport(df)
   report_ts.show()


Modifying sections
------------------

The report consists of sections.

In current version of edvart you can find following sections:

* TableOfContents

  - Provides table of contents with links to all other sections.
  - :py:meth:`~edvart.report.ReportBase.add_table_of_contents`

* DatasetOverview

  - Provides essential information about whole dataset
  - :py:meth:`~edvart.report.ReportBase.add_overview`

* UnivariateAnalysis

  - Provides analysis of individual columns
  - :py:meth:`~edvart.report.ReportBase.add_univariate_analysis`

* BivariateAnalysis

  - Provides analysis of pairs of columns
  - :py:meth:`~edvart.report.ReportBase.add_bivariate_analysis`

* MultivariateAnalysis

  - Provides analysis of all columns together. Currently features PCA, parallel coordinates and parallel categories subsections.
  - :py:meth:`~edvart.report.ReportBase.add_multivariate_analysis`

* GroupAnalysis

  - Provides analysis of each column when grouped a column or a set of columns. Includes basic information similar to dataset overview and univariate analysis, but on a per-group basis.
  - :py:meth:`~edvart.report.ReportBase.add_group_analysis`

* TimeseriesAnalysis

  - Provides analysis specific for time series.
  - :py:meth:`~edvart.report.TimeseriesReport.add_timeseries_analysis`


The edvart API allows you to choose which sections you want in the final report
or modifying sections settings.

Selection of sections
~~~~~~~~~~~~~~~~~~~~~

If you want to use only a subset of sections you have to set
`use_default_sections` parameter of report to `False` and then you can add your own sections.

.. code-block:: python

    # Shows only univariate analysis
    import edvart
    df = edvart.example_datasets.dataset_titanic()
    report = edvart.Report(df, use_default_sections=False)
    report.add_univariate_analysis()


Sections configuration
~~~~~~~~~~~~~~~~~~~~~~

Each section can be also configured.
For example you can define which columns should be used or omitted.

Or you can set section verbosity (described later).

.. code-block:: python

  # Configures sections to omit or use specific columns
  import edvart

  df = edvart.example_datasets.dataset_titanic()
  report = edvart.Report(df)

  report.add_overview(omit_columns=["PassengerId"]).add_univariate_analysis(
    use_columns=["Name", "Sex", "Age"]
  )



.. _verbosity:

Verbosity
---------

EDVART provides a concept of a verbosity that is used during *export* into jupyter notebook.
The verbosity helps us to generate a code with a specific level of detail.

edvart supports three levels of verbosity:

- verbosity 0
   - High level functions for whole sections are generated. User can modify the markdown description.
- verbosity 1
   - edvart functions are generated. User can modify parameters of these functions.
- verbosity 2
   - Raw code is generated. User can do very advanced modification such as changing visualisations style.

The verbosity can be set to whole report or to each section separately.

Examples:

.. code-block:: python

    # Set default verbosity for all sections to 1
    import edvart

    df = edvart.example_datasets.dataset_titanic()
    edvart.DefaultReport(df, verbosity=1).export_notebook("test-export.ipynb")


.. code-block:: python

    # Set default verbosity to 1 but use verbosity 2 for univariate analysis
    import edvart

    df = edvart.example_datasets.dataset_titanic()
    edvart.DefaultReport(df, verbosity=1, verbosity_univariate_analysis=2).export_notebook("test-export.ipynb")
