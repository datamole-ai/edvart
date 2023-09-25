Usage
=====

Quick Start
-----------

Show a Default Report in a Jupyter Notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import edvart


    df = edvart.example_datasets.dataset_titanic()
    edvart.DefaultReport(df).show()

Export the Report Code to a Jupyter Notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import edvart


    df = edvart.example_datasets.dataset_titanic()
    report = edvart.DefaultReport(df)
    report.export_notebook(
        "titanic_report.ipynb",
        dataset_name="Titanic",
        dataset_description="Dataset of 891 of the real Titanic passengers.",
    )

The exported notebook contains the code which generates the report.
It can be modified to fine-tune the report.
The code can be exported with different levels of detail (see :ref:`verbosity`).

Export a Report to HTML
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import edvart


    df = edvart.example_datasets.dataset_titanic()
    report = edvart.DefaultReport(df)
    report.export_html(
        html_filepath="titanic_report.html",
        dataset_name="Titanic",
        dataset_description="Dataset of 891 of the real Titanic passengers.",
    )


A :py:class:`~edvart.report.Report` can be directly exported
to HTML via the :py:meth:`~edvart.report.ReportBase.export_html` method.

Jupyter notebooks can be exported to other formats including HTML, using a tool
called `jupyter nbconvert` (https://nbconvert.readthedocs.io/en/latest/).
This can be useful to create a HTML report from a notebook which was exported
using the :py:meth:`~edvart.report.ReportBase.export_notebook` method.

Customizing the Report
----------------------

This section describes several concepts behind edvart and how a report
can be customized.

Report Class
~~~~~~~~~~~~

The :py:class:`~edvart.report.Report` class is central to the edvart API.
A *Report* consists of sections, which can be added via methods of the :py:class:`~edvart.report.Report` class.
The class :py:class:`~edvart.report.DefaultReport` is a subclass of :py:class:`~edvart.report.Report`,
which includes a default set of sections.

With an instance of :py:class:`~edvart.report.Report` you can:

1. Show the report directly in a Jupyter notebook using the :py:meth:`~edvart.report.Report.show` method.
2. Export the code which generates the report to a new Jupyter notebook using
   :py:meth:`~edvart.report.ReportBase.export_notebook` method.
   The code can be exported with different levels of :ref:`verbosity <verbosity>`.
   The notebook containing the exported code can be modified to fine-tune the report.
3. Export the output to a HTML file. You can specify an
   `nbconvert template
   <https://nbconvert.readthedocs.io/en/latest/customizing.html#selecting-a-template>`_
   to style the report.


Selection of Sections
~~~~~~~~~~~~~~~~~~~~~
You can add sections using section-specific methods ``add_*`` (e.g. :py:meth:`edvart.report.ReportBase.add_overview`)
or the general method `edvart.report.ReportBase.add_section` of the :py:class:`~edvart.report.Report` class.

.. code-block:: python

    # Include univariate and bivariate analysis
    import edvart


    df = edvart.example_datasets.dataset_titanic()
    report = (
        edvart.Report(df)
        .add_univariate_analysis()
        .add_bivariate_analysis()
    )

.. _sections-config:

Configuration of Sections
~~~~~~~~~~~~~~~~~~~~~~~~~

Each section can be also configured.
For example you can define which columns should be used or omitted.

.. code-block:: python

    import edvart
    from edvart.report_sections.dataset_overview import Overview
    from edvart.report_sections.add_univariate_analysis import UnivariateAnalysis


    df = edvart.example_datasets.dataset_titanic()
    report = (
        edvart.Report(df)
        .add_section(Overview(columns=["PassengerId"]))
        .add_section(UnivariateAnalysis(columns=["Name", "Sex", "Age"]))
    )


Subsections
***********

Some sections are made of subsections. For those, you can can configure which subsections are be included.

.. code-block:: python

    import edvart
    from edvart.report_sections.dataset_overview import Overview


    df = edvart.example_datasets.dataset_titanic()
    report = edvart.Report(df)

    report.add_overview(
        subsections=[
            Overview.OverviewSubsection.QuickInfo,
            Overview.OverviewSubsection.DataPreview,
        ]
    )


.. _verbosity:

Verbosity
~~~~~~~~~

A :py:class:`~edvart.report.Report` can be exported to a Jupyter notebook containing
the code which generates the report. The code can be exported with different levels of detail,
referred to as *verbosity*.

It can be set on the level of the whole report or on the level of each
section or subsection separately (see :ref:`sections-config`).

Specific verbosity overrides general verbosity, i.e. the verbosity set on a
subsection overrides the verbosity set on a section, which overrides
the verbosity set on the report.

EDVART supports three levels of verbosity:

LOW
   High level functions for whole sections are exported, i.e. the output
   of each section is generated by a single function call.
   Suitable for small modifications such as changing parameters of the functions,
   adding commentary to the report, adding visualizations which are not in EDVART, etc.

MEDIUM
   For report sections which consist of subsections, each subsection is
   exported to a separate function call.
   Same as LOW for report sections which do not consist of subsections.

HIGH
   The definitions of (almost) all functions are exported.
   The functions can be modified or used as a starting point for custom analysis.


Examples
********

.. code-block:: python

    # Set default verbosity for all sections to Verbosity.MEDIUM
    import edvart
    from edvart import Verbosity


    df = edvart.example_datasets.dataset_titanic()
    edvart.DefaultReport(df, verbosity=Verbosity.MEDIUM).export_notebook("test-export.ipynb")


.. code-block:: python

    import edvart
    from edvart import Verbosity


    # Set report verbosity to Verbosity.MEDIUM but use verbosity Verbosity.HIGH for univariate analysis
    df = edvart.example_datasets.dataset_titanic()
    edvart.DefaultReport(
        df,
        verbosity=Verbosity.MEDIUM,
        verbosity_univariate_analysis=Verbosity.HIGH,
    ).export_notebook("exported-report.ipynb")


Reports for Time Series Datasets
--------------------------------

The class :py:class:`~edvart.report.TimeseriesReport` is a version
of the :py:class:`~edvart.report.Report` class which is specific for creating
reports on time series datasets.
There is also a :py:class:`~edvart.report.DefaultTimeseriesReport`, which contains
a default set of sections, similar to :py:class:`~edvart.report.DefaultReport`.


The main differences compared to the report for tabular data are:

* a different set of default sections for :py:class:`~edvart.report.DefaultTimeseriesReport`
* :py:class:`~edvart.report_sections.TimeseriesAnalysis` section, which contains visualizations
  for analyzing time series data
* the assumption that the input data is time-indexed and sorted by time.

Helper functions :py:func:`edvart.utils.reindex_to_period` or :py:func:`edvart.utils.reindex_to_datetime`
can be used to index a DataFrame by a ``pd.PeriodIndex`` or a ``pd.DatetimeIndex`` respectively.

Each column in the input data is treated as a separate time series.

.. code-block:: python

   df = pd.DataFrame(
      data=[
            ["2018Q1", 120000, 11000],
            ["2018Q2", 150000, 13000],
            ["2018Q3", 100000, 12000],
            ["2018Q4", 110000, 11000],
            ["2019Q1", 120000, 13000],
            ["2019Q2", 110000, 12000],
            ["2019Q3", 120000, 14000],
            ["2019Q4", 160000, 12000],
            ["2020Q1", 130000, 12000],
      ],
      columns=["Quarter", "Revenue", "Profit"],
   )

   # Reindex using helper function to have 'Quarter' as index
   df = edvart.utils.reindex_to_datetime(df, datetime_column="Quarter")
   report_ts = edvart.DefaultTimeseriesReport(df)
   report_ts.show()
