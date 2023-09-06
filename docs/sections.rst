Report Sections
---------------

Dataset Overview
~~~~~~~~~~~~~~~~
  - Provides essential information about whole dataset, such as inferred
    data types, number of rows and columns, number of missing values, duplicates, etc.
  - See :py:meth:`edvart.report.ReportBase.add_overview`

Univariate Analysis
~~~~~~~~~~~~~~~~~~~
  - Provides analysis of individual columns. The analysis differs based on the data type of the column.
  - See :py:meth:`edvart.report.ReportBase.add_univariate_analysis`

Bivariate Analysis
~~~~~~~~~~~~~~~~~~
  - Provides analysis of pairs of columns, such as correlations, scatter plots, contingency tables, etc.
  - See :py:meth:`edvart.report.ReportBase.add_bivariate_analysis`


.. _multivariate_analysis:

Multivariate Analysis
~~~~~~~~~~~~~~~~~~~~~
  - Provides analysis of all columns together.
  - Currently features PCA, parallel coordinates and parallel categories subsections.
    Additionally, an UMAP section is included if the :ref:`extra<extras>` dependency ``umap`` is installed.
  - See :py:meth:`edvart.report.ReportBase.add_multivariate_analysis`

Group Analysis
~~~~~~~~~~~~~~
  - Provides analysis of each column when grouped by a column or a set of columns.
    Includes basic information similar to dataset overview and univariate analysis,
    but on a per-group basis.
  - See :py:meth:`edvart.report.ReportBase.add_group_analysis`

Timeseries Analysis
~~~~~~~~~~~~~~~~~~~
  - Provides analysis specific for time series.
  - Used with :py:class:`edvart.report.TimeseriesReport`
  - See :py:meth:`edvart.report.TimeseriesReport.add_timeseries_analysis`
