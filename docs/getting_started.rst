Getting started
===============

1. Start with default exploratory analysis in jupyter notebook.

.. code-block:: python

    import edvart
    df = edvart.example_datasets.dataset_titanic()
    edvart.Report(df).show()

2. Generate report notebook

.. code-block:: python

    import edvart
    df = edvart.example_datasets.dataset_titanic()
    report = edvart.Report(df)
    report.export_notebook("titanic_report.ipynb")

You can modify the generated notebook if you want to modify some settings.
For more advanced usage of edvart, please read the documentation section
:ref:`Advanced usage <advanced_usage>`.

3. Generate HTML report

.. code-block:: python

    import edvart
    df = edvart.example_datasets.dataset_titanic()
    report = edvart.Report(df)
    report.export_html(
        html_filepath="titanic_report.html",
        dataset_name="Titanic",
        dataset_description="Dataset that contains data for 891 of the real Titanic passengers.",
    )

