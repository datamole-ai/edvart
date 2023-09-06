Installation
============

``edvart`` is distributed as a Python package via `PyPI <https://pypi.org/project/edvart/>`_.
It can be installed using ``pip``:

.. code-block:: console

   $ pip install edvart

We recommend using `Poetry <https://python-poetry.org/>`_ for dependency management.
To add ``edvart`` into a Poetry environment, add the following snippet
to the ``pyproject.toml`` environment definition file:

.. parsed-literal::

   [tool.poetry.dependencies]
   python = ">=3.8, <3.12"
   edvart = "|VERSION|"


.. _extras:

Extras
------

Edvart has an optional dependency ``umap``, which adds a plot called `UMAP <https://umap-learn.readthedocs.io/en/latest/>`_
to :ref:`Multivariate Analysis <multivariate_analysis>`.

To install Edvart with the optional ``umap`` dependency via pip, run the following command:

.. code-block:: console

   $ pip install "edvart[umap]"

To install Edvart with the optional extra using Poetry, replace the snippet
of the ``pyproject.toml`` environment file above with the following snippet:

.. parsed-literal::

   [tool.poetry.dependencies]
   python = ">=3.8, <3.12"
   edvart = { version = "|VERSION|", extras = ["umap"] }

Rendering Plotly Interactive Plots
----------------------------------

Edvart uses `Plotly <https://plotly.com/python/>`_ to render interactive plots.

JupyterLab
~~~~~~~~~~

To display interactive plots which use Plotly in JupyterLab, you need to install some JupyterLab
extensions.

The extension ``jupyter-dash`` needs to be installed in order for Plotly plots
to be rendered correctly in JupyterLab.
It can be simply installed as a Python package, e.g. via ``pip``:

.. code-block:: console

   pip install jupyter-dash

to install `plotly-dash` to a Poetry environment, add the following line
under ``tool.poetry.dependencies`` in the ``pyproject.toml`` environment definition file:


.. code-block:: toml

   jupyter-dash = "^0.4.2"


See https://plot.ly/python/getting-started/ for more information.

Visual Studio Code
~~~~~~~~~~~~~~~~~~
The following extensions need to be installed to display Plotly
interactive plots in Visual Studio Code notebooks:

* `Jupyter <https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter>`_
   is required to
   run Jupyter notebooks in Visual Studio Code.
* `Jupyter Notebook Renderers <https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter-renderers>`_
   is required to render Plotly plots in Visual Studio Code notebooks.
