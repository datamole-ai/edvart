Installation
============

edvart is distributed via PyPI.
Example installation with pip:

.. code-block:: console

   $ pip install edvart

or you can add edvart into your environment file defined by `pyproject.toml`:

.. parsed-literal::

   [tool.poetry.dependencies]
   python = ">=3.8, <3.12"
   edvart = "|VERSION|"


Extras
------

edvart also has an optional dependency "umap", which adds a plot called UMAP
(Universal Manifold Approximation) to Multivariate Analysis. To install edvart with the optional
extra, replace the above snippet of the `pyproject.toml` environment file with the following
snippet:

.. parsed-literal::

   [tool.poetry.dependencies]
   python = ">=3.8, <3.12"
   edvart = { version = "|VERSION|", extras = ["umap"] }

To install edvart with the optional "umap" dependency via pip, run the following command:

.. code-block:: console

   $ pip install "edvart[umap]"


Plotly
======

JupyterLab
----------

To display interactive plots which use Plotly in JupyterLab, you need to install some JupyterLab
extensions.

To install the required extensions, you can follow the full guide at
https://plot.ly/python/getting-started/ or simply run the following commands
(inside the JupyterLab container if running in a container):

.. code-block:: console

   jupyter labextension install @jupyter-widgets/jupyterlab-manager@1.1 --no-build
   jupyter labextension install jupyterlab-plotly@1.5.2 --no-build
   jupyter labextension install plotlywidget@1.5.2 --no-build
   jupyter lab build

Visual Studio Code
------------------
To display interactive plots which use Plotly in Visual Studio Code notebooks,
you need to install the following extensions:

* `Jupyter <https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter>`_ is required to
  run Jupyter notebooks in Visual Studio Code.
* `Jupyter Notebook Renderers <https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter-renderers>`_ is required
  to render Plotly plots in Visual Studio Code notebooks.

