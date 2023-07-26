---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: edvart
    language: python
    name: edvart
---

```python
%load_ext autoreload
%autoreload 2

import edvart

import plotly.offline as py
py.init_notebook_mode()
```

```markdown
## Basic report
```

```python
dataset = edvart.example_datasets.dataset_titanic()
```

```python
dataset.head()
```

```python
report = edvart.Report(
    dataset,
    verbosity=0,
    columns_overview=['Name', 'Survived'],
    columns_univariate_analysis=['Name', 'Age', 'Pclass'],
    groupby='Survived',
)
```

```python
report.export_notebook('test-export.ipynb')
```

```python
report.export_html(
    html_filepath='test-export.html',
    dataset_name='Titanic',
    dataset_description='Dataset that contains data for 891 of the real Titanic passengers.'
)
```

```python
report.show()
```

```markdown
## Timeseries report
```

```python
dataset_ts = edvart.example_datasets.dataset_global_temp()
```

```python
report_ts = edvart.TimeseriesReport(
    dataset_ts,
    # Monthly data -> analyze yearly seasonality
    sampling_rate=12,
    stft_window_size=24,
)
```

```python
report_ts.export_notebook("test-export-ts.ipynb")
```

```python
report_ts.export_html("test-export-ts.html")
```

```python
report_ts.show()
```
