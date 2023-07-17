---
jupyter:
  jupytext:
    cell_metadata_json: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.0
  kernelspec:
    display_name: edvart
    language: python
    name: edvart
---

# [INSERT DATASET NAME] Report
[INSERT DATASET DESCRIPTION]

```python
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas  as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import warnings
import copy
from functools import partial

from IPython.display import display, HTML, Markdown
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from scipy import signal
import statsmodels.api as sm
import statsmodels.graphics.tsaplots as tsaplots
from statsmodels.tools.sm_exceptions import InterpolationWarning

import plotly
import plotly.graph_objects as go
import plotly.offline as py
py.init_notebook_mode()

import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import InterpolationWarning

import umap

from dart.pandas_formatting import *
from dart.data_types import is_numeric
from dart import utils
import dart
```

# Load Data

```python
dataset = dart.example_datasets.dataset_titanic()
```

# Report Configuration

```python
"""
Default list of names of columns which will be analyzed
"""
DEFAULT_COLUMN_SUBSET = dataset.columns

"""
Default dictionary of descriptive statistics that will be calculated for numerical columns
    Dicionary signature: 'StatisticName': stat_func
    stat_func signature: stat_func(series: pandas.Series) -> Any
"""
DEFAULT_DESCRIPTIVE_STATISTICS = {
    'Number of unique values': utils.num_unique_values,
    'Sum': utils.sum,
    'Mean': utils.mean,
    'Mode': utils.mode,
    'Standard deviation': utils.std,
    'Mean absolute deviation': utils.mad,
    'Median absolute deviation': utils.median_absolute_deviation,
    'Coefficient of variation': utils.coefficient_of_variation,
    'Kurtosis': utils.kurtosis,
    'Skewness': utils.skewness
}

"""
Default dictionary of quantile statistics that will be calculated for numerical columns
    Dicionary signature: 'StatisticName': stat_func
    stat_func signature: stat_func(series: pandas.Series) -> Any
"""
DEFAULT_QUANTILE_STATISTICS = {
    'Minimum': utils.min,
    'Maximum': utils.max,
    'Q1': utils.quartile1,
    'Median': utils.median,
    'Q3': utils.quartile3,
    'Range': utils.value_range,
    'IQR': utils.IQR,
}
```

# Dataset Overview

```python
# Dataset info calculation
missing_cells = dataset.isna().sum().sum()
missing_cells_percent = 100 * missing_cells / (dataset.shape[0] * dataset.shape[1])

zeros = (dataset == 0).sum().sum()
zeros_percent = 100 * zeros / (dataset.shape[0] * dataset.shape[1])

duplicate_rows = dataset.duplicated().sum()
duplicate_rows_percent = 100 * duplicate_rows / len(dataset)

dataset_info_rows = {
    'Rows': dataset.shape[0],
    'Columns': dataset.shape[1],
    'Missing cells': f'{missing_cells} ({missing_cells_percent:,.02f} %)',
    'Zeros': f'{zeros} ({zeros_percent:.02f} %)',
    'Duplicate rows': f'{duplicate_rows} ({duplicate_rows_percent:,.02f} %)'
}
```

## Quick Info

```python
# Render
render_dictionary(dataset_info_rows)
```

## Data Types

```python
# Get dtypes
dtypes = dataset.dtypes
dtypes_no_nans = (
    dataset
    .dropna(axis=1, how='all')
    .dropna(axis=0)
    .fillna(0, downcast='infer')
    .dtypes
)

# Convert result to frame for viewing
dtypes_frame = series_to_frame(
    series=dtypes,
    index_name='Column Name',
    column_name='Data Type'
)
dtypes_no_nans_frame = series_to_frame(
    series=dtypes_no_nans,
    index_name='Column Name',
    column_name='Data Type (after dropping NULL values)'
)
(
    dtypes_frame
    .merge(dtypes_no_nans_frame, on='Column Name', how='left')
    .style
    .hide_index()
)
```

## Dataset Sample

```python
N_HEAD    = 5
N_TAIL    = 5
N_SAMPLES = 10
```

### Dataset First Rows

```python
dataset.head(N_HEAD)
```

### Dataset Last Rows

```python
dataset.tail(N_TAIL)
```

### Dataset Random Sample

```python
dataset.sample(N_SAMPLES)
```

## Missing Values

```python
"""
Edit the COLUMN_SUBSET list to specify which columns will be considered in missing values counting
"""

COLUMN_SUBSET = DEFAULT_COLUMN_SUBSET

if isinstance(COLUMN_SUBSET, str):
    COLUMN_SUBSET = [COLUMN_SUBSET]
```

```python
MATPLOTLIB_BAR_PLOT_ARGS = {
    'figsize': (15, 6),
    'title': 'Missing Values Percentage of Each Column',
    'ylim': 0,
    'legend': False,
    'color': '#FFA07A'
}
```

```python
# Count null values
null_count = dataset[COLUMN_SUBSET].isna().sum()
null_percentage = 100 * null_count / len(dataset)

# Convert series to frames
null_count_frame = series_to_frame(
    series=null_count,
    index_name='Column Name',
    column_name='Null Count'
)
null_percentage_frame = series_to_frame(
    series=null_percentage,
    index_name='Column Name',
    column_name='Null %'
)
# Merge null count and percentage into one frame
null_stats_frame = (
    null_count_frame
    .merge(null_percentage_frame, on='Column Name')
    .sort_values('Null Count', ascending=False)
)
```

```python
# Render
(
    null_stats_frame
    .style
    .hide_index()
    .bar(color='#FFA07A', subset=['Null %'], vmax=100)
    .format({'Null %': '{:.03f}'})
)
```

```python
# Bar plot of missing values percentages for each column
(
    null_percentage_frame
    .sort_values('Null %', ascending=False)
    .plot
    .bar(x='Column Name', **MATPLOTLIB_BAR_PLOT_ARGS)
    .set_ylabel('Missing Values [%]')
);
```

```python
# generate large random dataframe
shape=(100000, 8)
na_prob = 0.1
zeros = np.zeros(shape)
zeros[np.random.uniform(size=shape) < na_prob] = None
df = pd.DataFrame(zeros, columns=(range(shape[1])))

def missing_values_matrix(df, fig_size=(15, 10), cmap=sns.color_palette(['#FFFFFF', '#FFA07A']), transpose=False):
    # Compute missing values matrix
    missing_values = df.isna()
    if transpose:
        missing_values = missing_values.transpose()

    # Plot missing values matrix
    ax = sns.heatmap(missing_values, cbar=False, cmap=cmap)

    # Add vertical/horizontal lines separating columns
    line_args = {'color': cmap[0], 'linewidth': 1}
    if transpose:
        ax.hlines(range(len(df.columns)), *ax.get_xlim(), **line_args)
    else:
        ax.vlines(range(len(df.columns)), *ax.get_ylim(), **line_args)

    # Set axes
    ticks = [0, df.shape[0]]
    if transpose:
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks)
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks)
    ax.figure.set_size_inches(*fig_size)
    ax.set_title('Missing values matrix')
    labels = ['Row number', 'Column Name']
    ax.set_xlabel(labels[1 - transpose])
    ax.set_ylabel(labels[transpose])

    # Display frame around the plot
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    plt.show()

missing_values_matrix(df, transpose=False)
```

### Constant Value

```python
CONSTANT = 0

if isinstance(COLUMN_SUBSET, str):
    COLUMN_SUBSET = [COLUMN_SUBSET]
```

```python
# Count constant counts
constant_count = (dataset[COLUMN_SUBSET] == CONSTANT).sum()
constant_percentage = 100 * constant_count / len(dataset)

# Convert series to frames
constant_count_frame = series_to_frame(
    series=constant_count,
    index_name='Column Name',
    column_name=f'"{CONSTANT}" Count')

constant_percentage_frame = series_to_frame(
    series=constant_percentage,
    index_name='Column Name',
    column_name=f'"{CONSTANT}" %'
)
# Merge absolute and relative counts
constant_stats_frame = (
    constant_count_frame
    .merge(constant_percentage_frame, on='Column Name')
    .sort_values(f'"{CONSTANT}" %', ascending=False)
)
```

```python
# Render
(
    constant_stats_frame
    .style
    .hide_index()
    .bar(color='#FFA07A', subset=[f'"{CONSTANT}" %'], vmax=100)
    .format({f'"{CONSTANT}" %': '{:.03f}'})
)
```

## Number of rows with at least one value missing

```python
COLUMN_SUBSET = DEFAULT_COLUMN_SUBSET

if isinstance(COLUMN_SUBSET, str):
    COLUMN_SUBSET = [COLUMN_SUBSET]
```

```python
num_rows_missing_value = (
    dataset[COLUMN_SUBSET]
    .isna()
    .any(axis=1)
    .sum()
)

percentage_rows_missing_value = 100 * num_rows_missing_value / len(dataset)

missing_value_rows_info = {
    'Missing value column subset' : str(list(COLUMN_SUBSET)),
    'Missing value row count': f'{num_rows_missing_value:,}',
    'Missing value row percentage': f'{percentage_rows_missing_value:.02f} %'
}
```

```python
# Render
render_dictionary(missing_value_rows_info)
```

## Number of duplicate rows

```python
COLUMN_SUBSET = DEFAULT_COLUMN_SUBSET

if isinstance(COLUMN_SUBSET, str):
    COLUMN_SUBSET = [COLUMN_SUBSET]
```

```python
num_duplicated_rows = (
    dataset
    .duplicated(subset=COLUMN_SUBSET)
    .sum()
)

percentage_duplicated_rows = 100 * num_duplicated_rows / len(dataset)

duplicate_rows_info = {
    'Duplicate rows column subset' : str(list(COLUMN_SUBSET)),
    'Duplicate row count': f'{num_duplicated_rows:,}',
    'Duplicate row percentage': f'{percentage_duplicated_rows:.02f} %'
}
```

```python
# Render
render_dictionary(duplicate_rows_info)
```

# Univariate analysis

```python
COLUMN_SUBSET = DEFAULT_COLUMN_SUBSET

if isinstance(COLUMN_SUBSET, str):
    COLUMN_SUBSET = [COLUMN_SUBSET]
```

```python
"""
To add or delete statistics to or from certain features use:
    descriptive_stats_funcs['ColumnName']['NewStatistic'] = new_statistic_func
    del descriptive_stats_funcs['ColumnName']['UnwantedStatistic']
"""

# Choose which statistics will be calculated for which columns
descriptive_stats_funcs = {}
quantile_stats_funcs = {}
for col_name in COLUMN_SUBSET:
    descriptive_stats_funcs[col_name] = copy.deepcopy(DEFAULT_DESCRIPTIVE_STATISTICS)
    quantile_stats_funcs[col_name] = copy.deepcopy(DEFAULT_QUANTILE_STATISTICS)
```

```python
"""
To change the way a column gets handled in the univariate analysis, change the properties
in the col_props dictionary for example by setting the is_categorical flag
col_props['ColumnName']['is_categorical'] = False
"""

# Set properties of each column/feature that impacts the way it gets analyzed
col_props = {}
for col_name in COLUMN_SUBSET:
    is_categorical = utils.is_categorical(dataset[col_name])
    col_props[col_name] = {'is_categorical': is_categorical}
```

```python
"""
The loop below iterates through each feature and outputs univariate analysis of that feature.

To add your own analysis either expand the loop with your own code and/or add HTML strings to
the html_table list (for example via html_table.append([df1.to_html(), df2.to_html()])). This
way of rendering HTML allows for example two dataframes to be rendered side by side. The html_table
list's elements (also lists) represent rows and elements within
those elements represent cells.
"""

# Iterate through columns and output univariate analysis results
for col_name in col_props.keys():
    # Print column name and basic info
    display(Markdown('---'))
    display(Markdown(f'## {col_name}'))
    display(Markdown('Categorical' if col_props[col_name]['is_categorical'] else 'Numerical'))

    html_table = []

    # Calculate statistics depending on variable type
    if col_props[col_name]['is_categorical']:
        common_values = utils.top_frequent_values(dataset[col_name], n_top=5)
        common_values_html = dict_to_html(common_values)
        common_values_html = add_html_heading(common_values_html, 'Most frequent values', 2)
        html_table.append([common_values_html])
    else:
        # Calculate statistics using chosen for the current feature
        desc_stats = {}
        quant_stats = {}
        for stat_name, stat_func in descriptive_stats_funcs[col_name].items():
            desc_stats[stat_name] = format_number(stat_func(dataset[col_name]), thousand_separator=' ')
        for stat_name, stat_func in quantile_stats_funcs[col_name].items():
            quant_stats[stat_name] = format_number(stat_func(dataset[col_name]), thousand_separator=' ')
        # Render statistics tables side by side
        desc_stats_html = add_html_heading(dict_to_html(desc_stats), 'Descriptive Statistics')
        quant_stats_html = add_html_heading(dict_to_html(quant_stats), 'Quantile Statistics')
        # Add tables to HTML table for rendering
        html_table.append([desc_stats_html, quant_stats_html])

    # Render statistics in an HTML table
    display(HTML(subcells_html(html_table)))

    # Plot column distribution
    # Don't plot if there are a lot of unique values in categorical column
    if col_props[col_name]['is_categorical'] and dataset[col_name].nunique() > 50:
        warnings.warn(f'Column "{col_name}" is categorical but has a lot of unique values, skipping distribution plotting')
    else:
        if col_props[col_name]['is_categorical']:
            fig = plt.figure(figsize=(20, 7))
            ax = dataset[col_name].value_counts().plot.bar(figure=fig)
            plt.ylabel('Frequency')
            plt.show()
        else:
            fig, (ax_box, ax_hist) = plt.subplots(
                nrows=2,
                sharex=True,
                gridspec_kw={'height_ratios': (.15, .85)},
                figsize=(20, 7)
            )
            sns.boxplot(dataset[col_name].dropna(), ax=ax_box)
            sns.distplot(dataset[col_name].dropna(), kde=False, ax=ax_hist)
            ax_box.set(xlabel='')
            plt.ylabel('Frequency')
            plt.show()
```

# Bivariate analysis

```python
COLUMNS_BIVARIATE = [c for c in COLUMN_SUBSET if not utils.is_categorical(dataset[c])]
```

```python
"""
Default dictionary of bivariate statistics that will be calculated for numerical columns
    Dicionary signature: 'CorrelationName': corr_func
    corr_func signature: stat_func(series: pandas.DataFrame) -> pandas.DataFrame
"""

DEFAULT_CORRELATIONS = {
    'pearson' : utils.pearson,
    'spearman': utils.spearman,
    'kendall' : utils.kendall,
}
```

```python
def plot_correlation(df, corr_name, fig_size=(10, 7), font_size=15, color_map="Blues"):
    # show header
    display(Markdown(f'## {corr_name.capitalize()} Correlation'))

    # calculate correlation between columns
    corr = DEFAULT_CORRELATIONS[corr_name](df)

    # plot correlation heatmap
    ax = sns.heatmap(corr.values, cmap=color_map)

    # set axes
    ax.set_xticks(range(len(COLUMNS_BIVARIATE)))
    ax.set_xticklabels(COLUMNS_BIVARIATE, fontsize=font_size, rotation=90)
    ax.set_yticklabels(COLUMNS_BIVARIATE, fontsize=font_size, rotation=0)

    # set size
    ax.figure.set_size_inches(*fig_size)
    plt.show()

for corr_name in DEFAULT_CORRELATIONS:
    plot_correlation(dataset[COLUMNS_BIVARIATE], corr_name)
```

## Pairplot

```python
utils.pair_plot(dataset[COLUMNS_BIVARIATE])
```

## Contingency Table

```python
def contingency_table(
    df, columns1, columns2, include_total=True, hide_zeros=True, scaling_func=np.cbrt, colormap='Blues', size_factor=0.7, fontsize=15
):
    if isinstance(columns1, str):
        columns1 = [columns1]
    if isinstance(columns2, str):
        columns2 = [columns2]
    table = pd.crosstab([df[col] for col in columns1], [df[col] for col in columns2], margins_name='Total', margins=include_total)
    annot = table.replace(0, '') if hide_zeros else table

    ax = sns.heatmap(
        scaling_func(table),
        annot=annot,
        fmt='',
        cbar=False,
        cmap=colormap,
        linewidths=0.1,
        xticklabels=1,
        yticklabels=1,
        annot_kws = {'fontsize': fontsize}
    )
    ax.figure.set_size_inches(
        size_factor * len(table.columns),
        size_factor * len(table)
    )

    ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)

    ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fontsize)

    ax.xaxis.tick_top()
    ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
    ax.xaxis.set_label_position('top')

    # Viusally separate the margins
    if include_total:
        ax.vlines(len(table.columns) - 1, ymin=0, ymax=len(table), color='grey')
        ax.hlines(len(table) - 1, xmin=0, xmax=len(table.columns), color='grey')

    plt.show()

contingency_table(dataset, 'Sex', 'Pclass')
```

# Group analysis

```python
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import colorlover as cl
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize, to_hex

from dart.data_types import DataType, infer_data_type
from dart.pandas_formatting import format_number
```

```python
DEFAULT_GROUP_DESCRIPTIVE_STATISTICS = {
    '# Unique values': utils.num_unique_values,
    'Sum': utils.sum,
    'Mode': utils.mode,
    'Mean': utils.mean,
    'Std': utils.std,
    'Mean abs dev': utils.mad,
    'Median abs dev': utils.median_absolute_deviation,
    'Relative Std': utils.coefficient_of_variation,
    'Kurtosis': utils.kurtosis,
    'Skewness': utils.skewness
}

DEFAULT_GROUP_QUANTILE_STATISTICS = {
    'Min': utils.min,
    'Q1': utils.quartile1,
    'Median': utils.median,
    'Q3': utils.quartile3,
    'Max': utils.max,
    'Range': utils.value_range,
    'IQR': utils.IQR
}
```

```python
def group_barplot(
    df,
    groupby,
    column,
    group_count_threshold=20,
    conditional_probability=True,
    xaxis_tickangle=0,
    alpha=0.5
):
    num_cat = df[column].nunique()
    if num_cat > group_count_threshold:
        warnings.warn(f'Too many categories ({num_cat}), not plotting distribution')
        return

    pivot = (
        df
        .pivot_table(
            index=groupby,
            columns=column,
            aggfunc='size',
            fill_value=0
        )
    )

    if conditional_probability:
        pivot = pivot.divide(pivot.sum(axis=1), axis=0)
        pivot.fillna(value=0)

    # Choose color palette
    colors = cl.scales['9']['qual']['Set1']
    color_idx = 0

    fig = go.Figure()
    for idx, row in pivot.iterrows():
        if hasattr(idx, '__len__') and not isinstance(idx, str):
            group_name = '_'.join([str(i) for i in idx])
        else:
            group_name = idx
        color = colors[color_idx % len(colors)]
        color_idx += 1
        fig.add_trace(
            go.Bar(
                x=pivot.columns,
                y=row,
                name=group_name,
                opacity=alpha,
                marker_color=color
            )
        )

    if conditional_probability:
        yaxis_title = f'P({column} | {groupby})'
    else:
        yaxis_title = f'Freq({column} | {groupby})'

    fig.update_layout(
        barmode='group',
        xaxis_tickangle=xaxis_tickangle,
        xaxis_title=column,
        yaxis_title=yaxis_title
    )
    fig.show()


def overlayed_histograms(
    df,
    groupby,
    column,
    bins=None,
    density=True,
    alpha=0.5
):
    # Modified Freedman-Diaconis bin number inference if bins is None
    if bins is None:
        IQR = df[column].quantile(0.75) - df[column].quantile(0.25)
        bin_width = 1 / np.cbrt(len(df)) * IQR
    else:
        bin_width = (df[column].max() - df[column].min()) / bins
    bin_config = {
        'start': df[column].min(),
        'end': df[column].max(),
        'size': bin_width
    }

    # Choose color palette
    colors = cl.scales['9']['qual']['Set1']
    color_idx = 0

    # Distribution plot
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.3, 0.7],
        vertical_spacing=0.02
    )
    for name, group in df.groupby(groupby):
        if hasattr(name, '__len__') and not isinstance(name, str):
            group_name = '_'.join([str(i) for i in name])
        else:
            group_name = name
        color = colors[color_idx % len(colors)]
        color_idx += 1
        # Add to boxplot
        fig.add_trace(
            go.Box(
                x=group[column],
                name=group_name,
                legendgroup=group_name,
                showlegend=False,
                marker_color=color
            ),
            row=1,
            col=1
        )
        # Add to histogram
        fig.add_trace(
            go.Histogram(
                x=group[column],
                name=group_name,
                legendgroup=group_name,
                xbins=bin_config,
                histnorm='density' if density else '',
                marker_color=color,
                opacity=alpha
            ),
            row=2,
            col=1
        )
    fig.update_layout(barmode='overlay')
    fig.update_xaxes(title_text=column, row=2, col=1)
    yaxis_title = 'Density' if density else 'Frequency'
    fig.update_yaxes(title_text=yaxis_title, row=2, col=1)
    fig.show()


def group_missing_matrix(
    df,
    groupby,
    round_decimals=2,
    heatmap=True,
    foreground_colormap='bone',
    background_colormap='OrRd',
    sort=True,
    sort_by=None,
    ascending=False
):
    gb = df.groupby(groupby)

    # Calculate number of samples in each group
    sizes = gb.size().rename('Group Size')

    # Calculate missing values percentage of each column for each group
    missing = gb.apply(lambda g: g.isna().sum(axis=0))
    missing = missing.divide(sizes, axis=0) * 100
    missing.fillna(value=0, inplace=True)
    missing = missing.round(decimals=round_decimals)

    if missing.sum().sum() == 0:
        print('There are no missing values')
        return

    # Concatenate group sizes and missing value percentages
    final_table = pd.concat([sizes, missing], axis=1)

    # Sort columns to better identify groups with missing data
    all_columns = [col for col in missing.columns if col != groupby and col not in groupby]
    if sort:
        if sort_by is None:
            sort_by = all_columns
        final_table.sort_values(
            by=sort_by,
            axis=0,
            ascending=ascending,
            inplace=True
        )

    # Drop columns with no missing data
    non_missing = final_table.sum(axis=0) == 0
    final_table = final_table.loc[:, ~non_missing]

    colored_columns = [col for col in final_table if col in all_columns]

    # Apply conditional formatting to each cell except group size column
    if heatmap:
        fg_cmap = cm.get_cmap(foreground_colormap)
        bg_cmap = cm.get_cmap(background_colormap)
        norm = Normalize(vmin=0, vmax=100)
        def color_cell(value):
            fg_hex = to_hex(fg_cmap(norm(value)))
            bg_hex = to_hex(bg_cmap(norm(value)))
            return f"""
                color: {fg_hex};
                background-color: {bg_hex};
            """
        render = (
            final_table
            .style
            .applymap(
                func=color_cell,
                subset=pd.IndexSlice[:, colored_columns]
            )
            .format(
                formatter='{0:.2f} %',
                subset=pd.IndexSlice[:, colored_columns]
            )
        )
    else:
        render = (
            final_table
            .style
            .format(
                formatter='{0:.2f} %',
                subset=pd.IndexSlice[:, colored_columns]
            )
        )

    # Render table
    display(render)


def within_group_stats(df, groupby, column, stats, round_decimals=2):
    gb = df.groupby(groupby)[column]
    group_stats = []
    for name, func in stats.items():
        group_stats.append(
            gb
            .apply(func)
            .rename(name)
        )
    stats_table = pd.concat(group_stats, axis=1)
    stats_table = stats_table.round(decimals=round_decimals)
    display(stats_table)


def group_analysis(
    df,
    groupby,
    within_group_statistics=True,
    conditioned_missing_values=True,
    distribution_plots=True
):
    if conditioned_missing_values:
        display(Markdown('## Missing values for each group'))
        group_missing_matrix(df, groupby)

    if distribution_plots or within_group_statistics:
        for col in df.columns:
            if col != groupby and col not in groupby:
                display(Markdown(f'---'))
                display(Markdown(f'### *{col}*'))
                datatype = infer_data_type(df[col])
                if datatype == DataType.NUMERIC:
                    if within_group_statistics:
                        within_group_stats(dataset, groupby, col, DEFAULT_GROUP_DESCRIPTIVE_STATISTICS)
                        within_group_stats(dataset, groupby, col, DEFAULT_GROUP_QUANTILE_STATISTICS)
                    overlayed_histograms(df, groupby, col)
                else:
                    group_barplot(df, groupby, col)
```

```python
group_analysis(df=dataset, groupby=['Sex', 'Pclass'])
```

# Multivariate Analysis


## Principal component analysis

```python
COLUMNS_PCA = [c for c in COLUMN_SUBSET if not utils.is_categorical(dataset[c])]
```

## First vs Second principal component

```python
def pca_first_vs_second(df, standardize=True, figsize=(10,7)):
    pca = PCA(n_components=2)
    if standardize:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df[COLUMNS_PCA])
        pca_components = pca.fit_transform(data_scaled)
    else:
        pca_components = pca.fit_transform(df[COLUMNS_PCA])

    fig = plt.figure(figsize=figsize)
    plt.scatter(pca_components[:, 0], pca_components[:, 1], figure=fig)
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.show()
    print(f'Explained variance ratio: {pca.explained_variance_ratio_[:2].sum() * 100 :.2f}%')

pca_first_vs_second(dataset)
```

## Explained variance ratio

```python
def pca_explained_variance(df, standardize=True, figsize=(10,7), show_grid=True):
    pca = PCA()
    if standardize:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df[COLUMNS_PCA])
        pca.fit(data_scaled)
    else:
        pca.fit(df[COLUMNS_MULTIVARIATE])

    fig = plt.figure(figsize=figsize)
    plt.plot(pca.explained_variance_ratio_, figure=fig)
    plt.plot(np.cumsum(pca.explained_variance_ratio_), figure=fig)

    plt.legend(['Individual component', 'Cumulative'])
    plt.xlabel('Principal component #')
    plt.ylabel('Explained variance ratio')
    plt.xticks(
        ticks=range(len(pca.explained_variance_ratio_)),
        labels=range(1, (len(pca.explained_variance_ratio_) + 1))
    )
    if show_grid:
        plt.grid()
    plt.show()

pca_explained_variance(dataset)
```

## UMAP

```python
def plot_umap(
    df,
    columns=None,
    color_col=None,
    interactive=True,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    random_state=42,
    figsize=(15, 15),
    opacity=0.8,
    show_message=True
):
    if columns is None:
        columns = [col for col in df.columns if is_numeric(df[col])]
    else:
        for col in columns:
            if not is_numeric(df[col]):
                raise ValueError(f'UMAP cannot be computed for non-numeric column {col}')
    embedder = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    df = df.dropna()
    embedded = embedder.fit_transform(df[columns])

    # Multiplier which makes plotly interactive plots (size in pixels) and
    # matplotlib plots (size in inches) about the same size
    INCHES_TO_PIXELS = 64
    if interactive:
        layout=dict(
            width=figsize[0] * INCHES_TO_PIXELS,
            height=figsize[1] * INCHES_TO_PIXELS,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            legend=dict(title=f'<b>{color_col}</b>')
        )

    if color_col is not None:
        is_color_categorical = not is_numeric(df[color_col]) or utils.is_categorical(df[color_col])
        if interactive:
            fig = go.Figure(
                layout=layout
            )
            if is_color_categorical:
                df = df.copy()
                x_name, y_name = '__dart_umap_x', '__dart_umap_y'
                df[x_name] = embedded[:, 0]
                df[y_name] = embedded[:, 1]
                for group_name, group in df.groupby(color_col):
                    fig.add_trace(
                        go.Scatter(
                            x=group[x_name],
                            y=group[y_name],
                            mode='markers',
                            marker=dict(opacity=opacity),
                            name=group_name,
                            text=[
                                '</br>'.join(f'{col_name}: {df.loc[row, col_name]}'
                                             for col_name
                                             in group.columns.drop([x_name, y_name])
                                )
                                for row
                                in group.index
                            ],
                        )
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=embedded[:, 0],
                        y=embedded[:, 1],
                        mode='markers',
                        marker=dict(
                            color=df[color_col],
                            opacity=opacity,
                            colorbar=dict(title=color_col)
                        ),
                        text=['</br>'.join(f'{col_name}: {df.loc[row, col_name]}' for col_name in df.columns) for row in df.index],
                    ),
                )
            fig.show()
        else:
            if is_color_categorical:
                color_categorical = pd.Categorical(df[color_col])
                color_codes = color_categorical.codes
            else:
                color_codes = df[color_col]

            fig, ax = plt.subplots(figsize=figsize)
            scatter = ax.scatter(embedded[:, 0], embedded[:, 1], c=color_codes, alpha=opacity)
            if is_color_categorical:
                legend_elements = scatter.legend_elements()
                ax.legend(legend_elements[0], color_categorical.categories, title=color_col)
            else:
                cbar = plt.colorbar(scatter)
                cbar.ax.set_ylabel(color_col)
            # Remove ticks - the exact locations of embedded points are irrelevant
            ax.set_xticks([])
            ax.set_yticks([])
            plt.show()
    else:
        if interactive:
            fig = go.Figure(
                go.Scatter(
                        x=embedded[:, 0],
                        y=embedded[:, 1],
                        mode='markers',
                        marker=dict(opacity=opacity),
                        text=['</br>'.join(f'{col_name}: {df.loc[row, col_name]}' for col_name in df.columns) for row in df.index],
                ),
                layout=layout
            )
            fig.show()
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(embedded[:, 0], embedded[:, 1], alpha=opacity)
            # Remove ticks - the exact locations of embedded points are irrelevant
            ax.set_xticks([])
            ax.set_yticks([])
            plt.show()

    if show_message:
        print('UMAP requires proper setting of hyperparameters. ')
        print('If results are unsatisfactory, consider trying different values of parameters `n_neighbors`, `min_dist` and `metric`.')

plot_umap(dataset, color_col='rep78', interactive=True)
```

## Parallel coordinates

```python
def discrete_colorscale(n, saturation=0.5, lightness=0.5):
    """Generate a colorscale of n discrete colors
    equally spaced around the HSL wheel
    with constant saturation and lightness
    """
    for i in range(n):
        color = f'hsl({(i / n) * 360 :.2f}, {saturation * 100 :.2f}%, {lightness * 100 :.2f}%)'
        yield (i / n, color)
        yield ((i + 1) / n, color)

def parallel_coordinates(df, columns=None, drop_na=False, hide_columns=None, color_col=None, show_colorscale=True):
    if columns is None:
        columns = list(df.columns)
    if hide_columns is not None:
        columns = list(filter(lambda x: x not in hide_columns, columns))
    df = df.copy()
    if drop_na:
        df = df.dropna()
    if color_col is not None:
        categorical_color = not is_numeric(df[color_col])

        if categorical_color:
            categories = df[color_col].unique()
            colorscale = list(discrete_colorscale(len(categories)))
            # encode categories into numbers
            color_series = pd.Series(pd.Categorical(df[color_col]).codes)
        else:
            color_series = df[color_col]
            colorscale = 'Bluered_r'

        line = {
            'color': color_series,
            'colorscale': colorscale,
            'showscale': show_colorscale,
            'colorbar': { 'title': color_col, 'lenmode': 'pixels', 'len': 300 }
        }

        if categorical_color:
            line['colorbar'].update({
                'tickvals': color_series.unique(),
                'ticktext': categories,
                'lenmode': 'pixels',
                'len': min(40 * len(categories), 300)
            })
    else:
        line = None

    numeric_columns = [c for c in columns if is_numeric(df[c])]
    categorical_columns = [c for c in columns if not is_numeric(df[c])]
    # Add numeric columns to dimensions
    dimensions = [
        {
            'label': col_name,
            'values': dataset[col_name],
        }
        for col_name in numeric_columns
    ]
    # Add categorical columns to dimensions
    for col_name in categorical_columns:
        categories = df[col_name].unique()
        values = pd.Series(pd.Categorical(df[col_name]).codes)
        dimensions.append({
            'label': col_name,
            'values': values,
            'tickvals': values.unique(),
            'ticktext': categories
        })

    fig = go.Figure(
        go.Parcoords(
            line = line,
            dimensions = dimensions
        )
    )

    fig.show()

parallel_coordinates(dataset, hide_columns=['make'], color_col='mpg')
```

# Parallel categories

```python
def parallel_categories(df, columns=None, hide_columns=None, color_col=None):
    if columns is None:
        columns = [col for col in df.columns if utils.is_categorical(df[col])]
    if hide_columns is not None:
        columns = [col for col in columns if col not in hide_columns]
    if color_col is not None:
        categorical_color = not is_numeric(df[color_col])

        if categorical_color:
            categories = df[color_col].unique()
            colorscale = list(discrete_colorscale(len(categories)))
            # encode categories into numbers
            color_series = pd.Series(pd.Categorical(df[color_col]).codes)
        else:
            color_series = df[color_col]
            colorscale = 'Bluered_r'

        line = {
            'color': color_series,
            'colorscale': colorscale,
            'colorbar': {'title': color_col}
        }

        if categorical_color:
            line['colorbar'].update({
                'tickvals': color_series.unique(),
                'ticktext': categories,
                'lenmode': 'pixels',
                'len': min(40 * len(categories), 300)
            })
    dimensions = [
        go.parcats.Dimension(values=df[col_name], label=col_name)
        for col_name in columns
    ]

    fig = go.Figure(
        go.Parcats(
            dimensions=dimensions,
            line=line
        )
    )
    fig.show()

parallel_categories(dataset, hide_columns=['make'], color_col='mpg')
```

# Time series analysis



## Boxplots over time intervals

```python
# Load time series example dataset
dataset_ts = dart.example_datasets.dataset_pollution()
```

## Time analysis plot

```python
def time_analysis_plot(df, columns=None, separate_plots=False, color_col=None):
    if color_col is not None:
        _time_analysis_colored_plot(df, columns=columns, color_col=color_col)
        return
    if columns is None:
        columns = [col for col in df.columns if is_numeric(df[col])]
    else:
        for col in columns:
            if not is_numeric(df[col]):
                raise ValueError(f'Cannot plot timeanalysis plot for non-numeric column {col}')

    data = [
        go.Scatter(
            x=df.index,
            y=df[col],
            name=col,
            mode='lines'
        )
        for col in columns
    ]

    layout = dict(
        xaxis_rangeslider_visible=True
    )
    if separate_plots:
        for trace in data:
            display(Markdown(f'---\n### {trace.name}'))
            go.Figure(data=trace, layout=layout).show()
    else:
        go.Figure(data=data, layout=layout).show()

def _time_analysis_colored_plot(df, columns=None, color_col=None):
    if columns is None:
        columns = [col for col in df.columns if is_numeric(df[col])]
    else:
        for col in columns:
            if not is_numeric(df[col]):
                raise ValueError(f'Cannot plot timeanalysis plot for non-numeric column {col}')
    layout = dict(
        xaxis_rangeslider_visible=True
    )
    if not utils.is_categorical(df[color_col]):
        raise ValueError(f'Cannot color by non-categorical column `{color_col}`')
    if df[color_col].nunique() > 20:
        warnings.warn('Coloring by categorical column with many unique values!')
    df_color_shifted = df[color_col].shift(-1)
    for col in columns:
        data = [
            go.Scatter(
                x=df.index,
                # GroupBy would normally be preferred, but we want a connected line
                # Therefore, we also plot a connecting line
                # to the next point where category changes
                y=df[col].mask((df[color_col] != category) & (df_color_shifted != category)),
                name = str(category),
                mode='lines',
                connectgaps=False
            )
            for category in df[color_col].unique()
        ]
        display(Markdown(f'---\n### {col}'))
        fig = go.Figure(data=data, layout=layout).show()

time_analysis_plot(dataset_ts[:5000], color_col='wnd_dir')
```

## Rolling statistics

```python
def rolling_statistics(
    df, columns=None,
    show_bands=True, band_width=1., show_std_dev=True, window_size=20,
    color_std='#CD5C5C', color_mean='#2040FF', color_band='#90E0FF'
):
    if columns is None:
        columns = [col for col in df.columns if is_numeric(df[col])]
    else:
        for col in columns:
            if not is_numeric(df[col]):
                raise ValueError(f'Cannot plot rolling statistics for non-numeric column `{col}`')

    df_rolling = df[columns].rolling(window_size)
    df_rolling_mean = df_rolling.mean()[window_size - 1:]
    df_rolling_std = df_rolling.std()[window_size - 1:]
    index = df.index[window_size - 1:]
    layout = dict(
        xaxis_rangeslider_visible=True
    )
    data = []
    for col in columns:
        data.append([])
        if show_std_dev:
            trace_std = go.Scatter(
                x=index,
                y=df_rolling_std[col],
                mode='lines',
                name='Rolling std. dev.',
                line={'color': color_std}
            )
            data[-1].append(trace_std)

        trace_mean = go.Scatter(
            x=index,
            y=df_rolling_mean[col],
            mode='lines',
            name='Rolling mean',
            line={'color': color_mean }
        )
        data[-1].append(trace_mean)

        if show_bands:
            # Plot upper band
            trace_mean_plus_std = go.Scatter(
                x=index,
                y=df_rolling_mean[col] + band_width * df_rolling_std[col],
                mode='lines',
                name='Rolling mean + {} rolling std. dev.'.format('' if band_width == 1 else str(band_width) + ' * '),
                line={'color': color_band}
            )
            # Plot lower band
            trace_mean_minus_std = go.Scatter(
                x=index,
                y=df_rolling_mean[col] - band_width * df_rolling_std[col],
                mode='lines',
                name='Rolling mean - {} rolling std. dev.'.format('' if band_width == 1 else str(band_width) + ' * '),
                line={'color': color_band}
            )
            data[-1].extend([trace_mean_plus_std, trace_mean_minus_std])

    for col_name, col_data in zip(columns, data):
        display(Markdown(f'---\n### {col_name}'))
        go.Figure(data=col_data, layout=layout).show()

rolling_statistics(dataset_ts)
```

## Boxplots over time intervals

```python
def default_grouping_functions():
    return {
        'Hour': lambda x: f'{x.day}/{x.month}/{x.year} {x.hour}:00',
        'Day': lambda x: f'{x.day}/{x.month}/{x.year}',
        'Week': lambda x: f'W{x.week}, {x.year if x.dayofweek < x.dayofyear else x.year - 1}',
        'Month': lambda x: f'{x.month_name()[:3]} {x.year}',
        'Quarter': lambda x: f'Q{x.quarter} {x.year}',
        'Year': lambda x: f'{x.year}',
        'Decade': lambda x: f'{x.year // 10 * 10}s'
    }

def get_default_grouping_func(df, max_nvalues=80):
    # find most granular grouping which does not produce too many values
    for name, func in default_grouping_functions().items():
        if df.index.to_series().apply(func).nunique() < max_nvalues:
            return name, func
    else:
        # If no grouping is rough enough, use the roughest available
        return name, func

def boxplots_over_time(df, columns=None, grouping_function = None, grouping_name = None, figsize=(20, 7), color=None):
    default_grouping_funcs = default_grouping_functions()
    if grouping_name in default_grouping_funcs:
        grouping_func = default_grouping_funcs[grouping_name]
    elif grouping_function is None:
        grouping_name, grouping_function = get_default_grouping_func(df)

    if columns is None:
        columns = [col for col in df.columns if is_numeric(df[col])]
    for col in columns:
            if not is_numeric(df[col]):
                raise ValueError(
                    f'Cannot plot rolling statistics for non-numeric column `{col}`'
                )

    for column in columns:
        if not is_numeric(df[column]):
            raise ValueError(f'Cannot plot boxplot for non-numeric column {column}')
        display(Markdown('---'))
        display(Markdown(f'## {column}'))
        ax = sns.boxplot(
            x=df.index.to_series().apply(grouping_function),
            y=df[column],
            color=color
        )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        if grouping_name is not None:
            ax.set_xlabel(grouping_name)
        ax.figure.set_size_inches(*figsize)

        plt.show()

boxplots_over_time(dataset_ts)
```

## Decomposition

```python
def timeseries_decomposition(df, columns=None, figsize=(20, 10), period=None):
    df = df.interpolate(method='time')
    if pd.infer_freq(df.index) is None and period is None:
        warnings.warn(f'Period could not be inferred, please set the period parameter to a suitable value. Decomposition will not be plotted.')
        return
    if columns is None:
        columns = [col for col in df.columns if is_numeric(df[col])]
    else:
        for col in columns:
            if not is_numeric(df[col]):
                raise ValueError(f'Only numeric columns are supported, column {col} does not appear to be numeric.')
    for col in columns:
        display(Markdown(f'---\n### {col}'))
        decomposition = sm.tsa.seasonal_decompose(df[col], period=period)
        fig = decomposition.plot()
        fig.set_size_inches(*figsize)
        fig.axes[0].set_title(None)
        fig.axes[0].set_ylabel('Original')
        fig.axes[-1].set_ylabel('Residual')
        plt.show()

timeseries_decomposition(dataset_ts, period=12)
```

## Stationarity tests

```python
def default_stationarity_tests():
    return {
        'KPSS Test (constant)': partial(sm.tsa.stattools.kpss, regression='c', nlags='auto'),
        'KPSS Test (trend)': partial(sm.tsa.stattools.kpss, regression='ct', nlags='auto'),
        'Augmented Dickey-Fuller Test': sm.tsa.stattools.adfuller
    }

def stationarity_tests(df, columns=None, kpss_const=True, kpss_trend=True, adfuller=True):
    df = df.copy().dropna()
    if columns is None:
        columns = df.columns
    stat_tests = default_stationarity_tests()
    if not kpss_const:
        stat_tests.pop('KPSS Test (constant)', None)
    if not kpss_trend:
        stat_tests.pop('KPSS Test (trend)', None)
    if not adfuller:
        stat_tests.pop('Augmented Dickey-Fuller Test', None)

    columns = [col for col in columns if is_numeric(df[col])]
    for col in columns:
        test_values_df = pd.DataFrame()
        display(Markdown(f'---\n### {col}'))
        for name, func in stat_tests.items():
            with warnings.catch_warnings(record=True) as w:
                test_vals = func(df[col])
            stat, pvalue = test_vals[:2]
            value_dict = {
                'Test statistic': format_number(stat, thousand_separator=' '),
                'P-value': ('<' if len(w) >= 1 else '') + format_number(pvalue, thousand_separator= ' '),
            }
            value_series = pd.Series(value_dict)
            test_values_df[name] = value_series
        display(test_values_df.style)

stationarity_tests(dataset_ts)
```

## Autocorrelation

```python
def plot_acf(df, columns=None, lags=None, figsize=(15, 5), partial=False):
    if columns is None:
        columns = [col for col in df.columns if is_numeric(df[col])]
    else:
        for col in columns:
            if not is_numeric(df[col]):
                raise ValueError(f'Cannot plot autocorrelation for non-numeric column `{col}`')

    plot_func = tsaplots.plot_pacf if partial else tsaplots.plot_acf
    for col in columns:
        display(Markdown(f'---\n### {col}'))
        fig = plot_func(df[col].dropna(), lags=lags)
        ax = fig.axes[0]
        ax.set_title('')
        ax.set_xlabel('Lag')
        ax.set_ylabel(('Partial ' if partial else '') + 'Autocorrelation')
        fig.set_size_inches(*figsize)
        plt.show()

plot_acf(dataset_ts)
```

## Partial autocorrelation

```python
plot_pacf = partial(plot_acf, partial=True)

plot_pacf(dataset_ts)
```

## Fourier transform

```python
def fft(df, sampling_rate, columns=None, figsize=(15, 6), log=False, freq_min=None, freq_max=None):
    if columns is None:
        columns = [col for col in df.columns if is_numeric(df[col])]
    else:
        for col in columns:
            if not is_numeric(df[col]):
                raise ValueError(f'Cannot perform Fourier transform for non-numeric column `{col}`')
    index_freq = pd.infer_freq(df.index) or ''
    for col in columns:
        # FFT requires samples at regular intervals
        df_col = df[col].interpolate(method='time')
        df_col_centered = df_col - df_col.mean()
        fft_result = np.fft.fft(df_col_centered)

        amplitude = np.abs(fft_result) * 2 / len(df)
        fft_freq = np.fft.fftfreq(len(amplitude), 1. / sampling_rate)
        idx_pos_freq = fft_freq > 0
        fft_freq, amplitude = fft_freq[idx_pos_freq], amplitude[idx_pos_freq]

        y = 10 * np.log10(amplitude) if log else amplitude
        fig, ax = plt.subplots(figsize=figsize)
        ax.stem(fft_freq, y, use_line_collection=True, markerfmt='')
        ax.set_xlabel(f'Frequency [1 / {sampling_rate}{index_freq}]')
        ax.set_ylabel('Amplitude' + (' [dB]' if log else ''))
        ax.set_xlim(freq_min, freq_max)
        display(Markdown(f'---\n### {col}'))
        plt.show()

# Hourly data with sampling rate 24 -> 24 daily samples
fft(dataset_ts, 24)
```

## Short-time Fourier transform

```python
def stft(
    df, sampling_rate, window_size, overlap=None, log=True,
    columns=None, window='hann', scaling='spectrum', figsize=(20, 7),
    freq_min=None, freq_max=None
):
    if columns is None:
        columns = [col for col in df.columns if is_numeric(df[col])]
    else:
        for col in columns:
            if not is_numeric(df[col]):
                raise ValueError(f'Cannot perform STFT for non-numeric column {col}')
    index_freq = pd.infer_freq(df.index) or ''
    for col in columns:
        display(Markdown(f'---\n### {col}'))
        freqs, times, Sx = signal.spectrogram(
            # interpolate to get samples at regular time intervals
            df[col].interpolate(method='time'),
            fs=sampling_rate,
            window=window,
            nperseg=window_size,
            # Overlap defaults to window_size // 8
            noverlap=overlap,
            scaling=scaling
        )

        # Add small positive value to avoid 0 in log
        y = 10 * np.log10(Sx + 1e-12) if log else Sx

        f, ax = plt.subplots(figsize=figsize)
        ax.pcolormesh(times, freqs, y, cmap='viridis')

        ax.set_ylabel(f'Frequency [1/({sampling_rate}{index_freq})]')
        ax.set_xlabel('Time')
        ax.set_ylim(freq_min, freq_max)
        # Show times from index in xticks
        ax.set_xticklabels(
            df.index[
                list(map(lambda time: int(time * sampling_rate), ax.get_xticks()[:-1]))
            ]
        )
        plt.show()

# Hourly data -> 24 samples/day with weekly windows
stft(dataset_ts, 24, 168)
```
