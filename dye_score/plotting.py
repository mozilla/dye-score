import os
from bokeh.models import (
    CDSView,
    ColumnDataSource,
    IndexFilter,
    LinearAxis,
    NumeralTickFormatter,
    Range1d,
)
from bokeh.palettes import inferno
from bokeh.plotting import figure
from numpy import histogram
from pandas import (
    read_csv as pd_read_csv
)


def plot_hist(title, hist, edges, y_axis_type='linear', bottom=0):
    p = figure(title=title, tools='', background_fill_color="#fafafa", y_axis_type=y_axis_type, width=800, height=400)
    p.quad(top=hist, bottom=bottom, left=edges[:-1], right=edges[1:], fill_color="navy", line_color="white", alpha=0.5)
    p.y_range.start = bottom
    p.grid.grid_line_color = "white"
    return p


def plot_key_leaky(percent_to_dye, key, y_axis_type='linear', bottom=0, bins=40):
    hist, edges = histogram(percent_to_dye, bins=bins)
    p = plot_hist(f'Distribution of leaky - {key}', hist, edges, y_axis_type=y_axis_type, bottom=bottom)
    p.width = 400
    p.height = 300
    p.xaxis.axis_label = '%age dye_site wants to dye'
    p.yaxis.axis_label = 'Frequency'
    return p


def get_pr_plot(
    pr_df, title, plot_opts, n_scripts_range,
    y_range=(0, 1), recall_color='black', n_scripts_color='firebrick',
):
    """Example code for plotting dye score threshold plots"""
    source = ColumnDataSource(pr_df)
    p = figure(title=title, y_range=y_range, **plot_opts)
    p.x_range.flipped = True
    p.line(
        x='dye_score_threshold', y='recall', color=recall_color, source=source, line_width=3)
    p.extra_y_ranges = {"n_scripts": Range1d(start=n_scripts_range[0], end=n_scripts_range[1])}
    p.line(
        x='dye_score_threshold', y='n_over_threshold', y_range_name='n_scripts', color=n_scripts_color, source=source)
    p.add_layout(
        LinearAxis(
            y_range_name="n_scripts",
            formatter=NumeralTickFormatter(format="0a"),
            axis_label_text_color=n_scripts_color,
            major_label_text_color=n_scripts_color
        ), 'right'
    )
    return p


def get_plots_for_thresholds(ds, thresholds, leaky_threshold, pr_plot_opts, filename_suffix='dye_snippets'):
    resultsdir = ds.config('DYESCORE_RESULTS_DIR')

    # Infile validation
    for threshold in thresholds:
        inpath = os.path.join(
            resultsdir, f'dye_score_plot_data_from_{filename_suffix}_{threshold}_leak_{leaky_threshold}.csv')
        ds.file_in_validation(inpath)

    plots = {}
    for threshold in thresholds:
        inpath = os.path.join(
            resultsdir, f'dye_score_plot_data_from_{filename_suffix}_{threshold}_leak_{leaky_threshold}.csv')
        if ds.s3:
            with ds.s3.open(inpath, 'r') as f:
                pr_df = pd_read_csv(f)
        else:
            pr_df = pd_read_csv(inpath)
        plots[threshold] = get_pr_plot(pr_df, title=f'{threshold}', pr_plot_opts=pr_plot_opts)
    return plots


def get_threshold_summary_plot(ds):
    resultsdir = ds.config('DYESCORE_RESULTS_DIR')
    inpath = os.path.join(resultsdir, f'recall_summary_plot_data.csv')
    ds.file_out_validation(inpath)
    if ds.s3:
        with ds.s3.open(inpath, 'r') as f:
            results_df = pd_read_csv(f)
    else:
        results_df = pd_read_csv(inpath)
    recall_thresholds = sorted(results_df.recall_threshold.unique())
    grouped_results_df = results_df.groupby('recall_threshold').agg(lambda x: list(x))
    palette = inferno(len(recall_thresholds) + 1)  # The yellow is often a little light
    source = ColumnDataSource(grouped_results_df)
    p = figure(
        title='Scripts captured by distance threshold for 5 recall thresholds (colored)',
        width=800, toolbar_location=None,
        tools='', y_range=Range1d(results_df.n_over_threshold.min(), results_df.n_over_threshold.max()),
        background_fill_color='slategray'
    )
    p.xaxis.axis_label = 'distance threshold'
    p.yaxis.axis_label = 'minimum n_scripts'
    p.yaxis.formatter = NumeralTickFormatter(format="0a")
    p.extra_y_ranges = {'percent': Range1d(results_df.percent.min(), results_df.percent.max())}
    p.add_layout(LinearAxis(
        y_range_name='percent', axis_label='minimum n_scripts (percent of total)',
        formatter=NumeralTickFormatter(format='0%')
    ), 'right')
    for i, recall_threshold in enumerate(recall_thresholds):
        view = CDSView(source=source, filters=[IndexFilter([i])])
        opts = dict(
            source=source, view=view, legend=str(recall_threshold),
            color=palette[i], line_width=5, line_alpha=0.6
        )
        p.multi_line(xs='distance_threshold', ys='n_over_threshold', **opts)
        p.multi_line(xs='distance_threshold', ys='percent',  y_range_name='percent', **opts)
    p.legend.click_policy = 'hide'
    return p
