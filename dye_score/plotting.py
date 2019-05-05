from bokeh.models import (
    ColumnDataSource,
    LinearAxis,
    NumeralTickFormatter,
    Range1d,
)
from bokeh.plotting import figure
from numpy import histogram

def plot_hist(title, hist, edges, y_axis_type='linear', bottom=0):
    p = figure(title=title, tools='', background_fill_color="#fafafa", y_axis_type=y_axis_type, width=800, height=400)
    p.quad(top=hist, bottom=bottom, left=edges[:-1], right=edges[1:], fill_color="navy", line_color="white", alpha=0.5)
    p.y_range.start = bottom
    p.grid.grid_line_color = "white"
    return p

def plot_key_leaky(percent_to_dye, key, y_axis_type='linear', bottom=0):
    hist, edges = histogram(percent_to_dye, bins=40)
    p = plot_hist(f'Distribution of leaky - {key}', hist, edges, y_axis_type=y_axis_type, bottom=bottom)
    p.width = 400
    p.height = 300
    p.xaxis.axis_label = '%age dye_site wants to dye'
    p.yaxis.axis_label = 'Frequency'
    return p

def get_pr_plot(pr_df, title, plot_opts, recall_color='black', n_scripts_color='firebrick'):
    """Example code for plotting dye score threshold plots"""
    source = ColumnDataSource(pr_df)
    p = figure(title=title, y_range=(0, 1), **plot_opts)
    p.x_range.flipped = True
    p.line(
        x='dye_score_threshold', y='recall', color=recall_color, source=source, line_width=3)
    p.extra_y_ranges = {"n_scripts": Range1d(start=0, end=35_000)}
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
