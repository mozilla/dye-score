from bokeh.models import (
    ColumnDataSource,
    LinearAxis,
    NumeralTickFormatter,
    Range1d,
)
from bokeh.plotting import figure


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
