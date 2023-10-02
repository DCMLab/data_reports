import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

plt.style.use("ggplot")

STD_LAYOUT = {
    "paper_bgcolor": "#FFFFFF",
    "plot_bgcolor": "#FFFFFF",
    "margin": {"l": 40, "r": 0, "b": 0, "t": 0, "pad": 0},
}


def cumulative_fraction(S, start_from_zero=False):
    """Accumulate the value counts of a Series so they can be plotted."""
    values_df = S.value_counts().to_frame("x").reset_index()
    total = values_df.x.sum()
    values_df["y"] = values_df.x.cumsum() / total
    if start_from_zero:
        return pd.concat(
            [pd.DataFrame({"chord": pd.NA, "x": 0, "y": 0.0}, index=[0]), values_df],
            ignore_index=True,
        )
    return values_df


def nest_level(obj, include_tuples=False):
    """Recursively calculate the depth of a nested list."""
    if obj.__class__ != list:
        if include_tuples:
            if obj.__class__ != tuple:
                return 0
        else:
            return 0
    max_level = 0
    for item in obj:
        max_level = max(max_level, nest_level(item, include_tuples=include_tuples))
    return max_level + 1


def plot_cum(
    S=None,
    cum=None,
    x_log=False,
    markersize=2,
    left_range=(-0.1, 4.40),
    right_range=(-0.023, 1.099),
    **kwargs,
):
    """Pass either a Series or cumulative_fraction(S).reset_index()"""
    if cum is None:
        cum = cumulative_fraction(S).reset_index()
        cum.index = cum.index + 1
    fig = make_subplots(
        specs=[
            [
                {
                    "secondary_y": True,
                }
            ]
        ]
    )
    ix = cum.index
    fig.add_trace(
        go.Scatter(
            x=ix,
            y=cum.x,
            text=cum["index"],
            name="Absolute count",
            mode="markers",
            marker=dict(size=markersize),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=ix,
            y=cum.y,
            text=cum["index"],
            name="Cumulative fraction",
            mode="markers",
            marker=dict(size=markersize),
        ),
        secondary_y=True,
    )
    fig.update_xaxes(
        title_text="Rank of chord label", zeroline=False, gridcolor="lightgrey"
    )
    if x_log:
        ranks = np.log(len(ix)) / np.log(10)
        fig.update_xaxes(type="log", range=(-0.01 * ranks, 1.01 * ranks))
    else:
        ranks = len(ix)
        fig.update_xaxes(range=(-0.02 * ranks, 1.02 * ranks))
    fig.update_yaxes(
        title_text="Absolute label count",
        secondary_y=False,
        type="log",
        gridcolor="grey",
        zeroline=True,
        dtick=1,
        range=left_range,
    )
    fig.update_yaxes(
        title_text="Cumulative fraction",
        secondary_y=True,
        gridcolor="lightgrey",
        zeroline=False,
        dtick=0.1,
        range=right_range,
    )
    fig.update_layout(**kwargs)
    return fig
