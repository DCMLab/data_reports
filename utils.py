import numpy as np
import colorlover
import plotly.express as px

STD_LAYOUT = {
 'paper_bgcolor': '#FFFFFF',
 'plot_bgcolor': '#FFFFFF',
 'margin': {'l': 40, 'r': 0, 'b': 0, 't': 40, 'pad': 0},
 'font': {'size': 15}
}

CADENCE_COLORS = dict(zip(('HC', 'PAC', 'PC', 'IAC', 'DC', 'EC'), colorlover.scales['6']['qual']['Set1']))
CORPUS_COLOR_SCALE = px.colors.qualitative.D3
TYPE_COLORS = dict(zip(('Mm7', 'M', 'o7', 'o', 'mm7', 'm', '%7', 'MM7', 'other'), colorlover.scales['9']['qual']['Paired']))

def color_background(x, color="#ffffb3"):
    """Format DataFrame cells with given background color."""
    return np.where(x.notna().to_numpy(), f"background-color: {color};", None)


def value_count_df(S, thing=None, counts='counts'):
    """Value counts as DataFrame where the index has the name of the given Series or ``thing`` and where the counts
    are given in the column ``counts``.
    """
    thing = S.name if thing is None else thing
    df = S.value_counts().rename(counts).to_frame()
    df.index.rename(thing, inplace=True)
    return df

