# ---
# jupyter:
#   jupytext:
#     formats: md:myst,ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: revamp
#     language: python
#     name: revamp
# ---

# %% [markdown]
# # New

# %%
# %load_ext autoreload
# %autoreload 2
import os
from functools import cache
from typing import List, Literal, Optional, Tuple

import ms3
import pandas as pd
import plotly.express as px
from dimcat import Pipeline, plotting

import utils

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)

# %%
RESULTS_PATH = os.path.abspath(os.path.join(utils.OUTPUT_FOLDER, "couperin_study"))
os.makedirs(RESULTS_PATH, exist_ok=True)


def make_output_path(
    filename: str,
    extension=None,
    path=RESULTS_PATH,
) -> str:
    return utils.make_output_path(filename=filename, extension=extension, path=path)


def save_figure_as(
    fig, filename, formats=("png", "pdf"), directory=RESULTS_PATH, **kwargs
):
    if formats is not None:
        for fmt in formats:
            plotting.write_image(fig, filename, directory, format=fmt, **kwargs)
    else:
        plotting.write_image(fig, filename, directory, **kwargs)


def style_plotly(fig, save_as=None, **layout):
    layout_args = dict(utils.STD_LAYOUT, **layout)
    fig.update_layout(**layout_args)
    fig.update_xaxes(gridcolor="lightgrey")
    fig.update_yaxes(gridcolor="lightgrey")
    if save_as:
        save_figure_as(fig, save_as)
    fig.show()


# %% [markdown]
# **Loading data**

# %%
D = utils.get_dataset("couperin_concerts", corpus_release="v2.2")
D

# %% [markdown]
# **Grouping data**

# %%
pipeline = Pipeline(["KeySlicer", "ModeGrouper"])
grouped_D = pipeline.process(D)
grouped_D

# %%
bass_notes = grouped_D.get_feature("bassnotes")
bass_notes.df

# %%
bass_notes.intervals_over_bass.iloc[0]

# %%
local_keys = grouped_D.get_feature("KeyAnnotations")
utils.print_heading("Key Segments")
print(local_keys.groupby("mode").size().to_string())
local_keys.head()

# %%
preceding = bass_notes.groupby(["piece", "localkey_slice"]).shift()
preceding.columns = "preceding_" + preceding.columns
subsequent = bass_notes.groupby(["piece", "localkey_slice"]).shift(-1)
subsequent.columns = "subsequent_" + subsequent.columns
BN = pd.concat([bass_notes, preceding, subsequent], axis=1)
BN["preceding_iv"] = BN.bass_note - BN.preceding_bass_note
BN["subsequent_iv"] = BN.subsequent_bass_note - BN.bass_note
BN["preceding_interval"] = ms3.transform(BN.preceding_iv, ms3.fifths2iv, smallest=True)
BN["subsequent_interval"] = ms3.transform(
    BN.subsequent_iv, ms3.fifths2iv, smallest=True
)
BN["preceding_iv_is_step"] = BN.preceding_iv.isin(
    (-5, -2, 2, 5)
).where(  # +m2, -M2, +M2, -m2
    BN.preceding_iv.notna()
)
BN["subsequent_iv_is_step"] = BN.subsequent_iv.isin((-5, -2, 2, 5)).where(
    BN.subsequent_iv.notna()
)
BN["preceding_iv_is_0"] = BN.preceding_iv == 0
BN["subsequent_iv_is_0"] = BN.subsequent_iv == 0
BN["preceding_movement"] = (
    BN.preceding_iv_is_step.map({True: "step", False: "leap"})
    .where(~BN.preceding_iv_is_0, "same")
    .where(BN.preceding_iv.notna(), "none")
)
BN["subsequent_movement"] = (
    BN.subsequent_iv_is_step.map({True: "step", False: "leap"})
    .where(~BN.subsequent_iv_is_0, "same")
    .where(BN.subsequent_iv.notna(), "none")
)
BN["preceding_movement_precise"] = BN.preceding_movement.where(
    BN.preceding_movement != "step", BN.preceding_interval
)
BN["subsequent_movement_precise"] = BN.subsequent_movement.where(
    BN.subsequent_movement != "step", BN.subsequent_interval
)
BN

# %%
ignore_mask = BN.subsequent_interval.isna() | BN.subsequent_interval.duplicated()
interval2fifths = (  # mapping that allows to order the x-axis with intervals according to LoF
    BN.loc[~ignore_mask, ["subsequent_interval", "subsequent_iv"]]
    .set_index("subsequent_interval")
    .iloc[:, 0]
    .sort_values()
)

# %% [markdown]
# ## Bass movement

# %%
interval_data = pd.concat(
    [
        BN.groupby("mode").subsequent_interval.value_counts(normalize=True),
        BN.groupby(["piece", "mode"])
        .subsequent_interval.value_counts(normalize=True)
        .groupby(["mode", "subsequent_interval"])
        .sem()
        .rename("std_err"),
    ],
    axis=1,
).reset_index()
fig = px.bar(
    interval_data,
    x="subsequent_interval",
    y="proportion",
    color="mode",
    barmode="group",
    error_y="std_err",
    color_discrete_map=utils.MAJOR_MINOR_COLORS,
    labels=dict(subsequent_interval="Interval"),
    title="Mode-wise proportion of how often a bass note moves by an interval",
    category_orders=dict(subsequent_interval=interval2fifths.index),
)
style_plotly(fig, "how_often_a_bass_note_moves_by_an_interval")

# %%
movement_data = pd.concat(
    [
        BN.groupby("mode").subsequent_movement.value_counts(
            normalize=True, dropna=False
        ),
        BN.groupby(["piece", "mode"])
        .subsequent_movement.value_counts(normalize=True, dropna=False)
        .groupby(["mode", "subsequent_movement"])
        .sem()
        .rename("std_err"),
    ],
    axis=1,
).reset_index()
movement_data.subsequent_movement = movement_data.subsequent_movement.fillna("none")
fig = px.bar(
    movement_data,
    x="subsequent_movement",
    y="proportion",
    color="mode",
    barmode="group",
    error_y="std_err",
    color_discrete_map=utils.MAJOR_MINOR_COLORS,
    labels=dict(subsequent_movement="Movement"),
    title="Mode-wise proportion of a bass note moving in a certain manner",
    category_orders=dict(subsequent_interval=interval2fifths.index),
)
style_plotly(fig, save_as="mode-wise_bass_motion")


# %%
def make_sankey_data(
    five_major, color_edges=True, precise=True
) -> Tuple[pd.DataFrame, List[str], List[str]] | Tuple[pd.DataFrame, List[str]]:
    preceding_movement = (
        "preceding_movement_precise" if precise else "preceding_movement"
    )
    subsequent_movement = (
        "subsequent_movement_precise" if precise else "subsequent_movement"
    )
    type_counts = five_major["intervals_over_bass"].value_counts()
    preceding_movement_counts = five_major[preceding_movement].value_counts()
    subsequent_movement_counts = five_major[subsequent_movement].value_counts()
    preceding_links = five_major.groupby(
        [preceding_movement]
    ).intervals_over_bass.value_counts()
    subsequent_links = five_major.groupby(
        [subsequent_movement]
    ).intervals_over_bass.value_counts()

    node_labels = []
    label_ids = dict()
    for key, node_sizes in (
        ("preceding", preceding_movement_counts),
        ("intervals", type_counts),
        ("subsequent", subsequent_movement_counts),
    ):
        for label in node_sizes.index:
            label_id = len(node_labels)
            node_labels.append(str(label))
            label_ids[(key, label)] = label_id

    edge_columns = ["source", "target", "value"]
    if color_edges:
        node_colors = utils.make_evenly_distributed_color_map(node_labels)
        edge_columns.append("color")

    links = []
    for (prec_mov, iv), cnt in preceding_links.items():
        source_id = label_ids.get(("preceding", prec_mov))
        target_id = label_ids.get(("intervals", iv))
        if color_edges:
            edge_color = node_colors[source_id]
            links.append((source_id, target_id, cnt, edge_color))
        else:
            links.append((source_id, target_id, cnt))

    for (subs_mov, iv), cnt in subsequent_links.items():
        source_id = label_ids.get(("intervals", iv))
        target_id = label_ids.get(("subsequent", subs_mov))
        if color_edges:
            edge_color = node_colors[target_id]
            links.append((source_id, target_id, cnt, edge_color))
        else:
            links.append((source_id, target_id, cnt))

    edge_data = pd.DataFrame(links, columns=edge_columns)
    if color_edges:
        return edge_data, node_labels, node_colors
    return edge_data, node_labels


def make_bass_degree_sankey(
    bass_degree: str, mode: Literal["major", "minor"], **layout
):
    edge_data, node_labels, node_colors = make_sankey_data(
        BN.loc[mode].query(f"bass_degree == '{bass_degree}'")
    )
    fig = utils.make_sankey(edge_data, node_labels, node_color=node_colors, **layout)
    return fig


# %% [markdown]
# ## Intervals over bass degree 1
# ### Major

# %%
make_bass_degree_sankey(1, "major")

# %% [markdown]
# ### Minor

# %%
make_bass_degree_sankey(1, "minor")

# %% [markdown]
# ## Intervals over bass degree 2
# ### Major

# %%
make_bass_degree_sankey(2, "major")

# %% [markdown]
# ### Minor

# %%
make_bass_degree_sankey(2, "minor")

# %% [markdown]
# ## Intervals over bass degree 3
# ### Major

# %%
make_bass_degree_sankey(3, "major")

# %% [markdown]
# ### Minor

# %%
make_bass_degree_sankey(3, "minor")

# %% [markdown]
# ## Intervals over bass degree 4
# ### Major

# %%
make_bass_degree_sankey(4, "major")

# %% [markdown]
# ### Minor

# %%
make_bass_degree_sankey(4, "minor")

# %% [markdown]
# ## Intervals over bass degree 5
# ### Major

# %%
make_bass_degree_sankey(5, "major")

# %% [markdown]
# ### Minor

# %%
make_bass_degree_sankey(5, "minor")

# %% [markdown]
# ## Intervals over bass degree 6
# ### Major

# %%
make_bass_degree_sankey(6, "major")

# %% [markdown]
# ### Minor

# %%
make_bass_degree_sankey(6, "minor")

# %% [markdown]
# ## Intervals over bass degree 7
# ### Major

# %%
make_bass_degree_sankey(7, "major")

# %% [markdown]
# ### Minor

# %%
make_bass_degree_sankey(7, "minor")

# %% [markdown]
# ## Explanatory power of the RoO

# %%
BN.groupby(["mode", "bass_degree"]).intervals_over_bass.apply(
    lambda S: S.value_counts().idxmax()
)

# %%
maj = ("M3", "P5")
maj6 = ("m3", "m6")
min = ("m3", "P5")
min6 = ("M3", "M6")
Mm56 = ("m3", "d5", "m6")
Mm34 = ("m3", "P4", "M6")
Mm24 = ("M2", "a4", "M6")
mm56 = ("M3", "P5", "M6")
hdim56 = ("m3", "P5", "M6")
hdim34 = ("M3", "a4", "M6")

regole = dict(
    ascending_major=[
        ("1", maj),  # most frequent
        ("2", Mm34),  # most frequent
        ("3", maj6),  # most frequent
        ("4", mm56),  # not most frequent
        ("5", maj),  # most frequent
        ("6", min6),  # not most frequent
        ("7", Mm56),  # most frequent
    ],
    descending_major=[
        ("1", maj),  # same
        ("7", maj6),  # different, not most frequent
        ("6", Mm34),  # different, not most frequent either
        ("5", maj),  # same
        ("4", Mm24),  # different, not most frequent either
        ("3", maj6),  # same
        ("2", Mm34),  # same
    ],
    ascending_minor=[
        ("1", min),  # most frequent
        ("2", Mm34),  # most frequent
        ("3", min6),  # most frequent
        ("4", hdim56),  # most frequent
        ("5", maj),  # most frequent
        ("#6", maj6),  # most frequent
        ("#7", Mm56),  # most frequent
    ],
    descending_minor=[
        ("1", min),  # same
        ("7", min6),  # different, most frequent
        ("6", hdim34),  # different, most frequent
        ("5", maj),  # same
        ("4", Mm24),  # different, not most frequent
        ("3", min6),  # same
        ("2", Mm34),  # same
    ],
)


# %%
@cache
def get_base_df(
    basis: Literal[
        "major_all", "minor_all", "major_diatonic", "minor_diatonic"
    ],  # minor_diatonic includes 6, #6, 7, #7
    query: Optional[str] = None,
):
    global BN
    try:
        mode, selection = basis.split("_")
    except Exception:
        raise ValueError(f"Invalid keyword for basis: {basis!r}")
    base = BN.loc[[mode]]
    if selection == "all":
        result = base
    elif selection == "diatonic":
        if mode == "major":
            result = base.query("bass_degree in ('1', '2', '3', '4', '5', '6', '7')")
        elif mode == "minor":
            result = base.query(
                "bass_degree in ('1', '2', '3', '4', '5', '6', '#6', '7', '#7')"
            )
    else:
        raise ValueError(f"Unknown keyword for selection: {selection!r}")
    if query:
        result = result.query(query)
    return result


@cache
def get_bass_degree_mask(
    basis: Literal[
        "major_all", "minor_all", "major_diatonic", "minor_diatonic"
    ],  # minor_diatonic includes 6, #6, 7, #7
    bass_degree: str,
    query: Optional[str] = None,
):
    base = get_base_df(basis, query=query)
    return base.bass_degree == bass_degree


@cache
def get_intervals_mask(
    basis: Literal[
        "major_all", "minor_all", "major_diatonic", "minor_diatonic"
    ],  # minor_diatonic includes 6, #6, 7, #7
    intervals: tuple,
    query: Optional[str] = None,
):
    base = get_base_df(basis, query=query)
    return base.intervals_over_bass == intervals


@cache
def get_chord_mask(
    basis: Literal[
        "major_all", "minor_all", "major_diatonic", "minor_diatonic"
    ],  # minor_diatonic includes 6, #6, 7, #7
    bass_degree: str,
    intervals: tuple,
    query: Optional[str] = None,
):
    bass_degree_mask = get_bass_degree_mask(
        basis=basis, bass_degree=bass_degree, query=query
    )
    intervals_mask = get_intervals_mask(basis=basis, intervals=intervals, query=query)
    return bass_degree_mask & intervals_mask


@cache
def get_chord_vocabulary_mask(
    basis: Literal[
        "major_all", "minor_all", "major_diatonic", "minor_diatonic"
    ],  # minor_diatonic includes 6, #6, 7, #7
    vocabulary: Tuple[Tuple[str, tuple], ...],
    query: Optional[str] = None,
) -> pd.Series:
    base = get_base_df(basis, query=query)
    mask = pd.Series(False, index=base.index, dtype="boolean")
    for bass_degree, intervals in vocabulary:
        mask |= get_chord_mask(
            basis=basis, bass_degree=bass_degree, intervals=intervals, query=query
        )
    return mask


def get_vocabulary_coverage(
    basis: Literal[
        "major_all", "minor_all", "major_diatonic", "minor_diatonic"
    ],  # minor_diatonic includes 6, #6, 7, #7
    vocabulary: Tuple[Tuple[str, tuple], ...],
    query: Optional[str] = None,
) -> float:
    mask = get_chord_vocabulary_mask(basis=basis, vocabulary=vocabulary, query=query)
    return mask.sum() / len(mask)


# %%
regola_vocabulary_major = tuple(
    set(regole["ascending_major"] + regole["descending_major"])
)
regola_vocabulary_minor = tuple(
    set(regole["ascending_minor"] + regole["descending_minor"])
)
get_vocabulary_coverage("major_all", regola_vocabulary_major)

# %%
get_vocabulary_coverage("major_diatonic", regola_vocabulary_major)

# %%
get_vocabulary_coverage("minor_all", regola_vocabulary_minor)

# %%
get_vocabulary_coverage("minor_diatonic", regola_vocabulary_minor)
