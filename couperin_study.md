---
jupytext:
  formats: md:myst,ipynb,py:percent
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: revamp
  language: python
  name: revamp
---

# Cou

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide imports
  code_prompt_show: Show imports
tags: [hide-cell]
---
%load_ext autoreload
%autoreload 2

import itertools
import os
from functools import cache
from typing import List, Literal, Optional, Tuple

import ms3
import numpy as np
import pandas as pd
import plotly.express as px
from dimcat import Pipeline, plotting

import utils

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
```

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide helpers
  code_prompt_show: Show helpers
tags: [hide-cell]
---
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


def style_plotly(
    fig,
    save_as=None,
    xaxes: Optional[dict] = None,
    yaxes: Optional[dict] = None,
    match_facet_yaxes=False,
    **layout,
):
    layout_args = dict(utils.STD_LAYOUT, **layout)
    fig.update_layout(**layout_args)
    xaxes_settings = dict(gridcolor="lightgrey")
    if xaxes:
        xaxes_settings.update(xaxes)
    fig.update_xaxes(**xaxes_settings)
    yaxes_settings = dict(gridcolor="lightgrey")
    if yaxes:
        yaxes_settings.update(yaxes)
    fig.update_yaxes(**yaxes_settings)
    if match_facet_yaxes:
        for row_idx, row_figs in enumerate(fig._grid_ref):
            for col_idx, col_fig in enumerate(row_figs):
                fig.update_yaxes(
                    row=row_idx + 1,
                    col=col_idx + 1,
                    matches="y" + str(len(row_figs) * row_idx + 1),
                )
    if save_as:
        save_figure_as(fig, save_as)
    fig.show()
```

**Loading data**

```{code-cell}
:tags: [hide-input]

D = utils.get_dataset("couperin_concerts", corpus_release="v2.2")
D
```

**Grouping data**

```{code-cell}
:tags: [hide-input]

pipeline = Pipeline(["KeySlicer", "ModeGrouper"])
grouped_D = D.apply_step(pipeline)
grouped_D
```

**Starting point: DiMCAT's BassNotes feature**

```{code-cell}
:tags: [hide-input]

bass_notes = D.apply_step(pipeline).get_feature("bassnotes")
bass_notes.df
```

**If needed, the `localkey_slice` intervals can be resolved using this table:**

```{code-cell}
:tags: [hide-input]

local_keys = grouped_D.get_feature("KeyAnnotations")
utils.print_heading("Key Segments Couperin")
print(local_keys.groupby("mode").size().to_string())
local_keys.head()
```

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide helpers
  code_prompt_show: Show helpers
tags: [hide-cell]
---
succession_map = dict(
    ascending_major={
        "1": "2",
        "2": "3",
        "3": "4",
        "4": "5",
        "5": "6",
        "6": "7",
        "7": "1",
    },
    ascending_minor={
        "1": "2",
        "2": "3",
        "3": "4",
        "4": "5",
        "5": "#6",
        "#6": "#7",
        "#7": "1",
    },
    descending={"1": "7", "2": "1", "3": "2", "4": "3", "5": "4", "6": "5", "7": "6"},
)


def inverse_dict(d):
    return {v: k for k, v in d.items()}


predecessor_map = dict(
    ascending_major=inverse_dict(succession_map["ascending_major"]),
    ascending_minor=inverse_dict(succession_map["ascending_minor"]),
    descending=inverse_dict(succession_map["descending"]),
)


def make_precise_preceding_movement_column(df):
    """Expects a dataframe containing the columns bass_degree, preceding_bass_degree, and preceding_movement,"""
    preceding_movement_precise = df.preceding_movement.where(
        df.preceding_movement != "step", df.preceding_interval
    )
    expected_ascending_degree = pd.concat(
        [
            df.loc[["major"], "bass_degree"].map(predecessor_map["ascending_major"]),
            df.loc[["minor"], "bass_degree"].map(predecessor_map["ascending_minor"]),
        ]
    )
    expected_descending_degree = df.bass_degree.map(predecessor_map["descending"])
    preceding_movement_precise = preceding_movement_precise.where(
        df.preceding_bass_degree != expected_ascending_degree, "ascending"
    )
    preceding_movement_precise = preceding_movement_precise.where(
        df.preceding_bass_degree != expected_descending_degree, "descending"
    )
    return preceding_movement_precise


def make_precise_subsequent_movement_column(df):
    """Expects a dataframe containing the columns bass_degree, subsequent_bass_degree, and subsequent_movement,"""
    subsequent_movement_precise = df.subsequent_movement.where(
        df.subsequent_movement != "step", df.subsequent_interval
    )
    expected_ascending_degree = pd.concat(
        [
            df.loc[["major"], "bass_degree"].map(succession_map["ascending_major"]),
            df.loc[["minor"], "bass_degree"].map(succession_map["ascending_minor"]),
        ]
    )
    expected_descending_degree = df.bass_degree.map(succession_map["descending"])
    subsequent_movement_precise = subsequent_movement_precise.where(
        df.subsequent_bass_degree != expected_ascending_degree, "ascending"
    )
    subsequent_movement_precise = subsequent_movement_precise.where(
        df.subsequent_bass_degree != expected_descending_degree, "descending"
    )
    return subsequent_movement_precise
```

**This is the main table of this notebook. It corresponds to the `BassNotes` features,
with a `preceding_` and a `subsequent_` copy of each column concatenated to the right.
The respective upward and downward shifts are performed within each localkey group,
leaving first bass degrees with undefined preceding values and last bass degrees without
undefined subsequent values.**

```{code-cell}
:tags: [hide-input]

def make_adjacency_table(bass_notes):
    preceding = bass_notes.groupby(["piece", "localkey_slice"]).shift()
    preceding.columns = "preceding_" + preceding.columns
    subsequent = bass_notes.groupby(["piece", "localkey_slice"]).shift(-1)
    subsequent.columns = "subsequent_" + subsequent.columns
    BN = pd.concat([bass_notes, preceding, subsequent], axis=1)
    BN["preceding_iv"] = BN.bass_note - BN.preceding_bass_note
    BN["subsequent_iv"] = BN.subsequent_bass_note - BN.bass_note
    BN["preceding_interval"] = ms3.transform(
        BN.preceding_iv, ms3.fifths2iv, smallest=True
    )
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
    BN["preceding_movement_precise"] = make_precise_preceding_movement_column(BN)
    BN["subsequent_movement_precise"] = make_precise_subsequent_movement_column(BN)
    return BN


BN = make_adjacency_table(bass_notes)
```

```{code-cell}
:tags: [hide-input]

ignore_mask = BN.subsequent_interval.isna() | BN.subsequent_interval.duplicated()
interval2fifths = (  # mapping that allows to order the x-axis with intervals according to LoF
    BN.loc[~ignore_mask, ["subsequent_interval", "subsequent_iv"]]
    .set_index("subsequent_interval")
    .iloc[:, 0]
    .sort_values()
)
```

## Overview of how the bass moves
### Intervals

```{code-cell}
:tags: [hide-input]

def plot_bass_movement(BN, corpus_name):
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
        title=f"Mode-wise proportion of how often a bass note moves by an interval in {corpus_name}",
        category_orders=dict(subsequent_interval=interval2fifths.index),
    )
    style_plotly(fig, f"how_often_a_bass_note_moves_by_an_interval_{corpus_name}")


plot_bass_movement(BN, "Couperin")
```

### Types of movement

**The values `ascending` and `descending` designate stepwise movement within the _regola_. Only non-chromatic scale
degrees can have these values with the exception of `#6` and `#7` which are considered diatonic in the context of
this study.**

```{code-cell}
:tags: [hide-input]

def plot_movement_types(BN, corpus_name, precise_categories=True):
    subsequent_movement = (
        "subsequent_movement_precise" if precise_categories else "subsequent_movement"
    )
    movement_data = pd.concat(
        [
            BN.groupby("mode")[subsequent_movement].value_counts(
                normalize=True, dropna=False
            ),
            BN.groupby(["piece", "mode"])[subsequent_movement]
            .value_counts(normalize=True, dropna=False)
            .groupby(["mode", subsequent_movement])
            .sem()
            .rename("std_err"),
        ],
        axis=1,
    ).reset_index()
    movement_data[subsequent_movement] = movement_data[subsequent_movement].fillna(
        "none"
    )
    fig = px.bar(
        movement_data,
        x=subsequent_movement,
        y="proportion",
        color="mode",
        barmode="group",
        error_y="std_err",
        color_discrete_map=utils.MAJOR_MINOR_COLORS,
        labels={subsequent_movement: "Movement"},
        title=f"Mode-wise proportion of a bass note moving in a certain manner in {corpus_name}",
        category_orders=dict(subsequent_interval=interval2fifths.index),
    )
    style_plotly(fig, save_as=f"mode-wise_bass_motion_{corpus_name}")


plot_movement_types(BN, "Couperin")
```

## Sankey diagrams showing movement types before and after each scale degree

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide helpers
  code_prompt_show: Show helpers
tags: [hide-cell]
---
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
    BN: pd.DataFrame,
    corpus: str,
    mode: Literal["major", "minor"],
    bass_degree: Optional[str | int] = None,
    **layout,
):
    """bass_degree None means all unigrams."""
    selected_unigrams = BN.loc[mode]
    if bass_degree:
        selected_unigrams = selected_unigrams.query(f"bass_degree == '{bass_degree}'")
        selection_text = f"bass degree {bass_degree}"
    else:
        selection_text = "any harmony"
    edge_data, node_labels, node_colors = make_sankey_data(selected_unigrams)

    title = f"Motions to and from {selection_text} in {corpus} ({mode})"
    fig = utils.make_sankey(
        edge_data, node_labels, node_color=node_colors, title=title, **layout
    )
    return fig
```

### All unigrams
#### Major

```{code-cell}
make_bass_degree_sankey(BN, "Couperin", "major")
```

#### Minor

```{code-cell}
make_bass_degree_sankey(BN, "Couperin", "minor")
```

### Intervals over bass degree 1
#### Major

```{code-cell}
:tags: [hide-input]

make_bass_degree_sankey(BN, "Couperin", "major", 1)
```

#### Minor

```{code-cell}
:tags: [hide-input]

make_bass_degree_sankey(BN, "Couperin", "minor", 1)
```

```{code-cell}
make_bass_degree_sankey(BN, "Corelli", "minor")
```

### Intervals over bass degree 2
#### Major

```{code-cell}
make_bass_degree_sankey(BN, "Couperin", "major", 2)
```

#### Minor

```{code-cell}
make_bass_degree_sankey(BN, "Couperin", "minor", 2)
```

### Intervals over bass degree 3
#### Major

```{code-cell}
make_bass_degree_sankey(BN, "Couperin", "major", 3)
```

#### Minor

```{code-cell}
make_bass_degree_sankey(BN, "Couperin", "minor", 3)
```

### Intervals over bass degree 4
#### Major

```{code-cell}
make_bass_degree_sankey(BN, "Couperin", "major", 4)
```

#### Minor

```{code-cell}
make_bass_degree_sankey(BN, "Couperin", "minor", 4)
```

### Intervals over bass degree 5
#### Major

```{code-cell}
make_bass_degree_sankey(BN, "Couperin", "major", 5)
```

#### Minor

```{code-cell}
make_bass_degree_sankey(BN, "Couperin", "minor", 5)
```

### Intervals over bass degree 6
#### Major

```{code-cell}
make_bass_degree_sankey(BN, "Couperin", "major", 6)
```

#### Minor (ascending)

```{code-cell}
make_bass_degree_sankey(BN, "Couperin", "minor", "#6")
```

#### Minor (descending)

```{code-cell}
make_bass_degree_sankey(BN, "Couperin", "minor", 6)
```

### Intervals over bass degree 7
#### Major

```{code-cell}
make_bass_degree_sankey(BN, "Couperin", "major", 7)
```

#### Minor (ascending)

```{code-cell}
make_bass_degree_sankey(BN, "Couperin", "minor", "#7")
```

#### Minor (descending)

```{code-cell}
make_bass_degree_sankey(BN, "Couperin", "minor", 7)
```

## Explanatory power of the RoO
### Defining the vocabulary

```{code-cell}
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
        ("6", maj6),  # not most frequent
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
regola_vocabulary_major = tuple(
    set(regole["ascending_major"] + regole["descending_major"])
)
regola_vocabulary_minor = tuple(
    set(regole["ascending_minor"] + regole["descending_minor"])
)
```

### Most frequent chords for each bass degree

```{code-cell}
:tags: [hide-input]

def summarize_groups_top_k_chords(df, column="intervals_over_bass", k=3):
    """Used in Groupby.apply()"""
    proportions = df[column].value_counts(normalize=True)
    entropy = -(proportions * np.log2(proportions)).sum()
    N = len(proportions)
    normalized_entropy = entropy / np.log2(N) if N > 1 else 0.0
    top_k = proportions.iloc[:k]
    rank_col = list(range(1, len(top_k) + 1))
    result = pd.DataFrame(
        dict(
            intervals_over_bass=top_k.index,
            proportion=top_k.values,
            normalized_entropy=normalized_entropy,
        ),
        index=rank_col,
    ).rename_axis("rank_chord")
    return result


def rank_bass_degrees(df: pd.Series):
    """Used in Groupby.apply()"""
    vc = df.bass_degree.value_counts(normalize=True).to_frame()
    vc["rank_bass"] = list(range(1, len(vc) + 1))
    return vc


def summarize_degree_wise_top_k(BN, column="intervals_over_bass", k=3):
    result = (
        BN.groupby(["mode", "bass_degree"]).apply(
            summarize_groups_top_k_chords, column=column, k=k
        )
    ).reset_index(level=-1)
    bass_proportions = BN.groupby("mode").apply(rank_bass_degrees)
    result = result.join(bass_proportions, lsuffix="_chord", rsuffix="_bass")
    return result


def degree_wise_top_k(BN, column="intervals_over_bass", k=3):
    summary = summarize_degree_wise_top_k(BN, column=column, k=k)
    result = []
    for mode, df in summary.groupby("mode"):
        vocab = regola_vocabulary_major if mode == "major" else regola_vocabulary_minor
        is_regola = (
            df.reset_index(level="bass_degree")[["bass_degree", "intervals_over_bass"]]
            .apply(tuple, axis=1)
            .isin(vocab)
        ).values
        df["is_regola"] = is_regola
        df = (
            df.sort_values(["rank_bass", "rank_chord"])
            .reset_index("bass_degree")
            .reset_index("mode", drop=True)
            .set_index(
                [
                    "rank_bass",
                    "bass_degree",
                    "proportion_bass",
                    "normalized_entropy",
                    "rank_chord",
                ]
            )
        )[["intervals_over_bass", "proportion_chord", "is_regola"]]
        result.append(df)
    return result


def style_rank_table(df: pd.DataFrame):

    def color_true_green(value):
        if value:
            return "background-color: lightgreen"
        return None

    new_index_names = dict(
        rank_bass="Rank",
        bass_degree="Bass Degree",
        proportion_bass="Proportion",
        normalized_entropy="Entropy",
        rank_chord="Top",
    )
    df = df.rename_axis(index=new_index_names)
    return (
        df.style.format({"proportion_chord": "{:.1%}"})
        .format_index(
            axis=0,
            formatter={
                "Proportion": "{:.1%}",  # Format as percentage with 1 decimal place
                "Entropy": "{:.3f}",  # Format as float with 3 decimal places
            },
        )
        .map(color_true_green, subset=["is_regola"])
        .relabel_index(["Chord", "Proportion", "Regola"], axis=1)
        # .format_index_names(new_index_names, axis=0) # available in a future pandas version
        # https://pandas.pydata.org/docs/dev/reference/api/pandas.io.formats.style.Styler.format_index_names.html
    )
```

#### Major

```{code-cell}
:tags: [hide-input]

major, minor = degree_wise_top_k(BN)
style_rank_table(major)
```

#### Minor

```{code-cell}
style_rank_table(minor)
```

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide helpers
  code_prompt_show: Show helpers
tags: [hide-cell]
---
name2BN = {"couperin": BN}


@cache
def get_base_df(
    bn_name: str,
    basis: Literal["major_all", "minor_all", "major_diatonic", "minor_diatonic"],
    query: Optional[str] = None,
):
    BN = name2BN[bn_name]
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
    bn_name: str,
    basis: Literal["major_all", "minor_all", "major_diatonic", "minor_diatonic"],
    bass_degree: str,
    query: Optional[str] = None,
):
    base = get_base_df(bn_name, basis, query=query)
    return base.bass_degree == bass_degree


@cache
def get_intervals_mask(
    bn_name,
    basis: Literal["major_all", "minor_all", "major_diatonic", "minor_diatonic"],
    intervals: tuple,
    query: Optional[str] = None,
):
    base = get_base_df(bn_name, basis, query=query)
    return base.intervals_over_bass == intervals


@cache
def get_chord_mask(
    bn_name,
    basis: Literal["major_all", "minor_all", "major_diatonic", "minor_diatonic"],
    bass_degree: str,
    intervals: tuple,
    query: Optional[str] = None,
):
    bass_degree_mask = get_bass_degree_mask(
        bn_name, basis=basis, bass_degree=bass_degree, query=query
    )
    intervals_mask = get_intervals_mask(
        bn_name, basis=basis, intervals=intervals, query=query
    )
    return bass_degree_mask & intervals_mask


@cache
def get_chord_vocabulary_mask(
    bn_name,
    basis: Literal["major_all", "minor_all", "major_diatonic", "minor_diatonic"],
    vocabulary: Tuple[Tuple[str, tuple], ...],
    query: Optional[str] = None,
) -> pd.Series:
    base = get_base_df(bn_name, basis, query=query)
    mask = pd.Series(False, index=base.index, dtype="boolean")
    for bass_degree, intervals in vocabulary:
        mask |= get_chord_mask(
            bn_name,
            basis=basis,
            bass_degree=bass_degree,
            intervals=intervals,
            query=query,
        )
    return mask


def inspect(
    bn_name,
    basis: Literal["major_all", "minor_all", "major_diatonic", "minor_diatonic"],
    vocabulary: Tuple[Tuple[str, tuple], ...],
    query: Optional[str] = None,
) -> pd.DataFrame:
    base = get_base_df(bn_name, basis, query=query)
    mask = get_chord_vocabulary_mask(
        bn_name, basis=basis, vocabulary=vocabulary, query=query
    )
    return base[mask]


def get_vocabulary_coverage(
    bn_name,
    basis: Literal["major_all", "minor_all", "major_diatonic", "minor_diatonic"],
    vocabulary: Tuple[Tuple[str, tuple], ...],
    query: Optional[str] = None,
) -> float:
    mask = get_chord_vocabulary_mask(
        bn_name, basis=basis, vocabulary=vocabulary, query=query
    )
    return mask.sum() / len(mask)


def get_coverage_values(
    bn_name,
    major_vocabulary: Optional[Tuple[Tuple[str, tuple], ...]] = None,
    minor_vocabulary: Optional[Tuple[Tuple[str, tuple], ...]] = None,
    **name2query,
) -> pd.Series:
    if not (major_vocabulary or minor_vocabulary):
        return pd.Series()
    results = {}
    if major_vocabulary:
        results.update(
            {
                ("major", "all"): get_vocabulary_coverage(
                    bn_name, "major_all", major_vocabulary
                ),
                ("major", "diatonic"): get_vocabulary_coverage(
                    bn_name, "major_diatonic", major_vocabulary
                ),
            }
        )
        for name, query in name2query.items():
            results[("major", name)] = get_vocabulary_coverage(
                bn_name, "major_diatonic", major_vocabulary, query=query
            )
    if minor_vocabulary:
        results.update(
            {
                ("minor", "all"): get_vocabulary_coverage(
                    bn_name, "minor_all", minor_vocabulary
                ),
                ("minor", "diatonic"): get_vocabulary_coverage(
                    bn_name, "minor_diatonic", minor_vocabulary
                ),
            }
        )
        for name, query in name2query.items():
            results[("minor", name)] = get_vocabulary_coverage(
                bn_name, "minor_diatonic", minor_vocabulary, query=query
            )
    result = pd.Series(results, name="proportion")
    result.index.names = ["mode", "coverage_of"]
    return result
```

### Which proportion of unigrams are "explained" by Campion's regola

The percentages are based on different sets of unigrams.
`from` means before/leading to a bass degree, `to` means after/following a bass degree.

* `all`: all bass degrees
* `diatonic`: all non-chromatic bass degrees (in minor, the chromatic scale degrees `#6` and `#7` are considered
  diatonic)
* `to_ascending`: all diatonic bass degrees that ascend within the regola
* `from_ascending`: all diatonic bass degrees that are reached by ascending within the regola
* `to_and_from_ascending`: all diatonic bass degrees that are reached by ascending within the regola and proceed
  ascending within the regola
* `to_and_from_either`: all diatonic bass degrees whose predecessor and successor are both upper or lower neighbors
  within the regola
* `to_leap`: all diatonic bass degrees followed by a leap
* `to_same`: all diatonic bass degrees followed by the same bass degree
* etc.

```{code-cell}
:tags: [hide-input]

regola_vocabulary_major = tuple(
    set(regole["ascending_major"] + regole["descending_major"])
)
regola_vocabulary_minor = tuple(
    set(regole["ascending_minor"] + regole["descending_minor"])
)

features = dict(
    to_ascending="subsequent_movement_precise == 'ascending'",
    to_descending="subsequent_movement_precise == 'descending'",
    to_either="subsequent_movement_precise == ['ascending', 'descending']",
    to_leap="subsequent_movement == 'leap'",
    to_same="subsequent_movement == 'same'",
    last_notes="subsequent_movement == 'none'",
    from_ascending="preceding_movement_precise == 'ascending'",
    from_descending="preceding_movement_precise == 'descending'",
    from_either="preceding_movement_precise == ['ascending', 'descending']",
    from_leap="preceding_movement == 'leap'",
    from_same="preceding_movement == 'same'",
    first_notes="preceding_movement == 'none'",
    to_and_from_ascending="subsequent_movement_precise == 'ascending' & preceding_movement_precise == 'ascending'",
    to_and_from_descending="subsequent_movement_precise == 'descending' & preceding_movement_precise == 'descending'",
    to_and_from_either="subsequent_movement_precise == ['ascending', 'descending'] & "
    "preceding_movement_precise == ['ascending', 'descending']",
    to_and_from_leap="subsequent_movement == 'leap' & preceding_movement == 'leap'",
    to_and_from_same="subsequent_movement == 'same' & preceding_movement == 'same'",
)

regola_coverage = get_coverage_values(
    "couperin", regola_vocabulary_major, regola_vocabulary_minor, **features
)
utils.print_heading(
    "What percentage of each unigram category the RoO covers in Couperin"
)
regola_coverage
```

### Comparing the regola against all "top k" vocabularies

**Campion's regola comprises 10 different chords for both major and minor.
For comparison, its values are shown at point 10.5 on the x-axis.
The lower two plots show how many unigrams are covered by individual chords.
Hover over the points to see the corresponding chords.**

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide helpers
  code_prompt_show: Show helpers
tags: [hide-cell]
---
def make_coverage_plot_data(
    bn_name, include_singular_vocabularies=True, **features
) -> pd.DataFrame:
    BN = name2BN[bn_name]
    all_chords = BN[["bass_degree", "intervals_over_bass"]].apply(tuple, axis=1)
    chord_ranking = all_chords.groupby("mode").value_counts(normalize=True)
    major_ranking, minor_ranking = (
        chord_ranking.loc["major"],
        chord_ranking.loc["minor"],
    )
    major_vocab, minor_vocab = [], []
    results = {}
    for i, (maj_chord, min_chord) in enumerate(
        itertools.zip_longest(major_ranking.index, minor_ranking.index), 1
    ):
        if maj_chord:
            major_vocab.append(maj_chord)
        if min_chord:
            minor_vocab.append(min_chord)
        key = ("cumulative", i) if include_singular_vocabularies else i
        values = get_coverage_values(
            bn_name, tuple(major_vocab), tuple(minor_vocab), **features
        )
        chord = pd.Series(str(maj_chord), index=values.index, name="chord")
        chord.loc["minor"] = str(min_chord)
        results[key] = pd.concat([values, chord], axis=1)
        if not include_singular_vocabularies:
            continue
        single_maj_vocab = (maj_chord,) if maj_chord else None
        single_min_vocab = (min_chord,) if min_chord else None
        values = get_coverage_values(
            bn_name, single_maj_vocab, single_min_vocab, **features
        )
        results[("single", i)] = pd.concat([values, chord], axis=1)
    index_levels = ["vocabulary", "rank"] if include_singular_vocabularies else ["rank"]
    return pd.concat(results, names=index_levels)
```

```{code-cell}
:tags: [hide-input]

def plot_regola_vs_top_k_coverage(bn_name):
    result = make_coverage_plot_data(bn_name, **features)
    regola_results = pd.concat(
        {("cumulative", 10.5): regola_coverage}, names=["vocabulary", "rank"]
    ).to_frame()
    regola_results.loc[:, "chord"] = "regola"
    result = pd.concat(
        [
            regola_results,
            result,
        ]
    ).sort_index()
    fig = px.line(
        result.reset_index(),
        x="rank",
        y="proportion",
        color="coverage_of",
        facet_col="mode",
        facet_row="vocabulary",
        hover_name="chord",
        log_x=True,
        title=f"How many {bn_name.title()} unigrams are covered by each top-k vocabulary",
    )
    style_plotly(
        fig,
        match_facet_yaxes=True,
        height=1500,
        legend=dict(
            orientation="h",
        ),
    )


plot_regola_vs_top_k_coverage("couperin")
```

**In order to inspect these plots you will want to hide traces.
Click on a legend item to toggle it, double-click on an item to toggle all others.**
