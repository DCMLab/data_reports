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

# New

```{code-cell}

%load_ext autoreload
%autoreload 2

import os
from typing import List, Literal, Tuple

import ms3
import pandas as pd
import plotly.express as px
from dimcat import Pipeline, plotting

import utils

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
```

```{code-cell}
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
```

**Loading data**

```{code-cell}
D = utils.get_dataset("couperin_concerts", corpus_release="v2.2")
D
```

**Grouping data**

```{code-cell}
pipeline = Pipeline(["KeySlicer", "ModeGrouper"])
grouped_D = pipeline.process(D)
grouped_D
```

```{code-cell}
bass_notes = grouped_D.get_feature("bassnotes")
bass_notes.df
```

```{code-cell}
bass_notes.intervals_over_bass.iloc[0]
```

```{code-cell}
local_keys = grouped_D.get_feature("KeyAnnotations")
utils.print_heading("Key Segments")
print(local_keys.groupby("mode").size().to_string())
local_keys.df
```

```{code-cell}
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
BN
```

```{code-cell}
ignore_mask = BN.subsequent_interval.isna() | BN.subsequent_interval.duplicated()
interval2fifths = (  # mapping that allows to order the x-axis with intervals according to LoF
    BN.loc[~ignore_mask, ["subsequent_interval", "subsequent_iv"]]
    .set_index("subsequent_interval")
    .iloc[:, 0]
    .sort_values()
)
```

```{code-cell}
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
```

```{code-cell}
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
```

```{code-cell}
def make_sankey_data(
    five_major, color_edges=True
) -> Tuple[pd.DataFrame, List[str], List[str]] | Tuple[pd.DataFrame, List[str]]:
    type_counts = five_major["intervals_over_bass"].value_counts()
    preceding_movement_counts = five_major["preceding_movement"].value_counts()
    subsequent_movement_counts = five_major["subsequent_movement"].value_counts()
    preceding_links = five_major.groupby(
        ["preceding_movement"]
    ).intervals_over_bass.value_counts()
    subsequent_links = five_major.groupby(
        ["subsequent_movement"]
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
```

## Intervals over bass degree 1
### Major

```{code-cell}
make_bass_degree_sankey(1, "major")
```

### Minor

```{code-cell}
make_bass_degree_sankey(1, "minor")
```

## Intervals over bass degree 2
### Major

```{code-cell}
make_bass_degree_sankey(2, "major")
```

### Minor

```{code-cell}
make_bass_degree_sankey(2, "minor")
```

## Intervals over bass degree 3
### Major

```{code-cell}
make_bass_degree_sankey(3, "major")
```

### Minor

```{code-cell}
make_bass_degree_sankey(3, "minor")
```

## Intervals over bass degree 4
### Major

```{code-cell}
make_bass_degree_sankey(4, "major")
```

### Minor

```{code-cell}
make_bass_degree_sankey(4, "minor")
```

## Intervals over bass degree 5
### Major

```{code-cell}
make_bass_degree_sankey(5, "major")
```

### Minor

```{code-cell}
make_bass_degree_sankey(5, "minor")
```

## Intervals over bass degree 6
### Major

```{code-cell}
make_bass_degree_sankey(6, "major")
```

### Minor

```{code-cell}
make_bass_degree_sankey(6, "minor")
```

## Intervals over bass degree 7
### Major

```{code-cell}
make_bass_degree_sankey(7, "major")
```

### Minor

```{code-cell}
make_bass_degree_sankey(7, "minor")
```

```{code-cell}

```
