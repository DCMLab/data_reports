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

import ms3
import pandas as pd
from dimcat import Pipeline, plotting
import plotly.express as px

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
bass_notes
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
BN["preceding_iv_is_step"] = (BN.preceding_iv.isin((-5, -2, 2, 5)) # +m2, -M2, +M2, -m2
                              .where(BN.preceding_iv.notna()))
BN["subsequent_iv_is_step"] = (BN.subsequent_iv.isin((-5, -2, 2, 5))
                               .where(BN.subsequent_iv.notna()))
BN["preceding_iv_is_0"] = BN.preceding_iv == 0
BN["subsequent_iv_is_0"] = BN.subsequent_iv == 0
BN["preceding_movement"] = (BN.preceding_iv_is_step.map({True: "step", False: "leap"})
                            .where(~BN.preceding_iv_is_0, "same")
                            .where(BN.preceding_iv.notna()))
BN["subsequent_movement"] = (BN.subsequent_iv_is_step.map({True: "step", False: "leap"})
                            .where(~BN.subsequent_iv_is_0, "same")
                            .where(BN.subsequent_iv.notna()))
BN
```

```{code-cell}
ignore_mask = BN.subsequent_interval.isna() | BN.subsequent_interval.duplicated()
interval2fifths = BN.loc[~ignore_mask, ["subsequent_interval", "subsequent_iv"]].set_index("subsequent_interval").iloc[:,0].sort_values()
interval2fifths
```

```{code-cell}
interval_data = BN.groupby("mode").subsequent_interval.value_counts(dropna=False, normalize=True).reset_index()
fig = px.bar(
    interval_data,
    x="subsequent_interval",
    y="proportion",
    color="mode",
    facet_row="mode",
    labels=dict(subsequent_interval="Interval"),
    title="Mode-wise proportion of how often a bass note moves by an interval",
    category_orders=dict(subsequent_interval=interval2fifths.index)
)
style_plotly(fig, "how_often_a_bass_note_moves_by_an_interval")
```

```{code-cell}
movement_data = BN.groupby("mode").subsequent_movement.value_counts(dropna=False, normalize=True).reset_index()
movement_data.subsequent_movement = movement_data.subsequent_movement.fillna("none")
movement_data
```

```{code-cell}
fig = px.bar(
    movement_data,
    x="subsequent_movement",
    y="proportion",
    color="mode",
    facet_row="mode",
    labels=dict(subsequent_movement="Movement"),
    title="Mode-wise proportion of a bass note moving in a certain manner",
    category_orders=dict(subsequent_interval=interval2fifths.index)
)
style_plotly(fig, save_as="mode-wise_bass_motion")
```

```{code-cell}

```
