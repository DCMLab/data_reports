---
jupytext:
  formats: md:myst,ipynb,py:percent
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
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
from typing import Iterable, List, Literal, Optional, Tuple

import ms3
import numpy as np
import pandas as pd
import plotly.express as px
from dimcat import Pipeline, plotting
from dimcat.data import resources

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
    font_size=30,
    **layout,
):
    layout_args = dict(utils.STD_LAYOUT, **layout)
    if font_size:
        font = layout_args.pop("font", {})
        font["size"] = font_size
        layout_args["font"] = font
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
    return fig
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
MAJOR_REGOLA_RN = {
    "I": "both",
    "V43": "both",
    "I6": "both",
    "ii65": "ascending",
    "V2": "descending",
    "V": "both",
    "IV6": "ascending",
    "V43/V": "descending",
    "V65": "ascending",
    "V6": "descending",
}
MINOR_REGOLA_RN = {
    "i": "both",
    "V43": "both",
    "i6": "both",
    "ii%65": "ascending",
    "V2": "descending",
    "V": "both",
    "IV6": "ascending",
    "ii%43": "descending",
    "V65": "ascending",
    "v6": "descending",
}


def make_regola_chord_column(df):
    chord_category = pd.concat(
        [
            df.loc[["major"], "chord"].map(MAJOR_REGOLA_RN),
            df.loc[["minor"], "chord"].map(MINOR_REGOLA_RN),
        ]
    )
    return chord_category.rename("roo_chord")


def make_regola_suspensions_column(df):
    major_roo_chords = tuple(MAJOR_REGOLA_RN.keys())
    minor_roo_chords = tuple(MINOR_REGOLA_RN.keys())
    major_roo_suspensions = tuple(chord + "(" for chord in major_roo_chords)
    minor_roo_suspensions = tuple(chord + "(" for chord in minor_roo_chords)
    major = df.loc[["major"], "chord"]
    major = major.where(~major.str.startswith(major_roo_suspensions), "RoO suspension")
    major = major.where(~major.isin(major_roo_chords), "RoO chord")
    major = major.where(major.isin(("RoO chord", "RoO suspension")), "Other")
    minor = df.loc[["minor"], "chord"]
    minor = minor.where(~minor.str.startswith(minor_roo_suspensions), "RoO suspension")
    minor = minor.where(~minor.isin(minor_roo_chords), "RoO chord")
    minor = minor.where(minor.isin(("RoO chord", "RoO suspension")), "Other")
    roo_suspensions = pd.concat([major, minor])
    return roo_suspensions.rename("roo_suspensions")


roo_succession_map = dict(
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


roo_predecessor_map = dict(
    ascending_major=inverse_dict(roo_succession_map["ascending_major"]),
    ascending_minor=inverse_dict(roo_succession_map["ascending_minor"]),
    descending=inverse_dict(roo_succession_map["descending"]),
)


def get_all_non_neighbours(lst, ix):
    N = len(lst)
    if ix >= N:
        return []
    left = (ix - 1) % N
    right = (ix + 1) % N
    return [item for i, item in enumerate(lst) if i not in (left, ix, right)]


def make_roo_leap_maps():
    steps = ["1", "2", "3", "4", "5", "6", "7"]
    asc_minor_steps = ["1", "2", "3", "4", "5", "#6", "#7"]
    major = {
        step: set(get_all_non_neighbours(steps, i)) for i, step in enumerate(steps)
    }
    minor = {
        step: set(get_all_non_neighbours(asc_minor_steps, i))
        for i, step in enumerate(asc_minor_steps)
    }
    for k, v in major.items():
        if k in minor:
            minor[k].update(v)
        else:
            minor[k] = v
    return major, minor


def make_roo_step_maps():
    major = {k: {v} for k, v in roo_succession_map["ascending_major"].items()}
    minor_preceding = {
        k: {v} for k, v in roo_predecessor_map["ascending_minor"].items()
    }
    minor_subsequent = {
        k: {v} for k, v in roo_succession_map["ascending_minor"].items()
    }
    for k, v in roo_predecessor_map["descending"].items():
        if k in minor_preceding:
            minor_preceding[k].add(v)
        else:
            minor_preceding[k] = {v}
    for k, v in roo_succession_map["descending"].items():
        major[k].add(v)
        if k in minor_subsequent:
            minor_subsequent[k].add(v)
        else:
            minor_subsequent[k] = {v}
    return major, minor_preceding, minor_subsequent


roo_leap_map_major, roo_leap_map_minor = make_roo_leap_maps()
roo_step_map_major, roo_step_map_minor_preceding, roo_step_map_minor_subsequent = (
    make_roo_step_maps()
)
```

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide helpers
  code_prompt_show: Show helpers
tags: [hide-cell]
---
def make_precise_preceding_movement_column(df):
    """Expects a dataframe containing the columns bass_degree, preceding_bass_degree, and preceding_movement,"""
    preceding_movement_precise = df.preceding_movement.where(
        df.preceding_movement != "Step", df.preceding_interval
    )
    expected_ascending_degree = pd.concat(
        [
            df.loc[["major"], "bass_degree"].map(
                roo_predecessor_map["ascending_major"]
            ),
            df.loc[["minor"], "bass_degree"].map(
                roo_predecessor_map["ascending_minor"]
            ),
        ]
    )
    expected_descending_degree = df.bass_degree.map(roo_predecessor_map["descending"])
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
        df.subsequent_movement != "Step", df.subsequent_interval
    )
    expected_ascending_degree = pd.concat(
        [
            df.loc[["major"], "bass_degree"].map(roo_succession_map["ascending_major"]),
            df.loc[["minor"], "bass_degree"].map(roo_succession_map["ascending_minor"]),
        ]
    )
    expected_descending_degree = df.bass_degree.map(roo_succession_map["descending"])
    subsequent_movement_precise = subsequent_movement_precise.where(
        df.subsequent_bass_degree != expected_ascending_degree, "ascending"
    )
    subsequent_movement_precise = subsequent_movement_precise.where(
        df.subsequent_bass_degree != expected_descending_degree, "descending"
    )
    return subsequent_movement_precise


def make_preceding_movement_category_column(df):
    """Expects a dataframe containing the columns bass_degree, subsequent_bass_degree, and subsequent_movement,"""
    preceding_movement_category = df.preceding_movement.copy()
    would_be_roo_leaps = pd.concat(
        [
            df.loc[["major"], "bass_degree"].map(roo_leap_map_major),
            df.loc[["minor"], "bass_degree"].map(roo_leap_map_minor),
        ]
    )
    is_regola_leap_mask = pd.Series(
        [
            False if pd.isnull(roo_leaps) else prec_bn in roo_leaps
            for prec_bn, roo_leaps in zip(df.preceding_bass_degree, would_be_roo_leaps)
        ],
        index=df.index,
    )
    preceding_movement_category = preceding_movement_category.where(
        ~is_regola_leap_mask, "Diatonic leap"
    ).replace("Leap", "Other leap")
    would_be_roo_steps = pd.concat(
        [
            df.loc[["major"], "bass_degree"].map(roo_step_map_major),
            df.loc[["minor"], "bass_degree"].map(roo_step_map_minor_preceding),
        ]
    )
    is_regola_step_mask = pd.Series(
        [
            False if pd.isnull(roo_steps) else prec_bn in roo_steps
            for prec_bn, roo_steps in zip(df.preceding_bass_degree, would_be_roo_steps)
        ],
        index=df.index,
    )
    preceding_movement_category = preceding_movement_category.where(
        ~is_regola_step_mask, "Diatonic step"
    ).replace("Step", "Other step")
    return preceding_movement_category.rename("preceding_movement_category")


def make_subsequent_movement_category_column(df):
    """Expects a dataframe containing the columns bass_degree, subsequent_bass_degree, and subsequent_movement,"""
    subsequent_movement_category = df.subsequent_movement.copy()
    would_be_roo_leaps = pd.concat(
        [
            df.loc[["major"], "bass_degree"].map(roo_leap_map_major),
            df.loc[["minor"], "bass_degree"].map(roo_leap_map_minor),
        ]
    )
    is_regola_leap_mask = pd.Series(
        [
            False if pd.isnull(roo_leaps) else subs_bn in roo_leaps
            for subs_bn, roo_leaps in zip(df.subsequent_bass_degree, would_be_roo_leaps)
        ],
        index=df.index,
    )
    subsequent_movement_category = subsequent_movement_category.where(
        ~is_regola_leap_mask, "Diatonic leap"
    ).replace("Leap", "Other leap")
    would_be_roo_steps = pd.concat(
        [
            df.loc[["major"], "bass_degree"].map(roo_step_map_major),
            df.loc[["minor"], "bass_degree"].map(roo_step_map_minor_subsequent),
        ]
    )
    is_regola_step_mask = pd.Series(
        [
            False if pd.isnull(roo_steps) else subs_bn in roo_steps
            for subs_bn, roo_steps in zip(df.subsequent_bass_degree, would_be_roo_steps)
        ],
        index=df.index,
    )
    subsequent_movement_category = subsequent_movement_category.where(
        ~is_regola_step_mask, "Diatonic step"
    ).replace("Step", "Other step")
    return subsequent_movement_category.rename("subsequent_movement_category")
```

**This is the main table of this notebook. It corresponds to the `BassNotes` features,
with a `preceding_` and a `subsequent_` copy of each column concatenated to the right.
The respective upward and downward shifts are performed within each localkey group,
leaving first bass degrees with undefined preceding values and last bass degrees without
undefined subsequent values.**

```{code-cell}
:tags: [hide-input]

def make_adjacency_table(bass_notes):
    bass_notes = pd.concat(
        [
            bass_notes,
            make_regola_chord_column(bass_notes),
            make_regola_suspensions_column(bass_notes),
        ],
        axis=1,
    )
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
    BN["subsequent_iv_is_step"] = BN.subsequent_iv.isin((-7, -5, -2, 2, 5, 7)).where(
        BN.subsequent_iv.notna()
    )
    BN["preceding_iv_is_0"] = BN.preceding_iv == 0
    BN["subsequent_iv_is_0"] = BN.subsequent_iv == 0
    BN["preceding_movement"] = (
        BN.preceding_iv_is_step.map({True: "Step", False: "Leap"})
        .where(~BN.preceding_iv_is_0, "Same")
        .where(BN.preceding_iv.notna(), "None")
    )
    BN["subsequent_movement"] = (
        BN.subsequent_iv_is_step.map({True: "Step", False: "Leap"})
        .where(~BN.subsequent_iv_is_0, "Same")
        .where(BN.subsequent_iv.notna(), "None")
    )
    BN["preceding_movement_precise"] = make_precise_preceding_movement_column(BN)
    BN["subsequent_movement_precise"] = make_precise_subsequent_movement_column(BN)
    BN["preceding_movement_category"] = make_preceding_movement_category_column(BN)
    BN["subsequent_movement_category"] = make_subsequent_movement_category_column(BN)
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

## Conditional probabilities
### p(RoO)

**The probability that a randomly picked chord is a RoO chord is 64.6 %**

```{code-cell}
BN.roo_suspensions.value_counts(normalize=True)
```

### p(RoO|bass)

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide helpers
  code_prompt_show: Show helpers
tags: [hide-cell]
---
def filter_diatonic_bass_degrees(
    base,
        mode: Optional[Literal["major", "minor"]] = None,
        roo_chords_only: bool = False
):
    if mode is None:
        query = (
            "(mode == 'major' & bass_degree in ('1', '2', '3', '4', '5', '6', '7')) | "
            "(mode == 'minor' & bass_degree in ('1', '2', '3', '4', '5', '6', '#6', '7', '#7'))"
        )
    elif mode == "major":
        query = "bass_degree in ('1', '2', '3', '4', '5', '6', '7')"
    elif mode == "minor":
        query = "bass_degree in ('1', '2', '3', '4', '5', '6', '#6', '7', '#7')"
    if roo_chords_only:
        query += " & roo_chord.notna()"
    result = base.query(query)
    return result


BN_dia = filter_diatonic_bass_degrees(BN)
print(f"len(BN_dia) = {len(BN)} - {len(BN) - len(BN_dia)} = {len(BN_dia)}")
```

**Probability that a diatonic bass degree is covered by the corresponding RoO chord: 65.7 %
(+3.4 % a RoO suspension)**

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide helpers
  code_prompt_show: Show helpers
tags: [hide-cell]
---
BN_dia.roo_suspensions.value_counts(normalize=True).to_frame().style.format("{:.1%}")
```

**Probability that a note essentielle is covered by the corresponding RoO chord: 73.7 %
(+5.2 % a RoO suspension)**\
**Probability that a note non-essentielle is covered by the corresponding RoO chord: 53.7 %
(+0.5 % a RoO suspension)**

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide helpers
  code_prompt_show: Show helpers
tags: [hide-cell]
---
notes_essentielles = ("1", "3", "5")
ess_selector = BN_dia.bass_degree.isin(notes_essentielles)
roo_selector = BN_dia.roo_chord.notna()
NE = BN_dia[ess_selector]
NN = BN_dia[~ess_selector]
pd.concat(
    [
        NE.roo_suspensions.value_counts(normalize=True).rename("Notes Essentielles"),
        NN.roo_suspensions.value_counts(normalize=True).rename(
            "Notes Non-Essentielles"
        ),
    ],
    axis=1,
).style.format("{:.1%}")
```

### p(#RoO = {2,1,0} | bass bigram)

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide helpers
  code_prompt_show: Show helpers
tags: [hide-cell]
---
all_bigrams = BN.query("subsequent_movement != 'None'")
all_steps = BN.query("subsequent_movement == 'Step'")
dia_steps = BN_dia.query("subsequent_movement_category == 'Diatonic step'")
n_bigrams, n_steps, n_dia_steps = len(all_bigrams), len(all_steps), len(dia_steps)
print(
    f"The Couperin dataset contains {n_bigrams} bigrams, "
    f"of which {n_steps} ({n_steps/n_bigrams:.1%}) are steps, "
    f"and {n_dia_steps} ({n_dia_steps/n_bigrams:.1%}) are diatonic steps."
)
all_leaps = BN.query("subsequent_movement == 'Leap'")
dia_leaps = BN_dia.query("subsequent_movement_category == 'Diatonic leap'")
n_leaps, n_dia_leaps = len(all_leaps), len(dia_leaps)
print(
    f"The Couperin dataset contains {n_bigrams} bigrams, "
    f"of which {n_leaps} ({n_leaps/n_bigrams:.1%}) are leaps, "
    f"and {n_dia_leaps} ({n_dia_leaps/n_bigrams:.1%}) are diatonic leaps."
)
```

**Given any bass bigram, the probability that**

* both bass notes carry RoO chords is **44.1 %** (**+ 5.7 %** that one is RoO, the other a suspension thereof);
* one of them carries an RoO chord is **34.3 %** (**+ 1.2 %** that one is an RoO suspension chord);
* none of them carries an RoO chord is **14.6 %**.

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide helpers
  code_prompt_show: Show helpers
tags: [hide-cell]
---
all_bigrams[["roo_suspensions", "subsequent_roo_suspensions"]].apply(
    lambda x: str(set(x)), axis=1  # str() is needed because of the styler
).value_counts(normalize=True).to_frame().style.format("{:.1%}")
```

### p(#RoO = {2,1,0} | diatonic leap)

**Given a diatonic leap, the probability that**

* both bass notes carry RoO chords is **46.1 %** (**+ 3.1 %** that one is RoO, the other a suspension thereof);
* one of them carries an RoO chord is **37.0 %** (**+ 0.3 %** that one is an RoO suspension chord);
* none of them carries an RoO chord is **13.4 %**.

```{code-cell}
dia_leaps[["roo_suspensions", "subsequent_roo_suspensions"]].apply(
    lambda x: str(set(x)), axis=1
).value_counts(normalize=True).to_frame().style.format("{:.1%}")
```

### p(#RoO = {2,1,0} | diatonic step)

**Given a diatonic step, the probability that**

* both bass notes carry RoO chords is **59.2 %** (**+ 4.5 %** that one is RoO, the other a suspension thereof);
* one of them carries an RoO chord is **25.0 %** (**+ 1.2 %** that one is an RoO suspension chord);
* none of them carries an RoO chord is **10.1 %**.

```{code-cell}
dia_steps[["roo_suspensions", "subsequent_roo_suspensions"]].apply(
    lambda x: str(set(x)), axis=1
).value_counts(normalize=True).to_frame().style.format("{:.1%}")
```

### p(#RoO = {2,1,0} | bass ∈ {1, 3, 5})

* both bass notes carry RoO chords is **64.9 %** (**+ 5.6 %** that one is RoO, the other a suspension thereof);
* one of them carries an RoO chord is **26.5 %** (**+ 0.2 %** that one is an RoO suspension chord);
* none of them carries an RoO chord is **2.8 %**.

```{code-cell}
bigrams_135_distinct = all_bigrams.query(
    "(bass_degree in @notes_essentielles) & (subsequent_bass_degree in @notes_essentielles) "
    "& bass_degree != subsequent_bass_degree"
)
bigrams_135_distinct[["roo_suspensions", "subsequent_roo_suspensions"]].apply(
    lambda x: str(set(x)), axis=1
).value_counts(normalize=True).to_frame().style.format("{:.1%}")
```

### p(movement = {leap,step,other} | bass ∈ {1, 3, 5})

NE := unigrams with diatonic bass degree ∈ {1, 3, 5} and RoO chord\
NN := unigrams with diatonic bass degree ∈ {2, 4, 6, 7} (and {#6, #7} in minor) and RoO chord

* probability to proceed by leap: NE = **58.1 %**; NN = **23.8 %**
* probability to proceed by step: NE = **22.6 %**; NN = **74.0 %**
* probability to remain: NE = **8.4 %**; NN = **1.5 %**
* probability to be last in key segment: NE = **10.9 %**; NN = **0.7 %**

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide helpers
  code_prompt_show: Show helpers
tags: [hide-cell]
---
NE_roo = BN_dia[roo_selector & ess_selector]
NN_roo = BN_dia[roo_selector & ~ess_selector]
pd.concat(
    [
        NE_roo.subsequent_movement.value_counts(normalize=True).rename(
            "Notes Essentielles"
        ),
        NN_roo.subsequent_movement.value_counts(normalize=True).rename(
            "Notes Non-Essentielles"
        ),
    ],
    axis=1,
).style.format("{:.1%}")
```

### p( RoO(subsequent) | bass ∈ {1, 3, 5} moves by {leap, step} )

Probability that the following chord is a RoO chord given

* a note essentielle proceeding by leap: **73.0 % (+4.7 % a suspension)**
* a note essentielle proceeding by step: **71.0 % (+0.6 % a suspension)**
* a note non-essentielle proceeding by leap: **50.0 % (+1.2 % a suspension)**
* a note non-essentielle proceeding by step: **80.5 % (+8.3 % a suspension)**

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide helpers
  code_prompt_show: Show helpers
tags: [hide-cell]
---
(
    pd.concat(
        {
            "Essentielles": NE_roo.groupby("subsequent_movement")
            .subsequent_roo_suspensions.value_counts(normalize=True)
            .unstack(),
            "Non-Essentielles": NN_roo.groupby("subsequent_movement")
            .subsequent_roo_suspensions.value_counts(normalize=True)
            .unstack(),
        }
    )
    .fillna(0.0)
    .rename_axis(["Notes", "Movement"])
    .rename_axis("Subsequent Chord", axis=1)
).style.format("{:.1%}")
```

### Degree-wise movement Sankey

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide helpers
  code_prompt_show: Show helpers
tags: [hide-cell]
---
def make_summary_sankey_data(
    BN, roo_chords_only=True, extend_right=True, color_edges=True
) -> Tuple[pd.DataFrame, List[str], List[str]] | Tuple[pd.DataFrame, List[str]]:
    BN_dia = filter_diatonic_bass_degrees(BN, roo_chords_only=roo_chords_only)
    preceding_movement = "preceding_movement"
    subsequent_movement = "subsequent_movement"
    middle_nodes_column = "bass_degree"
    BN_dia.loc[:, middle_nodes_column] = BN_dia[middle_nodes_column].str.replace(
        "#", ""
    )
    type_counts = BN_dia[middle_nodes_column].value_counts()
    preceding_movement_counts = BN_dia[preceding_movement].value_counts()
    subsequent_movement_counts = BN_dia[subsequent_movement].value_counts()
    preceding_links = BN_dia.groupby([preceding_movement])[
        middle_nodes_column
    ].value_counts()
    subsequent_links = BN_dia.groupby([subsequent_movement])[
        middle_nodes_column
    ].value_counts()
    node_value_counts = [
        ("preceding", preceding_movement_counts),
        ("intervals", type_counts),
        ("subsequent", subsequent_movement_counts),
    ]
    if extend_right:
        subsequent_roo_counts = BN_dia["subsequent_roo_suspensions"].value_counts()
        node_value_counts.append(("right", subsequent_roo_counts))
        subsequent_roo_links = BN_dia.groupby([subsequent_movement])[
            "subsequent_roo_suspensions"
        ].value_counts()

    node_labels = []
    label_ids = dict()
    for key, node_sizes in node_value_counts:
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

    if extend_right:
        for (subs_mov, roo), cnt in subsequent_roo_links.items():
            source_id = label_ids.get(("subsequent", subs_mov))
            target_id = label_ids.get(("right", roo))
            if color_edges:
                edge_color = node_colors[target_id]
                links.append((source_id, target_id, cnt, edge_color))
            else:
                links.append((source_id, target_id, cnt))

    edge_data = pd.DataFrame(links, columns=edge_columns)
    if color_edges:
        return edge_data, node_labels, node_colors
    return edge_data, node_labels


edge_data, node_labels, node_colors = make_summary_sankey_data(BN)
fig = utils.make_sankey(
    edge_data, node_labels, node_color=node_colors, font=dict(size=45)
)
save_figure_as(fig, "movement_summary_sankey", height=1000)
fig
```

## Overview of how the bass moves
### Intervals

```{code-cell}
:tags: [hide-input]

def plot_bass_movement(BN, corpus_name, **kwargs):
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
    return style_plotly(
        fig, f"how_often_a_bass_note_moves_by_an_interval_{corpus_name}", **kwargs
    )


fig = plot_bass_movement(BN, "Couperin")
save_figure_as(fig, "bass_intervals", height=1000)
fig
```

### Types of movement

**The values `ascending` and `descending` designate stepwise movement within the _regola_. Only non-chromatic scale
degrees can have these values with the exception of `#6` and `#7` which are considered diatonic in the context of
this study.**

```{code-cell}
:tags: [hide-input]

def plot_movement_types(
    BN,
    corpus_title: Optional[str] = None,
    column="subsequent_movement_category",
    **kwargs,
):
    movement_data = pd.concat(
        [
            BN.groupby("mode")[column].value_counts(normalize=True, dropna=False),
            BN.groupby(["piece", "mode"])[column]
            .value_counts(normalize=True, dropna=False)
            .groupby(["mode", column])
            .sem()
            .rename("std_err"),
        ],
        axis=1,
    ).reset_index()
    movement_data[column] = movement_data[column].fillna("None")
    figure_title = (
        None
        if corpus_title is None
        else f"Mode-wise proportion of how often a bass note moves in a certain manner in {corpus_title}"
    )
    fig = px.bar(
        movement_data,
        x=column,
        y="proportion",
        color="mode",
        barmode="group",
        error_y="std_err",
        color_discrete_map=utils.MAJOR_MINOR_COLORS,
        labels={column: "Movement"},
        title=figure_title,
        category_orders=dict(subsequent_interval=interval2fifths.index),
    )
    return style_plotly(fig, save_as=f"mode-wise_bass_motion_{corpus_title}", **kwargs)


fig = plot_movement_types(
    BN,
    None,
    font=dict(size=40),
    # xaxes=dict(tickvals=["Diatonic leap", "Diatonic step", "Same", "None", "Other step", "Other leap"],
    #            ticktext=["Diatonic leap", "Diatonic step", "Same", "None", "Other step", "Other leap"])
)
save_figure_as(fig, "bass_movements", height=1000)
fig
```

## Sankey diagrams showing movement types before and after each scale degree

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide helpers
  code_prompt_show: Show helpers
tags: [hide-cell]
---
CATEGORY2COLOR = dict(
    both="lightcoral",
    ascending="lightgreen",
    descending="lightblue",
)


def get_color(chord, mode):
    if mode == "major":
        category = MAJOR_REGOLA_RN.get(chord)
    elif mode == "minor":
        category = MINOR_REGOLA_RN.get(chord)
    if category:
        return CATEGORY2COLOR[category]


def style_unigram_table(df: pd.DataFrame):

    def color_regola_rows(row, mode):
        if pd.isna(row.iloc[0]):
            return None
        if color := get_color(row.iloc[0], mode):
            return [f"background-color: {color}"] * len(row)
        return None

    new_index = pd.MultiIndex.from_product(
        [["Major", "Minor"], ["Unigram", "Occurrences", "Proportion"]]
    )
    df = df.set_axis(new_index, axis=1)
    return df.style.apply(
        color_regola_rows, axis=1, subset=["Major"], mode="major"
    ).apply(color_regola_rows, axis=1, subset=["Minor"], mode="minor")


def make_sankey_data(
    BN, color_edges=True, precise=None, middle_nodes_column="intervals_over_bass"
) -> Tuple[pd.DataFrame, List[str], List[str]] | Tuple[pd.DataFrame, List[str]]:
    """
    precise=False -> preceding_movement / subsequent_movement
    precise=True -> preceding_movement_precise / subsequent_movement_precise
    precise=None -> preceding_movement_category / subsequent_movement_category
    """
    if precise is None:
        preceding_movement = "preceding_movement_category"
        subsequent_movement = "subsequent_movement_category"
    elif precise:
        preceding_movement = "preceding_movement_precise"
        subsequent_movement = "subsequent_movement_precise"
    else:
        preceding_movement = "preceding_movement"
        subsequent_movement = "subsequent_movement"
    type_counts = BN[middle_nodes_column].value_counts()
    preceding_movement_counts = BN[preceding_movement].value_counts()
    subsequent_movement_counts = BN[subsequent_movement].value_counts()
    preceding_links = BN.groupby([preceding_movement])[
        middle_nodes_column
    ].value_counts()
    subsequent_links = BN.groupby([subsequent_movement])[
        middle_nodes_column
    ].value_counts()

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
    corpus_title: Optional[str] = None,
    mode: Optional[Literal["major", "minor"]] = None,
    bass_degree: Optional[str | int] = None,
    precise=None,
    middle_nodes_column="intervals_over_bass",
    font_size=30,
    **layout,
):
    """
    Create Sankey diagram with values from `middle_nodes_column` as nodes stacked at the middle of the x axis.
    Nodes stacked at the left and right correspond to bass movements depending on the `precise` parameter.

    Args:
        BN (pd.DataFrame): Bass notes table used throughout this notebook.
        corpus_title (str, optional): Corpus title. If None, no title is added to the figure.
        mode (Literal["major", "minor"], optional):
            Usually, you need to select one mode for a meaningful diagram.
        bass_degree (str, int, optional): Bass degree for which to create the Sankey diagram.
        precise (bool, optional): Which movement columns to use for the left and right nodes.
            precise=False -> preceding_movement / subsequent_movement
            precise=True -> preceding_movement_precise / subsequent_movement_precise
            precise=None -> preceding_movement_category / subsequent_movement_category
        middle_nodes_column (str, optional): Column name from which to generate the middle nodes.
    """
    selected_unigrams = BN if mode is None else BN.loc[mode]
    if bass_degree:
        selected_unigrams = selected_unigrams.query(f"bass_degree == '{bass_degree}'")
        selection_text = f"bass degree {bass_degree}"
    else:
        selection_text = "any harmony"
    edge_data, node_labels, node_colors = make_sankey_data(
        selected_unigrams,
        precise=precise,
        middle_nodes_column=middle_nodes_column,
    )
    title = (
        None
        if corpus_title is None
        else f"Motions to and from {selection_text} in {corpus_title} ({mode})"
    )
    if font_size:
        font = layout.pop("font", {})
        font["size"] = font_size
        layout["font"] = font
    fig = utils.make_sankey(
        edge_data, node_labels, node_color=node_colors, title=title, **layout
    )
    return fig
```

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide helpers
  code_prompt_show: Show helpers
tags: [hide-cell]
---
fig = make_bass_degree_sankey(BN, None, None, middle_nodes_column="roo_suspensions")
save_figure_as(fig, "movement_sankey", height=700)
fig
```

```{code-cell}
BN.roo_suspensions.value_counts(normalize=True)  # Tabelle: gesamt, Dur, Moll
# Megatable zusammenfassen in 1 konditionale Wahrsch. pro Bassstufe, zusammengefasst zu 7 Stufen pro Modus
# --> aggregieren mit spread
# Nächster Schritt: Alle Step bigrams vs. alle Leap bigrams: In wie vielen Fällen tragen sie a) zwei, b) einen,
# oder c) null Regolaakkorde?
```

### Unigram Table

```{code-cell}
chord_labels = grouped_D.get_feature("HarmonyLabels")
unigram_occurrences = chord_labels.apply_step("Counter")
occurrence_ranking = unigram_occurrences.make_ranking_table(
    drop_cols=["chord_and_mode", "proportion"], top_k=0
)
style_unigram_table(occurrence_ranking)
```

### Unigram movement Sankey
#### Major

```{code-cell}
fig = make_bass_degree_sankey(BN, "Couperin", "major")
save_figure_as(fig, "couperin_sankey_complete_major")
fig
```

#### Minor

```{code-cell}
fig = make_bass_degree_sankey(BN, "Couperin", "minor")
save_figure_as(fig, "couperin_sankey_complete_minor")
fig
```

### Intervals over bass degree 1
#### Major

```{code-cell}
make_bass_degree_sankey(BN, "Couperin", "major", 1)
```

#### Minor

```{code-cell}
make_bass_degree_sankey(BN, "Couperin", "minor", 1)
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

def summarize_groups_top_k_chords(df, column="intervals_over_bass", k=None):
    """Used in Groupby.apply()"""
    proportions = df[column].value_counts(normalize=True)
    entropy = -(proportions * np.log2(proportions)).sum()
    N = len(proportions)
    normalized_entropy = entropy / np.log2(N) if N > 1 else 0.0
    if k is None:
        top_k = proportions
        n_rows = len(proportions)
    else:
        top_k = proportions.iloc[:k]
        n_rows = len(top_k)
        if len(proportions) > k:
            other = proportions.iloc[k:]
            n_rows += 1
            top_k["Other"] = other.sum()
    rank_col = list(range(1, n_rows + 1))
    result = pd.DataFrame(
        {
            column: top_k.index,
            "proportion": top_k.values,
            "normalized_entropy": normalized_entropy,
        },
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


def make_boolean_is_regola_chord_mask(
    df, mode_or_vocab: str | dict[str, Iterable[str]]
):
    """Mode can be "major" or "minor" or another vocabulary defined as {"bass_degree" => ["interval"]}."""
    if isinstance(mode_or_vocab, str):
        global regola_vocabulary_major, regola_vocabulary_minor
        vocab = (
            regola_vocabulary_major
            if mode_or_vocab == "major"
            else regola_vocabulary_minor
        )
    else:
        vocab = mode_or_vocab
    if "bass_degree" not in df.columns and "bass_degree" in df.index.names:
        df = df.reset_index(level="bass_degree")
    return df[["bass_degree", "intervals_over_bass"]].apply(tuple, axis=1).isin(vocab)


def degree_wise_top_k(BN, column="intervals_over_bass", k=None):
    summary = summarize_degree_wise_top_k(BN, column=column, k=k)
    result = []
    for mode, df in summary.groupby("mode"):
        is_regola = make_boolean_is_regola_chord_mask(df, mode).values
        df["is_regola"] = is_regola
        try:
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
        except Exception:
            print(f"{df.index=}, {df.columns=}")
            raise
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

major, minor = degree_wise_top_k(BN, k=3)
style_rank_table(major)
```

#### Minor

```{code-cell}
style_rank_table(minor)
```

### "Mega tables"

Equivalent to the two preceding tables but with additional heatmaps that show the predominant
movement types preceding and following any chord.

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide helpers
  code_prompt_show: Show helpers
tags: [hide-cell]
---
def summarize_groups_movements(
    df, column="preceding_movement_precise", normalize=False
):
    """Used in Groupby.apply()"""
    proportions = df[column].value_counts(normalize=True)
    entropy = -(proportions * np.log2(proportions)).sum()
    if normalize:
        movements = proportions
    else:
        movements = df[column].value_counts(normalize=False)
    N = len(movements)
    normalized_entropy = entropy / np.log2(N) if N > 1 else 0.0
    if column.endswith("_precise"):
        # differentiating between main categories and "Other" -- it would have been more clean to introduce this
        # alternative categorization as an additional column
        main_movements = [
            ix
            for ix in ("ascending", "descending", "Leap", "None")
            if ix in movements.index.values
        ]
        main_types = movements.loc[main_movements]
        if len(movements) > len(main_types):
            other = movements.loc[movements.index.difference(main_movements)]
            main_types["Other step"] = other.sum()
    else:
        main_types = movements
    main_types["movement_entropy"] = normalized_entropy
    value_column = "proportion" if normalize else "count"
    movements = pd.DataFrame(
        {
            column: main_types.index,
            value_column: main_types.values,
        },
    )
    return movements


def summarize_degree_wise_movement(
    BN, column="preceding_movement_precise", normalize=False
):
    result = (
        BN.groupby(["mode", "bass_degree", "intervals_over_bass"]).apply(
            summarize_groups_movements, column=column, normalize=normalize
        )
    ).droplevel(-1)
    movement_cols = list(
        BN[column].unique()
    )  # ["Leap", "ascending", "descending", "None", "Other step"]
    # column_order = ["movement_entropy"] + movement_cols
    value_column = "proportion" if normalize else "count"
    result = result.pivot(columns=column, values=value_column)  # [column_order]
    if not normalize:
        result = result.astype({col: "Int64" for col in movement_cols})
    return result
```

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide helpers
  code_prompt_show: Show helpers
tags: [hide-cell]
---
def aggregate_other_movements(df, normalize=True):
    columns = [col for col in df.columns if col != "movement_entropy"]
    result = df.loc[:, columns].sum()
    normalized = result / result.sum()
    non_zero = normalized[normalized > 0]
    entropy = -(non_zero * np.log2(non_zero)).sum()
    normalized_entropy = entropy / np.log2(len(columns))
    if normalized_entropy == 0:
        normalized_entropy = abs(normalized_entropy)  # to avoid -0.0
    if normalize:
        result = normalized
    result = pd.concat([pd.Series(dict(movement_entropy=normalized_entropy)), result])
    return result


def normalize_movement_columns(df):
    result = dict(ranking=df.loc[:, "ranking"])
    preceding = df.loc[:, "preceding"].copy()
    columns = [col for col in preceding.columns if col != "movement_entropy"]
    preceding.loc[:, columns] = preceding[columns].div(
        preceding[columns].sum(axis=1), axis=0
    )
    result["preceding"] = preceding
    subsequent = df.loc[:, "subsequent"].copy()
    subsequent.loc[:, columns] = subsequent[columns].div(
        subsequent[columns].sum(axis=1), axis=0
    )
    result["subsequent"] = subsequent
    return pd.concat(result, axis=1)


def cut_down_to_k(mega_table, k=3, normalize=True):
    results = []
    for bass, df in mega_table.groupby("bass_degree"):
        top_k_mask = df.loc[:, ("ranking", "rank_chord")] <= k
        result = df.loc[top_k_mask]
        if normalize:
            result = normalize_movement_columns(result)
        if not top_k_mask.all():
            other = df.loc[~top_k_mask]
            ranking_df = other.loc[:, "ranking"]
            ranking = ranking_df.iloc[0]
            ranking.proportion_chord = ranking_df.proportion_chord.sum()
            preceding = aggregate_other_movements(
                other.loc[:, "preceding"], normalize=normalize
            )
            subsequent = aggregate_other_movements(
                other.loc[:, "subsequent"], normalize=normalize
            )
            concatenated = pd.concat(
                dict(ranking=ranking, preceding=preceding, subsequent=subsequent)
            ).rename((ranking.name[0], "Other"))
            result = pd.concat([result, concatenated.to_frame().T])
        results.append(result)
    return pd.concat(results)


def make_mega_tables(BN=BN, k=None, precise: Optional[bool] = None):
    full_major, full_minor = degree_wise_top_k(BN, k=None)
    if precise is None:
        pm_col = "preceding_movement_category"
        sm_col = "subsequent_movement_category"
    elif precise:
        pm_col = "preceding_movement_precise"
        sm_col = "subsequent_movement_precise"
    else:
        pm_col = "preceding_movement"
        sm_col = "subsequent_movement"
    preceding_movements = summarize_degree_wise_movement(BN, column=pm_col)
    subsequent_movements = summarize_degree_wise_movement(BN, column=sm_col)
    pm_major, pm_minor = (
        preceding_movements.loc["major"],
        preceding_movements.loc["minor"],
    )
    sm_major, sm_minor = (
        subsequent_movements.loc["major"],
        subsequent_movements.loc["minor"],
    )
    reset_levels = [
        lvl
        for lvl in full_major.index.names
        if lvl not in ("bass_degree", "intervals_over_bass")
    ]
    mega_major = pd.concat(
        dict(
            ranking=full_major.reset_index(level=reset_levels).set_index(
                "intervals_over_bass", append=True
            ),
            preceding=pm_major,
            subsequent=sm_major,
        ),
        axis=1,
    )
    mega_minor = pd.concat(
        dict(
            ranking=full_minor.reset_index(level=reset_levels).set_index(
                "intervals_over_bass", append=True
            ),
            preceding=pm_minor,
            subsequent=sm_minor,
        ),
        axis=1,
    )
    if k is not None:
        mega_major = cut_down_to_k(mega_major, k=k)
        mega_minor = cut_down_to_k(mega_minor, k=k)
    return mega_major, mega_minor


mega_major, mega_minor = make_mega_tables(k=5)
mega_major
```

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide helpers
  code_prompt_show: Show helpers
tags: [hide-cell]
---
major_roo_chord2movement = dict(
    zip(
        sorted(regola_vocabulary_major),
        [
            "both",
            "both",
            "both",
            "descending",
            "ascending",
            "both",
            "descending",
            "ascending",
            "ascending",
            "descending",
        ],
    )
)
major_roo_chord2movement
```

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide helpers
  code_prompt_show: Show helpers
tags: [hide-cell]
---
minor_roo_chord2movement = dict(
    zip(
        sorted(regola_vocabulary_minor),
        [
            "ascending",
            "ascending",
            "both",
            "both",
            "both",
            "descending",
            "ascending",
            "both",
            "descending",
            "descending",
        ],
    )
)
minor_roo_chord2movement
```

```{code-cell}
:tags: [hide-input]

def style_mega_table(
    meta_table,
    mode,
):
    mega_styled = meta_table.set_index(meta_table.columns.to_list()[:4], append=True)
    mega_styled.index.names = ["B", "Chord", "R", "P", "E", "SR"]
    col2type = {
        col: "boolean" if col == ("ranking", "is_regola") else "Float64"
        for col in mega_styled.columns
    }
    mega_styled = (
        mega_styled.reorder_levels(["R", "B", "P", "E", "SR", "Chord"])
        .sort_index()
        .droplevel(0)
        .astype(col2type)
    )
    chords = mega_styled.index.get_level_values(-1)
    mega_styled.reset_index(level=-1, drop=True, inplace=True)
    mega_styled.insert(0, ("Chord", "Intervals"), chords)
    # mega_styled.columns = pd.MultiIndex.from_arrays(
    #     [
    #         3 * ["Chord"] + 6 * ["Preceding Movement"] + 6 * ["Subsequent Movement"],
    #         ["Intervals", "P", "RoO"]
    #         + 2 * ["E", "Leap", "RoO Asc", "RoO Desc", "None", "Other Step"],
    #     ]
    # )
    mega_styled.columns = pd.MultiIndex.from_arrays(
        [
            3 * ["Chord"] + 7 * ["Preceding Movement"] + 7 * ["Subsequent Movement"],
            ["Intervals", "P", "RoO"]
            + 2
            * ["None", "Other leap", "Other step", "Diatonic leap", "Diatonic step", "Same", "E"],
        ]
    )
    col2format = {
        "E": "{:.2}",
    }
    format_dict = {
        (l0, l1): "{:.1%}" if (f := col2format.get(l1)) is None else f
        for l0, l1 in mega_styled.columns
        if l1 not in ("RoO", "Intervals")
    }

    def get_chord_color(chord):
        nonlocal mode
        if mode == "major":
            movement = major_roo_chord2movement.get(chord)
        else:
            movement = minor_roo_chord2movement.get(chord)
        if movement is None:
            return
        return CATEGORY2COLOR[movement]

    def color_regola_rows(row):
        bass, intervals = row.name[0], row[("Chord", "Intervals")]
        chord = (bass, intervals)
        if color := get_chord_color(chord):
            return [f"background-color: {color}"] * len(row)
        return [None] * len(row)

    # roo_color_sublevels = ("Intervals", "P", "RoO", "E")
    roo_color_sublevels = (
        "Intervals",
        "P",
        "RoO",
        "E",
    )
    roo_color_columns = [
        (l0, l1) for l0, l1 in mega_styled.columns if l1 in roo_color_sublevels
    ]
    heatmap_color_columns = [
        (l0, l1) for l0, l1 in mega_styled.columns if l1 not in roo_color_sublevels
    ]
    return (
        mega_styled.style.format(format_dict)
        .format_index(
            axis=0,
            formatter={
                "P": "{:.1%}",  # Format as percentage with 1 decimal place
                "E": "{:.2f}",  # Format as float with 2 decimal places
            },
        )
        .apply(color_regola_rows, axis=1, subset=roo_color_columns)
        .background_gradient("Purples", subset=heatmap_color_columns, axis=None)
        .highlight_null(color="lightgrey")
        .set_table_styles(
            {
                ("Chord", "Intervals"): [
                    {"selector": "", "props": "border-left: 1px solid"}
                ],
                ("Preceding Movement", "None"): [
                    {"selector": "", "props": "border-left: 1px solid"},
                    {"selector": "td", "props": "border-left: 1px solid #000066"},
                ],
                ("Subsequent Movement", "None"): [
                    {"selector": "", "props": "border-left: 1px solid"},
                    {"selector": "td", "props": "border-left: 1px solid #000066"},
                ],
            }
        )
    )


style_mega_table(mega_major, "major")
```

```{code-cell}
:tags: [hide-input]

style_mega_table(mega_minor, "minor")
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
        result = filter_diatonic_bass_degrees(base, mode)
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
    to_leap="subsequent_movement == 'Leap'",
    to_same="subsequent_movement == 'Same'",
    last_notes="subsequent_movement == 'None'",
    from_ascending="preceding_movement_precise == 'ascending'",
    from_descending="preceding_movement_precise == 'descending'",
    from_either="preceding_movement_precise == ['ascending', 'descending']",
    from_leap="preceding_movement == 'Leap'",
    from_same="preceding_movement == 'Same'",
    first_notes="preceding_movement == 'None'",
    to_and_from_ascending="subsequent_movement_precise == 'ascending' & preceding_movement_precise == 'ascending'",
    to_and_from_descending="subsequent_movement_precise == 'descending' & preceding_movement_precise == 'descending'",
    to_and_from_either="subsequent_movement_precise == ['ascending', 'descending'] & "
    "preceding_movement_precise == ['ascending', 'descending']",
    to_and_from_leap="subsequent_movement == 'Leap' & preceding_movement == 'Leap'",
    to_and_from_same="subsequent_movement == 'Same' & preceding_movement == 'Same'",
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



def make_coverage_plot_data_with_regola(bn_name):
    global features, regola_coverage
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
    return result.reset_index()


def plot_regola_vs_top_k_coverage(bn_name):
    result = make_coverage_plot_data_with_regola(bn_name)
    plot_coverage_data(result, bn_name)


def plot_coverage_data(
    coverage_data,
    bn_name="couperin",
    facet_row="vocabulary",
    xaxes: Optional[dict] = None,
    yaxes: Optional[dict] = None,
):
    if facet_row != "vocabulary":
        coverage_data = coverage_data.query("vocabulary == 'cumulative'")
    fig = px.line(
        coverage_data,
        x="rank",
        y="proportion",
        markers=True,
        color="coverage_of",
        facet_col="mode",
        facet_row=facet_row,
        hover_name="chord",
        log_x=True,
        title=f"How many {bn_name.title()} unigrams are covered by each top-k vocabulary",
    )
    return style_plotly(
        fig,
        match_facet_yaxes=True,
        height=1500,
        legend=dict(
            orientation="h",
        ),
        xaxes=xaxes,
        yaxes=yaxes,
    )


def plot_coverage_data_categorical(
    coverage_data,
    bn_name="couperin",
    facet_row="comparison",
    xaxes: Optional[dict] = None,
    yaxes: Optional[dict] = None,
):
    coverage_data = coverage_data.copy()
    coverage_data["vocab"] = (
        "top " + coverage_data["rank"].astype(int).astype(str)
    ).where(coverage_data["rank"] != 10.5, "RoO")
    if facet_row != "vocabulary":
        coverage_data = coverage_data.query("vocabulary == 'cumulative'")
    fig = px.line(
        coverage_data,
        x="vocab",
        y="proportion",
        markers=True,
        color="coverage_of",
        facet_col="mode",
        facet_row=facet_row,
        hover_name="chord",
        title=f"How many {bn_name.title()} unigrams are covered by each top-k vocabulary",
    )
    return style_plotly(
        fig,
        match_facet_yaxes=True,
        height=1500,
        legend=dict(
            orientation="h",
        ),
        xaxes=xaxes,
        yaxes=yaxes,
    )
```

```{code-cell}
feature_group = dict(
    to_same="less",
    to_and_from_same="less",
    from_same="less",
    to_and_from_leap="less",
    to_leap="less",
    all="less",
    diatonic="less",
    from_leap="less",
    from_ascending="less",
    to_descending="more",
    from_either="less",
    first_notes="less",
    to_either="more",
    from_descending="more",
    to_ascending="more",
    last_notes="less",
    to_and_from_ascending="more",
    to_and_from_either="more",
    to_and_from_descending="more",
)
coverage_data = make_coverage_plot_data_with_regola("couperin")
coverage_data["comparison"] = coverage_data.coverage_of.map(feature_group)
plot_coverage_data_categorical(
    coverage_data,
)
```

```{code-cell}
r0, r1 = 7, 14
selection = coverage_data.query("@r0 <= rank <= @r1")
fig = plot_coverage_data_categorical(
    selection,
)
save_figure_as(fig, "couperin_top_k_coverage")
fig
```

**The following table shows for which subsets the regola performs better (positive values) or
worse (negative values) than the top-10 vocabulary.**

```{code-cell}
unigram_subset_sizes = {}
for mode in ("major", "minor"):
    basis_dia, basis_all = mode + "_diatonic", mode + "_all"
    for f_name, query in features.items():
        mask = get_chord_vocabulary_mask(
            "couperin", basis=basis_dia, vocabulary=(), query=query
        )
        unigram_subset_sizes[(mode, f_name)] = len(mask)
    mask = get_chord_vocabulary_mask(
        "couperin", basis=basis_dia, vocabulary=(), query=None
    )
    unigram_subset_sizes[(mode, "diatonic")] = len(mask)
    mask = get_chord_vocabulary_mask(
        "couperin", basis=basis_all, vocabulary=(), query=None
    )
    unigram_subset_sizes[(mode, "all")] = len(mask)

unigram_subset_sizes = pd.Series(unigram_subset_sizes, name="N").rename_axis(
    ["mode", "coverage_of"]
)
unigram_subset_sizes
```

```{code-cell}
def prep_cov_data(S):
    return (
        S.reset_index(drop=True)
        .set_index(["vocabulary", "mode", "coverage_of"])
        .proportion
    )


roo_vals = prep_cov_data(
    coverage_data.query("rank == 10.5 & vocabulary == 'cumulative'")
)
top_10_vals = prep_cov_data(
    coverage_data.query("rank == 10 & vocabulary == 'cumulative'")
)
difference_roo_top10 = roo_vals - top_10_vals
merged = pd.merge(
    unigram_subset_sizes,
    roo_vals.rename("RoO"),
    on=["mode", "coverage_of"],
    how="right",
)
merged.index = roo_vals.index
inspect_difference = pd.concat(
    [merged, top_10_vals.rename("top-10"), difference_roo_top10.rename("difference")],
    axis=1,
)
inspect_difference.sort_values("difference", ascending=False)
```

```{code-cell}
plot_regola_vs_top_k_coverage("couperin")
```

**In order to inspect these plots you will want to hide traces.
Click on a legend item to toggle it, double-click on an item to toggle all others.**

+++

## Regola chords and movement types
### All regola chords
**The following table shows absolute counts and proportion of movement types preceding and
succeeding all RoO chords.**

```{code-cell}
:tags: [hide-input]

def get_BN_reg(BN, regola_only=True):
    """A version of BN filtered on regola chords only."""
    result = []
    for mode, df in BN.groupby("mode"):
        is_regola = make_boolean_is_regola_chord_mask(df, mode)
        df["is_regola"] = is_regola.values
        result.append(df)
    result_df = pd.concat(result)
    if regola_only:
        return result_df[result_df.is_regola]
    return result_df


def tally_movement_per_chord(BN_reg, degree_wise=False):
    if degree_wise:
        gpb = BN_reg.groupby(["mode", "bass_degree", "intervals_over_bass"])
    else:
        gpb = BN_reg.groupby(["mode"])
    return pd.concat(
        [
            gpb.preceding_movement_precise.value_counts().rename("preceding"),
            gpb.preceding_movement_precise.value_counts(normalize=True).rename(
                "preceding_%"
            )
            * 100,
            gpb.subsequent_movement_precise.value_counts().rename("subsequent"),
            gpb.subsequent_movement_precise.value_counts(normalize=True).rename(
                "subsequent_%"
            )
            * 100,
        ],
        axis=1,
    ).astype(dict(preceding="Int64", subsequent="Int64"))


BN_reg = get_BN_reg(BN)
tally_movement_per_chord(BN_reg)
```

### Degree-wise
**The following table shows absolute counts and proportion of movement types preceding and
succeeding each individual RoO chord.**

```{code-cell}
:tags: [hide-input]

regola_chord_movement = tally_movement_per_chord(BN_reg, degree_wise=True)
regola_chord_movement
```

### As Sankey diagrams
#### Major

```{code-cell}
fig = make_bass_degree_sankey(BN_reg, "Couperin", "major")
save_figure_as(fig, "couperin_sankey_regola_major")
fig
```

#### Minor

```{code-cell}
fig = make_bass_degree_sankey(BN_reg, "Couperin", "minor")
save_figure_as(fig, "couperin_sankey_regola_minor")
fig
```

### Intervals over bass degree 1
#### Major

```{code-cell}
:tags: [hide-input]

make_bass_degree_sankey(BN_reg, "Couperin", "major", 1)
```

#### Minor

```{code-cell}
make_bass_degree_sankey(BN_reg, "Couperin", "minor", 1)
```

### Intervals over bass degree 2
#### Major

```{code-cell}
make_bass_degree_sankey(BN_reg, "Couperin", "major", 2)
```

#### Minor

```{code-cell}
make_bass_degree_sankey(BN_reg, "Couperin", "minor", 2)
```

### Intervals over bass degree 3
#### Major

```{code-cell}
make_bass_degree_sankey(BN_reg, "Couperin", "major", 3)
```

#### Minor

```{code-cell}
make_bass_degree_sankey(BN_reg, "Couperin", "minor", 3)
```

### Intervals over bass degree 4
#### Major

```{code-cell}
make_bass_degree_sankey(BN_reg, "Couperin", "major", 4)
```

#### Minor

```{code-cell}
make_bass_degree_sankey(BN_reg, "Couperin", "minor", 4)
```

### Intervals over bass degree 5
#### Major

```{code-cell}
make_bass_degree_sankey(BN_reg, "Couperin", "major", 5)
```

#### Minor

```{code-cell}
make_bass_degree_sankey(BN_reg, "Couperin", "minor", 5)
```

### Intervals over bass degree 6
#### Major

```{code-cell}
make_bass_degree_sankey(BN_reg, "Couperin", "major", 6)
```

#### Minor (ascending)

```{code-cell}
make_bass_degree_sankey(BN_reg, "Couperin", "minor", "#6")
```

#### Minor (descending)

```{code-cell}
make_bass_degree_sankey(BN_reg, "Couperin", "minor", 6)
```

### Intervals over bass degree 7
#### Major

```{code-cell}
make_bass_degree_sankey(BN_reg, "Couperin", "major", 7)
```

#### Minor (ascending)

```{code-cell}
make_bass_degree_sankey(BN_reg, "Couperin", "minor", "#7")
```

#### Minor (descending)

```{code-cell}
make_bass_degree_sankey(BN_reg, "Couperin", "minor", 7)
```

## Bigrams

```{code-cell}
chord_bgt: resources.NgramTable = chord_labels.apply_step("BigramAnalyzer")
chord_bigrams = chord_bgt.make_bigram_tuples("chord")
bgt = chord_bigrams.make_ranking_table()
bgt.drop(columns=[("major", "proportion_%"), ("minor", "proportion_%")]).style.format(
    {col: "{:.2%}" for col in bgt.columns if col[1] == "proportion"}
)
```
