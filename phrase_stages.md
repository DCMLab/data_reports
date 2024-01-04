---
jupytext:
  formats: md:myst,ipynb,py:percent
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: revamp
  language: python
  name: revamp
---

# Detecting diatonic bands

ToDo

* n01op18-1_01, phrase_id 4, viio/vi => #viio/
* 07-1, phrase_id 2415, vi/V in D would be f# but this is clearly in a. It is a minor key, so bVI should be VI

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide imports
  code_prompt_show: Show imports
tags: [hide-cell]
---
import os
from random import choice
from typing import Hashable, Optional

import dimcat as dc
import ms3
import numpy as np
import pandas as pd
import plotly.express as px
from dimcat import resources
from dimcat.data.resources.utils import (
    make_adjacency_groups,
    make_group_start_mask,
    merge_columns_into_one,
)
from dimcat.plotting import make_box_plot, write_image
from git import Repo

import utils
from create_gantt import create_gantt, fill_yaxis_gaps
from utils import (
    DEFAULT_OUTPUT_FORMAT,
    OUTPUT_FOLDER,
    get_repo_name,
    print_heading,
    resolve_dir,
)

%load_ext autoreload
%autoreload 2


pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
```

```{code-cell}
RESULTS_PATH = os.path.abspath(os.path.join(OUTPUT_FOLDER, "phrases"))
os.makedirs(RESULTS_PATH, exist_ok=True)


def make_output_path(filename, extension=None):
    if extension:
        extension = "." + extension.lstrip(".")
    else:
        extension = DEFAULT_OUTPUT_FORMAT
    return os.path.join(RESULTS_PATH, f"{filename}{extension}")


def save_figure_as(fig, filename, directory=RESULTS_PATH, **kwargs):
    write_image(fig, filename, directory, **kwargs)
```

```{code-cell}
:tags: [hide-input]

package_path = resolve_dir(
    "~/distant_listening_corpus/distant_listening_corpus.datapackage.json"
)
repo = Repo(os.path.dirname(package_path))
print_heading("Data and software versions")
print(f"Data repo '{get_repo_name(repo)}' @ {repo.commit().hexsha[:7]}")
print(f"dimcat version {dc.__version__}")
print(f"ms3 version {ms3.__version__}")
D = dc.Dataset.from_package(package_path)
chronological_corpus_names = D.get_metadata().get_corpus_names(func=None)
D
```

```{code-cell}
phrase_annotations: resources.PhraseAnnotations = D.get_feature("PhraseAnnotations")
phrase_annotations
```

```{code-cell}
CRITERIA = dict(
    chord_reduced_and_localkey=["chord_reduced", "localkey"],
    chord_reduced_and_mode=["chord_reduced_and_mode"],
    bass_degree=["bass_note"],
    root_roman=["root_roman", "localkey_mode"],
    root_degree=["root"],
    numeral_or_applied_to_numeral=["numeral_or_applied_to_numeral", "localkey_mode"],
    effective_localkey=["effective_localkey"],
)
criterion2stages = utils.make_criterion_stages(phrase_annotations, CRITERIA)
```

```{code-cell}
uncompressed_lengths = utils.get_criterion_phrase_lengths(
    criterion2stages["uncompressed"]
)
uncompressed_lengths.groupby("corpus").describe()
```

```{code-cell}
make_box_plot(
    uncompressed_lengths,
    x_col="corpus",
    y_col="phrase_length",
    height=800,
    category_orders=dict(corpus=chronological_corpus_names),
)
```

```{code-cell}
def get_phrase_chord_tones(phrase_annotations) -> resources.PhraseData:
    chord_tones = phrase_annotations.get_phrase_data(
        reverse=True,
        columns=[
            "chord",
            "duration_qb",
            "localkey",
            "globalkey",
            "globalkey_is_minor",
            "effective_localkey",
            "chord_tones",
        ],
        drop_levels="phrase_component",
    )
    df = chord_tones.df
    df.chord_tones.where(df.chord_tones != (), inplace=True)
    df.chord_tones.ffill(inplace=True)
    df = ms3.transpose_chord_tones_by_localkey(df, by_global=True)
    df["lowest_tpc"] = df.chord_tones.map(min)
    highest_tpc = df.chord_tones.map(max)
    df["tpc_width"] = highest_tpc - df.lowest_tpc
    df["highest_tpc"] = highest_tpc
    return chord_tones.from_resource_and_dataframe(chord_tones, df)


def group_operation(group_df):
    return utils._compute_smallest_fifth_ranges(
        group_df.lowest_tpc.values, group_df.tpc_width.values
    )


def _make_diatonics_criterion(
    chord_tones,
) -> pd.DataFrame:
    lowest, width = zip(
        *chord_tones.groupby("phrase_id", sort=False, group_keys=False).apply(
            group_operation
        )
    )
    lowest = np.concatenate(lowest)
    width = np.concatenate(width)
    result = pd.DataFrame(
        {"lowest_tpc": lowest, "tpc_width": width}, index=chord_tones.index
    )
    return result


def make_diatonics_criterion(
    chord_tones,
    join_str: Optional[str | bool] = None,
    fillna: Optional[Hashable] = None,
) -> pd.Series:
    result = _make_diatonics_criterion(chord_tones)
    result = merge_columns_into_one(result, join_str=join_str, fillna=fillna)
    return result.rename("diatonics")
```

```{code-cell}
numeral_type_effective_key = phrase_annotations.get_phrase_data(
    reverse=True,
    columns=[
        "effective_localkey",
        "effective_localkey_resolved",
        "effective_localkey_is_minor",
        "globalkey_is_minor",
        "label",
        "numeral",
        "root_roman",
        "chord_type",
    ],
    drop_levels="phrase_component",
)

dominant_selector = utils.make_dominant_selector(numeral_type_effective_key)

expected_root = ms3.transform(
    numeral_type_effective_key,
    ms3.roman_numeral2fifths,
    ["effective_localkey", "globalkey_is_minor"],
).rename("expected_root")
expected_root = expected_root.where(dominant_selector).astype("Int64")

effective_numeral = pd.concat(
    [
        ms3.transform(
            numeral_type_effective_key,
            ms3.rel2abs_key,
            ["numeral", "effective_localkey", "globalkey_is_minor"],
        ).rename("effective_numeral"),
        numeral_type_effective_key.globalkey_is_minor,
    ],
    axis=1,
)
subsequent_root = (
    ms3.transform(
        effective_numeral,
        ms3.roman_numeral2fifths,
    )
    .shift()
    .astype("Int64")
    .rename("subsequent_root")
)
all_but_ultima_selector = ~make_group_start_mask(subsequent_root, "phrase_id")
subsequent_root.where(all_but_ultima_selector, inplace=True)

subsequent_root_roman = numeral_type_effective_key.root_roman.shift().rename(
    "subsequent_root_roman"
)
subsequent_root_roman.where(all_but_ultima_selector, inplace=True)
numeral_type_effective_key = pd.concat(
    [numeral_type_effective_key, subsequent_root, subsequent_root_roman], axis=1
)

merge_with_previous = (expected_root == subsequent_root).fillna(False)
copy_decision_from_previous = (expected_root.eq(expected_root.shift())).fillna(False)
fill_preparation_chain = (
    copy_decision_from_previous & merge_with_previous.shift().fillna(False)
) & all_but_ultima_selector
keep_root = ~(merge_with_previous | fill_preparation_chain)
criterion = numeral_type_effective_key.root_roman.where(
    keep_root, subsequent_root_roman
)
criterion = criterion.where(~copy_decision_from_previous).ffill()


numeral_type_effective_key = pd.concat([numeral_type_effective_key, criterion], axis=1)

root_roman_or_its_dominant = criterion2stages["uncompressed"].regroup_phrases(criterion)
# criterion2stages[
#     "effective_numeral_or_its_dominant"
# ] = effective_numeral_or_its_dominant
# numeral_type_effective_key.head(100)
root_roman_or_its_dominant.head(100)
```

```{code-cell}
grouping, group_values = make_adjacency_groups(
    numeral_type_effective_key.expected_root.fillna(1000)
)
grouping.where(dominant_selector, inplace=True)


def dominant_resolves_as_expected(group_df):
    exp_root, nxt_root, nxt_numeral = group_df.iloc[0]
    if exp_root == nxt_root:
        return [nxt_numeral] * len(group_df)
    return [pd.NA] * len(group_df)


numeral_type_effective_key[
    ["expected_root", "subsequent_root", "subsequent_root_roman"]
].groupby(grouping, group_keys=False).apply(dominant_resolves_as_expected)
```

```{code-cell}
chord_tones = get_phrase_chord_tones(phrase_annotations)
chord_tones.head()
```

```{code-cell}
chord_tones.tpc_width.value_counts()
```

```{code-cell}
diatonics_criterion = make_diatonics_criterion(chord_tones)
diatonics_stages = chord_tones.regroup_phrases(diatonics_criterion)
criterion2stages["diatonics"] = diatonics_stages
```

```{code-cell}
utils.compare_criteria_metrics(criterion2stages, height=1000)
```

```{code-cell}
utils._compare_criteria_stage_durations(
    criterion2stages, chronological_corpus_names=chronological_corpus_names
)
```

```{code-cell}
utils._compare_criteria_phrase_lengths(
    criterion2stages, chronological_corpus_names=chronological_corpus_names
)
```

```{code-cell}
utils._compare_criteria_entropies(
    criterion2stages, chronological_corpus_names=chronological_corpus_names
)
```

```{code-cell}
diatonics_stages
```

```{code-cell}
LT_DISTANCE2SCALE_DEGREE = {
    0: "7 (#7)",
    1: "3 (#3)",
    2: "6 (#6)",
    3: "2",
    4: "5",
    5: "1",
    6: "4",
    7: "b7 (7)",
    8: "b3 (3)",
    9: "b6 (6)",
    10: "b2",
    11: "b5",
    12: "b1",
    13: "b4",
    14: "bb7",
    15: "bb3",
    16: "bb6",
}


COLOR_NAMES = {
    0: "BLUE_600",  # 7 (#7) (leading tone)
    1: "FUCHSIA_600",  # 3 (#3) (mediant (major))
    2: "AMBER_500",  # 6 (#6) (submediant (major))
    3: "CYAN_300",  # 2 (supertonic)
    4: "VIOLET_900",  # 5 (dominant)
    5: "GREEN_500",  # 1 (tonic)
    6: "RED_500",  # 4 (subdominant)
    7: "STONE_500",  # b7 (7) (subtonic (minor))
    8: "FUCHSIA_800",  # b3 (3) (mediant (minor))
    9: "YELLOW_400",  # b6 (6) (submediant (minor))
    10: "TEAL_600",  # b2
    11: "PINK_600",  # b5 (diminished dominant)
    12: "INDIGO_900",  # b1 (diminished tonic)
    13: "LIME_600",  # b4 (diminished subdominant)
    14: "GRAY_500",  # bb7 (diminished subtonic)
    15: "GRAY_900",  # bb3 (diminished mediant)
    16: "GRAY_300",  # bb6 (diminished submediant)
}
DEGREE2COLOR = {
    degree: utils.TailwindColorsHex.get_color(COLOR_NAMES[dist])
    for dist, degree in LT_DISTANCE2SCALE_DEGREE.items()
}


def _make_start_finish(phrase_df):
    starts = (-phrase_df.duration_qb.cumsum()).rename(
        "Start"
    )  # .astype("datetime64[s]")
    ends = starts.shift().fillna(1).rename("Finish")  # .astype("datetime64[s]")
    return pd.DataFrame({"Start": starts, "Finish": ends})


def make_timeline_data(chord_tones):
    timeline_data = pd.concat(
        [
            chord_tones,
            chord_tones.groupby("phrase_id", group_keys=False, sort=False).apply(
                _make_start_finish
            ),
            _make_diatonics_criterion(chord_tones).rename(
                columns=dict(
                    lowest_tpc="diatonics_lowest_tpc", tpc_width="diatonics_tpc_width"
                )
            ),
        ],
        axis=1,
    )
    exploded_chord_tones = chord_tones.chord_tones.explode()
    exploded_chord_tones = pd.DataFrame(
        dict(
            chord_tone=exploded_chord_tones,
            Task=ms3.transform(exploded_chord_tones, ms3.fifths2name),
        ),
        index=exploded_chord_tones.index,
    )
    timeline_data = pd.merge(
        timeline_data, exploded_chord_tones, left_index=True, right_index=True
    )
    n_below_leading_tone = (
        timeline_data.diatonics_lowest_tpc
        + timeline_data.diatonics_tpc_width
        - timeline_data.chord_tone
    ).rename("n_below_leading_tone")

    resource = pd.DataFrame(
        dict(
            n_below_leading_tone=n_below_leading_tone,
            Resource=n_below_leading_tone.map(LT_DISTANCE2SCALE_DEGREE),
        ),
        index=n_below_leading_tone.index,
    )
    timeline_data = pd.concat([timeline_data, resource], axis=1).rename(
        columns=dict(chord="Description")
    )
    return timeline_data
```

```{code-cell}
timeline_data = make_timeline_data(chord_tones)
timeline_data.head()
```

```{code-cell}


def make_rectangle_shape(group_df, y_min):
    result = dict(
        type="rect",
        x0=group_df.Start.min(),
        x1=group_df.Finish.max(),
        fillcolor="LightSalmon",
        opacity=0.5,
        line_width=0,
        layer="below",
    )
    first_row = group_df.iloc[0]
    lowest_tpc = first_row.diatonics_lowest_tpc
    tpc_width = first_row.diatonics_tpc_width
    highest_tpc = lowest_tpc + tpc_width
    result["y0"] = lowest_tpc - y_min - 0.5
    result["y1"] = highest_tpc - y_min + 0.5
    diatonic = ms3.fifths2name(highest_tpc - 5)
    try:
        text = diatonic if tpc_width < 7 else f"{diatonic}/{diatonic.lower()}"
        result["label"] = dict(
            text=text,
            textposition="top left",
        )
    except AttributeError:
        raise
    return result


def make_diatonics_rectangles(phrase_timeline_data):
    shapes = []
    diatonics = merge_columns_into_one(
        phrase_timeline_data[["diatonics_lowest_tpc", "diatonics_tpc_width"]]
    )
    rectangle_grouper, _ = make_adjacency_groups(diatonics)
    min_y = phrase_timeline_data.chord_tone.min()
    for group, group_df in phrase_timeline_data.groupby(rectangle_grouper):
        shapes.append(make_rectangle_shape(group_df, y_min=min_y))
    return shapes


def plot_phrase(
    phrase_timeline_data,
    colorscale=None,
    add_shapes=True,
):
    shapes = make_diatonics_rectangles(phrase_timeline_data)
    dummy_resource_value = phrase_timeline_data.Resource.iat[0]
    phrase_timeline_data = fill_yaxis_gaps(
        phrase_timeline_data, "chord_tone", Resource=dummy_resource_value
    )
    print(phrase_timeline_data.Task.value_counts())
    if phrase_timeline_data.Task.isna().any():
        names = ms3.transform(phrase_timeline_data.chord_tone, ms3.fifths2name)
        phrase_timeline_data.Task.fillna(names, inplace=True)
    # return phrase_timeline_data
    corpus, piece, phrase_id, *_ = phrase_timeline_data.index[0]
    title = f"Phrase {phrase_id} from {corpus}/{piece}"
    kwargs = dict(title=title, colors=colorscale)
    if add_shapes:
        kwargs["shapes"] = shapes
    fig = create_gantt(
        phrase_timeline_data.sort_values("chord_tone", ascending=False), **kwargs
    )
    fig.update_layout(hovermode="x unified", legend_traceorder="grouped")
    # fig.update_traces(hovertemplate="Task: %{text}<br>Start: %{x}<br>Finish: %{y}")
    return fig
```

```{code-cell}
n_phrases = max(timeline_data.index.levels[2])
# {degree: utils.TailwindColorsHex.get_color(DEGREE2COLOR[degree]) for degree in phrase_timeline_data.Resource.unique()}
colorscale = DEGREE2COLOR
```

```{code-cell}
phrase_timeline_data = timeline_data.query(f"phrase_id == {choice(range(n_phrases))}")
```

```{code-cell}
fig = plot_phrase(phrase_timeline_data, colorscale=colorscale)
plot_phrase(phrase_timeline_data, colorscale=colorscale, add_shapes=False).show()
fig
```

```{code-cell}
fig.add_shape(
    type="rect",
    line_width=2,
    x0=-4,
    y0=3,
    x1=-2,
    y1=1,
    label=dict(
        text="Keys of G or g",
        textposition="top left",
        font=dict(color="black", size=20),
    ),
    showlegend=True,
)
fig
```

```{code-cell}
fig = px.timeline(
    phrase_timeline_data,
    x_start="Start",
    x_end="Finish",
    y="Task",
    color="Resource",
    color_discrete_map=colorscale,
)
# fig.update_xaxes(
#   tickformat="%S",
# )
# fig.update_layout(dict(xaxis_type=None))
fig
```

```{code-cell}

```
