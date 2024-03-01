---
jupytext:
  formats: md:myst,ipynb,py:percent
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Detecting diatonic bands

ToDo

* n01op18-1_01, phrase_id 4, viio/vi => #viio/
* 07-1, phrase_id 2415, vi/V in D would be f# but this is clearly in a. It is a minor key, so bVI should be VI

```{code-cell} ipython3
---
mystnb:
  code_prompt_hide: Hide imports
  code_prompt_show: Show imports
tags: [hide-cell]
---
%load_ext autoreload
%autoreload 2
import os
from random import choice
from typing import Dict, Hashable, Iterable, Optional

import dimcat as dc
import ms3
import numpy as np
import pandas as pd
import plotly.express as px
from dimcat import plotting, resources
from dimcat.data.resources.utils import make_adjacency_groups, merge_columns_into_one
from git import Repo

import utils

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
```

```{code-cell} ipython3
RESULTS_PATH = os.path.expanduser("~/git/diss/33_phrases/figs")
os.makedirs(RESULTS_PATH, exist_ok=True)


def make_output_path(filename, extension=None):
    if extension:
        extension = "." + extension.lstrip(".")
    else:
        extension = utils.DEFAULT_OUTPUT_FORMAT
    return os.path.join(RESULTS_PATH, f"{filename}{extension}")


def save_figure_as(
    fig, filename, formats=("png", "pdf"), directory=RESULTS_PATH, **kwargs
):
    if formats is not None:
        for fmt in formats:
            plotting.write_image(fig, filename, directory, format=fmt, **kwargs)
    else:
        plotting.write_image(fig, filename, directory, **kwargs)
```

```{code-cell} ipython3
:tags: [hide-input]

package_path = utils.resolve_dir(
    "~/distant_listening_corpus/distant_listening_corpus.datapackage.json"
)
repo = Repo(os.path.dirname(package_path))
utils.print_heading("Data and software versions")
print(f"Data repo '{utils.get_repo_name(repo)}' @ {repo.commit().hexsha[:7]}")
print(f"dimcat version {dc.__version__}")
print(f"ms3 version {ms3.__version__}")
D = dc.Dataset.from_package(package_path)
chronological_corpus_names = D.get_metadata().get_corpus_names(func=None)
D
```

```{code-cell} ipython3
phrase_annotations: resources.PhraseAnnotations = D.get_feature("PhraseAnnotations")
phrase_annotations
```

```{code-cell} ipython3
codette = phrase_annotations.get_phrase_data(
    ["label", "phraseend", "cadence"], components="codetta"
)
new_style_codette = codette.groupby("phrase_id").filter(
    lambda df: ~df.label.str.contains(r"\\").any()
)
n_new_style_phrases = new_style_codette.index.get_level_values("phrase_id").nunique()
has_cadence = new_style_codette.groupby("phrase_id").apply(
    lambda df: df.label.str.contains("\|").any()  # noqa W605
)
n_with_cadence = has_cadence.sum()
print(
    f"{n_with_cadence} out of {n_new_style_phrases} eligible phrases ({n_with_cadence/n_new_style_phrases:.1%}) "
    f"include a cadence."
)
```

```{code-cell} ipython3
has_cadence
```

```{code-cell} ipython3
phrase_4873 = phrase_annotations.query("phrase_id == 4873").iloc[:-1].iloc[::-1].copy()
renaming = dict(
    label="label",
    mn="m.",
    mn_onset="onset",
    duration_qb="duration",
    localkey="local key",
    effective_localkey="tonicized",
    localkey_mode="mode",
    chord="chord",
    chord_reduced="reduced",
    root_roman="roman",
    root="root",
    bass_note="bass",
    numeral_or_applied_to_numeral="numeral/applied",
)
phrase_4873.rename(columns=renaming, inplace=True)
phrase_4873["numeral/dominant"] = [
    "v",
    "v",
    "v",
    "#vi",
    "III",
    "v",
    "VI",
    "v",
    "v",
    "i",
    "i",
    "i",
    "III/i",
    "III/i",
    "III",
    "III/i",
    "iv",
    "iv/i",
    "i",
]
phrase_4873["I/V"] = (
    phrase_4873["numeral"]
    .where(phrase_4873["numeral"].isin({"I", "i", "V"}))
    .ffill()
    .str.upper()
)
phrase_4873.reset_index(drop=True)[
    list(renaming.values()) + ["numeral/dominant", "I/V"]
]
```

```{code-cell} ipython3
CRITERIA = dict(
    # chord_reduced_and_localkey=["chord_reduced", "localkey"],
    chord_reduced_and_mode=["chord_reduced_and_mode"],
    bass_degree=["bass_note"],
    root_roman=["root_roman", "localkey_mode"],
    root_degree=["root"],
    numeral_or_applied_to_numeral=["numeral_or_applied_to_numeral", "localkey_mode"],
    effective_localkey=["effective_localkey"],
    localkey=["localkey"],
)
criterion2stages = utils.make_criterion_stages(phrase_annotations, CRITERIA)
```

```{code-cell} ipython3
phi0 = criterion2stages["uncompressed"]
```

```{code-cell} ipython3
def show_stage(df, stage: int, **kwargs):
    """Pie chart for a given stage."""
    stage = df.query(f"stage == {stage}")
    if isinstance(stage, pd.DataFrame):
        column = stage.iloc(axis=1)[0]
    else:
        column = stage
    vc = column.value_counts().to_frame()
    settings = dict(
        traces_settings=dict(
            textposition="inside",
            textinfo="value+percent",
        ),
        layout=dict(uniformtext_minsize=20, uniformtext_mode="hide"),
        color_discrete_sequence=px.colors.qualitative.Light24,
    )
    settings.update(kwargs)
    return plotting.make_pie_chart(vc, **settings)


phi0_0 = show_stage(
    phi0,
    stage=0,
    layout=dict(
        margin=dict(t=0, r=0, b=0, l=0),
    ),
)
# save_figure_as(phi0_0, "phi0_0_pie_chart", height=800, width=1000)
phi0_0
```

```{code-cell} ipython3
phi0_0_counts = phi0.query("stage == 0").chord_and_mode.value_counts()
n_phrases = phi0_0_counts.sum()
phi0_0_counts.iloc[:7].sum() / n_phrases
```

```{code-cell} ipython3
def show_stages(df, stages: int | Iterable[int] = (0, 1, 2), **kwargs):
    """Pie chart for a given stage."""
    column_name = df.columns[0]
    if isinstance(stages, int):
        stages = [stages]
    else:
        stages = list(stages)
    data = df.query("stage in @stages").reset_index()
    data["aligned stage"] = "S<sub>" + data.stage.astype(str) + "</sub>"
    pie_data = data.groupby("aligned stage")[column_name].value_counts().reset_index()
    settings = dict(
        x_col=column_name,
        traces_settings=dict(
            textposition="inside",
            textinfo="value+percent",
        ),
        layout=dict(
            uniformtext_minsize=20,
            uniformtext_mode="hide",
            color_discrete_sequence=px.colors.qualitative.Light24,
            # showlegend=False,
            # legend=dict(orientation="h")
        ),
        facet_col="aligned stage",
    )
    settings.update(kwargs)
    return plotting.make_pie_chart(pie_data, **settings)


phi0_facets = show_stages(
    phi0,
    layout=dict(
        margin=dict(t=38, r=0, b=0, l=0),
    ),
)
save_figure_as(phi0_facets, "phi0_pies", height=600, width=2000)
phi0_facets
```

```{code-cell} ipython3
phi0_first3 = phi0.query("stage in (0,1,2)")
ix = phi0_first3.chord_and_mode.value_counts().index
selected_labels = ix[ix.str.match("^(I,|i,|V,)")]
print(f"Showing fraction of selected labels {set(selected_labels)} for each stage.")
phi0_first3.groupby("stage").chord_and_mode.apply(
    lambda S: S.value_counts().loc[selected_labels].sum() / n_phrases
)
```

```{code-cell} ipython3
trigrams = (
    phi0_first3.chord_and_mode.str.split(",", expand=True)
    .iloc[:, 0]
    .rename("trigram")
    .groupby("phrase_id")
    .apply(tuple)
)
trigram_distribution = utils.value_count_df(trigrams, rank_index=True)
trigram_distribution
```

```{code-cell} ipython3
100 - trigram_distribution.iloc[:20]["%"].sum()
```

```{code-cell} ipython3
ix = phi0_first3.chord_and_mode.value_counts().index
```

```{code-cell} ipython3

```

```{code-cell} ipython3


def group_operation(group_df):
    return utils._compute_smallest_fifth_ranges(
        group_df.lowest_tpc.values, group_df.tpc_width.values, verbose=False
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

```{code-cell} ipython3
numeral_type_effective_key = phrase_annotations.get_phrase_data(
    reverse=True,
    columns=[
        "numeral",
        "chord_type",
        "effective_localkey",
        "effective_localkey_is_minor",
    ],
    drop_levels="phrase_component",
)
dominant_selector = utils.make_dominant_selector(numeral_type_effective_key)
effective_numeral = ms3.transform(
    numeral_type_effective_key,
    ms3.rel2abs_key,
    ["numeral", "effective_localkey", "effective_localkey_is_minor"],
).rename("effective_numeral_or_its_dominant")
effective_numeral.where(
    ~dominant_selector,
    numeral_type_effective_key.effective_localkey,
    inplace=True,
)
effective_numeral_or_its_dominant = criterion2stages["uncompressed"].regroup_phrases(
    effective_numeral
)
criterion2stages[
    "effective_numeral_or_its_dominant"
] = effective_numeral_or_its_dominant
effective_numeral_or_its_dominant.head(100)
```

```{code-cell} ipython3
effective_numeral_or_its_dominant.query(
    "phrase_id == 4873"
).effective_numeral_or_its_dominant.to_list()
```

```{code-cell} ipython3
chord_tones = utils.get_phrase_chord_tones(phrase_annotations)
chord_tones.head()
```

```{code-cell} ipython3
chord_tones.tpc_width.value_counts()
```

```{code-cell} ipython3
diatonics_criterion = make_diatonics_criterion(chord_tones)
diatonics_stages = chord_tones.regroup_phrases(diatonics_criterion)
criterion2stages["diatonics"] = diatonics_stages
```

```{code-cell} ipython3


def compare_criteria_metrics(
    name2phrase_data: Dict[str, resources.PhraseData], **kwargs
):
    bar_data = utils.get_metrics_means(name2phrase_data).reset_index()
    models = dict(
        uncompressed="Φ°",
        # chord_reduced_and_localkey="Φ<sup>reduced ∧ local key</sup>",
        chord_reduced_and_mode="Φ<sup>reduced ∧ mode</sup>",
        bass_degree="Φ<sup>bass</sup>",
        root_roman="Φ<sup>roman</sup>",
        root_degree="Φ<sup>root</sup>",
        numeral_or_applied_to_numeral="Φ<sup>numeral/borrowed</sup>",
        effective_numeral_or_its_dominant="Φ<sup>numeral/dominant</sup>",
        effective_localkey="Φ<sup>tncz</sup>",
        diatonics="Φ<sup>hull</sup>",
        localkey="Φ<sup>local key</sup>",
    )
    bar_data.criterion = bar_data.criterion.replace(models)
    layout = dict(showlegend=False)
    if more_layout := kwargs.pop("layout", None):
        layout.update(more_layout)
    return plotting.make_bar_plot(
        bar_data,
        facet_row="metric",
        color="criterion",
        x_col="mean",
        y_col="criterion",
        x_axis=dict(matches=None, showticklabels=True),
        category_orders=dict(criterion=models.values()),
        color_discrete_sequence=px.colors.qualitative.Dark24,
        layout=layout,
        error_x="sem",
        orientation="h",
        labels=dict(
            criterion="stage model",
            entropy="entropy of stage distributions in bits",
            corpus="",
        ),
        **kwargs,
    )


height = 1000
width = 600
fig = compare_criteria_metrics(
    criterion2stages,
    layout=dict(
        margin=dict(t=0, r=20),
        showlegend=False,
    ),
    height=height,
    facet_row_spacing=0.07,
    font_size=17,
)
# each subplot should have its own axis title
fig.update_xaxes(title_text="entropy (bits)", row=1)
fig.update_xaxes(title_text="number of stages", row=2)
fig.update_xaxes(title_text="duration in ♩", row=3)
save_figure_as(fig, "comparison_of_stage_merging_criteria", height=height, width=width)
fig
```

```{code-cell} ipython3
utils._compare_criteria_stage_durations(
    criterion2stages, chronological_corpus_names=chronological_corpus_names
)
```

```{code-cell} ipython3
utils._compare_criteria_phrase_lengths(
    criterion2stages, chronological_corpus_names=chronological_corpus_names
)
```

```{code-cell} ipython3
utils._compare_criteria_entropies(
    criterion2stages, chronological_corpus_names=chronological_corpus_names
)
```

```{code-cell} ipython3
diatonics_stages.query("phrase_id == 4873")
```

```{code-cell} ipython3
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


def make_timeline_data(chord_tones):
    timeline_data = pd.concat(
        [
            chord_tones,
            chord_tones.groupby(
                "phrase_id", group_keys=False, sort=False
            ).duration_qb.apply(utils.make_start_finish),
            _make_diatonics_criterion(chord_tones).rename(
                columns=dict(
                    lowest_tpc="diatonics_lowest_tpc", tpc_width="diatonics_tpc_width"
                )
            ),
        ],
        axis=1,
    )
    exploded_chord_tones = chord_tones.chord_tone_tpcs.explode()
    exploded_chord_tones = pd.DataFrame(
        dict(
            chord_tone_tpc=exploded_chord_tones,
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
        - timeline_data.chord_tone_tpc
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

```{code-cell} ipython3
timeline_data = make_timeline_data(chord_tones)
timeline_data.head()
```

```{code-cell} ipython3
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
    min_y = phrase_timeline_data.chord_tone_tpc.min()
    for group, group_df in phrase_timeline_data.groupby(rectangle_grouper):
        shapes.append(make_rectangle_shape(group_df, y_min=min_y))
    return shapes


# def make_diatonics_rectangles(phrase_timeline_data):
#     shapes = []
#     diatonics = merge_columns_into_one(
#         phrase_timeline_data[["diatonics_lowest_tpc", "diatonics_tpc_width"]]
#     )
#     rectangle_grouper, _ = make_adjacency_groups(diatonics)
#     y_min = phrase_timeline_data.chord_tone_tpc.min()
#     for group, group_df in phrase_timeline_data.groupby(rectangle_grouper):
#         first_row = group_df.iloc[0]
#         lowest_tpc = first_row.diatonics_lowest_tpc
#         tpc_width = first_row.diatonics_tpc_width
#         highest_tpc = lowest_tpc + tpc_width
#         x0, x1 = first_row.Start.min(), first_row.Finish.max()
#         y0 = lowest_tpc - y_min - 0.5
#         y1 = highest_tpc - y_min + 0.5
#         diatonic = ms3.fifths2name(highest_tpc - 5)
#         text = diatonic if tpc_width < 7 else f"{diatonic}/{diatonic.lower()}"
#         shapes.append(utils.make_rectangle_shape(x0=x0, x1=x1, y0=y0, y1=y1, text=text))
#     return shapes
```

```{code-cell} ipython3
n_phrases = max(timeline_data.index.levels[2])
# {degree: utils.TailwindColorsHex.get_color(DEGREE2COLOR[degree]) for degree in phrase_timeline_data.Resource.unique()}
colorscale = DEGREE2COLOR
```

```{code-cell} ipython3
PIN_PHRASE_ID = None
# 827
# 2358
# 5932
# 9649

if PIN_PHRASE_ID is None:
    current_id = choice(range(n_phrases))
else:
    current_id = PIN_PHRASE_ID
phrase_timeline_data = timeline_data.query(f"phrase_id == {current_id}")
```

```{code-cell} ipython3
fig = utils.plot_phrase(
    phrase_timeline_data,
    colorscale=colorscale,
    shapes=make_diatonics_rectangles(phrase_timeline_data),
)
# utils.plot_phrase(
#     phrase_timeline_data,
#     colorscale=colorscale,
# ).show()
fig
```

```{code-cell} ipython3
plot_phrase_id = 4873  # 10403
plot_phrase_data = timeline_data.query(f"phrase_id == {plot_phrase_id}")
fig = utils.plot_phrase(
    plot_phrase_data,
    colorscale=colorscale,
    shapes=make_diatonics_rectangles(plot_phrase_data),
    title="",
    x_axis=dict(
        tickmode="array", tickvals=[-2, -6, -10, -14, -18], ticktext=[9, 8, 7, 6, 5]
    ),
)
fig.update_shapes(label=dict(textposition="middle center", font_size=30))
fig.update_layout(
    dict(
        legend=dict(
            orientation="h",
            # font_size=17
        )
    )
)
save_figure_as(
    fig,
    f"diatonic_hull_for_phrase_{plot_phrase_id}",
    formats=["pdf"],
    width=1280,
    height=720,
)
fig
```

```{code-cell} ipython3

```

```{code-cell} ipython3
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

```{code-cell} ipython3
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

### Demo values

```{code-cell} ipython3
for criterion, stages in criterion2stages.items():
    utils.print_heading(criterion)
    values = stages.df.query("phrase_id == 9773").iloc[:, 0].to_list()
    print(values[28])
    print()
```
