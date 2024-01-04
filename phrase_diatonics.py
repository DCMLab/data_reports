# ---
# jupyter:
#   jupytext:
#     formats: md:myst,ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: revamp
#     language: python
#     name: revamp
# ---

# %% [markdown]
# # Detecting diatonic bands
#
# ToDo
#
# * n01op18-1_01, phrase_id 4, viio/vi => #viio/vi

# %% mystnb={"code_prompt_hide": "Hide imports", "code_prompt_show": "Show imports"} tags=["hide-cell"]
import os
from random import choice
from typing import Hashable, Optional

import dimcat as dc
import ms3
import numpy as np
import pandas as pd
import plotly.express as px
from dimcat import resources
from dimcat.data.resources.utils import merge_columns_into_one
from dimcat.plotting import write_image
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

# %load_ext autoreload
# %autoreload 2


pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)

# %%
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


# %% tags=["hide-input"]
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

# %%
phrase_annotations: resources.PhraseAnnotations = D.get_feature("PhraseAnnotations")
phrase_annotations

# %%
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


# %%
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


# %%
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
is_dominant = numeral_type_effective_key.numeral.eq(
    "V"
) & numeral_type_effective_key.chord_type.isin({"Mm7", "M"})
leading_tone_is_root = (
    numeral_type_effective_key.numeral.eq("#vii")
    & numeral_type_effective_key.effective_localkey_is_minor
) | (
    numeral_type_effective_key.numeral.eq("vii")
    & ~numeral_type_effective_key.effective_localkey_is_minor
)
is_rootless_dominant = (
    leading_tone_is_root & numeral_type_effective_key.chord_type.isin({"o", "o7", "%7"})
)
dominants_and_resolutions = ms3.transform(
    numeral_type_effective_key,
    ms3.rel2abs_key,
    ["numeral", "effective_localkey", "effective_localkey_is_minor"],
).rename("effective_numeral_or_its_dominant")
dominants_and_resolutions.where(
    ~(is_dominant | is_rootless_dominant),
    numeral_type_effective_key.effective_localkey,
    inplace=True,
)
effective_numeral_or_its_dominant = criterion2stages["uncompressed"].regroup_phrases(
    dominants_and_resolutions
)
criterion2stages[
    "effective_numeral_or_its_dominant"
] = effective_numeral_or_its_dominant
effective_numeral_or_its_dominant.head(100)

# %%
chord_tones = get_phrase_chord_tones(phrase_annotations)
chord_tones.head()

# %%
chord_tones.tpc_width.value_counts()

# %%
diatonics_criterion = make_diatonics_criterion(chord_tones)
diatonics_stages = chord_tones.regroup_phrases(diatonics_criterion)
criterion2stages["diatonics"] = diatonics_stages

# %%
utils.compare_criteria_metrics(criterion2stages, height=1000)

# %%
utils._compare_criteria_stage_durations(
    criterion2stages, chronological_corpus_names=chronological_corpus_names
)

# %%
utils._compare_criteria_phrase_lengths(
    criterion2stages, chronological_corpus_names=chronological_corpus_names
)

# %%
utils._compare_criteria_entropies(
    criterion2stages, chronological_corpus_names=chronological_corpus_names
)

# %%
diatonics_stages


# %%
def _make_timeline_data(phrase_df):
    starts = (-phrase_df.duration_qb.cumsum()).rename(
        "Start"
    )  # .astype("datetime64[s]")
    ends = starts.shift().fillna(1).rename("Finish")  # .astype("datetime64[s]")
    timeline_data = pd.concat(
        [
            phrase_df,
            starts,
            ends,
        ],
        axis=1,
    )
    timeline_data = pd.merge(
        timeline_data,
        phrase_df.chord_tones.explode().rename("Task"),
        left_index=True,
        right_index=True,
    )
    return timeline_data


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
    timeline_data["Resource"] = (
        timeline_data.diatonics_lowest_tpc
        + timeline_data.diatonics_tpc_width
        - timeline_data.chord_tone
    )
    return timeline_data.rename(columns=dict(chord="Description"))


timeline_data = make_timeline_data(chord_tones)
timeline_data.head()

# %%

n_phrases = max(timeline_data.index.levels[2])
phrase_timeline_data = timeline_data.query(f"phrase_id == {choice(range(n_phrases))}")
phrase_timeline_data


# %%
def plot_phrase(phrase_timeline_data):
    phrase_timeline_data = fill_yaxis_gaps(
        phrase_timeline_data, "chord_tone", Resource=0
    )
    if phrase_timeline_data.Task.isna().any():
        names = ms3.transform(phrase_timeline_data.chord_tone, ms3.fifths2name)
        phrase_timeline_data.Task.fillna(names, inplace=True)
    # return phrase_timeline_data
    colorscale = dict(zip(range(10), px.colors.qualitative.G10))
    fig = create_gantt(
        phrase_timeline_data.sort_values("chord_tone", ascending=False),
        colors=colorscale,
    )
    fig.update_layout(hovermode="x unified", legend_traceorder="grouped")
    # fig.update_traces(hovertemplate="Task: %{text}<br>Start: %{x}<br>Finish: %{y}")
    return fig


fig = plot_phrase(phrase_timeline_data)
fig

# %%
fig["data"]

# %%
fig["layout"]

# %%
scatter = px.scatter(
    phrase_timeline_data,
    x="Start",
    y="Finish",
    color="Task",
    custom_data="chord_tones",
    hover_name="chord",
)
scatter.update_traces(
    hovertemplate="<b>%{hovertext}</b><br><br>"
    "Task=0<br>"
    "Start=%{x}<br>"
    "Finish=%{y}<br>"
    "chord_tones=%{customdata}"
    "<extra></extra>"
)
scatter

# %%
for scatter_trace in fig["data"]:
    scatter.update()

# %%

fig = px.timeline(phrase_timeline_data, x_start="Start", x_end="Finish", y="Task")
# fig.update_xaxes(
#   tickformat="%S",
# )
fig.update_layout(dict(xaxis_type=None))
fig
