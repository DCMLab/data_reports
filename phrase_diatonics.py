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
# %load_ext autoreload
# %autoreload 2

import os
from typing import Dict, Hashable, List, Optional

import dimcat as dc
import ms3
import numpy as np
import pandas as pd
from dimcat import resources
from dimcat.data.resources.results import _entropy
from dimcat.data.resources.utils import merge_columns_into_one
from dimcat.plotting import make_bar_plot, make_box_plot, write_image
from git import Repo

import utils
from utils import (
    DEFAULT_OUTPUT_FORMAT,
    OUTPUT_FOLDER,
    get_repo_name,
    make_stage_data,
    print_heading,
    resolve_dir,
)

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


# %%
def make_criterion(
    phrase_feature,
    criterion_name: Optional[str] = None,
    columns="chord",
    components="body",
    drop_levels=3,
    reverse=True,
    level_name="stage",
    query=None,
    join_str: Optional[str | bool] = None,
    fillna: Optional[Hashable] = None,
) -> pd.Series:
    """Function sets the defaults for the stage TSVs produced in the following."""
    phrase_data = phrase_feature.get_phrase_data(
        columns=columns,
        components=components,
        drop_levels=drop_levels,
        reverse=reverse,
        level_name=level_name,
        wide_format=False,
        query=query,
    )
    if not isinstance(columns, str) and len(columns) > 1:
        phrase_data = merge_columns_into_one(
            phrase_data, join_str=join_str, fillna=fillna
        )
        if criterion_name is None:
            criterion_name = "_and_".join(columns)
    else:
        phrase_data = phrase_data.iloc(axis=1)[0]
        if criterion_name is None:
            if isinstance(columns, str):
                criterion_name = columns
            else:
                criterion_name = columns[0]
    result = phrase_data.rename(criterion_name)
    return result


def make_criterion_stages(
    phrase_annotations: resources.PhraseAnnotations,
    criteria_dict: Dict[str, str | List[str]],
    join_str=True,
):
    """Takes a {name -> [columns]} dict."""
    uncompressed = make_stage_data(
        phrase_annotations,
        columns=["chord_and_mode", "duration_qb"],
        wide_format=False,
    )
    name2phrase_data = {"uncompressed": uncompressed}
    for name, columns in criteria_dict.items():
        criterion = make_criterion(
            phrase_annotations,
            columns=columns,
            criterion_name=name,
            join_str=join_str,
        )
        name2phrase_data[name] = uncompressed.regroup_phrases(criterion)
    return name2phrase_data


def get_stage_durations(phrase_data: resources.PhraseData):
    return phrase_data.groupby(
        ["corpus", "piece", "phrase_id", "stage"]
    ).duration_qb.sum()


def get_criterion_phrase_lengths(phrase_data: resources.PhraseData):
    """In terms of number of stages after merging."""
    stage_index = phrase_data.index.to_frame(index=False)
    phrase_id_col = stage_index.columns.get_loc("phrase_id")
    groupby = stage_index.columns.to_list()[: phrase_id_col + 1]
    stage_lengths = stage_index.groupby(groupby).stage.max() + 1
    return stage_lengths.rename("phrase_length")


def get_criterion_phrase_entropies(
    phrase_data: resources.PhraseData, criterion_name: Optional[str] = None
):
    if not criterion_name:
        criterion_name = phrase_data.columns.to_list()[0]
    criterion_distributions = phrase_data.groupby(
        ["corpus", criterion_name]
    ).duration_qb.sum()
    return criterion_distributions.groupby("corpus").agg(_entropy).rename("entropy")


def get_metrics_means(name2phrase_data: Dict[str, resources.PhraseData]):
    criterion_metric2value = {}
    for name, stages in name2phrase_data.items():
        stage_durations = get_stage_durations(stages)
        criterion_metric2value[
            (name, "mean stage duration", "mean")
        ] = stage_durations.mean()
        criterion_metric2value[
            (name, "mean stage duration", "sem")
        ] = stage_durations.sem()
        phrase_lengths = get_criterion_phrase_lengths(stages)
        criterion_metric2value[
            (name, "mean phrase length", "mean")
        ] = phrase_lengths.mean()
        criterion_metric2value[
            (name, "mean phrase length", "sem")
        ] = phrase_lengths.sem()
        phrase_entropies = get_criterion_phrase_entropies(stages)
        criterion_metric2value[
            (name, "mean phrase entropy", "mean")
        ] = phrase_entropies.mean()
        criterion_metric2value[
            (name, "mean phrase entropy", "sem")
        ] = phrase_entropies.sem()
    metrics = pd.Series(criterion_metric2value, name="value").unstack(sort=False)
    metrics.index.names = ["criterion", "metric"]
    return metrics


def compare_criteria_metrics(
    name2phrase_data: Dict[str, resources.PhraseData], **kwargs
):
    metrics = get_metrics_means(name2phrase_data).reset_index()
    return make_bar_plot(
        metrics,
        facet_row="metric",
        color="criterion",
        x_col="mean",
        y_col="criterion",
        x_axis=dict(matches=None, showticklabels=True),
        layout=dict(showlegend=False),
        error_x="sem",
        orientation="h",
        labels=dict(entropy="entropy of stage distributions in bits", corpus=""),
        **kwargs,
    )


def plot_corpuswise_criteria_means(
    criterion2values: Dict[str, pd.Series],
    category_title="stage_type",
    y_axis_label="mean duration of stages in ♩",
    **kwargs,
):
    """Takes a {trace_name -> values} dict where each entry will be turned into a bar plot trace for comparison."""
    aggregated = {
        name: durations.groupby("corpus").agg(["mean", "sem"])
        for name, durations in criterion2values.items()
    }
    df = pd.concat(aggregated, names=[category_title])
    corpora = df.index.get_level_values("corpus").unique()
    corpus_order = [
        corpus for corpus in chronological_corpus_names if corpus in corpora
    ]
    return make_bar_plot(
        df,
        x_col="corpus",
        y_col="mean",
        error_y="sem",
        color=category_title,
        category_orders=dict(corpus=corpus_order),
        labels=dict(mean=y_axis_label, corpus=""),
        **kwargs,
    )


def plot_corpuswise_criteria(
    criterion2values,
    category_title="stage_type",
    y_axis_label="entropy of stage distributions in bits",
    **kwargs,
):
    """Takes a {trace_name -> PhraseData} dict where each entry will be turned into a bar plot trace for comparison."""
    df = pd.concat(criterion2values, names=[category_title])
    corpora = df.index.get_level_values("corpus").unique()
    corpus_order = [
        corpus for corpus in chronological_corpus_names if corpus in corpora
    ]
    return make_bar_plot(
        df,
        x_col="corpus",
        y_col="entropy",
        color=category_title,
        category_orders=dict(corpus=corpus_order),
        labels=dict(entropy=y_axis_label, corpus=""),
        **kwargs,
    )


def _compare_criteria_stage_durations(
    name2phrase_data: Dict[str, resources.PhraseData]
):
    durations_dict = {
        name: get_stage_durations(stages) for name, stages in name2phrase_data.items()
    }
    return plot_corpuswise_criteria_means(durations_dict, height=800)


def compare_criteria_stage_durations(
    phrase_annotations: resources.PhraseAnnotations,
    criteria_dict: Dict[str, str | List[str]],
    join_str=True,
):
    name2phrase_data = make_criterion_stages(
        phrase_annotations, criteria_dict, join_str=join_str
    )
    return _compare_criteria_stage_durations(name2phrase_data)


def _compare_criteria_phrase_lengths(name2phrase_data: Dict[str, resources.PhraseData]):
    lengths_dict = {
        name: get_criterion_phrase_lengths(durations)
        for name, durations in name2phrase_data.items()
    }
    return plot_corpuswise_criteria_means(
        lengths_dict, y_axis_label="mean number of stages per phrase", height=800
    )


def compare_criteria_phrase_lengths(
    phrase_annotations: resources.PhraseAnnotations,
    criteria_dict: Dict[str, str | List[str]],
    join_str=True,
):
    name2phrase_data = make_criterion_stages(
        phrase_annotations, criteria_dict, join_str=join_str
    )
    return _compare_criteria_phrase_lengths(name2phrase_data)


def _compare_criteria_entropies(name2phrase_data: Dict[str, resources.PhraseData]):
    entropies = {
        name: get_criterion_phrase_entropies(durations)
        for name, durations in name2phrase_data.items()
    }
    return plot_corpuswise_criteria(entropies, height=800)


def compare_criteria_entropies(phrase_annotations, criteria_dict, join_str=True):
    name2phrase_data = make_criterion_stages(
        phrase_annotations, criteria_dict, join_str=join_str
    )
    return _compare_criteria_entropies(name2phrase_data)


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
criterion2stages = make_criterion_stages(phrase_annotations, CRITERIA)


# %%
def get_phrase_chord_tones(phrase_annotations):
    chord_tones = phrase_annotations.get_phrase_data(
        reverse=True,
        columns=[
            "chord",
            "localkey",
            "globalkey",
            "globalkey_is_minor",
            "chord_tones",
        ],
        drop_levels="phrase_component",
    )
    chord_tones.df.chord_tones.where(chord_tones.chord_tones != (), inplace=True)
    chord_tones.df.chord_tones.ffill(inplace=True)
    chord_tones = ms3.transpose_chord_tones_by_localkey(chord_tones.df, by_global=True)
    chord_tones["lowest_tpc"] = chord_tones.chord_tones.map(min)
    highest_tpc = chord_tones.chord_tones.map(max)
    chord_tones["tpc_width"] = highest_tpc - chord_tones.lowest_tpc
    chord_tones["highest_tpc"] = highest_tpc
    return chord_tones


def group_operation(group_df):
    return utils._compute_smallest_fifth_ranges(
        group_df.lowest_tpc.values, group_df.tpc_width.values
    )


def make_diatonics_criterion(
    chord_tones,
    join_str: Optional[str | bool] = None,
    fillna: Optional[Hashable] = None,
):
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
criterion2stages["diatonics"] = criterion2stages["uncompressed"].regroup_phrases(
    diatonics_criterion
)

# %%
compare_criteria_metrics(criterion2stages, height=1000)

# %%
_compare_criteria_stage_durations(criterion2stages)

# %%
_compare_criteria_phrase_lengths(criterion2stages)

# %%
_compare_criteria_entropies(criterion2stages)

# %%
uncompressed_lengths = get_criterion_phrase_lengths(criterion2stages["uncompressed"])
uncompressed_lengths.groupby("corpus").describe()

# %%
make_box_plot(
    uncompressed_lengths,
    x_col="corpus",
    y_col="phrase_length",
    height=800,
    category_orders=dict(corpus=chronological_corpus_names),
)
