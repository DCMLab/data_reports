# ---
# jupyter:
#   jupytext:
#     formats: md:myst,ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: revamp
#     language: python
#     name: revamp
# ---

# %% [markdown]
# # Phrase Alignment
#
# ToDo: in phrase 14634 the progression `V/IV I6 IV` looks like an obvious mistake that should read `V6/IV` instead of
# `I6`.

# %% mystnb={"code_prompt_hide": "Hide imports", "code_prompt_show": "Show imports"} tags=["hide-cell"]
# %load_ext autoreload
# %autoreload 2
import os
from collections import Counter, defaultdict
from itertools import accumulate
from typing import Dict, List, Tuple

import dimcat as dc
import ms3
import numpy as np
import pandas as pd
from dimcat import resources
from dimcat.data.resources.utils import (
    append_index_levels,
    make_phrase_start_mask,
    make_range_index_from_boolean_mask,
)
from dimcat.plotting import write_image
from git import Repo
from sksequitur import Grammar, Parser, parse

import utils

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)

# %%
RESULTS_PATH = os.path.expanduser("~/git/diss/33_phrases/figs")
os.makedirs(RESULTS_PATH, exist_ok=True)


def make_output_path(
    filename: str,
    extension=None,
    path=RESULTS_PATH,
) -> str:
    return utils.make_output_path(filename=filename, extension=extension, path=path)


def save_figure_as(fig, filename, directory=RESULTS_PATH, **kwargs):
    write_image(fig, filename, directory, **kwargs)


# %% tags=["hide-input"]
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

# %%
composition_years = D.get_metadata().get_composition_years()
composition_years.head()

# %%
phrase_annotations: resources.PhraseAnnotations = D.get_feature("PhraseAnnotations")
phrase_annotations

# %%
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
phrase_4873

# %%
phrase_4873.reset_index(drop=True)[
    list(renaming.values()) + ["numeral/dominant", "I/V"]
]

# %%
labels = ["1", "2", "3"]
data = pd.DataFrame(
    dict(
        source=[0, 1, 2],
        target=[0, 0, 0],
        value=[1, 2, 3],
    )
)
utils.make_sankey(
    data,
    labels=labels,
    x=[0.9, 0.5, 0.0001],
    y=[0.5, 0.25, 0.75],
    arrangement="fixed",
)

# %%
stage_data = utils.make_stage_data(
    phrase_annotations,
    columns=["chord", "numeral_or_applied_to_numeral", "localkey", "duration_qb"],
    wide_format=False,
)
stage_data._df["numeral_fifths"] = stage_data.numeral_or_applied_to_numeral.map(
    ms3.roman_numeral2fifths
)
split = stage_data.numeral_or_applied_to_numeral.str.extract(r"(b*)(.+)")
stage_data._df.numeral_or_applied_to_numeral = split[0] + split[1].str.upper()
stage_data.head(50)

# %%
stage_data.query("phrase_id == 14633")

# %%


def make_regrouped_stage_index(
    df: pd.DataFrame,
    inner_grouping: pd.Series,
    outer_grouping: pd.Series,
    level_names: Tuple[str, str] = ("stage", "substage"),
) -> D:
    """Adapted from dimcat.data.resources.utils.make_regrouped_stage_index"""
    assert len(inner_grouping.shape) == 1, "Expecting a Series."
    phrase_start_mask = make_phrase_start_mask(df)
    substage_start_mask = (
        (inner_grouping != inner_grouping.shift()).fillna(True).to_numpy(dtype=bool)
    ) | phrase_start_mask
    substage_level = make_range_index_from_boolean_mask(substage_start_mask)
    # make new stage level that restarts at phrase starts and increments at substage starts
    stage_start_mask = (
        (outer_grouping != outer_grouping.shift()).fillna(True).to_numpy(dtype=bool)
    ) | phrase_start_mask
    stage_level = make_range_index_from_boolean_mask(
        substage_start_mask, stage_start_mask
    )
    # create index levels as dataframe in order to concatenate them to existing levels
    primary, secondary = level_names
    new_index = pd.DataFrame({primary: stage_level, secondary: substage_level})
    return new_index


tondom_criterion = stage_data.numeral_or_applied_to_numeral
tondom_criterion = tondom_criterion.where(tondom_criterion.isin({"I", "i", "V"}))
tondom_criterion = (
    tondom_criterion.groupby("phrase_id").ffill().rename("tondom_segment")
)
numeral_criterion = utils.make_criterion(
    phrase_annotations,
    columns="numeral_or_applied_to_numeral",
    # criterion_name=name,
)
tonic_dominant = stage_data.regroup_phrases(
    tondom_criterion, level_names=("tondom_stage", "substage")
)

substage_index_levels = make_regrouped_stage_index(
    tonic_dominant,
    numeral_criterion,
    tondom_criterion,
    ("numeral_stage", "chord_stage"),
)
tonic_dominant._df.index = append_index_levels(
    tonic_dominant.index, substage_index_levels, drop_levels=-1
)
tondom = (
    tonic_dominant.join(composition_years)
    .groupby("phrase_id")
    .filter(
        lambda phrase: not (
            phrase.tondom_segment.isna().any() or phrase.numeral_fifths.isna().any()
        )
    )
)
tondom.head(50)

# %%
phrase_ids = tondom.index.get_level_values("phrase_id").unique()
remaining = phrase_ids.nunique()
n_phrases = phrase_ids.max()
removed = n_phrases - remaining
print(
    f"{removed} phrases have been removed ({removed / n_phrases:.1%}), {remaining} are remaining."
)

# %%
n = 15
print(f"Pieces with more than {n} tonic-dominant segments:")
tondom.query(f"tondom_stage > {n}").index.droplevel([2, 3, 4, 5]).unique().to_list()


# %%


def get_node_info(tondom_nodes, stage_nodes, offset=1e-09):
    """Offset is a workaround for the bug in Plotly preventing coordinates from being zero."""
    # labels
    node2label = {
        node: label for nodes in stage_nodes.values() for label, node in nodes.items()
    }
    node2label.update({node: label for (_, label), node in tondom_nodes.items()})
    labels = [node2label[node] for node in range(len(node2label))]

    # y-coordinates (label-wise)
    label2fifths = dict(zip(labels, map(ms3.roman_numeral2fifths, labels)))
    smallest = min(label2fifths.values())
    label2norm_fifths = {k: v - smallest for k, v in label2fifths.items()}
    norm_fifths2y = pd.Series(
        np.linspace(offset, 1, max(label2norm_fifths.values()) + 1)
    )
    label2y = {k: norm_fifths2y[v] for k, v in label2norm_fifths.items()}

    # x-coordinates (substage-wise); the evenly spaced positions are called ranks and are counted from right to left
    substages_per_stage = Counter(tondom_stage for (tondom_stage, _) in stage_nodes)
    stage_distances = [0] + [
        substages_per_stage[tondom_stage - 1]
        + 1  # distance = n_substages of previous + 1 (the previous itself)
        for tondom_stage in range(1, len(tondom_nodes))
    ]
    stage_ranks = list(accumulate(stage_distances))
    node2rank_and_label = {
        node: (stage_ranks[tondom_stage], label)
        for (tondom_stage, label), node in tondom_nodes.items()
    }
    node2rank_and_label.update(
        {
            node: (stage_ranks[tondom_stage] + numeral_stage, label)
            for (tondom_stage, numeral_stage), nodes in stage_nodes.items()
            for label, node in nodes.items()
        }
    )
    n_ranks = max((rank for rank, _ in node2rank_and_label.values())) + 1
    rank2x = pd.Series(np.linspace(1, offset, n_ranks))
    node_pos = {
        node: (rank2x[rank], label2y[label])
        for node, (rank, label) in node2rank_and_label.items()
    }
    return labels, node_pos


def tondom_stages2graph_data(
    stages, ending_on=None, stop_at_modulation=False, cut_at_stage=None
):
    """Sankey graph data based on two nested stage levels, one for the tonic-dominant criterion and one for the numeral
    criterion.
    """
    tondom_nodes = (
        {}
    )  # {(tondom_stage, tondom) -> node} <-- targets for all numeral_stage == 1 nodes
    stage_nodes = defaultdict(
        dict
    )  # {(tondom_stage, numeral_stage) -> {numeral -> node}} <-- for every sub-stage, the paradigmatic choice of
    # numerals
    edge_weights = Counter()  # {(source_node, target_node) -> weight}
    node_counter = 0
    if ending_on is not None:
        if isinstance(ending_on, str):
            ending_on = {ending_on}
        else:
            ending_on = set(ending_on)
    for phrase_id, progression in stages.groupby("phrase_id"):
        previous_node = None
        for (tondom_stage, numeral_stage), stage_df in progression.groupby(
            ["tondom_stage", "numeral_stage"]
        ):
            if cut_at_stage and tondom_stage > cut_at_stage:
                break
            first_row = stage_df.iloc[0]
            current_tondom = first_row.tondom_segment
            current_numeral = first_row.numeral_or_applied_to_numeral
            # noinspection PyUnboundLocalVariable
            if tondom_stage == 0 and numeral_stage == 0:
                if ending_on is not None and current_numeral not in ending_on:
                    break
                if stop_at_modulation:
                    localkey = first_row.localkey
            elif stop_at_modulation and first_row.localkey != localkey:
                break
            if numeral_stage == 0:
                tondom_node_id = (tondom_stage, current_tondom)
                if tondom_node_id in tondom_nodes:
                    current_node = tondom_nodes[tondom_node_id]
                else:
                    tondom_nodes[tondom_node_id] = current_node = node_counter
                    node_counter += 1
            else:
                if current_numeral in stage_nodes[(tondom_stage, numeral_stage)]:
                    current_node = stage_nodes[(tondom_stage, numeral_stage)][
                        current_numeral
                    ]
                else:
                    stage_nodes[(tondom_stage, numeral_stage)][
                        current_numeral
                    ] = current_node = node_counter
                    node_counter += 1
            if previous_node is not None:
                edge_weights.update([(current_node, previous_node)])
            previous_node = current_node
    labels, node_pos = get_node_info(tondom_nodes, stage_nodes)
    return edge_weights, labels, node_pos


edge_weights, labels, node_pos = tondom_stages2graph_data(
    tondom, ending_on={"I"}, stop_at_modulation=True, cut_at_stage=2
)


# %%
def tondom_graph_data2sankey(
    edge_weights: Dict[Tuple[int, int], int],
    labels: List[str],
    node_pos,
    **kwargs,
):
    """Create a Sankey diagram from the nodes of every stage and the edge weights between them."""
    data = pd.DataFrame(
        [(u, v, w) for (u, v), w in edge_weights.items()],
        columns=["source", "target", "value"],
    )
    return utils.make_sankey(data, labels, node_pos=node_pos, **kwargs)


fig = tondom_graph_data2sankey(
    edge_weights, labels, node_pos, height=800, arrangement="fixed"
)
fig

# %%
save_figure_as(fig, "tondom_sankey_draft", width=5000)

# %%
print(parse(["I", "IV", "V", "vi", "I", "IV", "V", "I"]))

# %%
tondom_stage_parser = Parser()
no_repeats = (
    tondom.numeral_or_applied_to_numeral.groupby(level=[0, 1, 2, 3, 4])
    .first()
    .iloc[::-1]
)
for _, progression in no_repeats.groupby(["phrase_id", "tondom_stage"]):
    tondom_stage_parser.feed(progression + "-")

# %%
print(Grammar(tondom_stage_parser.tree))

# %%
phrase_components = phrase_annotations.extract_feature("PhraseComponents")
phrase_components.head(30)

# %%
bodies = phrase_components.query("phrase_component == 'body' & n_modulations == 0")
bodies

# %%
I_selector = ~bodies.chords.map(
    lambda c: pd.isnull(c[-1]) or c[-1] not in ("I", "i"), na_action="ignore"
)
bodies_I = bodies[I_selector]
bodies_I.sort_values("n_chords").query("n_chords > 1")
