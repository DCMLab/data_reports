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
# * n01op18-1_01, phrase_id 4, viio/vi => #viio/
# * 07-1, phrase_id 2415, vi/V in D would be f# but this is clearly in a. It is a minor key, so bVI should be VI

# %% mystnb={"code_prompt_hide": "Hide imports", "code_prompt_show": "Show imports"} tags=["hide-cell"]
# %load_ext autoreload
# %autoreload 2
import os
from random import choice
from typing import Hashable, Optional

import dimcat as dc
import ms3
import numpy as np
import pandas as pd
from dimcat import resources
from dimcat.data.resources.utils import (
    make_adjacency_groups,
    make_group_start_mask,
    merge_columns_into_one,
    subselect_multiindex_from_df,
)
from dimcat.plotting import make_box_plot, write_image
from git import Repo

import utils
from utils import (
    DEFAULT_OUTPUT_FORMAT,
    OUTPUT_FOLDER,
    get_repo_name,
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
uncompressed_lengths = utils.get_criterion_phrase_lengths(
    criterion2stages["uncompressed"]
)
uncompressed_lengths.groupby("corpus").describe()

# %%
make_box_plot(
    uncompressed_lengths,
    x_col="corpus",
    y_col="phrase_length",
    height=800,
    category_orders=dict(corpus=chronological_corpus_names),
)


# %%
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
def make_root_roman_or_its_dominants_criterion(
    phrase_annotations: resources.PhraseAnnotations,
    query: Optional[str] = None,
    inspect_masks: bool = False,
):
    """For computing this criterion, dominants take on the numeral of their expected tonic chord except when they are
    part of a dominant chain that resolves as expected. In this case, the entire chain takes on the numeral of the
    last expected tonic chord. This results in all chords that are adjacent to their corresponding dominant or to a
    chain of dominants resolving into the respective chord are grouped into a stage. All other numeral groups form
    individual stages.
    """
    # region prepare required chord features
    phrase_data = utils.get_phrase_chord_tones(
        phrase_annotations,
        additional_columns=[
            "relativeroot",
            "localkey_resolved",
            "localkey_is_minor",
            "effective_localkey_resolved",
            "effective_localkey_is_minor",
            "timesig",
            "mn_onset",
            "numeral",
            "root_roman",
            "chord_type",
        ],
        query=query,
    )
    localkey_tonic_fifths = ms3.transform(
        phrase_data,
        ms3.roman_numeral2fifths,
        ["localkey_resolved", "globalkey_is_minor"],
    )
    localkey_tonic_tpc = localkey_tonic_fifths.add(
        ms3.transform(phrase_data.globalkey, ms3.name2fifths)
    ).rename("localkey_tonic_tpc")
    expected_root_tpc = ms3.transform(
        phrase_data,
        ms3.roman_numeral2fifths,
        ["effective_localkey", "globalkey_is_minor"],
    ).rename("expected_root_tpc")
    is_dominant = utils.make_dominant_selector(phrase_data)
    expected_root_tpc = expected_root_tpc.where(is_dominant).astype("Int64")
    expected_tonic = phrase_data.relativeroot.fillna(
        phrase_data.effective_localkey_is_minor.map({True: "i", False: "I"})
    )  # this is equivalent to the effective_localkey (relative to the global tonic), but relative to the localkey
    effective_numeral = (
        ms3.transform(
            phrase_data,
            ms3.rel2abs_key,
            ["numeral", "effective_localkey_resolved", "globalkey_is_minor"],
        )
        .astype("string")
        .rename("effective_numeral")
    )
    subsequent_root = (
        ms3.transform(
            pd.concat(
                [
                    effective_numeral,
                    phrase_data.globalkey_is_minor,
                ],
                axis=1,
            ),
            ms3.roman_numeral2fifths,
        )
        .shift()
        .astype("Int64")
        .rename("subsequent_root")
    )
    # set ultima rows (first of phrase_id groups) to NA
    all_but_ultima_selector = ~make_group_start_mask(subsequent_root, "phrase_id")
    subsequent_root.where(all_but_ultima_selector, inplace=True)
    subsequent_root_roman = phrase_data.root_roman.shift().rename(
        "subsequent_root_roman"
    )
    subsequent_root_roman.where(all_but_ultima_selector, inplace=True)

    # endregion prepare required chord features
    # region prepare masks

    # naming can be confusing: phrase data is expected to be reversed, i.e. the first row is a phrase's ultima chord
    # hence, when column names say "subsequent", the corresponding variable names say "previous" to avoid confusion
    # regarding the direction of the shift. In other words, .shift() yields previous values (values of preceding rows)
    # that correspond to subsequent chords
    merge_with_previous = (expected_root_tpc == subsequent_root).fillna(False)
    copy_decision_from_previous = expected_root_tpc.eq(
        expected_root_tpc.shift()
    ).fillna(
        False
    )  # has same expectation as previous dominant and will take on its value if the end of a dominant chain
    # is resolved; otherwise it keeps its own expected tonic as value

    dominant_grouper, _ = make_adjacency_groups(is_dominant, groupby="phrase_id")
    dominant_group_resolves = (
        merge_with_previous.groupby(dominant_grouper).first().to_dict()
    )  # True for those dominant groups where the first 'merge_with_previous' is True, the other groups are left alone
    potential_dominant_chain_mask = merge_with_previous | copy_decision_from_previous
    dominant_chains_groupby = potential_dominant_chain_mask.groupby(dominant_grouper)
    dominant_chain_fill_indices = []
    for (group, dominant_chain), index in zip(
        dominant_chains_groupby, dominant_chains_groupby.indices.values()
    ):
        if not dominant_group_resolves[group]:
            continue
        for do_fill, ix in zip(dominant_chain[1:], index[1:]):
            # collect all indices following the first (which is already merged and will provide the root_numeral to be
            # propagated) which are either the same dominant (copy_decision_from_previous) or the previous dominant's
            # dominant (merge_with_previous), but stop when the chain is broken, leaving unconnected dominants alone
            # with their own expected resolutions
            if not do_fill:
                break
            dominant_chain_fill_indices.append(ix)
    dominant_chain_fill_mask = np.zeros_like(potential_dominant_chain_mask, bool)
    dominant_chain_fill_mask[dominant_chain_fill_indices] = True

    # endregion prepare masks

    # region make criterion

    root_roman_criterion = expected_tonic.where(
        is_dominant, phrase_data.root_roman
    ).rename("root_roman_or_its_dominants")
    root_roman_criterion.where(
        ~merge_with_previous, subsequent_root_roman, inplace=True
    )
    root_roman_criterion = root_roman_criterion.where(~dominant_chain_fill_mask).ffill()

    # endregion make criterion

    concatenate_this = [
        localkey_tonic_tpc,
        phrase_data,
        effective_numeral,
        expected_root_tpc,
        subsequent_root,
        subsequent_root_roman,
    ]
    if inspect_masks:
        concatenate_this += [
            merge_with_previous.rename("merge_with_previous"),
            copy_decision_from_previous.rename("copy_decision_from_previous"),
            potential_dominant_chain_mask.rename("potential_dominant_chain_mask"),
            pd.Series(
                dominant_chain_fill_mask,
                index=phrase_data.index,
                name="dominant_chain_fill_mask",
            ),
        ]

    phrase_data = phrase_data.from_resource_and_dataframe(
        phrase_data,
        pd.concat(concatenate_this, axis=1),
    )
    return phrase_data.regroup_phrases(root_roman_criterion)


root_roman_or_its_dominants = make_root_roman_or_its_dominants_criterion(
    phrase_annotations
)
criterion2stages["root_roman_or_its_dominants"] = root_roman_or_its_dominants
root_roman_or_its_dominants.head(100)

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
def make_simple_resource_column(timeline_data, name="Resource"):
    is_dominant = timeline_data.expected_root_tpc.notna()
    group_levels = is_dominant.index.names[:-1]
    stage_has_dominant = is_dominant.groupby(group_levels).any()
    is_tonic_resolution = ~is_dominant & stage_has_dominant.reindex(timeline_data.index)
    resource_column = pd.Series("other", index=timeline_data.index, name=name)
    resource_column.where(~is_dominant, "dominant", inplace=True)
    resource_column.where(~is_tonic_resolution, "tonic resolution", inplace=True)
    return resource_column


def make_detailed_resource_column(timeline_data, name="Resource"):
    V_is_root = timeline_data.numeral.eq("V")
    is_dominant_triad = V_is_root & timeline_data.chord_type.eq("M")
    is_dominant_seventh = V_is_root & timeline_data.chord_type.eq("Mm7")
    in_minor = timeline_data.effective_localkey_is_minor
    leading_tone_is_root = (timeline_data.numeral.eq("#vii") & in_minor) | (
        timeline_data.numeral.eq("vii") & ~in_minor
    )
    is_dim = leading_tone_is_root & timeline_data.chord_type.eq("o")
    is_dim7 = leading_tone_is_root & timeline_data.chord_type.eq("o7")
    if_halfdim7 = timeline_data.chord_type.eq("%7")
    is_dominant = timeline_data.expected_root_tpc.notna()
    group_levels = is_dominant.index.names[:-1]
    stage_has_dominant = is_dominant.groupby(group_levels).any()
    is_tonic_resolution = ~is_dominant & stage_has_dominant.reindex(timeline_data.index)
    is_minor_resolution = timeline_data.effective_numeral.str.islower()
    resource_column = pd.Series("other", index=timeline_data.index, name=name)
    resource_column.where(~is_dominant_triad, "D", inplace=True)
    resource_column.where(~is_dim, "rootless D7", inplace=True)
    resource_column.where(~is_dominant_seventh, "D7", inplace=True)
    resource_column.where(~if_halfdim7, "rootless D79", inplace=True)
    resource_column.where(~is_dim7, "rootless D7b9", inplace=True)
    resource_column.where(
        ~(is_tonic_resolution & is_minor_resolution),
        "minor tonic resolution",
        inplace=True,
    )
    resource_column.where(
        ~(is_tonic_resolution & ~is_minor_resolution),
        "major tonic resolution",
        inplace=True,
    )
    return resource_column


def make_timeline_data(root_roman_or_its_dominants, detailed=False):
    timeline_data = pd.concat(
        [
            root_roman_or_its_dominants,
            root_roman_or_its_dominants.groupby(
                "phrase_id", group_keys=False, sort=False
            ).duration_qb.apply(utils.make_start_finish),
            ms3.transform(
                root_roman_or_its_dominants,
                ms3.roman_numeral2fifths,
                ["effective_localkey_resolved", "globalkey_is_minor"],
            ).rename("effective_local_tonic_tpc"),
        ],
        axis=1,
    )
    exploded_chord_tones = root_roman_or_its_dominants.chord_tone_tpcs.explode()
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
    if detailed:
        resource_col = make_detailed_resource_column(timeline_data)
        function_col = make_simple_resource_column(
            timeline_data, name="simple_function"
        )
    else:
        resource_col = make_simple_resource_column(timeline_data)
        function_col = make_detailed_resource_column(
            timeline_data, name="detailed_function"
        )
    timeline_data = pd.concat(
        [
            timeline_data,
            function_col,
            resource_col,
        ],
        axis=1,
    ).rename(columns=dict(chord="Description"))
    return timeline_data


# %%
DETAILED_FUNCTIONS = True
timeline_data = make_timeline_data(
    root_roman_or_its_dominants, detailed=DETAILED_FUNCTIONS
)
timeline_data.head(50)

# %%
n_phrases = max(timeline_data.index.levels[2])


def make_function_colors(detailed=False):
    if detailed:
        colorscale = {
            resource: utils.TailwindColorsHex.get_color(color_name)
            for resource, color_name in [
                ("other", "GRAY_500"),
                ("D", "RED_400"),
                ("rootless D7", "RED_500"),
                ("D7", "RED_600"),
                ("rootless D79", "RED_700"),
                ("rootless D7b9", "RED_900"),
                ("minor tonic resolution", "PURPLE_700"),
                ("major tonic resolution", "SKY_500"),
            ]
        }
    else:
        color_shade = 500
        colorscale = {
            resource: utils.TailwindColorsHex.get_color(color_name, color_shade)
            for resource, color_name in zip(
                ("dominant", "tonic resolution", "other"), ("red", "blue", "gray")
            )
        }
    return colorscale


def make_tonic_line(y_root, x0, x1):
    return dict(
        type="line",
        x0=x0,
        x1=x1,
        y0=y_root,
        y1=y_root,
        line_width=1,
        line_dash="dash",
    )


def make_major_shapes(y_root, x0, x1, text):
    result = []
    y0_maj = y_root - 1.5
    y1_maj = y_root + 5.5
    result.append(
        utils.make_rectangle_shape(
            x0=x0, x1=x1, y0=y0_maj, y1=y1_maj, text=text, legendgroup="localkey"
        )
    )
    result.append(make_tonic_line(y_root, x0, x1))
    if y_root > 1:
        y1_min = y0_maj
        y0_min = max(-0.5, y1_min - 3)
        result.append(
            utils.make_rectangle_shape(
                x0=x0,
                x1=x1,
                y0=y0_min,
                y1=y1_min,
                text="parallel minor",
                line_dash="dot",
                legendgroup="localkey",
            )
        )
    return result


def make_minor_shapes(y_root, x0, x1, text):
    result = []
    y0_min = y_root - 4.5
    y1_min = y_root + 2.5
    result.append(
        utils.make_rectangle_shape(
            x0=x0, x1=x1, y0=y0_min, y1=y1_min, text=text, legendgroup="localkey"
        )
    )
    result.append(make_tonic_line(y_root, x0, x1))
    y0_maj = y1_min
    y1_maj = y0_maj + 3
    result.append(
        utils.make_rectangle_shape(
            x0=x0,
            x1=x1,
            y0=y0_maj,
            y1=y1_maj,
            text="parallel major",
            line_dash="dot",
            legendgroup="localkey",
        )
    )
    return result


def make_localkey_rectangles(phrase_timeline_data):
    shapes = []
    rectangle_grouper, _ = make_adjacency_groups(phrase_timeline_data.localkey)
    y_min = phrase_timeline_data.chord_tone_tpc.min()
    for group, group_df in phrase_timeline_data.groupby(rectangle_grouper):
        x0, x1 = group_df.Start.min(), group_df.Finish.max()
        first_row = group_df.iloc[0]
        y_root = first_row.localkey_tonic_tpc - y_min
        text = first_row.localkey
        if first_row.localkey_is_minor:
            localkey_shapes = make_minor_shapes(y_root, x0=x0, x1=x1, text=text)
        else:
            localkey_shapes = make_major_shapes(y_root, x0=x0, x1=x1, text=text)
        shapes.extend(localkey_shapes)
    shapes[0].update(dict(showlegend=True, name="local key"))
    shapes[1].update(dict(showlegend=True, name="local tonic"))
    return shapes


colorscale = make_function_colors(detailed=DETAILED_FUNCTIONS)

# %%
PIN_PHRASE_ID = 2358
# 5932

if PIN_PHRASE_ID is None:
    phrase_timeline_data = timeline_data.query(
        f"phrase_id == {choice(range(n_phrases))}"
    )
else:
    phrase_timeline_data = timeline_data.query(f"phrase_id == {PIN_PHRASE_ID}")

# %%
fig = utils.plot_phrase(
    phrase_timeline_data,
    colorscale=colorscale,
    shapes=make_localkey_rectangles(phrase_timeline_data),
)
fig

# %%
phrase_2358 = make_root_roman_or_its_dominants_criterion(
    phrase_annotations, query="phrase_id == 2358", inspect_masks=True
)
phrase_2358.iloc[-20:]

# %%
phrase_2358.iloc[-20:, -8:]

# %%
make_localkey_rectangles(phrase_timeline_data)


# %%
def subselect_dominant_stages(timeline_data):
    """Returns a copy where all remaining stages contain at least one dominant."""
    dominant_stage_mask = (
        timeline_data.expected_root_tpc.notna().groupby(level=[0, 1, 2, 3]).any()
    )
    dominant_stage_index = dominant_stage_mask[dominant_stage_mask].index
    all_dominant_stages = subselect_multiindex_from_df(
        timeline_data, dominant_stage_index
    )
    return all_dominant_stages


all_dominant_stages = subselect_dominant_stages(timeline_data)
all_dominant_stages

# %%
gpb = all_dominant_stages.groupby(level=[0, 1, 2, 3])
expected_root_tpcs = gpb.expected_root_tpc.nunique()
expected_root_tpcs[expected_root_tpcs.gt(1)]

# %%
unique_resource_vals = gpb.Resource.unique()
unique_resource_vals.head()

# %%
n_root_roman = gpb.root_roman_or_its_dominants.nunique()
n_root_roman[n_root_roman.gt(1)]
