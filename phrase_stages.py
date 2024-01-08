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
from numbers import Number
from random import choice
from typing import List, Optional, Tuple

import dimcat as dc
import ms3
import numpy as np
import pandas as pd
from dimcat import resources
from dimcat.base import FriendlyEnum
from dimcat.data.resources.utils import (
    make_adjacency_groups,
    make_group_start_mask,
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
def _make_root_roman_or_its_dominants_criterion(
    phrase_data: resources.PhraseData,
    inspect_masks: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    # region prepare required chord features
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
    ).rename(
        "expected_tonic"
    )  # this is equivalent to the effective_localkey (which is relative to the global tonic),
    # but relative to the localkey
    effective_numeral = (
        ms3.transform(
            phrase_data,
            ms3.rel2abs_key,
            ["numeral", "effective_localkey_resolved", "globalkey_is_minor"],
        )
        .astype("string")
        .rename("effective_numeral")
    )
    subsequent_root_tpc = (
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
        .rename("subsequent_root_tpc")
    )
    # set ultima rows (first of phrase_id groups) to NA
    all_but_ultima_selector = ~make_group_start_mask(subsequent_root_tpc, "phrase_id")
    subsequent_root_tpc.where(all_but_ultima_selector, inplace=True)
    subsequent_root_roman = phrase_data.root_roman.shift().rename(
        "subsequent_root_roman"
    )
    subsequent_root_roman.where(all_but_ultima_selector, inplace=True)
    subsequent_numeral_is_minor = (
        effective_numeral.str.islower().shift().rename("subsequent_numeral_is_minor")
    )
    subsequent_numeral_is_minor.where(all_but_ultima_selector, inplace=True)
    # endregion prepare required chord features
    # region prepare masks
    # naming can be confusing: phrase data is expected to be reversed, i.e. the first row is a phrase's ultima chord
    # hence, when column names say "subsequent", the corresponding variable names say "previous" to avoid confusion
    # regarding the direction of the shift. In other words, .shift() yields previous values (values of preceding rows)
    # that correspond to subsequent chords
    merge_with_previous = (expected_root_tpc == subsequent_root_tpc).fillna(False)
    copy_decision_from_previous = effective_numeral.where(is_dominant)
    copy_decision_from_previous = copy_decision_from_previous.eq(
        copy_decision_from_previous.shift()
    ).fillna(False)
    copy_chain_decision_from_previous = expected_root_tpc.eq(
        expected_root_tpc.shift()
    ).fillna(
        False
    )  # has same expectation as previous dominant and will take on its value if the end of a dominant chain
    # is resolved; otherwise it keeps its own expected tonic as value
    dominant_grouper, _ = make_adjacency_groups(is_dominant, groupby="phrase_id")
    dominant_group_resolves = (
        merge_with_previous.groupby(dominant_grouper).first().to_dict()
    )  # True for those dominant groups where the first 'merge_with_previous' is True, the other groups are left alone
    potential_dominant_chain_mask = (
        merge_with_previous | copy_chain_decision_from_previous
    )
    dominant_chains_groupby = potential_dominant_chain_mask.groupby(dominant_grouper)
    dominant_chain_fill_indices = []
    for (group, dominant_chain), index in zip(
        dominant_chains_groupby, dominant_chains_groupby.indices.values()
    ):
        if not dominant_group_resolves[group]:
            continue
        for do_fill, ix in zip(dominant_chain[1:], index[1:]):
            # collect all indices following the first (which is already merged and will provide the root_numeral to be
            # propagated) which are either the same dominant (copy_chain_decision_from_previous) or the previous
            # dominant's dominant (merge_with_previous), but stop when the chain is broken, leaving unconnected
            # dominants alone with their own expected resolutions
            if not do_fill:
                break
            dominant_chain_fill_indices.append(ix)
    dominant_chain_fill_mask = np.zeros_like(potential_dominant_chain_mask, bool)
    dominant_chain_fill_mask[dominant_chain_fill_indices] = True
    # endregion prepare masks
    # region make criteria

    # without dominant chains
    # for this criterion, only those dominants that resolve as expected take on the value of their expected tonic, so
    # that, otherwise, they are available for merging with their own dominant
    root_dominant_criterion = expected_tonic.where(
        is_dominant & merge_with_previous, phrase_data.root_roman
    )
    root_dominant_criterion.where(
        ~merge_with_previous, subsequent_root_roman, inplace=True
    )
    root_dominant_criterion = (
        root_dominant_criterion.where(~copy_decision_from_previous)
        .ffill()
        .rename("root_roman_or_its_dominant")
    )

    # with dominant chains
    # for this criterion, all dominants
    root_dominants_criterion = expected_tonic.where(is_dominant, phrase_data.root_roman)
    root_dominants_criterion.where(
        ~merge_with_previous, subsequent_root_roman, inplace=True
    )
    root_dominants_criterion = (
        root_dominants_criterion.where(~dominant_chain_fill_mask)
        .ffill()
        .rename("root_roman_or_its_dominants")
    )
    # endregion make criteria
    concatenate_this = [
        localkey_tonic_tpc,
        phrase_data,
        effective_numeral,
        expected_tonic,
        expected_root_tpc,
        subsequent_root_tpc,
        subsequent_root_roman,
        subsequent_numeral_is_minor,
    ]
    if inspect_masks:
        concatenate_this += [
            merge_with_previous.rename("merge_with_previous"),
            copy_decision_from_previous.rename("copy_decision_from_previous"),
            copy_chain_decision_from_previous.rename(
                "copy_chain_decision_from_previous"
            ),
            potential_dominant_chain_mask.rename("potential_dominant_chain_mask"),
            pd.Series(
                dominant_chain_fill_mask,
                index=phrase_data.index,
                name="dominant_chain_fill_mask",
            ),
        ]
    phrase_data_df = pd.concat(concatenate_this, axis=1)
    return phrase_data_df, root_dominant_criterion, root_dominants_criterion


def make_root_roman_or_its_dominants_criterion(
    phrase_annotations: resources.PhraseAnnotations,
    merge_dominant_chains: bool = True,
    query: Optional[str] = None,
    inspect_masks: bool = False,
):
    """For computing this criterion, dominants take on the numeral of their expected tonic chord except when they are
    part of a dominant chain that resolves as expected. In this case, the entire chain takes on the numeral of the
    last expected tonic chord. This results in all chords that are adjacent to their corresponding dominant or to a
    chain of dominants resolving into the respective chord are grouped into a stage. All other numeral groups form
    individual stages.
    """
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
    (
        phrase_data_df,
        root_dominant_criterion,
        root_dominants_criterion,
    ) = _make_root_roman_or_its_dominants_criterion(phrase_data, inspect_masks)
    if merge_dominant_chains:
        phrase_data_df = pd.concat([root_dominant_criterion, phrase_data_df], axis=1)
        regroup_by = root_dominants_criterion
    else:
        phrase_data_df = pd.concat([root_dominants_criterion, phrase_data_df], axis=1)
        regroup_by = root_dominant_criterion
    phrase_data = phrase_data.from_resource_and_dataframe(
        phrase_data,
        phrase_data_df,
    )
    return phrase_data.regroup_phrases(regroup_by)


root_roman_or_its_dominants = make_root_roman_or_its_dominants_criterion(
    phrase_annotations
)
criterion2stages["root_roman_or_its_dominants"] = root_roman_or_its_dominants
root_roman_or_its_dominants.head(100)


# %% [raw]
# utils.compare_criteria_metrics(criterion2stages, height=1000)

# %% [raw]
# utils._compare_criteria_stage_durations(
#     criterion2stages, chronological_corpus_names=chronological_corpus_names
# )

# %% [raw]
# utils._compare_criteria_phrase_lengths(
#     criterion2stages, chronological_corpus_names=chronological_corpus_names
# )

# %% [raw]
# utils._compare_criteria_entropies(
#     criterion2stages, chronological_corpus_names=chronological_corpus_names
# )

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


class DetailedFunction(FriendlyEnum):
    I = "major tonic resolution"  # noqa: E741
    i = "minor tonic resolution"
    V = "D"
    vii = "rootless D7"
    V7 = "D7"
    vii07 = "rootless D79"
    viio7 = "rootless D7b9"
    OTHER = "other"


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
    resource_column = pd.Series(
        DetailedFunction.OTHER.value, index=timeline_data.index, name=name
    )
    resource_column.where(~is_dominant_triad, DetailedFunction.V.value, inplace=True)
    resource_column.where(~is_dim, DetailedFunction.vii.value, inplace=True)
    resource_column.where(~is_dominant_seventh, DetailedFunction.V7.value, inplace=True)
    resource_column.where(~if_halfdim7, DetailedFunction.vii07.value, inplace=True)
    resource_column.where(~is_dim7, DetailedFunction.viio7.value, inplace=True)
    resource_column.where(
        ~(is_tonic_resolution & is_minor_resolution),
        DetailedFunction.i.value,
        inplace=True,
    )
    resource_column.where(
        ~(is_tonic_resolution & ~is_minor_resolution),
        DetailedFunction.I.value,
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
                (DetailedFunction.i.value, "PURPLE_700"),
                (DetailedFunction.I.value, "SKY_500"),
                (DetailedFunction.V.value, "RED_400"),
                (DetailedFunction.vii.value, "RED_500"),
                (DetailedFunction.V7.value, "RED_600"),
                (DetailedFunction.vii07.value, "RED_700"),
                (DetailedFunction.viio7.value, "RED_900"),
                (DetailedFunction.OTHER.value, "GRAY_500"),
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


def make_tonic_line(y_root: int, x0: Number, x1: Number, line_dash="dash"):
    return dict(
        type="line",
        x0=x0,
        x1=x1,
        y0=y_root,
        y1=y_root,
        line_width=1,
        line_dash=line_dash,
    )


def get_major_y_coordinates(y_root):
    y0_primary = y_root - 1.5
    y1_primary = y_root + 5.5
    if y_root > 1:
        y1_secondary = y0_primary
        y0_secondary = max(-0.5, y1_secondary - 3)
    else:
        y0_secondary = None
        y1_secondary = None
    return y0_primary, y1_primary, y0_secondary, y1_secondary


def get_minor_y_coordinates(y_root):
    y0_primary = y_root - 4.5
    y1_primary = y_root + 2.5
    y0_secondary = y1_primary
    y1_secondary = y0_secondary + 3
    return y0_primary, y1_primary, y0_secondary, y1_secondary


def _make_localkey_shapes(
    y_root: int, is_minor: bool, x0: Number, x1: Number, text: Optional[str] = None
) -> List[dict]:
    result = []
    if is_minor:
        y0_primary, y1_primary, y0_secondary, y1_secondary = get_minor_y_coordinates(
            y_root
        )
    else:
        y0_primary, y1_primary, y0_secondary, y1_secondary = get_major_y_coordinates(
            y_root
        )
    result.append(
        utils.make_rectangle_shape(
            x0=x0,
            x1=x1,
            y0=y0_primary,
            y1=y1_primary,
            text=text,
            legendgroup="localkey",
        )
    )
    result.append(make_tonic_line(y_root, x0, x1))
    text = "parallel major" if is_minor else "parallel minor"
    if y0_secondary is not None:
        result.append(
            utils.make_rectangle_shape(
                x0=x0,
                x1=x1,
                y0=y0_secondary,
                y1=y1_secondary,
                text=text,
                line_dash="dot",
                legendgroup="localkey",
            )
        )
    return result


def make_localkey_shapes(phrase_timeline_data):
    shapes = []
    rectangle_grouper, _ = make_adjacency_groups(phrase_timeline_data.localkey)
    y_min = phrase_timeline_data.chord_tone_tpc.min()
    for group, group_df in phrase_timeline_data.groupby(rectangle_grouper):
        x0, x1 = group_df.Start.min(), group_df.Finish.max()
        first_row = group_df.iloc[0]
        y_root = first_row.localkey_tonic_tpc - y_min
        text = first_row.localkey
        localkey_shapes = _make_localkey_shapes(
            y_root, is_minor=first_row.localkey_is_minor, x0=x0, x1=x1, text=text
        )
        shapes.extend(localkey_shapes)
    shapes[0].update(dict(showlegend=True, name="local key"))
    shapes[1].update(dict(showlegend=True, name="local tonic"))
    return shapes


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


SHARPWISE_COLORS = [
    "RED",
    "ROSE",
    "ORANGE",
    "PINK",
    "AMBER",
    "FUCHSIA",
    "YELLOW",
    "SLATE",
    "STONE",
]
FLATWISE_COLORS = [
    "LIME",
    "GREEN",
    "BLUE",
    "CYAN",
    "EMERALD",
    "INDIGO",
    "TEAL",
    "VIOLET",
    "SLATE",
    "STONE",
]


def _make_shape_data_for_numeral(
    numeral: str,
    globalkey: str,
    globalkey_is_minor: bool,
    x0: Number,
    x1: Number,
    y_min: int,
    local_tonic_tpc: int,
):
    numeral_tpc = ms3.roman_numeral2fifths(
        numeral, globalkey_is_minor
    ) + ms3.name2fifths(globalkey)
    y_root = numeral_tpc - y_min
    text = numeral
    first_numeral_component = numeral.split("/")[0]
    tonicized_is_minor = first_numeral_component.islower()
    shape_data = dict(
        x0=x0,
        x1=x1,
        y_root=y_root,
        is_minor=tonicized_is_minor,
        text=text,
    )
    distance_to_local_tonic = numeral_tpc - local_tonic_tpc
    if distance_to_local_tonic == 0:
        if tonicized_is_minor:
            primary_color = ("PURPLE", 700)
        else:
            primary_color = ("SKY", 500)
    else:
        color_index = abs(distance_to_local_tonic) - 1
        if distance_to_local_tonic > 0:
            color = SHARPWISE_COLORS[color_index]
        else:
            color = FLATWISE_COLORS[color_index]
        primary_color = (color, 500)
    shape_data["primary_color"] = utils.TailwindColorsHex.get_color(*primary_color)
    if tonicized_is_minor:
        color_name, color_shade = primary_color
        color_shade -= 300
        shape_data["secondary_color"] = utils.TailwindColorsHex.get_color(
            color_name, color_shade
        )
    return shape_data


def _get_tonicization_area_shape_data(all_dominant_stages, groupby_levels):
    area_shape_data = []
    y_min = phrase_timeline_data.chord_tone_tpc.min()
    for _, group_df in all_dominant_stages.groupby(groupby_levels):
        first_row = group_df.iloc[0]
        numeral = first_row.root_roman_or_its_dominants
        add_stage_area = not (numeral == "i" and first_row.localkey_is_minor) and not (
            numeral == "I" and not first_row.localkey_is_minor
        )
        other_resolved_dominants = group_df.expected_root_tpc.eq(
            group_df.subsequent_root_tpc
        ) & group_df.root_roman_or_its_dominant.ne(group_df.root_roman_or_its_dominants)
        add_tonicized_areas = other_resolved_dominants.any()
        if not (add_stage_area or add_tonicized_areas):
            continue
        x0, x1 = group_df.Start.min(), group_df.Finish.max()
        if add_stage_area:
            shape_data = _make_shape_data_for_numeral(
                numeral=numeral,
                globalkey=first_row.globalkey,
                globalkey_is_minor=first_row.globalkey_is_minor,
                x0=x0,
                x1=x1,
                y_min=y_min,
                local_tonic_tpc=first_row.localkey_tonic_tpc,
            )
            shape_data["legendgroup"] = "stage"
            area_shape_data.append(shape_data)
        if add_tonicized_areas:
            unique_substages = group_df[other_resolved_dominants].index.unique()
            stage_name, substage_name = unique_substages.names[-2:]
            for *_, stage, substage in unique_substages:
                rectangle_data = group_df.query(
                    f"{stage_name} == {stage} & {substage_name} in [{substage - 1}, {substage}]"
                )
                x0, x1 = rectangle_data.Start.min(), rectangle_data.Finish.max()
                last_row = rectangle_data.iloc[-1]
                numeral = last_row.root_roman_or_its_dominant
                shape_data = _make_shape_data_for_numeral(
                    numeral=numeral,
                    globalkey=last_row.globalkey,
                    globalkey_is_minor=last_row.globalkey_is_minor,
                    x0=x0,
                    x1=x1,
                    y_min=y_min,
                    local_tonic_tpc=last_row.localkey_tonic_tpc,
                )
                shape_data["legendgroup"] = "tonicization"
                area_shape_data.append(shape_data)
    return area_shape_data


def _make_tonicization_shapes(
    y_root: int,
    is_minor: bool,
    x0: Number,
    x1: Number,
    legendgroup: str,
    primary_color: str,
    secondary_color: Optional[str] = None,
    text: Optional[str] = None,
) -> List[dict]:
    result = []
    if is_minor:
        y0_primary, y1_primary, y0_secondary, y1_secondary = get_minor_y_coordinates(
            y_root
        )
    else:
        y0_primary, y1_primary, y0_secondary, y1_secondary = get_major_y_coordinates(
            y_root
        )
    rectangle_settings = dict(
        legendgroup=legendgroup,
        line_width=0,
        opacity=0.3,
        layer="below",
        textposition="middle center",
        label=dict(font=dict(size=100)),
    )
    result.append(
        utils.make_rectangle_shape(
            x0=x0,
            x1=x1,
            y0=y0_primary,
            y1=y1_primary,
            text=text,
            fillcolor=primary_color,
            **rectangle_settings,
        )
    )
    result.append(make_tonic_line(y_root, x0, x1, line_dash="dot"))
    if is_minor:
        if y0_secondary is not None:
            result.append(
                utils.make_rectangle_shape(
                    x0=x0,
                    x1=x1,
                    y0=y0_secondary,
                    y1=y1_secondary,
                    fillcolor=secondary_color,
                    **rectangle_settings,
                )
            )
    return result


def get_tonicization_data(phrase_timeline_data):
    all_dominant_stages = subselect_dominant_stages(phrase_timeline_data)
    groupby_levels = all_dominant_stages.index.names[:-1]
    area_shape_data = _get_tonicization_area_shape_data(
        all_dominant_stages, groupby_levels
    )
    shapes = []
    for shape_data in area_shape_data:
        shapes.extend(_make_tonicization_shapes(**shape_data))
    if len(shapes):
        shapes[0].update(dict(showlegend=True, name="tonicized area"))
        shapes[1].update(dict(showlegend=True, name="tonicized pitch class"))
    return shapes


colorscale = make_function_colors(detailed=DETAILED_FUNCTIONS)

# %%
PIN_PHRASE_ID = 6832
# 2358
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
    shapes=make_localkey_shapes(phrase_timeline_data)
    + get_tonicization_data(phrase_timeline_data),
)
fig

# %%
make_root_roman_or_its_dominants_criterion(
    phrase_annotations, query="phrase_id == 6832", inspect_masks=True
)

# %%
phrase_timeline_data

# %%
make_localkey_shapes(phrase_timeline_data)

# %%

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
