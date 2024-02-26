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
# # Detecting diatonic bands
#
# ToDo
#
# * n01op18-1_01, phrase_id 4, viio/vi => #viio/
# * 07-1, phrase_id 2415, vi/V in D would be f# but this is clearly in a. It is a minor key, so bVI should be VI
# * phrase806_n14op131_05_1-79 clearly too long, begins with sequenced segments ending on HCs

# %% mystnb={"code_prompt_hide": "Hide imports", "code_prompt_show": "Show imports"} tags=["hide-cell"]
# %load_ext autoreload
# %autoreload 2
import os
import warnings
from numbers import Number
from random import choice
from typing import List, Literal, Optional, Tuple

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


def save_figure_as(
    fig, filename, formats=("png", "pdf"), directory=RESULTS_PATH, **kwargs
):
    if formats is not None:
        for fmt in formats:
            write_image(fig, filename, directory, format=fmt, **kwargs)
    else:
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
    globalkey_tpc = ms3.transform(phrase_data.globalkey, ms3.name2fifths)
    localkey_tonic_tpc = localkey_tonic_fifths.add(globalkey_tpc).rename(
        "localkey_tonic_tpc"
    )
    expected_root_tpc = (
        ms3.transform(
            phrase_data,
            ms3.roman_numeral2fifths,
            ["effective_localkey", "globalkey_is_minor"],
        )
        .add(globalkey_tpc)
        .rename("expected_root_tpc")
    )
    is_dominant = utils.make_dominant_selector(phrase_data)
    expected_root_tpc = expected_root_tpc.where(is_dominant).astype("Int64")
    expected_numeral = phrase_data.relativeroot.fillna(
        phrase_data.effective_localkey_is_minor.map({True: "i", False: "I"})
    ).rename(
        "expected_numeral"
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
    root_tpc = (
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
        .add(globalkey_tpc)
        .astype("Int64")
        .rename("root_tpc")
    )
    subsequent_root_tpc = root_tpc.shift().rename("subsequent_root_tpc")
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
    first_merge_index_in_group = {
        group: mask.values.argmax()
        for group, mask in merge_with_previous.groupby(dominant_grouper)
        if mask.any()
    }  # for each dominant group where at least one dominant resolves ('merge_with_previous' is True), the index of
    # the revelant row within the group, which serves as signal for a potential dominant chain starting with this index
    # to be fille with the expected resolution of its first member
    potential_dominant_chain_mask = (
        merge_with_previous | copy_chain_decision_from_previous
    )
    dominant_chains_groupby = potential_dominant_chain_mask.groupby(dominant_grouper)
    dominant_chain_fill_indices = []
    for (group, dominant_chain), index in zip(
        dominant_chains_groupby, dominant_chains_groupby.indices.values()
    ):
        fill_after = first_merge_index_in_group.get(group)
        if fill_after is None:
            continue
        fill_from = fill_after + 1
        for do_fill, ix in zip(dominant_chain[fill_from:], index[fill_from:]):
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
    root_dominant_criterion = expected_numeral.where(
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
    root_dominants_criterion = expected_numeral.where(
        is_dominant & all_but_ultima_selector, phrase_data.root_roman
    )
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
        expected_numeral,
        expected_root_tpc,
        root_tpc,
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
    phrase_annotations,  # query=f"phrase_id == 9649", inspect_masks=True
)
criterion2stages["root_roman_or_its_dominants"] = root_roman_or_its_dominants
root_roman_or_its_dominants.head(100)


# %% [raw] jupyter={"outputs_hidden": false}
# utils.compare_criteria_metrics(criterion2stages, height=1000)

# %% [raw] jupyter={"outputs_hidden": false}
# utils._compare_criteria_stage_durations(
#     criterion2stages, chronological_corpus_names=chronological_corpus_names
# )

# %% [raw] jupyter={"outputs_hidden": false}
# utils._compare_criteria_phrase_lengths(
#     criterion2stages, chronological_corpus_names=chronological_corpus_names
# )

# %% [raw] jupyter={"outputs_hidden": false}
# utils._compare_criteria_entropies(
#     criterion2stages, chronological_corpus_names=chronological_corpus_names
# )


# %% jupyter={"outputs_hidden": false}
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
    aug6 = "augmented 6th"
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
    if_halfdim7 = leading_tone_is_root & timeline_data.chord_type.eq("%7")
    is_dominant = timeline_data.expected_root_tpc.notna()
    is_aug6 = timeline_data.chord_type.isin(("Fr", "Ger", "It"))
    group_levels = is_dominant.index.names[:-1]  # groupby substages by stage
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
    resource_column.where(~is_aug6, DetailedFunction.aug6.value, inplace=True)
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


def make_timeline_data(root_roman_or_its_dominants, detailed=True):
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


# %% jupyter={"outputs_hidden": false}
def make_function_colors(detailed=True):
    if detailed:
        colorscale = {
            resource: utils.TailwindColorsHex.get_color(color_name)
            for resource, color_name in [
                (DetailedFunction.i.value, "PURPLE_700"),
                (DetailedFunction.I.value, "SKY_500"),
                (DetailedFunction.V.value, "RED_300"),
                (DetailedFunction.vii.value, "RED_400"),
                (DetailedFunction.V7.value, "RED_500"),
                (DetailedFunction.vii07.value, "RED_600"),
                (DetailedFunction.viio7.value, "RED_700"),
                (DetailedFunction.aug6.value, "RED_800"),
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


def make_tonic_line(y_root: int, x0: Number, x1: Number, line_dash="solid"):
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
    result.append(make_tonic_line(y_root, x0, x1, line_dash="solid"))
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
    if len(dominant_stage_index) == 0:
        return pd.DataFrame()
    all_dominant_stages = subselect_multiindex_from_df(
        timeline_data, dominant_stage_index
    )
    return all_dominant_stages


SHARPWISE_COLORS = [
    "RED",  # localkey + 1 fifths
    "ROSE",  # + 2
    "ORANGE",  # + 3 etc.
    "PINK",
    "AMBER",
    "FUCHSIA",
    "YELLOW",
    "SLATE",
    "STONE",
]
FLATWISE_COLORS = [
    "LIME",  # localkey - 1 fifths
    "GREEN",  # - 2
    "BLUE",  # - 3 etc.
    "CYAN",
    "EMERALD",
    "INDIGO",
    "TEAL",
    "VIOLET",
    "SLATE",
    "STONE",
]


def style_shape_data_by_root(
    root_tpc: int,
    minor: bool,
    localkey_tonic_tpc: int,
    y_min: int,
    x0: Number,
    x1: Number,
    **kwargs,
):
    y_root = root_tpc - y_min
    shape_data = dict(x0=x0, x1=x1, y_root=y_root, is_minor=minor, **kwargs)
    distance_to_local_tonic = int(root_tpc - localkey_tonic_tpc)
    if distance_to_local_tonic == 0:
        if minor:
            primary_color = ("PURPLE", 700)
        else:
            primary_color = ("SKY", 500)
    else:
        color_index = abs(distance_to_local_tonic) - 1
        if distance_to_local_tonic > 0:
            color = SHARPWISE_COLORS[color_index]
        else:
            color = FLATWISE_COLORS[color_index]
        primary_color = (color, 600)
    shape_data["primary_color"] = utils.TailwindColorsHex.get_color(*primary_color)
    if minor:
        color_name, color_shade = primary_color
        color_shade -= 300
        shape_data["secondary_color"] = utils.TailwindColorsHex.get_color(
            color_name, color_shade
        )
    return shape_data


def make_shape_data_for_numeral(
    numeral: str,
    local_tonic_tpc: str,
    globalkey_is_minor: bool,
    x0: Number,
    x1: Number,
    y_min: int,
):
    numeral_tpc = (
        ms3.roman_numeral2fifths(numeral, globalkey_is_minor) + local_tonic_tpc
    )
    first_numeral_component = numeral.split("/")[0]
    tonicized_is_minor = first_numeral_component.islower()
    return style_shape_data_by_root(
        numeral_tpc, tonicized_is_minor, local_tonic_tpc, y_min, x0, x1, text=numeral
    )


def get_stage_shape_data(all_dominant_stages, groupby_levels, y_min):
    """Returns filled rectangle shapes for all non-tonic stages"""
    area_shape_data = []
    for _, group_df in all_dominant_stages.groupby(groupby_levels):
        first_row = group_df.iloc[0]
        numeral = first_row.root_roman_or_its_dominants
        # add_stage_area = not (numeral == "i" and first_row.localkey_is_minor) and not (
        #     numeral == "I" and not first_row.localkey_is_minor
        # )
        # other_resolved_dominants = group_df.expected_root_tpc.eq(
        #     group_df.subsequent_root_tpc
        # ) & group_df.root_roman_or_its_dominant.ne(group_df.root_roman_or_its_dominants)
        # add_tonicized_areas = other_resolved_dominants.any()
        # if not (add_stage_area or add_tonicized_areas):
        #     continue
        x0, x1 = group_df.Start.min(), group_df.Finish.max()
        # if add_stage_area:
        shape_data = make_shape_data_for_numeral(
            numeral=numeral,
            local_tonic_tpc=first_row.localkey_tonic_tpc,
            globalkey_is_minor=first_row.globalkey_is_minor,
            x0=x0,
            x1=x1,
            y_min=y_min,
        )
        shape_data["legendgroup"] = "stage"
        area_shape_data.append(shape_data)
        # if add_tonicized_areas:
        #     unique_substages = group_df[other_resolved_dominants].index.unique()
        #     stage_name, substage_name = unique_substages.names[-2:]
        #     for *_, stage, substage in unique_substages:
        #         rectangle_data = group_df.query(
        #             f"{stage_name} == {stage} & {substage_name} in [{substage - 1}, {substage}]"
        #         )
        #         x0, x1 = rectangle_data.Start.min(), rectangle_data.Finish.max()
        #         last_row = rectangle_data.iloc[-1]
        #         numeral = last_row.root_roman_or_its_dominant
        #         shape_data = make_shape_data_for_numeral(
        #             numeral=numeral,
        #             local_tonic_tpc=first_row.localkey_tonic_tpc,
        #             globalkey_is_minor=first_row.globalkey_is_minor,
        #             x0=x0,
        #             x1=x1,
        #             y_min=y_min,
        #         )
        #         shape_data["legendgroup"] = "tonicization"
        #         area_shape_data.append(shape_data)
    return area_shape_data


def make_tonicization_shapes(
    y_root: int,
    is_minor: bool,
    x0: Number,
    x1: Number,
    legendgroup: Literal["stage"] | str,
    primary_color: str,
    secondary_color: Optional[str] = None,
    primary_line_dash: str = "solid",
    secondary_line_dash: str = "dot",
    text: Optional[str] = None,
) -> List[dict]:
    """Turns 'shape data' dicts into the corresponding rectangle shapes, applying different styles according to
    'legendgroup' and 'is_minor'.

    Args:
        y_root: Coordinate of the root pitch, i.e. tonic_tpc - y_min.
        is_minor:
            If True and 'secondary_color' is not None, a secondary rectangle is added to account for minor's +3 range.
        x0: Leftmost x coordinate.
        x1: Rightmost x coordinate.
        legendgroup: If 'stage', the rectangle(s) will be filled, otherwise they will be outlined.
        primary_color: Color of the rectangle.
        secondary_color:
            Only relevant when 'is_minor' is True: If a secondary color is specified, a secondary rectangle
            will be added to account for minor's +3 range.
        text: The tonicized numeral.

    Returns:
        A list of rectangle shapes that can be added to a Plotly figure.
    """
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
        x0=x0,
        x1=x1,
        y0=y0_primary,
        y1=y1_primary,
        line_dash=primary_line_dash,
        text=text,
        legendgroup=legendgroup,
        layer="below",
        textposition="middle center",
        label=dict(font=dict(size=70)),
    )
    if legendgroup == "stage":
        rectangle_settings["fillcolor"] = primary_color
        rectangle_settings["line_width"] = 0
        rectangle_settings["opacity"] = 0.3
    else:
        rectangle_settings["line_color"] = primary_color
        rectangle_settings["line_width"] = 3
    result.append(
        utils.make_rectangle_shape(
            **rectangle_settings,
        )
    )
    if legendgroup == "stage":
        tonic_line_dash = "longdash"
    elif primary_line_dash == "solid":
        tonic_line_dash = "longdashdot"
    else:
        tonic_line_dash = "dashdot"
    result.append(make_tonic_line(y_root, x0, x1, line_dash=tonic_line_dash))
    if is_minor and y0_secondary is not None:
        del rectangle_settings["text"]
        rectangle_settings["y0"] = y0_secondary
        rectangle_settings["y1"] = y1_secondary
        rectangle_settings["line_dash"] = secondary_line_dash
        if legendgroup == "stage":
            rectangle_settings["fillcolor"] = secondary_color
        else:
            rectangle_settings["line_color"] = secondary_color
        result.append(
            utils.make_rectangle_shape(
                **rectangle_settings,
            )
        )
    return result


def get_tonicization_shape_data(phrase_timeline_data):
    """Uses :func:`_make_shape_data_for_numeral` to create shape data for all resolving tonicizations that are not
    already covered by a stage.
    """
    out_of_stage_tonicizations = phrase_timeline_data.expected_root_tpc.eq(
        phrase_timeline_data.subsequent_root_tpc
    ) & phrase_timeline_data.root_roman_or_its_dominant.ne(
        phrase_timeline_data.root_roman_or_its_dominants
    )
    if not out_of_stage_tonicizations.any():
        return []
    area_shape_data = []
    y_min = phrase_timeline_data.chord_tone_tpc.min()
    grouping, _ = make_adjacency_groups(
        phrase_timeline_data.root_roman_or_its_dominant, groupby="phrase_id"
    )
    groups_to_consider = out_of_stage_tonicizations.groupby(grouping).any().to_dict()
    for group, rectangle_data in phrase_timeline_data.groupby(grouping):
        if not groups_to_consider[group]:
            continue
        x0, x1 = rectangle_data.Start.min(), rectangle_data.Finish.max()
        last_row = rectangle_data.iloc[-1]
        numeral = last_row.root_roman_or_its_dominant
        shape_data = make_shape_data_for_numeral(
            numeral=numeral,
            local_tonic_tpc=last_row.localkey_tonic_tpc,
            globalkey_is_minor=last_row.globalkey_is_minor,
            x0=x0,
            x1=x1,
            y_min=y_min,
        )
        shape_data["legendgroup"] = "tonicization"
        area_shape_data.append(shape_data)
    return area_shape_data


def get_tonicization_data(
    phrase_timeline_data,
    stages: bool = True,
    tonicizations: bool = False,
):
    """Collects shape data from the selected functions and turns it into corresponding Plotly shapes using
    :func:`_make_tonicization_shapes`.


    Args:
        phrase_timeline_data: Chords of a single phrase with exploded chord tones.
        stages: By default (True), non-tonic stages are highlighted by filled rectangles.
        tonicizations:
            By default (True), tonicizations are highlighted by outlined rectangles. Implementation
            superseded by :func:`get_extended_tonicization_data`.

    Returns:

    """
    all_dominant_stages = subselect_dominant_stages(phrase_timeline_data)
    groupby_levels = all_dominant_stages.index.names[:-1]
    y_min = phrase_timeline_data.chord_tone_tpc.min()
    area_shape_data = []
    if stages:
        area_shape_data += get_stage_shape_data(
            all_dominant_stages, groupby_levels, y_min=y_min
        )
    if tonicizations:
        area_shape_data += get_tonicization_shape_data(phrase_timeline_data)
    shapes = []
    for shape_data in area_shape_data:
        shapes.extend(make_tonicization_shapes(**shape_data))
    if len(shapes):
        shapes[0].update(dict(showlegend=True, name="tonicized area"))
        shapes[1].update(dict(showlegend=True, name="tonicized pitch class"))
    return shapes


def make_extended_tonicization_shape_data(stage_inspection_data):
    dominant_grouper, group2expected_root_tpc = ms3.adjacency_groups(
        stage_inspection_data.expected_root_tpc, na_values="bfill", prevent_merge=False
    )
    stage_data = pd.concat(
        [
            stage_inspection_data,
            utils.make_start_finish(stage_inspection_data.df.duration_qb),
            dominant_grouper.rename("dominant_group"),
            dominant_grouper.map(group2expected_root_tpc).rename(
                "group_expects_root_tpc"
            ),
        ],
        axis=1,
    )
    shape_data = []

    def new_shape_skeleton():
        return dict(
            x0=None,
            x1=None,
            localkey_tonic_tpc=None,
            root_tpc=None,
            # text=None,
            minor=None,
            primary_line_dash=None,
            legendgroup="tonicization",
        )

    def initialize_next_shape():
        nonlocal current_shape
        if current_shape["x0"] is not None:
            shape_data.append(dict(current_shape))
        current_shape = new_shape_skeleton()

    def new_solid_shape(row, is_dominant=True):
        nonlocal current_shape
        initialize_next_shape()
        current_shape["x0"] = row.Start
        current_shape["x1"] = row.Finish
        current_shape["localkey_tonic_tpc"] = row.localkey_tonic_tpc
        if is_dominant:
            current_shape["root_tpc"] = row.expected_root_tpc
            current_shape["minor"] = row.expected_numeral.islower()
            # current_shape["text"] = row.expected_numeral
        else:
            current_shape["root_tpc"] = row.root_tpc
            current_shape["minor"] = row.effective_numeral.islower()
        current_shape["primary_line_dash"] = "solid"

    def new_dashed_shape(row, minor):
        nonlocal current_shape
        initialize_next_shape()
        current_shape["x0"] = row.Start
        current_shape["x1"] = row.Finish
        current_shape["localkey_tonic_tpc"] = row.localkey_tonic_tpc
        current_shape["root_tpc"] = row.group_expects_root_tpc
        current_shape["minor"] = minor
        current_shape["primary_line_dash"] = "dash"

    def prolong_solid_shape(row, is_dominant=True):
        nonlocal current_shape
        if is_dominant:
            minor = row.expected_numeral.islower()
        else:
            minor = row.effective_numeral.islower()
        if current_shape["minor"] == minor:
            current_shape["x1"] = row.Finish
        else:
            new_solid_shape(row, is_dominant=is_dominant)

    def prolong_dashed_shape(row, minor):
        nonlocal current_shape
        if current_shape["minor"] == minor:
            current_shape["x1"] = row.Finish
        else:
            new_dashed_shape(row, minor)

    def chord_fits_range(row, root_tpc) -> Optional[bool]:
        leading_tone_tpc = root_tpc + 5
        if row.highest_tpc > leading_tone_tpc:
            return
        lowest_tpc_major = leading_tone_tpc - 6
        lowest_tpc_minor = leading_tone_tpc - 9
        if row.lowest_tpc >= lowest_tpc_major:
            # fits in the major key range
            return False
        elif row.lowest_tpc >= lowest_tpc_minor:
            # fits in the minor key range
            return True
        return

    current_shape = new_shape_skeleton()
    for row in stage_data.iloc[::-1].itertuples(index=False):
        if not pd.isnull(
            row.expected_root_tpc
        ):  # this is a dominant and starts or prolongs a solid-line rectangle
            if current_shape["x0"] is None:
                # no shape currently active
                new_solid_shape(row)
            else:
                if current_shape["primary_line_dash"] == "solid":
                    if current_shape["root_tpc"] in (
                        row.expected_root_tpc,
                        row.root_tpc,
                    ):
                        prolong_solid_shape(row, is_dominant=True)
                        if current_shape["root_tpc"] != row.expected_root_tpc:
                            # this dominant resolves the previous one, so it prolonges the solid rectangle, but
                            # then starts a new one for the new expected root
                            new_solid_shape(row, is_dominant=True)
                    else:
                        new_solid_shape(row, is_dominant=True)
                else:
                    new_solid_shape(row, is_dominant=True)
        elif current_shape["x0"] is None:
            # no shape currently active, skip the gap until the next dominant
            continue
        elif current_shape["primary_line_dash"] == "solid":
            if row.group_expects_root_tpc == row.root_tpc:
                # this is a resolution of the previous dominant (in terms of the chord root)
                prolong_solid_shape(row, is_dominant=False)
            else:
                # check whether this chord's TPC range fits in as a prolongation of a tonicized key
                mode_range = chord_fits_range(row, current_shape["root_tpc"])
                if mode_range is None:
                    # does not fit in either range
                    initialize_next_shape()
                    continue
                new_dashed_shape(row, minor=mode_range)
        else:
            # current shape is dashed, check whether this chord prolongs its TPC range or its parallel key's TPC range
            mode_range = chord_fits_range(row, current_shape["root_tpc"])
            if mode_range is None:
                # does not fit in either range
                initialize_next_shape()
                continue
            if row.group_expects_root_tpc == current_shape["root_tpc"]:
                # prolongs the tonicized key's range
                prolong_dashed_shape(row, minor=mode_range)
            else:
                initialize_next_shape()
    initialize_next_shape()
    return shape_data


def get_extended_tonicization_shape_data(stage_inspection_data, y_min):
    tonicization_shapes = make_extended_tonicization_shape_data(stage_inspection_data)
    shapes = []
    for shape_info in tonicization_shapes:
        shape_data = style_shape_data_by_root(y_min=y_min, **shape_info)
        shapes.extend(make_tonicization_shapes(**shape_data))
    return shapes


# %% jupyter={"outputs_hidden": false}
DETAILED_FUNCTIONS = True
timeline_data = make_timeline_data(
    root_roman_or_its_dominants, detailed=DETAILED_FUNCTIONS
)
n_phrases = max(timeline_data.index.levels[2])
colorscale = make_function_colors(detailed=DETAILED_FUNCTIONS)


# %% jupyter={"outputs_hidden": false}
def plot_phrase_stages(
    phrase_annotations,
    phrase_id,
    localkey_shapes: bool = True,
    stage_shapes: bool = True,
    tonicization_shapes: bool = True,
    detailed_functions=True,
    **kwargs,
):
    stage_data = make_root_roman_or_its_dominants_criterion(
        phrase_annotations, query=f"phrase_id == {phrase_id}"
    )
    phrase_timeline_data = make_timeline_data(stage_data, detailed=detailed_functions)
    colorscale = make_function_colors(detailed=detailed_functions)
    shapes = []
    if localkey_shapes:
        shapes.extend(make_localkey_shapes(phrase_timeline_data))
    if stage_shapes:
        shapes.extend(
            get_tonicization_data(
                phrase_timeline_data, stages=True, tonicizations=False
            )
        )
    if tonicization_shapes:
        shapes.extend(
            get_extended_tonicization_shape_data(
                stage_data, y_min=phrase_timeline_data.chord_tone_tpc.min()
            )
        )
    fig = utils.plot_phrase(
        phrase_timeline_data, colorscale=colorscale, shapes=shapes, **kwargs
    )
    return fig


fig = plot_phrase_stages(
    phrase_annotations,
    title="",
    x_axis=dict(
        tickmode="array", tickvals=[-2, -6, -10, -14, -18], ticktext=[9, 8, 7, 6, 5]
    ),
    phrase_id=4873,  # 5932
)
fig.update_shapes(
    dict(label_text=""), selector=dict(line_dash="dot", legendgroup="stage")
)
fig.update_shapes(dict(x0=-18), dict(x0=-18.5, legendgroup="stage"))
fig.update_shapes(dict(x0=-18), dict(x0=-18.5, type="line"))
fig.update_shapes(dict(y0=-0.5), dict(y0=-1.5))
fig.update_layout(dict(legend=dict(orientation="h", font_size=17)))
save_figure_as(
    fig, "modulation_structure_for_phrase_4873", formats=["pdf"], width=1280, height=720
)
fig

# %%
fig.layout.shapes

# %%
[shape for shape in fig.layout.shapes if shape.x0 == -18.0]

# %% jupyter={"outputs_hidden": false}
PIN_PHRASE_ID = 10403
# 24 good Bach
# 87 short and simple ABC example, e-a-G with pivot chord, pretty ideal
# 638 exciting ABC
# 806 from ABC is fantastic but I would actually see it as more than one phrase
# 827
# 2338 Beethoven Eb-f-c
# 2358
# 2983 cool Waldstein
# 3339
# 4739 Corelli
# 5103 Corelli with visible encoding error (pops out)
# 5122 super cool Couperin but the beginning is a bit verkorkst, modulation debatable
# 5932
# 9649
# 141102 Liederkreis great but has modulation to IV for a single measure. Discuss?
# 14031 Kinderszenen with pivot chord

# bugfix: 4157

warnings.simplefilter(action="ignore", category=FutureWarning)
if PIN_PHRASE_ID is None:
    current_id = choice(range(n_phrases))
else:
    current_id = PIN_PHRASE_ID
plot_phrase_stages(phrase_annotations, phrase_id=current_id)

# %% jupyter={"outputs_hidden": false}
modulating_phrases = phrase_annotations.groupby("phrase_id").filter(
    lambda df: df.localkey.nunique() > 1
)
modulating_ids = modulating_phrases.index.get_level_values("phrase_id").unique()

# %% jupyter={"outputs_hidden": false}
selected_modulating_id = choice(modulating_ids)
plot_phrase_stages(phrase_annotations, phrase_id=selected_modulating_id)

# %% jupyter={"outputs_hidden": false}
# plot_phrase_stages(phrase_annotations, phrase_id=2358)

# %% [raw] jupyter={"outputs_hidden": false}
# from pandas.core.indexers.objects import BaseIndexer
# import numpy.typing as npt
#
#
# class DominantsToEndIndexer(BaseIndexer):
#
#      def get_window_bounds(
#         self,
#         num_values: int = 0,
#         min_periods: int | None = None,
#         center: bool | None = None,
#         closed: str | None = None,
#         step: int | None = None,
#     ) -> Tuple[npt.NDArray, npt.NDArray]:
#         return (
#             np.zeros(num_values, dtype=np.int64),
#             np.arange(1, num_values + 1, dtype=np.int64),
#         )
#
# indexer = DominantsToEndIndexer()

# %% jupyter={"outputs_hidden": false}
criterion2stages["uncompressed"]
