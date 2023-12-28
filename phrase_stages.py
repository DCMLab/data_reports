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
# # Phrases in the DLC
#
# ToDo: Wrong `duration_qb` in phrase ID 14628


# %% mystnb={"code_prompt_hide": "Hide imports", "code_prompt_show": "Show imports"} tags=["hide-cell"]
# %load_ext autoreload
# %autoreload 2

import os
import re
from typing import Hashable, Optional

import dimcat as dc
import ms3
import pandas as pd
from dimcat import resources
from dimcat.data.resources.results import _entropy
from dimcat.data.resources.utils import merge_columns_into_one
from dimcat.plotting import make_bar_plot, make_box_plot, make_pie_chart, write_image
from git import Repo

from utils import (
    DEFAULT_OUTPUT_FORMAT,
    OUTPUT_FOLDER,
    get_repo_name,
    print_heading,
    resolve_dir,
    value_count_df,
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


def make_and_store_stage_data(
    phrase_feature,
    name: Optional[str] = None,
    columns="chord",
    components="body",
    drop_levels=3,
    reverse=True,
    level_name="stage",
    wide_format=True,
    query=None,
) -> resources.PhraseData:
    """Function sets the defaults for the stage TSVs produced in the following."""
    phrase_data = phrase_feature.get_phrase_data(
        columns=columns,
        components=components,
        drop_levels=drop_levels,
        reverse=reverse,
        level_name=level_name,
        wide_format=wide_format,
        query=query,
    )
    if name:
        phrase_data.to_csv(make_output_path(name, "tsv"), sep="\t")
    return phrase_data


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
phrase_annotations = D.get_feature("PhraseAnnotations")
phrase_annotations


# %%
def make_criteria_dict(phrase_annotations, criteria_dict, join_str=True):
    """Takes a {name -> [columns]} dict."""
    uncompressed = make_and_store_stage_data(
        phrase_annotations,
        columns=["chord_and_mode", "duration_qb"],
        wide_format=False,
    )
    name2stages = {"uncompressed": uncompressed}
    for name, columns in criteria_dict.items():
        criterion = make_criterion(
            phrase_annotations,
            columns=columns,
            criterion_name=name,
            join_str=join_str,
        )
        name2stages[name] = uncompressed.regroup_phrases(criterion)
    return name2stages


def get_stage_durations(df):
    return df.groupby(["corpus", "piece", "phrase_id", "stage"]).duration_qb.sum()


def compare_mean_stage_durations(durations_dict, category_title="stage_type", **kwargs):
    """Takes a {trace_name -> durations} dict where each entry will be turned into a bar plot trace for comparison."""
    aggregated = {
        name: durations.groupby("corpus").agg(["mean", "sem"])
        for name, durations in durations_dict.items()
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
        labels=dict(mean="mean duration of stages in ♩", corpus=""),
        **kwargs,
    )


def compare_criteria_stage_durations(phrase_annotations, criteria_dict, join_str=True):
    name2stages = make_criteria_dict(
        phrase_annotations, criteria_dict, join_str=join_str
    )
    durations_dict = {
        name: get_stage_durations(stages) for name, stages in name2stages.items()
    }
    return compare_mean_stage_durations(durations_dict, height=800)


compare_criteria_stage_durations(
    phrase_annotations,
    dict(
        chord_reduced_and_mode=["chord_reduced_and_mode"],
        bass_degree=["bass_note"],
        root_degree=["root"],
        root_roman=["root_roman", "localkey_mode"],
        numeral_or_applied_to=["numeral_or_applied_to", "localkey_mode"],
    ),
)


# %%
def get_criterion_stage_entropies(df, criterion_name=None):
    if not criterion_name:
        criterion_name = df.columns.to_list()[0]
    criterion_distributions = df.groupby(["corpus", criterion_name]).duration_qb.sum()
    return criterion_distributions.groupby("corpus").agg(_entropy).rename("entropy")


def compare_entropies(stages_dict, category_title="stage_type", **kwargs):
    """Takes a {trace_name -> PhraseData} dict where each entry will be turned into a bar plot trace for comparison."""
    entropies = {
        name: get_criterion_stage_entropies(durations)
        for name, durations in stages_dict.items()
    }
    df = pd.concat(entropies, names=[category_title])
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
        labels=dict(entropy="entropy of stage distributions in bits", corpus=""),
        **kwargs,
    )


def compare_criteria_entropies(phrase_annotations, criteria_dict, join_str=True):
    name2stages = make_criteria_dict(
        phrase_annotations, criteria_dict, join_str=join_str
    )
    return compare_entropies(name2stages, height=800)


compare_criteria_entropies(
    phrase_annotations,
    dict(
        chord_reduced_and_mode=["chord_reduced_and_mode"],
        bass_degree=["bass_note"],
        root_degree=["root"],
        root_roman=["root_roman", "localkey_mode"],
        numeral_or_applied_to=["numeral_or_applied_to", "localkey_mode"],
    ),
)

# %%
criteria = make_criteria_dict(
    phrase_annotations,
    dict(
        chord_reduced_and_mode=["chord_reduced_and_mode"],
        bass_degree=["bass_note"],
        root_degree=["root"],
        root_roman=["root_roman", "localkey_mode"],
        numeral_or_applied_to=["numeral_or_applied_to", "localkey_mode"],
    ),
)


# %%
def get_stage_lengths(df):
    stage_index = df.index.to_frame(index=False)
    phrase_id_col = stage_index.columns.get_loc("phrase_id")
    groupby = stage_index.columns.to_list()[: phrase_id_col + 1]
    stage_lengths = stage_index.groupby(groupby).stage.max() + 1
    return stage_lengths


uncompressed_lengths = get_stage_lengths(criteria["uncompressed"])
uncompressed_lengths

# %%
uncompressed_lengths.groupby("corpus").describe()

# %%


make_box_plot(
    uncompressed_lengths,
    x_col="corpus",
    y_col="stage",
    height=800,
    category_orders=dict(corpus=chronological_corpus_names),
)

# %%
phrases = phrase_annotations.extract_feature("PhraseLabels")
phrases

# %%
vc = value_count_df(phrases.end_chord, rank_index=True)
vc.head(50)

# %%
stages = make_and_store_stage_data(
    phrases, name="stages", columns=["localkey", "chord"]
)
stages.head()

# %%
onekey_major = make_and_store_stage_data(
    phrases,
    name="onekey_major",
    query="body_n_modulations == 0 & localkey_mode == 'major'",
)
onekey_minor = make_and_store_stage_data(
    phrases,
    name="onekey_minor",
    query="body_n_modulations == 0 & localkey_mode == 'minor'",
)
one_key_major_I = make_and_store_stage_data(
    phrases,
    name="onekey_major_I",
    query="body_n_modulations == 0 & localkey_mode == 'major' & end_chord == 'I'",
)  # end_chord.str.contains('^I(?![iIvV\/])')")
one_key_minor_i = make_and_store_stage_data(
    phrases,
    name="one_key_minor_i",
    query="body_n_modulations == 0 & localkey_mode == 'minor' & end_chord == 'i'",
)


# %%
def show_stage(df, column: int, **kwargs):
    """Pie chart for a given stage."""
    stage = df.loc(axis=1)[column]
    if len(stage.shape) > 1:
        stage = stage.iloc(axis=1)[0]
    vc = stage.value_counts().to_frame()
    settings = dict(
        traces_settings=dict(textposition="inside"),
        layout=dict(uniformtext_minsize=20, uniformtext_mode="hide"),
    )
    settings.update(kwargs)
    return make_pie_chart(vc, **settings)


show_stage(one_key_major_I, 1)

# %%
numeral_regex = r"^(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none)"


def get_numeral(label: str) -> str:
    """From a chord label, get the root or, if an applied chord, the root that the chord is applied to."""
    considered = label.split("/")[-1]
    try:
        return re.match(numeral_regex, considered).group()
    except AttributeError:
        return pd.NA


numeral_criterion = one_key_major_I.dataframe.chord.map(
    get_numeral, na_action="ignore"
).rename("root_numeral")
result = one_key_major_I.regroup_phrases(numeral_criterion)
result.to_csv(make_output_path("one_key_major_I.stages", "tsv"), sep="\t")
result


# %%
# regrouped = result._format_dataframe(
#     result.drop(columns="root_numeral"), "WIDE"
# )  # ToDo: Unstack needs to take into account the new index levels
regrouped = result["chord"].unstack(level=[-2, -1]).sort_index(axis=1)
show_stage(regrouped, 1)

# %%
regrouped

# %% [raw]
# # this recursive approach is very inefficient
#
# @cache
# def make_regexes(numeral) -> Tuple[str, str]:
#     """Returns two regular expressions, one that matches chords with a given root and are not applied chords, and one
#     that matches chords that are applied to that root.
#     """
#     return rf"^{numeral}[^iIvV\/]*$", rf"^.+/{numeral}\]?$"
#
#
# def _merge_subsequent_into_stage(df, *regex, fill_value=".", printout=False):
#     """Considers the left-most column as the current stage, the head, and the adjacent column (the first column of the
#     tail) as subsequent stage. The items from the latter that match any of the given regular expressions are merged
#     into the former by inserting a new column containing these items representing a substage (filled with the
#     fill_value). The corresponding rows in the tail, from which items were merged, are rolled to the left and newly
#     empty columns at the end of tail are dropped. Then the tail is processed with recursively_merge_adjacent_roots()
#     before concatenating it to the substages on the left.
#     """
#     _, n_cols = df.shape
#     if n_cols < 2:
#         return df
#     substage = itertools.count(start=1)
#     column_iloc = df.iloc(axis=1)
#     stage_column = column_iloc[0]
#     stage = stage_column.name[0]
#     head = [stage_column]
#     tail = column_iloc[1:]
#     right = column_iloc[1]
#
#     def make_mask(series):
#         mask = np.zeros_like(right, bool)
#         for rgx in regex:
#             mask |= series.str.contains(rgx, na=False)
#         return mask
#
#     mask = make_mask(right)
#     while mask.any():
#         new_column = right.where(mask, other=fill_value).rename((stage, next(substage)))
#         head.append(new_column)
#         tail.loc[mask] = tail.loc[mask].shift(-1, axis=1)
#         right = tail.iloc(axis=1)[0]
#         mask = make_mask(right)
#     tail_curtailed = tail.dropna(how="all", axis=1)
#     if printout:
#         print(
#             f"Added {len(head)} substages to {stage}, reducing the tail from {n_cols} to {tail_curtailed.shape[1]}"
#         )
#     recursively_merged_tail = recursively_merge_adjacent_roots(
#         tail_curtailed, printout=printout
#     )
#     return pd.concat(head + [recursively_merged_tail], axis=1)
#
#
# def recursively_merge_adjacent_roots(phrase_data, printout=False):
#     """Recursively inserts substage columns by merging labels from later stages into earlier ones when they have
#     the same root. Chords applied to the root are also merged."""
#     n_rows, n_cols = phrase_data.shape
#     if printout:
#         print(f"merge_adjacent_roots({n_rows}x{n_cols})")
#     if n_cols < 2:
#         return phrase_data
#     numeral_groups = phrase_data.iloc(axis=1)[0].map(get_numeral, na_action="ignore")
#     results = []
#     for numeral, df in phrase_data.groupby(numeral_groups):
#         regex_a, regex_b = make_regexes(numeral)
#         df_curtailed = df.dropna(how="all", axis=1)
#         if printout:
#             print(f"_merge_subsequent_into_stage({len(df)}/{n_rows}) for {numeral}")
#         numeral_df = _merge_subsequent_into_stage(
#             df_curtailed, regex_a, regex_b, printout=printout
#         )
#         results.append(numeral_df)
#     return pd.concat(results).sort_index(axis=1)
#
# result = recursively_merge_adjacent_roots(phrase_data)
