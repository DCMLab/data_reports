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

# %% mystnb={"code_prompt_hide": "Hide imports", "code_prompt_show": "Show imports"} tags=["hide-cell"]
# # %load_ext autoreload
# # %autoreload 2

import os
from typing import Hashable, Optional

import dimcat as dc
import ms3
import pandas as pd
from dimcat import resources
from dimcat.data.resources.utils import merge_columns_into_one
from dimcat.plotting import write_image
from git import Repo

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
phrase_bodies = phrase_annotations.get_phrase_data(
    ["bass_note", "intervals_over_bass"], drop_levels="phrase_component"
)
phrase_bodies.head(20)

# %%
bgt = phrase_bodies.apply_step("BigramAnalyzer")
bgt.loc(axis=1)["b", "bass_note"] -= bgt.loc(axis=1)["a", "bass_note"]
bgt

# %%
chord_type_pairs = bgt.make_bigram_tuples(
    "intervals_over_bass", None, terminal_symbols="DROP"
)
chord_type_pairs.make_ranking_table()

# %%
chord_type_transitions = bgt.get_transitions(
    "intervals_over_bass", None, terminal_symbols="DROP"
)
chord_type_transitions.head(50)

# %%
new_idx = chord_type_transitions.index.copy()
antecedent_counts = (
    chord_type_transitions.groupby("antecedent")["count"].sum().to_dict()
)
level_0_values = pd.Series(
    new_idx.get_level_values(0).map(antecedent_counts.get),
    index=new_idx,
    name="antecedent_count",
)
pd.concat([level_0_values, chord_type_transitions], axis=1).sort_values(
    ["antecedent_count", "count"], ascending=False
)

# %%
chord_type_transitions.sort_values("count", ascending=False)

# %%
bgt.make_bigram_tuples(terminal_symbols="DROP").make_ranking_table()
