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
# # Chord Profiles

# %% mystnb={"code_prompt_hide": "Hide imports", "code_prompt_show": "Show imports"} tags=["hide-cell"]
# # %load_ext autoreload
# # %autoreload 2
import os
from typing import Tuple

import dimcat as dc
import ms3
import pandas as pd
import plotly.express as px
from dimcat.data.resources import Durations
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
RESULTS_PATH = os.path.abspath(os.path.join(OUTPUT_FOLDER, "chord_profiles"))
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
D

# %%
pipeline = [
    "HasHarmonyLabelsFilter",
    "ModeGrouper",
    "CorpusGrouper",
]
analyzed_D = D.apply_step(*pipeline)
harmony_labels = analyzed_D.get_feature("HarmonyLabels")
harmony_labels

# %%
harmony_labels.loc[
    harmony_labels["scale_degrees_major"] == ("1", "3", "5", "b7"), "chord"
].value_counts()

# %%
sd_maj_sonorities = (
    harmony_labels.groupby("scale_degrees_major")
    .duration_qb.sum()
    .sort_values(ascending=False)
)
pd.concat(
    [
        sd_maj_sonorities,
        (sd_maj_sonorities / sd_maj_sonorities.sum()).rename("proportion"),
    ],
    axis=1,
)

# %%
sd_major_occurrences = (
    harmony_labels.groupby("scale_degrees_major")
    .size()
    .rename("frequency")
    .sort_values(ascending=False)
    .reset_index()
)
sd_major_occurrences.index = sd_major_occurrences.index.rename("rank") + 1
sd_major_occurrences = sd_major_occurrences.reset_index()
px.scatter(sd_major_occurrences, x="rank", y="frequency", log_y=True)

# %%
px.scatter(sd_major_occurrences, x="rank", y="C")

# %%


def find_index_of_r1_r2(C: pd.Series) -> Tuple[int, int]:
    """Takes a Series representing C = 1 / (frequency(rank) - rank) and returns the indices of r1 and r2, left and
    right of the discontinuity."""
    r1_i = C.idxmax()
    r2_i = C.lt(0).idxmax()
    assert (
        r2_i == r1_i + 1
    ), f"Expected r1 and r2 to be one apart, but got r1_i = {r1_i}, r2_i = {r2_i}"
    return r1_i, r2_i


def compute_h(df) -> int | float:
    """Computes the h-point of a DataFrame with columns "rank" and "frequency" and returns the rank of the h-point.
    Returns a rank integer if a value with r = f(r) exists, otherwise rank float.
    """
    if (mask := df.frequency.eq(df["rank"])).any():
        h_ix = df.index[mask][0]
        return df.at[h_ix, "rank"]
    C = 1 / (sd_major_occurrences.frequency - sd_major_occurrences["rank"])
    r1_i, r2_i = find_index_of_r1_r2(C)
    (r1, f_r1), (r2, f_r2) = df.loc[[r1_i, r2_i], ["rank", "frequency"]].values
    return (f_r1 * r2 - f_r2 * r1) / (r2 - r1 + f_r1 - f_r2)


compute_h(sd_major_occurrences)

# %%
sd_major_occurrences.iloc[130:150]

# %%
sd_sonorities = (
    harmony_labels.groupby("scale_degrees")
    .duration_qb.sum()
    .sort_values(ascending=False)
)
pd.concat(
    [sd_sonorities, (sd_sonorities / sd_sonorities.sum()).rename("proportion")], axis=1
)

# %%
chord_proportions: Durations = harmony_labels.apply_step("Proportions")
chord_proportions.make_ranking_table(drop_cols="chord_and_mode").iloc[:50, :50]

# %%
chord_proportions.make_ranking_table(["mode"], drop_cols="chord_and_mode")

# %%
piece2profile = {
    group: profile["duration_qb"].droplevel(
        ["corpus", "piece", "mode", "chord_and_mode"]
    )
    for group, profile in chord_proportions.groupby(["corpus", "piece", "mode"])
}
piece_profiles = pd.concat(piece2profile, axis=1, names=["corpus", "piece", "mode"])
piece_profiles.sort_index(
    key=lambda _: pd.Index(piece_profiles.notna().sum(axis=1)), ascending=False
).iloc[:50, :50]

# %%
piece_profiles.sort_index(
    key=lambda _: pd.Index(piece_profiles.sum(axis=1)), ascending=False
).iloc[:50, :50]

# %%
corpus_proportions = chord_proportions.combine_results().droplevel("chord_and_mode")
corpus_proportions

# %%
corpus2profile = {
    group: profile["duration_qb"].droplevel(["corpus", "mode"])
    for group, profile in corpus_proportions.groupby(["corpus", "mode"])
}
corpus_profiles = pd.concat(corpus2profile, axis=1, names=["corpus", "mode"])
chord_occurrence_mask = corpus_profiles.notna()
corpus_frequency = chord_occurrence_mask.sum(axis=1)
chord_occurrence_mask = chord_occurrence_mask.sort_index(
    key=lambda _: pd.Index(corpus_frequency), ascending=False
)
chord_occurrence_mask = chord_occurrence_mask.sort_index(
    key=lambda _: pd.Index(chord_occurrence_mask.sum()), ascending=False, axis=1
)
corpus_profiles.sort_index(key=lambda _: pd.Index(corpus_frequency), ascending=False)

# %%
mask_with_sum = pd.concat(
    [chord_occurrence_mask, chord_occurrence_mask.sum(axis=1).rename("sum")], axis=1
)
mask_with_sum.to_csv(make_output_path("chord_occurrence_mask", "tsv"), sep="\t")

# %%
