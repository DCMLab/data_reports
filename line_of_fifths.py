# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md:myst,py:percent
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
# ## Line of Fifth plots
#
# Notebook adapted from the one used for the presentation at HCI 2023 in Copenhagen

# %%
# %load_ext autoreload
# %autoreload 2
import os

import dimcat as dc
import ms3
from dimcat import groupers, resources
from git import Repo

# %%
from utils import (
    DEFAULT_OUTPUT_FORMAT,
    OUTPUT_FOLDER,
    get_repo_name,
    print_heading,
    resolve_dir,
)

RESULTS_PATH = os.path.abspath(os.path.join(OUTPUT_FOLDER, "line_of_fifths"))
os.makedirs(RESULTS_PATH, exist_ok=True)


def make_output_path(filename):
    return os.path.join(RESULTS_PATH, f"{filename}{DEFAULT_OUTPUT_FORMAT}")


# %% [markdown]
# **Loading data**

# %%
package_path = resolve_dir(
    "~/distant_listening_corpus/couperin_concerts/couperin_concerts.datapackage.json"
)
repo = Repo(os.path.dirname(package_path))
print_heading("Data and software versions")
print(f"Data repo '{get_repo_name(repo)}' @ {repo.commit().hexsha[:7]}")
print(f"dimcat version {dc.__version__}")
print(f"ms3 version {ms3.__version__}")
D = dc.Dataset.from_package(package_path)
D

# %% [markdown]
# ## Pitch class distribution for the beginning of La Mer

# %%
la_mer_notes = resources.Notes.from_descriptor_path("La_Mer_1-84.notes.resource.json")
la_mer_notes.load()
la_mer_notes

# %%
la_mer_notes.plot_grouped(
    title="Pitch-class distribution in Claude Debussy's 'La Mer' (mm. 1-84)",
    output=make_output_path(
        "debussy_la_mer_beginning_pitch_class_distribution_bars",
    ),
    height=800,
)

# %%
grouped_la_mer = groupers.MeasureGrouper().process(la_mer_notes)
grouped_la_mer.plot_grouped(
    title="Normalized measure-wise pitch-class distribution in 'La Mer' (mm. 1-84)",
    layout=dict(yaxis_type="linear"),
    output=make_output_path(
        "debussy_la_mer_beginning_barwise_pitch_class_distributions_bubbles"
    ),
    width=1200,
)

# %% [markdown]
# ## Pitch class distributions for the datapackage

# %%
notes = D.get_feature("notes")
notes.plot_grouped(
    title=f"Pitch-class distribution for {notes.resource_name.split('.')[0]}",
    output=make_output_path("complete_pitch_class_distribution_absolute_bars"),
    height=800,
)

# %%
grouped_D = groupers.YearGrouper().process(D)
grouped_notes = grouped_D.get_feature("notes")
grouped_notes

# %%
grouped_notes.plot(
    output=make_output_path("all_pitch_class_distributions_piecewise_bubbles"),
    title=f"Normalized piece-wise pitch-class distributions for {grouped_notes.resource_name.split('.')[0]}",
    width=1200,
)

# %%
grouped_notes.plot_grouped(
    output=make_output_path("all_pitch_class_distributions_yearwise_bubbles"),
    title=f"Normalized year-wise pitch-class distributions for {grouped_notes.resource_name.split('.')[0]}",
    width=1200,
)
