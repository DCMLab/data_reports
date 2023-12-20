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
# %load_ext autoreload
# %autoreload 2
import os
from typing import Optional

import dimcat as dc
import ms3
import pandas as pd
from dimcat.plotting import write_image
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
phrases = D.get_feature("PhraseLabels")
phrases

# %%
vc = value_count_df(phrases.end_chord, rank_index=True)
vc.head(50)


# %%
def make_and_store_phrase_data(
        name: Optional[str] = None,
        columns = "chord",
        components = "body",
        droplevels = 3,
        reverse = True,
        new_level_name= "stage",
        wide_format = True,
        query = None,
):
    """Function sets the defaults for the stage TSVs produced in the following."""
    phrase_data = phrases.filter_phrase_data(
        columns=columns,
        components=components,
        droplevels=droplevels,
        reverse=reverse,
        new_level_name=new_level_name,
        wide_format=wide_format,
        query=query
    )
    if name:
        phrase_data.to_csv(make_output_path(name, "tsv"), sep='\t')
    return phrase_data

stages = make_and_store_phrase_data("stages", columns=["localkey", "chord"])
stages.head()

# %%
onekey_major = make_and_store_phrase_data("onekey_major", query="body_n_modulations == 0 & localkey_mode == 'major'")
onekey_minor = make_and_store_phrase_data("onekey_minor", query="body_n_modulations == 0 & localkey_mode == 'minor'")
one_key_major_I = make_and_store_phrase_data("onekey_major_I", query="body_n_modulations == 0 & localkey_mode == 'major' & end_chord == 'I'") # end_chord.str.contains('^I(?![iIvV\/])')")
one_key_minor_i = make_and_store_phrase_data("one_key_minor_i", query="body_n_modulations == 0 & localkey_mode == 'minor' & end_chord == 'i'")
