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
phrase_annotations = D.get_feature("PhraseAnnotations")
phrase_annotations

# %%
overall_ranking = (
    D.get_feature("HarmonyLabels").get_default_analysis().make_ranking_table()
)
within_phrase_ranking = phrase_annotations.get_default_analysis().make_ranking_table()
pd.concat(
    [overall_ranking, within_phrase_ranking], keys=["overall", "within_phrase"], axis=1
)

# %%
bigram_table = phrase_annotations.apply_step("bigramanalyzer")
bigram_table

# %%
bgt = bigram_table.make_bigram_tuples(join_str=True, terminal_symbols="DROP")
bgt.head(50)

# %%
bgt.make_ranking_table()

# %%
phrase_components = phrase_annotations.extract_feature("PhraseComponents")
phrase_components

# %%
phrases = phrase_components.extract_feature("PhraseLabels")
phrases

# %%
vc = value_count_df(phrases.end_chord, rank_index=True)
vc.head(50)

# %%
