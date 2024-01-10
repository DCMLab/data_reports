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
from dimcat import resources
from dimcat.plotting import write_image
from git import Repo

import utils

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)

# %%
RESULTS_PATH = os.path.abspath(os.path.join(utils.OUTPUT_FOLDER, "phrases"))
os.makedirs(RESULTS_PATH, exist_ok=True)


def make_output_path(
    filename: str,
    extension=None,
    path=RESULTS_PATH,
) -> str:
    return utils.make_output_path(
        filename,
        extension=extension,
        path=path,
    )


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
phrase_annotations = D.get_feature("PhraseAnnotations")
phrase_annotations

# %%
test = phrase_annotations.query("piece == 'n01op18-1_01'").copy()
test[test.pedal.notna()]

# %%
phrase_labels = phrase_annotations.extract_feature("PhraseLabels")
phrase_labels


# %%
def add_bass_progressions(
    phrase_bodies: resources.PhraseData,
    reverse=False,
) -> resources.NgramTable:
    bgt: resources.NgramTable = phrase_bodies.apply_step("BigramAnalyzer")
    if reverse:
        bgt.loc(axis=1)["b", "bass_note"] = (
            bgt.loc(axis=1)["a", "bass_note"] - bgt.loc(axis=1)["b", "bass_note"]
        )
    else:
        bgt.loc(axis=1)["b", "bass_note"] -= bgt.loc(axis=1)["a", "bass_note"]
    new_index = pd.MultiIndex.from_tuples(
        [
            ("b", "bass_progression") if t == ("b", "bass_note") else t
            for t in bgt.columns
        ]
    )
    bgt.df.columns = new_index
    return bgt


ending_on_tonic_data = phrase_labels.get_phrase_data(
    ["bass_note", "intervals_over_bass"],
    drop_levels="phrase_component",
    reverse=True,
    query="end_chord == ['I', 'i']",
)
ending_on_tonic = add_bass_progressions(ending_on_tonic_data, reverse=True)
ending_on_tonic

# %%
sonority_progression_tuples = ending_on_tonic.make_bigram_tuples(
    "intervals_over_bass", None, terminal_symbols="DROP"
)
sonority_progression_tuples

# %%
sonority_progression_tuples.query("i == 0").n_gram.value_counts()

# %%
ending_on_tonic = phrase_labels.get_phrase_data(
    ["chord"],
    drop_levels="phrase_component",
    reverse=True,
    query="end_chord == ['I', 'i']",
)
ending_on_tonic.query("i == 1").chord.value_counts()

# %%
bodies_reversed = phrase_labels.get_phrase_data(
    ["chord", "numeral"],
    drop_levels="phrase_component",
    reverse=True,
)
bgt = bodies_reversed.apply_step("BigramAnalyzer")
bgt

# %%
bgt.query("@bgt.a.numeral == 'V'").b.chord.value_counts()

# %%
phrase_bodies = phrase_annotations.get_phrase_data(
    ["bass_note", "intervals_over_bass"], drop_levels="phrase_component"
)
bgt = add_bass_progressions(phrase_bodies)
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
chord_type_transitions.sort_values("count", ascending=False).iloc[:100]

# %%
bgt.make_bigram_tuples(terminal_symbols="DROP").make_ranking_table()
