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
# # Chord Profiles

# %% mystnb={"code_prompt_hide": "Hide imports", "code_prompt_show": "Show imports"} tags=["hide-cell"]
# %load_ext autoreload
# %autoreload 2
import os
from typing import Dict, List, Tuple

import dimcat as dc
import ms3
import pandas as pd
from dimcat import resources
from dimcat.plotting import write_image
from git import Repo
from matplotlib import pyplot as plt

import utils

plt.rcParams["figure.dpi"] = 300

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)

# %%
RESULTS_PATH = os.path.expanduser("~/git/diss/31_profiles/figs")
os.makedirs(RESULTS_PATH, exist_ok=True)


def make_output_path(filename, extension=None):
    if extension:
        extension = "." + extension.lstrip(".")
    else:
        extension = utils.DEFAULT_OUTPUT_FORMAT
    return os.path.join(RESULTS_PATH, f"{filename}{extension}")


def save_figure_as(fig, filename, directory=RESULTS_PATH, **kwargs):
    if not any(key in kwargs for key in ("height", "width")):
        kwargs["width"] = 1280
        kwargs["height"] = 720
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
D

# %%
chord_slices = utils.get_sliced_notes(D)
chord_slices.head(5)

# %%
features = {
    "root_per_globalkey": (  # baseline globalkey-roots without any note information
        ["root_per_globalkey", "intervals_over_root"],
        "chord symbols (root per globalkey + intervals)",
    ),
    "root_per_localkey": (  # baseline localkey-roots without any note information
        ["root", "intervals_over_root"],
        "chord symbols (root per localkey + intervals)",
    ),
    "root_per_tonicization": (  # baseline root over tonicized key without any note information
        ["root_per_tonicization", "intervals_over_root"],
        "chord symbols (root per tonicization + intervals)",
    ),
    "globalkey_profiles": (  # baseline notes - globalkey
        ["fifths_over_global_tonic"],
        "tone profiles as per global key",
    ),
    "localkey_profiles": (  # baseline notes - localkey
        ["fifths_over_local_tonic"],
        "tone profiles as per local key",
    ),
    "tonicization_profiles": (  # baseline notes - tonicized key
        ["fifths_over_tonicization"],
        "tone profiles as per tonicized key",
    ),
    "global_root_ct": (
        ["root_per_globalkey", "fifths_over_root"],
        "chord-tone profiles over root-per-globalkey",
    ),
    "local_root_ct": (
        ["root", "fifths_over_root"],
        "chord-tone profiles over root-per-localkey",
    ),
    "tonicization_root_ct": (
        [
            "root_per_tonicization",
            "fifths_over_root",
        ],
        "chord-tone profiles over root-per-tonicization",
    ),
}


analyzer_config = dc.DimcatConfig(
    "PrevalenceAnalyzer",
    index=["corpus", "piece"],
)


def make_data(
    chord_slices: resources.DimcatResource,
    features: Dict[str, Tuple[str | List[str], str]],
) -> Dict[str, resources.PrevalenceMatrix]:
    data = {}
    for feature_name, (feature_columns, info) in features.items():
        print(f"Computing prevalence matrix for {info}")
        analyzer_config.update(columns=feature_columns)
        prevalence_matrix = chord_slices.apply_step(analyzer_config)
        data[feature_name] = prevalence_matrix
    return data


data = make_data(chord_slices, features)

# %% [markdown]
# ## Document frequencies of chord features

# %%
info2features = {key: feature_columns for feature_columns, key in features.values()}
ranking_table = utils.compare_corpus_frequencies(chord_slices, info2features)
ranking_table.columns.rename("feature", level=0, inplace=True)
ranking_table.index = ranking_table.index.rename("rank") + 1
ranking_table.iloc[:30]

# %% [markdown] jupyter={"outputs_hidden": false}
# ## Principal Component Analyses
# ### Chord profiles

# %% jupyter={"outputs_hidden": false}
metadata = D.get_metadata()
PIECE_YEARS = metadata.get_composition_years().rename("mean_composition_year")
PIECE_MODE = metadata.annotated_key.str.islower().map({True: "minor", False: "major"})


def show_pca(feature, **kwargs):
    global data, features
    prevalence_matrix = data[feature]
    info = features[feature][1]
    return utils.plot_pca(prevalence_matrix.relative, info=info, **kwargs)


show_pca("root_per_globalkey", color=PIECE_YEARS, symbol=PIECE_MODE)

# %%
show_pca("root_per_localkey", color=PIECE_YEARS, symbol=PIECE_MODE)

# %%
show_pca("root_per_tonicization", color=PIECE_YEARS, symbol=PIECE_MODE)

# %% [markdown]
# ## Pitch-class profiles

# %%
show_pca("globalkey_profiles", color=PIECE_YEARS, symbol=PIECE_MODE)

# %%
show_pca("localkey_profiles", color=PIECE_YEARS, symbol=PIECE_MODE)

# %%
show_pca("tonicization_profiles", color=PIECE_YEARS, symbol=PIECE_MODE)

# %% [markdown]
# ### Chord-tone profiles

# %%
show_pca("global_root_ct", color=PIECE_YEARS, symbol=PIECE_MODE)

# %%
show_pca("local_root_ct", color=PIECE_YEARS, symbol=PIECE_MODE)

# %%
show_pca("tonicization_root_ct", color=PIECE_YEARS, symbol=PIECE_MODE)

# %% [raw] jupyter={"outputs_hidden": false}
# ## Create chord-tone profiles for multiple chord features
#
# Tokens are `(feature, ..., chord_tone)` tuples.

# %% [raw] jupyter={"outputs_hidden": false}
# chord_reduced: resources.PrevalenceMatrix = chord_slices.apply_step(
#     dc.DimcatConfig(
#         "PrevalenceAnalyzer",
#         columns=["chord_reduced_and_mode", "fifths_over_local_tonic"],
#         index=["corpus", "piece"],
#     )
# )
# print(f"Shape: {chord_reduced.shape}")
# utils.replace_boolean_column_level_with_mode(chord_reduced)

# %% [raw] jupyter={"outputs_hidden": false}
# numerals: resources.PrevalenceMatrix = chord_slices.apply_step(
#     dc.DimcatConfig(
#         "PrevalenceAnalyzer",
#         columns=["effective_localkey_is_minor", "numeral", "fifths_over_local_tonic"],
#         index=["corpus", "piece"],
#     )
# )
# print(f"Shape: {numerals.shape}")
# utils.replace_boolean_column_level_with_mode(numerals)

# %% [raw] jupyter={"outputs_hidden": false}
# roots: resources.PrevalenceMatrix = chord_slices.apply_step(
#     dc.DimcatConfig(
#         "PrevalenceAnalyzer",
#         columns=["root", "fifths_over_local_tonic"],
#         index=["corpus", "piece"],
#     )
# )
# print(f"Shape: {roots.shape}")

# %% [raw] jupyter={"outputs_hidden": false}
# root_per_globalkey = chord_slices.apply_step(
#     dc.DimcatConfig(
#         "PrevalenceAnalyzer",
#         columns=["root_per_globalkey", "fifths_over_local_tonic"],
#         index=["corpus", "piece"],
#     )
# )
# print(f"Shape: {root_per_globalkey.shape}")

# %% [raw] jupyter={"outputs_hidden": false}
# fig = utils.plot_document_frequency(chord_reduced)
# save_figure_as(fig, "document_frequency_of_chord_tones")
# fig

# %% [raw] jupyter={"outputs_hidden": false}
# utils.plot_document_frequency(numerals, info="numerals")

# %% [raw] jupyter={"outputs_hidden": false}
# utils.plot_document_frequency(roots, info="roots")

# %% [raw] jupyter={"outputs_hidden": false}
# utils.plot_document_frequency(
#     root_per_globalkey, info="root relative to global tonic"
# )

# %% [raw] jupyter={"outputs_hidden": false}
# ## Principal Component Analyses

# %% [raw] jupyter={"outputs_hidden": false}
# # chord_reduced.query("piece in ['op03n12a', 'op03n12b']").dropna(axis=1, how='all')

# %% [raw] jupyter={"outputs_hidden": false}
# metadata = D.get_metadata()
# CORPUS_YEARS = utils.corpus_mean_composition_years(metadata)
# PIECE_YEARS = metadata.get_composition_years().rename("mean_composition_year")
# utils.plot_pca(
#     chord_reduced.relative,
#     info="chord-tone profiles of reduced chords",
#     color=PIECE_YEARS,
# )

# %% [raw] jupyter={"outputs_hidden": false}
# utils.plot_pca(
#     chord_reduced.combine_results("corpus").relative,
#     info="chord-tone profiles of reduced chords",
#     color=CORPUS_YEARS,
#     size=5,
# )

# %% [raw] jupyter={"outputs_hidden": false}
# utils.plot_pca(
#     numerals.relative, info="numeral profiles of numerals", color=PIECE_YEARS
# )

# %% [raw] jupyter={"outputs_hidden": false}
# utils.plot_pca(
#     numerals.combine_results("corpus").relative,
#     info="chord-tone profiles of numerals",
#     color=CORPUS_YEARS,
#     size=5,
# )

# %% [raw] jupyter={"outputs_hidden": false}
# utils.plot_pca(
#     roots.relative, info="root profiles of chord roots (local)", color=PIECE_YEARS
# )

# %% [raw] jupyter={"outputs_hidden": false}
# utils.plot_pca(
#     roots.combine_results("corpus").relative,
#     info="chord-tone profiles of chord roots (local)",
#     color=CORPUS_YEARS,
#     size=5,
# )

# %% [raw] jupyter={"outputs_hidden": false}
# utils.plot_pca(
#     root_per_globalkey.relative,
#     info="root profiles of chord roots (global)",
#     color=PIECE_YEARS,
# )

# %% [raw] jupyter={"outputs_hidden": false}
# utils.plot_pca(
#     root_per_globalkey.combine_results("corpus").relative,
#     info="chord-tone profiles of chord roots (global)",
#     color=CORPUS_YEARS,
#     size=5,
# )
