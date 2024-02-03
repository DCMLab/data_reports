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
# # Chopin Profiles
#
# Motivation: Chopin's dominant is often attributed a special characteristic due to the characteristic 13

# %% mystnb={"code_prompt_hide": "Hide imports", "code_prompt_show": "Show imports"} tags=["hide-cell"]
# %load_ext autoreload
# %autoreload 2

import math
import os
import re

import dimcat as dc
import ms3
import numpy as np
import pandas as pd
from dimcat import resources
from dimcat.data.resources.results import compute_entropy_of_occurrences
from dimcat.plotting import make_bar_plot, write_image
from git import Repo
from IPython.display import display
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
harmony_labels = D.get_feature("harmonylabels")
print(f"{harmony_labels.index.droplevel(-1).nunique()} annotated pieces")
harmony_labels.query(
    "changes.str.contains('13') & corpus != 'bartok_bagatelles'"
).chord.value_counts().sort_values()

# %% [markdown]
# ## VI43(13) example
#
# The only occurrence in its context:

# %%
harmony_labels.loc(axis=0)["medtner_tales", "op35n02", 325:332]

# %% [markdown]
# ### Without the 13: 3 pieces

# %%
harmony_labels[harmony_labels.chord == "VI43"]

# %% [markdown]
# ### Without inversion: 96 pieces

# %%
VI7_chords = harmony_labels[
    (harmony_labels.intervals_over_root == ("M3", "P5", "m7"))
    & harmony_labels.root.eq(-4)
]
VI7_chords

# %%
print(
    f"-4, (M3, M5, m7) occurs in {VI7_chords.index.droplevel(-1).nunique()} difference pieces, "
    f"often as dominant of neapolitan"
)


# %% [markdown]
# ## Reduction of vocabulary size


# %%
def normalized_entropy_of_prevalence(value_counts):
    return compute_entropy_of_occurrences(value_counts) / math.log2(len(value_counts))


chord_and_mode_prevalence = harmony_labels.groupby("chord_and_mode").duration_qb.agg(
    ["sum", "count"]
)
print(
    f"Chord + mode: n = {len(chord_and_mode_prevalence)}, h = \n"
    f"{compute_entropy_of_occurrences(chord_and_mode_prevalence)}"
)

# %%
type_inversion_change = harmony_labels.groupby(
    ["chord_type", "figbass", "changes"], dropna=False
).duration_qb.agg(["sum", "count"])
print(
    f"Chord type + inversion + change: n = {len(type_inversion_change)}, h = \n"
    f"{compute_entropy_of_occurrences(type_inversion_change)}"
)

# %% [markdown]
# Negligible difference between the two different ways of calculating, probably due to an inconsistent indication of
# changes. But the second one is the one that also allows filtering out the changes >= 8


# %%
def show_stats(groupby, info, k=5):
    prevalence = groupby.duration_qb.agg(["sum", "count"])
    entropies = compute_entropy_of_occurrences(prevalence).rename("entropy")
    entropies_norm = normalized_entropy_of_prevalence(prevalence).rename(
        "normalized entropy"
    )
    ent = pd.concat([entropies, entropies_norm], axis=1)
    print(f"{info}: n = {len(prevalence)}, h = \n{ent}")
    n_pieces_per_token = groupby.apply(
        lambda df: len(df.groupby(["corpus", "piece"])), include_groups=False
    )
    print("Token that appears in the highest number of pieces:")
    display(n_pieces_per_token.iloc[[n_pieces_per_token.argmax()]])
    n_pieces_per_token_vc = n_pieces_per_token.value_counts().sort_index()
    n_pieces_per_token_vc.index.rename("occuring in # pieces", inplace=True)
    n_pieces_per_token_vc = pd.concat(
        [
            n_pieces_per_token_vc.rename("tokens"),
            n_pieces_per_token_vc.rename("proportion") / n_pieces_per_token_vc.sum(),
        ],
        axis=1,
    )
    selection = n_pieces_per_token_vc.iloc[np.r_[0:k, -k:0]]
    print(
        f"\nTokens occurring in only {k} or fewer pieces: {selection.tokens.sum()} ({selection.proportion.sum():.1%})"
    )
    display(selection)
    print(
        "Quantiles indicating fractions of all tokens which occur in # or less pieces"
    )
    display(
        n_pieces_per_token.quantile([0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
    )


type_inversion_change = pd.concat(
    [
        harmony_labels[["duration_qb", "chord_type", "figbass"]],
        harmony_labels.changes.map(
            lambda ch: tuple(
                sorted(
                    (ch_tup[0] for ch_tup in ms3.changes2list(ch)),
                    key=lambda s: int(re.search(r"\d+$", s).group()),
                )
            ),
            na_action="ignore",
        ),
    ],
    axis=1,
)

show_stats(
    type_inversion_change.groupby(["chord_type", "figbass", "changes"], dropna=False),
    "Chord type + inversion + change",
)

# %%
change_max_7 = harmony_labels.changes.map(
    lambda ch: tuple(
        sorted(
            (ch_tup[0] for ch_tup in ms3.changes2list(ch) if int(ch_tup[-1]) < 8),
            key=lambda s: int(re.search(r"\d+$", s).group()),
        )
    ),
    na_action="ignore",
)
typ_inv_change_max_7 = pd.concat(
    [harmony_labels[["duration_qb", "chord_type", "figbass"]], change_max_7], axis=1
)
show_stats(
    typ_inv_change_max_7.groupby(["chord_type", "figbass", "changes"], dropna=False),
    "Chord type + inversion + changes < 8",
)

# %%
typ_change_max_7 = pd.concat(
    [harmony_labels[["duration_qb", "chord_type"]], change_max_7], axis=1
)
show_stats(
    typ_change_max_7.groupby(["chord_type", "changes"], dropna=False),
    "Chord type + changes < 8",
)

# %%
show_stats(
    harmony_labels.groupby(["intervals_over_root", "figbass"], dropna=False),
    "intervals over root + inversion",
)

# %%
ior_inversion_doc_freqs = harmony_labels.groupby(
    ["intervals_over_root", "figbass"], dropna=False
).apply(lambda df: len(df.groupby(["corpus", "piece"])), include_groups=False)
ior_inversion_doc_freqs.sort_values(ascending=False).iloc[:10]

# %%
show_stats(harmony_labels.groupby("intervals_over_root"), "intervals over root")

# %%
root_per_localkey: resources.PrevalenceMatrix = harmony_labels.apply_step(
    dict(
        dtype="prevalenceanalyzer",
        index=["corpus", "piece"],
        columns=["root", "intervals_over_root"],
    )
)
root_per_localkey.document_frequencies()

# %%
ior_doc_freqs = harmony_labels.groupby(["intervals_over_root"], dropna=False).apply(
    lambda df: len(df.groupby(["corpus", "piece"])), include_groups=False
)
ior_doc_freqs.sort_values(ascending=False).iloc[:10]

# %% [markdown]
# ### counter-comparison: chord-type + inversion

# %%
show_stats(
    harmony_labels.groupby(["chord_type", "figbass"], dropna=False),
    "Chord type + inversion",
)

# %%
typ_inv_doc_freqs = harmony_labels.groupby(
    ["chord_type", "figbass"], dropna=False
).apply(lambda df: len(df.groupby(["corpus", "piece"])), include_groups=False)
typ_inv_doc_freqs.sort_values(ascending=False).iloc[:10]

# %% [markdown]
# ### Difference between `intervals_over_root` and `chord_type + changes <8`

# %%
pd.concat([typ_change_max_7, harmony_labels.intervals_over_root], axis=1).groupby(
    "intervals_over_root"
)[["chord_type", "changes"]].value_counts()

# %% [raw]
# harmony_labels = D.get_feature("harmonylabels")
# harmony_labels.head()
# raw_labels = harmony_labels.numeral.str.upper() + harmony_labels.figbass.fillna('')
# ll = raw_labels.to_list()
# from suffix_tree import Tree
# sfx_tree = Tree({"dlc": ll})
# # query = ["I", "I6", "VII6", "II6", "I", "V7"]
# query = ["VII6", "II6", "I"]
# sfx_tree.find_all(query)

# %%
chord_slices = utils.get_sliced_notes(D)
chord_slices.head(5)

# %% [markdown]
# ## 3 root entropies

# %%
analyzer_config = dc.DimcatConfig(
    "PrevalenceAnalyzer",
    index=["corpus", "piece"],
)
roots_only = {}
for root_type in ("root_per_globalkey", "root", "root_per_tonicization"):
    analyzer_config.update(columns=root_type)
    roots_only[root_type] = chord_slices.apply_step(analyzer_config)

# %%
for root_type, prevalence_matrix in roots_only.items():
    print(root_type)
    occurring_roots = sorted(prevalence_matrix.columns.map(int))
    print(occurring_roots)
    ent = normalized_entropy_of_prevalence(prevalence_matrix.absolute.sum())
    print(ent)


# %% [markdown]
# **First intuition: Compare `V7` chord profiles**

# %%
chord_tone_profiles = utils.make_chord_tone_profile(chord_slices)
chord_tone_profiles.head()

# %%
utils.plot_chord_profiles(chord_tone_profiles, "V7, major")

# %% [markdown]
# **It turns out that the scale degree in question (3) is more frequent in `V7` chords in `bach_solo` and
# `peri_euridice` than in Chopin's Mazurkas. We might suspect that the Chopin chord is not included because it is
# highlighted as a different label, 7e.g. `V7(13)`.**

# %%
utils.plot_chord_profiles(chord_tone_profiles, "V7(13), major")

# %% [markdown]
# **From here, it is interesting to ask, either, if these special labels show up more frequently in Chopin's corpus
# than in others, and if 3 shows up prominently in Chopin's dominants if we combine all dominant chord profiles with
# each other.**

# %%
all_V7 = harmony_labels.query("numeral == 'V' & figbass == '7'")
all_V7.head()

# %%
all_V7["tonicization_chord"] = all_V7.chord.str.split("/").str[0]

# %%
all_V7_absolute = all_V7.groupby(["corpus", "tonicization_chord"]).duration_qb.agg(
    ["sum", "size"]
)
all_V7_absolute.columns = ["duration_qb", "count"]
all_V7_absolute

# %%
all_V7_relative = all_V7_absolute / all_V7_absolute.groupby("corpus").sum()
make_bar_plot(
    all_V7_relative.reset_index(),
    x_col="tonicization_chord",
    y_col="duration_qb",
    color="corpus",
    log_y=True,
)

# %%
all_V7_relative.loc["chopin_mazurkas"].sort_values("count", ascending=False) * 100

# %% [markdown]
# **This is not a good way of comparing dominant chords. We could now start summing up all the different chords we
# consider to be part of the "Chopin chord" category. Chord-tone-profiles are probably the better way to see.**

# %% [markdown]
# ## Create chord-tone profiles for multiple chord features
#
# Tokens are `(feature, ..., chord_tone)` tuples.

# %%
tonicization_profiles: resources.PrevalenceMatrix = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["root_per_tonicization", "fifths_over_tonicization"],
        index="corpus",
    )
)
tonicization_profiles._df.columns = ms3.map2elements(
    tonicization_profiles.columns, int
).set_names(["root_per_tonicization", "fifths_over_tonicization"])

# %%
tonicization_profiles.head()
dominant_ct = tonicization_profiles.loc(axis=1)[[1]].stack()
dominant_ct.columns = ["duration_qb"]
dominant_ct["proportion"] = dominant_ct["duration_qb"] / dominant_ct.groupby(
    "corpus"
).duration_qb.agg("sum")
dominant_ct.head()

# %%
fig = make_bar_plot(
    dominant_ct.reset_index(),
    x_col="fifths_over_tonicization",
    y_col="proportion",
    facet_row="corpus",
    facet_row_spacing=0.001,
    height=10000,
    y_axis=dict(matches=None),
)
fig

# %% [markdown]
# **Chopin does not show a specifically high bar for 4 (the major third of the scale), Mozart's is higher, for example.
# This could have many reasons, e.g. that the pieces are mostly in minor, or that the importance of scale degree 3 as
# lower neighbor to the dominant seventh is statistically more important, or that the "characteristic 13" is not
# actually important duration-wise.**

# %%
tonic_thirds_in_dominants = (
    dominant_ct.loc[(slice(None), [4, -3]), "proportion"].groupby("corpus").sum()
)

# %%
make_bar_plot(
    tonic_thirds_in_dominants,
    x_col="corpus",
    y_col="proportion",
    category_orders=dict(corpus=D.get_metadata().get_corpus_names(func=None)),
    title="Proportion of scale degree 3 in dominant chords, chronological order",
)

# %% [markdown]
# **No chronological trend visible.**
