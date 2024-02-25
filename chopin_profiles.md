---
jupytext:
  formats: md:myst,ipynb,py:percent
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: revamp
  language: python
  name: revamp
---

# Chopin Profiles

Motivation: Chopin's dominant is often attributed a special characteristic due to the characteristic 13

```{code-cell} ipython3
---
mystnb:
  code_prompt_hide: Hide imports
  code_prompt_show: Show imports
tags: [hide-cell]
---
%load_ext autoreload
%autoreload 2

import math
import os
import re
from typing import Dict, Literal

import dimcat as dc
import ms3
import numpy as np
import pandas as pd
from dimcat import resources
from dimcat.data.resources.results import compute_entropy_of_occurrences
from dimcat.plotting import (
    make_bar_plot,
    make_line_plot,
    make_scatter_plot,
    write_image,
)
from git import Repo
from IPython.display import display

import utils

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
```

```{code-cell} ipython3
RESULTS_PATH = os.path.expanduser("~/git/diss/32_profiles/figs")
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
```

```{code-cell} ipython3
:tags: [hide-input]

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
```

```{code-cell} ipython3
harmony_labels = D.get_feature("harmonylabels")
print(f"{harmony_labels.index.droplevel(-1).nunique()} annotated pieces")
harmony_labels.query(
    "changes.str.contains('13') & corpus != 'bartok_bagatelles'"
).chord.value_counts().sort_values()
```

## VI43(13) example

The only occurrence in its context:

```{code-cell} ipython3
harmony_labels.loc(axis=0)["medtner_tales", "op35n02", 325:332]
```

### Without the 13: 3 pieces

```{code-cell} ipython3
harmony_labels[harmony_labels.chord == "VI43"]
```

### Without inversion: 96 pieces

```{code-cell} ipython3
VI7_chords = harmony_labels[
    (harmony_labels.intervals_over_root == ("M3", "P5", "m7"))
    & harmony_labels.root.eq(-4)
]
VI7_chords
```

```{code-cell} ipython3
print(
    f"-4, (M3, M5, m7) occurs in {VI7_chords.index.droplevel(-1).nunique()} difference pieces, "
    f"often as dominant of neapolitan"
)
```

## Reduction of vocabulary size

```{code-cell} ipython3
def normalized_entropy_of_prevalence(value_counts):
    return compute_entropy_of_occurrences(value_counts) / math.log2(len(value_counts))


chord_and_mode_prevalence = harmony_labels.groupby("chord_and_mode").duration_qb.agg(
    ["sum", "count"]
)
print(
    f"Chord + mode: n = {len(chord_and_mode_prevalence)}, h = \n"
    f"{compute_entropy_of_occurrences(chord_and_mode_prevalence)}"
)
```

```{code-cell} ipython3
type_inversion_change = harmony_labels.groupby(
    ["chord_type", "figbass", "changes"], dropna=False
).duration_qb.agg(["sum", "count"])
print(
    f"Chord type + inversion + change: n = {len(type_inversion_change)}, h = \n"
    f"{compute_entropy_of_occurrences(type_inversion_change)}"
)
```

Negligible difference between the two different ways of calculating, probably due to an inconsistent indication of
changes. But the second one is the one that also allows filtering out the changes >= 8

```{code-cell} ipython3
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
```

```{code-cell} ipython3
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
```

```{code-cell} ipython3
typ_change_max_7 = pd.concat(
    [harmony_labels[["duration_qb", "chord_type"]], change_max_7], axis=1
)
show_stats(
    typ_change_max_7.groupby(["chord_type", "changes"], dropna=False),
    "Chord type + changes < 8",
)
```

```{code-cell} ipython3
show_stats(
    harmony_labels.groupby(["intervals_over_root", "figbass"], dropna=False),
    "intervals over root + inversion",
)
```

```{code-cell} ipython3
ior_inversion_doc_freqs = harmony_labels.groupby(
    ["intervals_over_root", "figbass"], dropna=False
).apply(lambda df: len(df.groupby(["corpus", "piece"])), include_groups=False)
ior_inversion_doc_freqs.sort_values(ascending=False).iloc[:10]
```

```{code-cell} ipython3
show_stats(harmony_labels.groupby("intervals_over_root"), "intervals over root")
```

```{code-cell} ipython3
def show_which_pieces_dont_include(harmony_labels, columns, value):
    for gr, df in harmony_labels.groupby(["corpus", "piece"]):
        tokens = set(df[columns].itertuples(index=False, name=None))
        if value in tokens:
            continue
        print(gr)


show_which_pieces_dont_include(
    harmony_labels, ["root", "intervals_over_root"], (1, ("M3", "P5"))
)
```

```{code-cell} ipython3
ior_doc_freqs = harmony_labels.groupby(["intervals_over_root"], dropna=False).apply(
    lambda df: len(df.groupby(["corpus", "piece"])), include_groups=False
)
ior_doc_freqs.sort_values(ascending=False).iloc[:10]
```

### counter-comparison: chord-type + inversion

```{code-cell} ipython3
show_stats(
    harmony_labels.groupby(["chord_type", "figbass"], dropna=False),
    "Chord type + inversion",
)
```

```{code-cell} ipython3
typ_inv_doc_freqs = harmony_labels.groupby(
    ["chord_type", "figbass"], dropna=False
).apply(lambda df: len(df.groupby(["corpus", "piece"])), include_groups=False)
typ_inv_doc_freqs.sort_values(ascending=False).iloc[:10]
```

### Difference between `intervals_over_root` and `chord_type + changes <8`

```{code-cell} ipython3
pd.concat([typ_change_max_7, harmony_labels.intervals_over_root], axis=1).groupby(
    "intervals_over_root"
)[["chord_type", "changes"]].value_counts()
```

```{raw-cell}
harmony_labels = D.get_feature("harmonylabels")
harmony_labels.head()
raw_labels = harmony_labels.numeral.str.upper() + harmony_labels.figbass.fillna('')
ll = raw_labels.to_list()
from suffix_tree import Tree
sfx_tree = Tree({"dlc": ll})
# query = ["I", "I6", "VII6", "II6", "I", "V7"]
query = ["VII6", "II6", "I"]
sfx_tree.find_all(query)
```

```{code-cell} ipython3
chord_slices = utils.get_sliced_notes(D)
chord_slices.head(5)
```

## 3 root entropies

```{code-cell} ipython3
analyzer_config = dc.DimcatConfig(
    "PrevalenceAnalyzer",
    index=["corpus", "piece"],
)
roots_only = {}
for root_type in ("root_per_globalkey", "root", "root_per_tonicization"):
    analyzer_config.update(columns=root_type)
    roots_only[root_type] = chord_slices.apply_step(analyzer_config)
```

```{code-cell} ipython3
root_prevalences = {}
for root_type, prevalence_matrix in roots_only.items():
    print(root_type)
    occurring_roots = sorted(prevalence_matrix.columns.map(int))
    print(occurring_roots, f"({len(occurring_roots)})")
    ent = normalized_entropy_of_prevalence(prevalence_matrix.absolute.sum())
    print(ent)
    length_of_range = max(occurring_roots) - min(occurring_roots) + 1
    n_values = len(occurring_roots)
    prevalence = prevalence_matrix.type_prevalence().rename("duration in ♩")
    if length_of_range != n_values:
        missing = set(range(min(occurring_roots), max(occurring_roots) + 1)) - set(
            occurring_roots
        )
        prevalence = pd.concat([prevalence, pd.Series(pd.NA, index=missing)]).rename(
            "duration in ♩"
        )
    prevalence = pd.concat(
        [prevalence, prevalence.rename("proportion") / prevalence.sum()], axis=1
    )
    if root_type == "root_per_globalkey":
        key = "R<sup>G</sup>"
    elif root_type == "root":
        key = "R<sup>L</sup>"
    else:
        key = "R<sup>T</sup>"
    root_prevalences[key] = prevalence
```

```{code-cell} ipython3
root_prev_data = pd.concat(
    root_prevalences, names=["type of root", "root"]
).reset_index()
root_prev_data.root = root_prev_data.root.astype(int)
root_prev_data = root_prev_data.sort_values("root")
fig = make_line_plot(
    root_prev_data,
    x_col="root",
    y_col="duration in ♩",
    hover_data=["proportion"],
    color="type of root",
    markers=True,
    category_orders={
        "type of root": ["R<sup>G</sup>", "R<sup>L</sup>", "R<sup>T</sup>"]
    },
    x_axis=dict(dtick=1, zerolinecolor="lightgrey"),
    log_y=True,
)
save_figure_as(fig, "root_prevalences.pdf", width=1280, height=600)
fig
```

Probably mistakes:

- `-11` from `bVI` in minor
  ```
    for c, p, m in (
            harmony_labels
                    .query("numeral == 'bVI' & localkey_is_minor")
                    .reset_index()[["corpus", "piece", "mn"]]
                    .itertuples(index=False, name=None)):
        print(f"- {c}, {p}, m. {m}")
  ```
    - ABC, n04op18-4_01, m. 106
    - ABC, n04op18-4_01, m. 127
    - ABC, n07op59-1_02, m. 365
    - ABC, n09op59-3_04, m. 105
    - ABC, n10op74_02, m. 113
    - ABC, n10op74_02, m. 113
    - bartok_bagatelles, op06n06, m. 12
    - beethoven_piano_sonatas, 03-1, m. 218
    - beethoven_piano_sonatas, 07-1, m. 63
    - beethoven_piano_sonatas, 07-1, m. 244
    - beethoven_piano_sonatas, 32-2, m. 92
    - chopin_mazurkas, BI157-1op59-1, m. 4
    - chopin_mazurkas, BI157-1op59-1, m. 28
    - chopin_mazurkas, BI157-1op59-1, m. 82
    - chopin_mazurkas, BI157-1op59-1, m. 106
    - grieg_lyric_pieces, op38n01, m. 53
    - grieg_lyric_pieces, op38n01, m. 57
    - mahler_kindertotenlieder, kindertotenlieder_03_wenn_dein_mutterlein, m. 25
    - mahler_kindertotenlieder, kindertotenlieder_03_wenn_dein_mutterlein, m. 56
    - medtner_tales, op35n02, m. 113
    - medtner_tales, op48n01, m. 326
    - mendelssohn_quartets, 02op13d, m. 176
    - rachmaninoff_piano, op42_03, m. 14
    - rachmaninoff_piano, op42_07, m. 4
    - rachmaninoff_piano, op42_07, m. 4
    - rachmaninoff_piano, op42_07, m. 6
    - rachmaninoff_piano, op42_07, m. 6
    - rachmaninoff_piano, op42_07, m. 11

```{code-cell} ipython3
ctp = {}
for root_type in ("root_per_globalkey", "root", "root_per_tonicization"):
    analyzer_config.update(columns=[root_type, "fifths_over_root"])
    prevalence_matrix: resources.PrevalenceMatrix = chord_slices.apply_step(
        analyzer_config
    )
    ctp[root_type] = prevalence_matrix
    print(
        f"{root_type}: n = {prevalence_matrix.n_types}, "
        f"normalized entropy = {normalized_entropy_of_prevalence(prevalence_matrix.type_prevalence())}"
    )
    display(prevalence_matrix.document_frequencies().iloc[:5])
```

```{code-cell} ipython3
show_which_pieces_dont_include(
    chord_slices, ["root_per_tonicization", "fifths_over_root"], ("0", "1")
)
```

```{code-cell} ipython3
strange = chord_slices.loc(axis=0)[
    "kleine_geistliche_konzerte", "op09n05swv310_Ich_liege_und_schlafe"
]
strange.query("root_per_globalkey == '0' & fifths_over_root == '1'")
```

```{code-cell} ipython3
strange.query("numeral in ('I', 'i')")
```

## Intervals over root

```{code-cell} ipython3
harmony_labels.intervals_over_root.map(len).value_counts()
```

```{code-cell} ipython3
ior_prevalence: resources.PrevalenceMatrix = harmony_labels.apply_step(
    dict(
        dtype="prevalenceanalyzer",
        index=["corpus", "piece"],
        columns=["intervals_over_root"],
    )
)
ior_types = ior_prevalence.type_prevalence().rename("duration in ♩")
ior_types = pd.concat(
    [
        ior_types,
        ior_types.rename("proportion") / ior_types.sum(),
        ior_prevalence.document_frequencies().rename("piece frequency (n = 1219)"),
    ],
    axis=1,
)
ior_types.index.rename("sonority", inplace=True)
ior_types = ior_types.reset_index().sort_values(
    "piece frequency (n = 1219)", ascending=False
)
ior_types.iloc[:20].to_clipboard()
ior_types.head(20).style.format(
    {
        "duration in ♩": "{:.1f}",
        "proportion": "{:.1%}",
    }
)
```

```{code-cell} ipython3
ior_prevalence.n_types
```

```{code-cell} ipython3
ior_doc_freq = ior_prevalence.document_frequencies()
five_or_less = len(ior_doc_freq[ior_doc_freq < 6])
print(
    f"{five_or_less} ({five_or_less/ior_prevalence.n_types:.1%}) occur in 5 or less pieces."
)
only_one = len(ior_doc_freq[ior_doc_freq == 1])
print(
    f"{only_one} ({only_one/ior_prevalence.n_types:.1%}) occur only in a single piece."
)
```

### Chord profile stats

```{code-cell} ipython3
chord_profiles = {}
for root_type in ("root_per_globalkey", "root", "root_per_tonicization"):
    analyzer_config.update(columns=[root_type, "intervals_over_root"])
    cps: resources.PrevalenceMatrix = chord_slices.apply_step(analyzer_config)
    chord_profiles[root_type] = cps
    print(
        f"{root_type}: n = {cps.n_types}, "
        f"normalized entropy = {normalized_entropy_of_prevalence(cps.type_prevalence())}"
    )
```

```{code-cell} ipython3
def get_vocabulary_name(profile_type, root_type):
    if root_type == "root_per_globalkey":
        key = f"V<sup>G &#215; {profile_type}</sup>"
    elif root_type == "root":
        key = f"V<sup>L &#215; {profile_type}</sup>"
    else:
        key = f"V<sup>T &#215; {profile_type}</sup>"
    return key


def make_cumulative_profile_stats(
    chord_profiles: Dict[str, resources.PrevalenceMatrix],
    profile_type: Literal["son", "ct"],
    legend_title="type of chord profile",
):
    cps_frequencies = {}
    for root_type, cps in chord_profiles.items():
        type_prev = cps.type_prevalence()
        cumul_abs = type_prev.sort_values().cumsum()
        cumul_rel = cumul_abs / type_prev.sum()
        freqs = pd.concat(
            [
                type_prev.rename("duration in ♩"),
                cumul_rel.rename("cumulative proportion"),
                cps.document_frequencies(name="piece frequency"),
            ],
            axis=1,
        )
        key = get_vocabulary_name(profile_type, root_type)
        cps_frequencies[key] = freqs
    cps_frequencies = pd.concat(
        cps_frequencies, names=[legend_title, "root", "sonority"]
    )
    return cps_frequencies


cps_frequencies = make_cumulative_profile_stats(chord_profiles, "son")
cps_frequencies.head()
```

```{code-cell} ipython3
make_line_plot(
    cps_frequencies,
    x_col="cumulative proportion",
    y_col="piece frequency",
    color="type of chord profile",
    hover_data=["root", "sonority", "duration in ♩"],
    category_orders={
        "type of chord profile": [
            "V<sup>G &#215; son</sup>",
            "V<sup>L &#215; son</sup>",
            "V<sup>T &#215; son</sup>",
        ]
    },
    markers=True,
    log_x=True,
    x_axis=dict(autorange="reversed"),
    title="Document frequency of chord-profile tokens over their cumulative proportion on a log scale",
)
```

```{code-cell} ipython3
def plot_piece_frequency(
    chord_profiles: Dict[str, resources.PrevalenceMatrix],
    profile_type: Literal["son", "ct"],
):
    cps_doc_freqs = {}
    keys = []
    for root_type, cps in chord_profiles.items():
        doc_freq = cps.document_frequencies(name="piece frequency")
        doc_freq.index.set_names(["root", "sonority"], inplace=True)
        doc_freq = doc_freq.reset_index()
        doc_freq.index = doc_freq.index.rename("rank") + 1
        key = get_vocabulary_name(profile_type, root_type)
        keys.append(key)
        cps_doc_freqs[key] = doc_freq
    cps_doc_freqs = pd.concat(
        cps_doc_freqs, names=["type of chord profile"]
    ).reset_index()
    return make_scatter_plot(
        cps_doc_freqs,
        x_col="rank",
        y_col="piece frequency",
        color="type of chord profile",
        hover_data=cps_doc_freqs.columns.to_list(),
        category_orders={"type of chord profile": keys},
        log_x=True,
        layout=dict(legend=dict(orientation="h", y=1.15)),
        # title="Document frequency of chord-profile tokens over their rank on a log scale"
    )


fig = plot_piece_frequency(chord_profiles, "son")
save_figure_as(fig, "chord_profile_tokens.pdf", width=1280, height=500)
fig
```

## Chord-tone profile stats

```{code-cell} ipython3
for_prevalence: resources.PrevalenceMatrix = chord_slices.apply_step(
    dict(
        dtype="prevalenceanalyzer",
        index=["corpus", "piece"],
        columns=["fifths_over_root"],
    )
)
for_types = for_prevalence.type_prevalence().rename("duration in ♩")
for_types = pd.concat(
    [
        for_types,
        for_types.rename("proportion") / for_types.sum(),
        for_prevalence.document_frequencies().rename("piece frequency (n = 1219)"),
    ],
    axis=1,
)
for_types.index.rename("interval over root", inplace=True)
for_types = for_types.reset_index().sort_values(
    "piece frequency (n = 1219)", ascending=False
)
for_types.loc(axis=1)["interval over root"] = (
    for_types["interval over root"]
    + " ("
    + ms3.transform(for_types["interval over root"], ms3.fifths2iv)
    + ")"
)
for_types.iloc[:20].to_clipboard()
for_types.head(20).style.format(
    {
        "duration in ♩": "{:.1f}",
        "proportion": "{:.1%}",
    }
)
```

```{code-cell} ipython3
for_prevalence.n_types
```

```{code-cell} ipython3
for_doc_freq = for_prevalence.document_frequencies()
five_or_less = len(for_doc_freq[for_doc_freq < 6])
print(
    f"{five_or_less} ({five_or_less/for_prevalence.n_types:.1%}) occur in 5 or less pieces."
)
only_one = len(for_doc_freq[for_doc_freq == 1])
print(
    f"{only_one} ({only_one/for_prevalence.n_types:.1%}) occur only in a single piece."
)
```

```{code-cell} ipython3
fig = plot_piece_frequency(ctp, "ct")
save_figure_as(fig, "chord_tone_profile_tokens.pdf", width=1280, height=500)
fig
```

## Chopin's dominant

**First intuition: Compare `V7` chord profiles**

```{code-cell} ipython3
chord_tone_profiles = utils.make_chord_tone_profile(chord_slices)
chord_tone_profiles.head()
```

```{code-cell} ipython3
utils.plot_chord_profiles(chord_tone_profiles, "V7, major")
```

**It turns out that the scale degree in question (3) is more frequent in `V7` chords in `bach_solo` and
`peri_euridice` than in Chopin's Mazurkas. We might suspect that the Chopin chord is not included because it is
highlighted as a different label, 7e.g. `V7(13)`.**

```{code-cell} ipython3
utils.plot_chord_profiles(chord_tone_profiles, "V7(13), major")
```

**From here, it is interesting to ask, either, if these special labels show up more frequently in Chopin's corpus
than in others, and if 3 shows up prominently in Chopin's dominants if we combine all dominant chord profiles with
each other.**

```{code-cell} ipython3
all_V7 = harmony_labels.query("numeral == 'V' & figbass == '7'")
all_V7.head()
```

```{code-cell} ipython3
all_V7["tonicization_chord"] = all_V7.chord.str.split("/").str[0]
```

```{code-cell} ipython3
all_V7_absolute = all_V7.groupby(["corpus", "tonicization_chord"]).duration_qb.agg(
    ["sum", "size"]
)
all_V7_absolute.columns = ["duration_qb", "count"]
all_V7_absolute
```

```{code-cell} ipython3
all_V7_relative = all_V7_absolute / all_V7_absolute.groupby("corpus").sum()
make_bar_plot(
    all_V7_relative.reset_index(),
    x_col="tonicization_chord",
    y_col="duration_qb",
    color="corpus",
    log_y=True,
)
```

```{code-cell} ipython3
all_V7_relative.loc["chopin_mazurkas"].sort_values("count", ascending=False) * 100
```

**This is not a good way of comparing dominant chords. We could now start summing up all the different chords we
consider to be part of the "Chopin chord" category. Chord-tone-profiles are probably the better way to see.**

+++

## Create chord-tone profiles for multiple chord features

Tokens are `(feature, ..., chord_tone)` tuples.

```{code-cell} ipython3
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
```

```{code-cell} ipython3
tonicization_profiles.head()
dominant_ct = tonicization_profiles.loc(axis=1)[[1]].stack()
dominant_ct.columns = ["duration_qb"]
dominant_ct["proportion"] = dominant_ct["duration_qb"] / dominant_ct.groupby(
    "corpus"
).duration_qb.agg("sum")
dominant_ct.head()
```

```{code-cell} ipython3
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
```

**Chopin does not show a specifically high bar for 4 (the major third of the scale), Mozart's is higher, for example.
This could have many reasons, e.g. that the pieces are mostly in minor, or that the importance of scale degree 3 as
lower neighbor to the dominant seventh is statistically more important, or that the "characteristic 13" is not
actually important duration-wise.**

```{code-cell} ipython3
tonic_thirds_in_dominants = (
    dominant_ct.loc[(slice(None), [4, -3]), "proportion"].groupby("corpus").sum()
)
```

```{code-cell} ipython3
make_bar_plot(
    tonic_thirds_in_dominants,
    x_col="corpus",
    y_col="proportion",
    category_orders=dict(corpus=D.get_metadata().get_corpus_names(func=None)),
    title="Proportion of scale degree 3 in dominant chords, chronological order",
)
```

**No chronological trend visible.**
