---
jupytext:
  formats: md:myst,ipynb,py:percent
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: revamp
  language: python
  name: revamp
---

# Harmonies

```{code-cell}
%load_ext autoreload
%autoreload 2
import os
from collections import Counter
from statistics import mean

import dimcat as dc
import ms3
import pandas as pd
from dimcat import plotting
from dimcat.utils import grams, make_transition_matrix
import utils

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
```

```{code-cell}
RESULTS_PATH = os.path.abspath(os.path.join(utils.OUTPUT_FOLDER, "harmonies"))
os.makedirs(RESULTS_PATH, exist_ok=True)


def make_output_path(
    filename: str,
    extension=None,
    path=RESULTS_PATH,
) -> str:
    return utils.make_output_path(filename=filename, extension=extension, path=path)


def save_figure_as(
    fig, filename, formats=("png", "pdf"), directory=RESULTS_PATH, **kwargs
):
    if formats is not None:
        for fmt in formats:
            plotting.write_image(fig, filename, directory, format=fmt, **kwargs)
    else:
        plotting.write_image(fig, filename, directory, **kwargs)
```

**Loading data**

```{code-cell}
D = utils.get_dataset("couperin_concerts", corpus_release="v2.2")
D
```

**All labels**

```{code-cell}
labels = D.get_feature("harmonylabels")
labels
```

```{code-cell}
metadata = D.get_metadata()
is_annotated_mask = metadata.label_count > 0
is_annotated_index = dc.PieceIndex(metadata.index[is_annotated_mask])
annotated_notes = D.get_feature("notes").subselect(is_annotated_index)
print(f"The annotated pieces have {len(annotated_notes)} notes.")
```

**Delete @none labels**
This creates progressions between the label before and after the `@none` label that might not actually be perceived
as transitions!

```{code-cell}
df = utils.remove_none_labels(labels.df)
```

**Delete non-chord labels (typically, phrase labels)**

```{code-cell}
df = utils.remove_non_chord_labels(df)
```

```{code-cell}
key_region_groups, key_region2key = ms3.adjacency_groups(df.localkey)
df["key_regions"] = key_region_groups
```

## Unigrams

```{code-cell}
k = 25
df.chord.value_counts().iloc[:k]
```

```{code-cell}
font_dict = {"font": {"size": 20}}
H_LAYOUT = utils.STD_LAYOUT.copy()
H_LAYOUT.update(
    {
        "legend": dict(
            {"orientation": "h", "itemsizing": "constant", "x": -0.05}, **font_dict
        )
    }
)
```

```{code-cell}
fig = utils.plot_cum(
    df.chord,
    x_log=True,
    markersize=4,
    left_range=(-0.03, 3.7),
    right_range=(-0.01, 1.11),
    **H_LAYOUT,
)
save_figure_as(fig, "chord_label_unigram_distribution")
fig
```

### Unigrams in major segments

```{code-cell}
minor, major = df[df.localkey_is_minor], df[~df.localkey_is_minor]
print(
    f"{len(major)} tokens ({len(major.chord.unique())} types) in major and {len(minor)} "
    f"({len(minor.chord.unique())} types) in minor."
)
```

```{code-cell}
major.chord.value_counts().iloc[:k]
```

```{code-cell}
fig = utils.plot_cum(
    major.chord,
    x_log=True,
    markersize=4,
    left_range=(-0.03, 3.7),
    right_range=(-0.01, 1.11),
    **H_LAYOUT,
)
save_figure_as(fig, "chord_label_unigram_distribution_in_major")
fig.show()
```

### Unigrams in minor segments

```{code-cell}
print(
    f"{len(major)} tokens ({len(major.chord.unique())} types) in major and {len(minor)} "
    f"({len(minor.chord.unique())} types) in minor."
)
```

```{code-cell}
minor.chord.value_counts().iloc[:k]
```

```{code-cell}
fig = utils.plot_cum(
    minor.chord,
    x_log=True,
    markersize=4,
    left_range=(-0.03, 3.7),
    right_range=(-0.01, 1.11),
    **H_LAYOUT,
)
save_figure_as(fig, "chord_label_unigram_distribution_in_minor")
fig.show()
```

## Bigrams

```{code-cell}
chord_successions = [s.to_list() for _, s in df.groupby("key_regions").chord]
```

```{code-cell}
gs = grams(chord_successions)
c = Counter(gs)
```

```{code-cell}
dict(sorted(c.items(), key=lambda a: a[1], reverse=True)[:k])
```

### Absolute Counts (read from index to column)

```{code-cell}
make_transition_matrix(chord_successions, k=k, distinct_only=True)
```

### Normalized Counts

```{code-cell}
make_transition_matrix(
    chord_successions, k=k, distinct_only=True, normalize=True, decimals=2
)
```

### Entropy

```{code-cell}
make_transition_matrix(
    chord_successions, k=k, IC=True, distinct_only=True, smooth=1, decimals=2
)
```

### Minor vs. Major

```{code-cell}
region_is_minor = (
    df.groupby("key_regions")
    .localkey_is_minor.unique()
    .map(lambda values: values[0])
    .to_dict()
)
region_key = (
    df.groupby("key_regions").localkey.unique().map(lambda values: values[0]).to_dict()
)
```

```{code-cell}
key_chords = {
    ix: s.to_list()
    for ix, s in df.reset_index().groupby(["piece", "key_regions"]).chord
}
major, minor = [], []
for chords, is_minor in zip(key_chords.values(), region_is_minor.values()):
    (major, minor)[is_minor].append(chords)
```

```{code-cell}
make_transition_matrix(major, k=k, distinct_only=True, normalize=True)
```

```{code-cell}
make_transition_matrix(minor, k=k, distinct_only=True, normalize=True)
```

### Chord progressions without suspensions

Here called *plain chords*, which consist only of numeral, inversion figures, and relative keys.

```{code-cell}
df["plain_chords"] = (
    df.numeral + df.figbass.fillna("") + ("/" + df.relativeroot).fillna("")
)
```

```{code-cell}
df.plain_chords.iloc[:k]
```

**Consecutive identical labels are merged**

```{code-cell}
def remove_subsequent_identical(col):
    return col[col != col.shift()].to_list()


key_regions_plain_chords = (
    df.reset_index()
    .groupby(["piece", "key_regions"])
    .plain_chords.apply(remove_subsequent_identical)
)
key_plain_chords = {ix: s for ix, s in key_regions_plain_chords.items()}
major_plain, minor_plain = [], []
for chords, is_minor in zip(key_plain_chords.values(), region_is_minor.values()):
    (major_plain, minor_plain)[is_minor].append(chords)
```

```{code-cell}
plain_chords_per_segment = {k: len(v) for k, v in key_plain_chords.items()}
```

```{code-cell}
print(
    f"The local key segments have {sum(plain_chords_per_segment.values())} 'plain chords' without immediate "
    f"repetitions, yielding {len(grams(list(key_plain_chords.values())))} bigrams.\n{sum(map(len, major_plain))} "
    f"chords are in major, {sum(map(len, minor_plain))} in minor."
)
```

```{code-cell}
{
    segment: chord_count
    for segment, chord_count in list(
        {
            (
                piece,
                region_key[key] + (" minor" if region_is_minor[key] else " major"),
            ): v
            for (piece, key), v in plain_chords_per_segment.items()
        }.items()
    )[:k]
}
```

```{code-cell}
print(
    f"Segments being in the same local key have a mean length of {round(mean(plain_chords_per_segment.values()), 2)} "
    f"plain chords."
)
```

#### Most frequent 3-, 4-, and 5-grams in major

```{code-cell}
utils.sorted_gram_counts(major_plain, 3)
```

```{code-cell}
utils.sorted_gram_counts(major_plain, 4)
```

```{code-cell}
utils.sorted_gram_counts(major_plain, 5)
```

#### Most frequent 3-, 4-, and 5-grams in minor

```{code-cell}
utils.sorted_gram_counts(minor_plain, 3)
```

```{code-cell}
utils.sorted_gram_counts(minor_plain, 4)
```

```{code-cell}
utils.sorted_gram_counts(minor_plain, 5)
```

### Counting particular progressions

```{code-cell}
MEMORY = {}
chord_progressions = list(key_plain_chords.values())


def look_for(n_gram):
    n = len(n_gram)
    if n in MEMORY:
        n_grams = MEMORY[n]
    else:
        n_grams = grams(chord_progressions, n)
        MEMORY[n] = n_grams
    matches = n_grams.count(n_gram)
    total = len(n_grams)
    return f"{matches} ({round(100*matches/total, 3)} %)"
```

```{code-cell}
look_for(("i", "v6"))
```

```{code-cell}
look_for(("i", "v6", "iv6"))
```

```{code-cell}
look_for(("i", "v6", "iv6", "V"))
```

```{code-cell}
look_for(("i", "V6", "v6"))
```

```{code-cell}
look_for(("V", "IV6", "V65"))
```

### Chord progressions preceding phrase endings

```{code-cell}
def phraseending_progressions(df, n=3, k=k):
    selector = (
        df.groupby(level=0, group_keys=False)
        .phraseend.apply(lambda col: col.notna().shift().fillna(True))
        .cumsum()
    )
    print(f"{selector.max()} phrases overall.")
    phraseends = (
        df.groupby(selector)
        .apply(lambda df: df.chord.iloc[-n:].reset_index(drop=True))
        .unstack()
    )
    return (
        phraseends.groupby(phraseends.columns.to_list())
        .size()
        .sort_values(ascending=False)
        .iloc[:k]
    )
```

```{code-cell}
phraseending_progressions(df)
```

```{code-cell}
phraseending_progressions(df, 4)
```
