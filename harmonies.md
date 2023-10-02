---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.0
kernelspec:
  display_name: coup
  language: python
  name: coup
---

```{code-cell} ipython3
%load_ext autoreload
%autoreload 2
import os
from collections import Counter
import pandas as pd
from helpers import grams, plot_cum, sorted_gram_counts, transition_matrix, STD_LAYOUT
import ms3
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
CORPUS_PATH = "~/all_subcorpora/couperin_concerts"
RESULTS_PATH = os.path.join('..', 'results')
```

```{code-cell} ipython3
corpus_obj = ms3.Corpus(CORPUS_PATH)
corpus_obj.view.include('facet', "expanded")
corpus_obj.parse_tsv()
corpus_obj
```

```{code-cell} ipython3
df = corpus_obj.expanded()
df['key_regions'] = df.groupby(level=0, group_keys=False).localkey.apply(lambda col: col != col.shift()).cumsum()
df.head(20)
```

# Unigrams

```{code-cell} ipython3
k = 25
df.chord.value_counts().iloc[:k]
```

```{code-cell} ipython3
font_dict = {'font': {'size': 20}}
H_LAYOUT = STD_LAYOUT.copy()
H_LAYOUT.update({'legend': dict({'orientation': 'h', 'itemsizing':'constant', 'x': -0.05}, **font_dict)})
```

```{code-cell} ipython3
fig = plot_cum(df.chord, x_log=True, markersize=4, left_range=(-0.03, 3.7), right_range=(-0.01,1.11), **H_LAYOUT)
fig.write_image(os.path.join(RESULTS_PATH, 'type_distribution.png'))
fig
```

## Unigrams in major segments

```{code-cell} ipython3
minor, major = df[df.localkey_is_minor], df[~df.localkey_is_minor]
print(f"{len(major)} tokens ({len(major.chord.unique())} types) in major and {len(minor)} ({len(minor.chord.unique())} types) in minor.")
```

```{code-cell} ipython3
major.chord.value_counts().iloc[:k]
```

```{code-cell} ipython3
fig = plot_cum(major.chord, x_log=True, markersize=4, left_range=(-0.03, 3.7), right_range=(-0.01,1.11), **H_LAYOUT)
fig.write_image(os.path.join(RESULTS_PATH, 'unigrams_major.png'))
fig.show()
```

## Unigrams in minor segments

```{code-cell} ipython3
print(f"{len(major)} tokens ({len(major.chord.unique())} types) in major and {len(minor)} ({len(minor.chord.unique())} types) in minor.")
```

```{code-cell} ipython3
minor.chord.value_counts().iloc[:k]
```

```{code-cell} ipython3
fig = plot_cum(minor.chord, x_log=True, markersize=4, left_range=(-0.03, 3.7), right_range=(-0.01,1.11), **H_LAYOUT)
fig.write_image(os.path.join(RESULTS_PATH, 'unigrams_minor.png'))
fig.show()
```

# Bigrams

```{code-cell} ipython3
chord_successions = [s.to_list() for _, s in df.groupby('key_regions').chord]
```

```{code-cell} ipython3
gs = grams(chord_successions)
c = Counter(gs)
```

```{code-cell} ipython3
dict(sorted(c.items(), key=lambda a: a[1], reverse=True)[:k])
```

## Absolute Counts (read from index to column)

```{code-cell} ipython3
transition_matrix(chord_successions, k=k, dist_only=True)
```

## Normalized Counts

```{code-cell} ipython3
transition_matrix(chord_successions, k=k, dist_only=True, normalize=True, decimals=2)
```

## Entropy

```{code-cell} ipython3
transition_matrix(chord_successions, k=k, IC=True, dist_only=True, smooth=1, decimals=2)
```

## Minor vs. Major

```{code-cell} ipython3
region_is_minor = df.groupby('key_regions').localkey_is_minor.unique().map(lambda l: l[0]).to_dict()
region_key = df.groupby('key_regions').localkey.unique().map(lambda l: l[0]).to_dict()
```

```{code-cell} ipython3
key_chords = {ix: s.to_list() for ix, s in df.reset_index().groupby(['piece', 'key_regions']).chord}
major, minor = [], []
for chords, is_minor in zip(key_chords.values(), region_is_minor.values()):
    (major, minor)[is_minor].append(chords)
```

```{code-cell} ipython3
transition_matrix(major, k=k, dist_only=True, normalize=True)
```

```{code-cell} ipython3
transition_matrix(minor, k=k, dist_only=True, normalize=True)
```

## Chord progressions without suspensions

Here called *plain chords*, which consist only of numeral, inversion figures, and relative keys.

```{code-cell} ipython3
df['plain_chords'] = df.numeral + df.figbass.fillna('') + ('/' + df.relativeroot).fillna('')
```

```{code-cell} ipython3
df.plain_chords.iloc[:k]
```

**Consecutive identical labels are merged**

```{code-cell} ipython3
def remove_subsequent_identical(col):
    return col[col != col.shift()].to_list()
key_regions_plain_chords = (df.reset_index().groupby(['piece', 'key_regions']).plain_chords.apply
                            (remove_subsequent_identical))
key_plain_chords = {ix: s for ix, s in key_regions_plain_chords.items()}
major_plain, minor_plain = [], []
for chords, is_minor in zip(key_plain_chords.values(), region_is_minor.values()):
    (major_plain, minor_plain)[is_minor].append(chords)
```

```{code-cell} ipython3
plain_chords_per_segment = {k: len(v) for k, v in key_plain_chords.items()}
```

```{code-cell} ipython3
print(f"The local key segments have {sum(plain_chords_per_segment.values())} 'plain chords' without immediate repetitions, \
yielding {len(grams(list(key_plain_chords.values())))} bigrams.\n{sum(map(len, major_plain))} chords are in major, {sum(map(len, minor_plain))} in minor.")
```

```{code-cell} ipython3
{segment: chord_count  for segment, chord_count in list({(piece, region_key[key] + (' minor' if region_is_minor[key] else ' major')): v for (piece, key), v in plain_chords_per_segment.items()}.items())[:k]}
```

```{code-cell} ipython3
from statistics import mean
print(f"Segments being in the same local key have a mean length of {round(mean(plain_chords_per_segment.values()), 2)} plain chords.")
```

### Most frequent 3-, 4-, and 5-grams in major

```{code-cell} ipython3
sorted_gram_counts(major_plain, 3)
```

```{code-cell} ipython3
sorted_gram_counts(major_plain, 4)
```

```{code-cell} ipython3
sorted_gram_counts(major_plain, 5)
```

### Most frequent 3-, 4-, and 5-grams in minor

```{code-cell} ipython3
sorted_gram_counts(minor_plain, 3)
```

```{code-cell} ipython3
sorted_gram_counts(minor_plain, 4)
```

```{code-cell} ipython3
sorted_gram_counts(minor_plain, 5)
```

## Counting particular progressions

```{code-cell} ipython3
MEMORY = {}
l = list(key_plain_chords.values())
def look_for(n_gram):
    n = len(n_gram)
    if n in MEMORY:
        n_grams = MEMORY[n]
    else:
        n_grams = grams(l, n)
        MEMORY[n] = n_grams
    matches = n_grams.count(n_gram)
    total = len(n_grams)
    return f"{matches} ({round(100*matches/total, 3)} %)"
```

```{code-cell} ipython3
look_for(('i', 'v6'))
```

```{code-cell} ipython3
look_for(('i', 'v6', 'iv6'))
```

```{code-cell} ipython3
look_for(('i', 'v6', 'iv6', 'V'))
```

```{code-cell} ipython3
look_for(('i', 'V6', 'v6'))
```

```{code-cell} ipython3
look_for(('V', 'IV6', 'V65'))
```

## Chord progressions preceding phrase endings

```{code-cell} ipython3
def phraseending_progressions(df, n=3, k=k):
    selector = (df.groupby(level=0, group_keys=False).phraseend.apply(lambda col: col.notna().shift().fillna(True))
                .cumsum())
    print(f"{selector.max()} phrases overall.")
    phraseends = df.groupby(selector).apply(lambda df: df.chord.iloc[-n:].reset_index(drop=True)).unstack()
    return phraseends.groupby(phraseends.columns.to_list()).size().sort_values(ascending=False).iloc[:k]
```

```{code-cell} ipython3
phraseending_progressions(df)
```

```{code-cell} ipython3
phraseending_progressions(df, 4)
```

```{code-cell} ipython3

```
