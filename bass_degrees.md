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

# Bass degrees

```{code-cell}

%load_ext autoreload
%autoreload 2
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import ms3
import pandas as pd
import plotly.express as px
from dimcat import groupers, plotting
from dimcat.data.resources.utils import make_adjacency_groups

import utils

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
```

```{code-cell}
RESULTS_PATH = os.path.abspath(os.path.join(utils.OUTPUT_FOLDER, "bass_degrees"))
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

**Grouping data**

```{code-cell}
grouped_D = groupers.ModeGrouper().process(D)
grouped_D
```

```{code-cell}
bass_notes = grouped_D.get_feature("bassnotes")
bass_notes.df
```

```{code-cell}
bass_notes.plot(
    output=make_output_path("piecewise_bass_degrees"), width=1100, height=2500
)
```

```{code-cell}
bass_notes.plot_grouped(output=make_output_path("bass_degrees_major_minor"))
```

## Bass degree unigrams
As expressed by the annotation labels.

```{code-cell}
labels = D.get_feature("harmonylabels")
df = labels.df
df.head(20)
```

```{code-cell}
bd = df.bass_note.value_counts()
print(f"N = {len(df)}")
fig = px.bar(
    x=ms3.fifths2sd(bd.index.to_list()),
    y=bd.values,
    labels=dict(x="Scale degree in relation to major scale", y="count"),
    title="Distribution of all bass degrees",
    color_discrete_sequence=["grey"] * len(bd),
)
fig.update_xaxes(type="category")
save_figure_as(fig, "bass_degree_unigrams")
fig.show()
```

```{code-cell}
localkey_is_major = ~df.localkey_is_minor
print(f"N = {localkey_is_major.sum()}")
bd_maj = df[localkey_is_major].bass_note.value_counts()
fig = px.bar(
    bd_maj,
    x=ms3.fifths2sd(bd_maj.index.to_list(), minor=False),
    y=bd_maj.values,
    labels=dict(x="Scale degree in relation to major scale", y="count"),
    title="Distribution of bass degrees in major segments",
    color_discrete_sequence=["blue"] * len(bd_maj),
)
fig.update_xaxes(type="category")
save_figure_as(fig, "bass_degree_unigrams_major")
fig.show()
```

```{code-cell}
print(f"N = {df.localkey_is_minor.sum()}")
bd_min = df[df.localkey_is_minor].bass_note.value_counts()
fig = px.bar(
    bd_min,
    x=ms3.fifths2sd(bd_min.index.to_list(), minor=True),
    y=bd_min.values,
    labels=dict(x="Scale degree in relation to minor scale", y="count"),
    title="Distribution of bass degrees in minor segments",
    color_discrete_sequence=["red"] * len(bd_min),
)
fig.update_xaxes(type="category")
save_figure_as(fig, "bass_degree_unigrams_minor")
fig.show()
```

**Note: When dropping immediate repetitions of the same bass degree (see transition matrices below), minor segments,
too, have 1 as the most frequent bass degree. This can be linked to frequent suspensions over bass degree 5.**

+++

## Intervals between adjacent bass notes
### Preparing the data
#### Get localkey segments

```{code-cell}
df.groupby(level=0, group_keys=False).localkey.apply(
    lambda col: col != col.shift()
).cumsum()
```

```{code-cell}
key_region_groups, key_region2key = make_adjacency_groups(df.localkey, groupby="piece")
df["key_regions"] = key_region_groups
df["bass_degree"] = ms3.transform(df, ms3.fifths2sd, ["bass_note", "localkey_is_minor"])
```

```{code-cell}
segment_lengths = df.groupby("key_regions").size()
fig = px.histogram(
    segment_lengths,
    log_y=True,
    labels=dict(value="#labels in a key segment", count="number of key segments"),
)
save_figure_as(fig, "n_labels_per_key_segment_histogram")
fig.show()
```

```{code-cell}
# Show all segments of length L
L = 1
selected = segment_lengths[segment_lengths == L].index
df[df.key_regions.isin(selected)]
```

**Delete @none labels**
This creates progressions between the label before and after the `@none` label that might not actually be perceived
as transitions!

```{code-cell}
df = utils.remove_none_labels(df)
```

**Delete non-chord labels (typically, phrase labels)**

```{code-cell}
df = utils.remove_non_chord_labels(df)
```

### Get bass degree progressions & intervals
All scale degrees are expressed as fifth-intervals to the local tonic:

| fifths-interval | interval    |
|-----------------|-------------|
| -3              | m3          |
| -2              | m7          |
| -1              | P4          |
| 0               | local tonic |
| 1               | P5          |
| 2               | M2          |
| 3               | M6          |

```{code-cell}
bd_series = {seg: bn for seg, bn in df.groupby("key_regions").bass_note}
bd_intervals = {seg: bd - bd.shift() for seg, bd in bd_series.items()}
df["bass_interval"] = df.groupby("key_regions", group_keys=False).bass_note.apply(
    lambda bd: bd.shift(-1) - bd
)
print("Example output for the intervals of the first key segment:")
df.loc[df.key_regions == 1, ["bass_note", "bass_interval"]]
```

#### Count bass intervals

```{code-cell}
interval_counter = Counter()
for S in bd_intervals.values():
    interval_counter.update(S.dropna())
interval_counter
```

```{code-cell}
bar_data = pd.Series(interval_counter).sort_values(ascending=False)
fig = px.bar(
    x=ms3.fifths2iv(bar_data.index.to_list()),
    y=bar_data.values,
    labels=dict(x="Interval between adjacent bass notes", y="count"),
    title="Distribution of all bass intervals (within the same localkey)",
    color_discrete_sequence=["grey"] * len(bd),
)
fig.update_xaxes(type="category")
save_figure_as(fig, "bass_progression_intervals_within_all_key_segments_sorted_bars")
fig.show()
```

**

```{code-cell}
px.bar(
    x=interval_counter.keys(),
    y=interval_counter.values(),
    labels=dict(x="interval in fifths", y="count"),
    title="Orderd on the line of fifths",
)
```

```{code-cell}
iv_order = [
    -6,
    1,
    8,
    -11,
    -4,
    3,
    10,
    -9,
    -2,
    5,
    12,
    -7,
    0,
    7,
    -5,
    2,
    9,
    -10,
    -3,
    4,
    11,
    -8,
    -1,
    6,
]  # do not occur: -11, -10, 11
iv_counter = {
    ms3.fifths2iv(i, True): interval_counter[i]
    for i in iv_order
    if i in interval_counter
}
px.bar(
    x=iv_counter.keys(),
    y=iv_counter.values(),
    labels=dict(x="interval", y="count"),
    title="Ordered by base " "interval",
)
```

```{code-cell}
iv_order = [
    12,
    10,
    -9,
    -6,
    -7,
    8,
    -4,
    5,
    1,
    3,
    -2,
    0,
    -1,
    2,
    -5,
    4,
    -3,
    6,
    7,
    -8,
    9,
]  # do not occur: -11, -10, 11
iv_counter = {
    ms3.fifths2iv(i): interval_counter[i] for i in iv_order if i in interval_counter
}
px.bar(
    x=iv_counter.keys(),
    y=iv_counter.values(),
    labels=dict(x="interval", y="count"),
    title="Ordered around the unison",
)
```

```{code-cell}
px.bar(
    x=sorted(iv_counter.keys(), key=lambda k: iv_counter[k], reverse=True),
    y=sorted(iv_counter.values(), reverse=True),
    labels=dict(x="interval", y="count"),
    title="Descending frequency",
)
```

### Checking rare intervals

**occurrences of -9 (-A2)**

* `c02n01_prelude` m. 12: `ii%` was corrected to `iii%`

The other two were correct:

* `c07n05_gavote` m. 27: progression `V65 VIM7` in minor
* `c03n05_gavotte` m. 22: progression `V65 ii%43` (in minor)

```{code-cell}
# see all key regions containing a certain interval in the bass
bass_interval = -9
selected = df.loc[df.bass_interval == bass_interval].key_regions.unique()
df.loc[
    df.key_regions.isin(selected), ["mc", "mn", "chord", "bass_degree", "bass_interval"]
]
```

### Inspecting stepwise bass movement

**Add column with bass interval in semitones.**
It's called `bass_interval_pc` as in "pitch class"

```{code-cell}
pc_ivs = ms3.transform(df, ms3.fifths2pc, ["bass_interval"])
df["bass_interval_pc"] = pc_ivs.where(pc_ivs <= 6, pc_ivs % -6)
```

```{code-cell}
df[["mc", "mn", "chord", "bass_degree", "bass_interval", "bass_interval_pc"]].head(15)
```

#### Create key region summary

**With one row per key region and the following columns**

| **column**            | what it contains                                               |
|-----------------------|----------------------------------------------------------------|
| **globalkey**         | the global key of the segment                                  |
| **localkey**          | the local key of the segment                                   |
| **length**            | number of labels                                               |
| **length_norepeat**   | number of labels without immediate repetitions                 |
| **n_stepwise**        | number of stepwise bass progressions                           |
| **%_stepwise**        | percentage of stepwise progressions (based on length_norepeat) |
| **n_ascending**       | number of stepwise ascending bass progressions                 |
| **n_descending**      | number of stepwise descending bass progressions                |
| **bd**                | full sequence of bass degrees                                  |
| **stepwise_bd**       | all sequences of consecutive stepwise bass progressions        |
| **stepwise_chords**   | the corresponding chord sequences                              |
| **ascending_bd**      | the subset of ascending bass progressions                      |
| **ascending_chords**  | the corresponding chord sequences                              |
| **descending_bd**     | the subset of descending bass progressions                     |
| **descending_chords** | the corresponding chord sequences                              |
| **ixa**               | index of the segment's first row                               |
| **ixb**               | index of the segment's last row                                |

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
key_regions = utils.make_key_region_summary_table(df, "key_regions")
key_regions.head(10)
```

**Store to file `key_regions.tsv` for easier inspection.**

```{code-cell}
key_regions.to_csv(os.path.join(RESULTS_PATH, "key_regions.tsv"), sep="\t")
```

```{code-cell}
print(
    f"{key_regions.n_stepwise.sum() / key_regions.length_norepeat.sum():.1%} of all bass movements are stepwise."
)
```

```{code-cell}
def remove_immediate_repetitions(s):
    res = []
    last = ""
    for word in s.split():
        if word != last:
            res.append(word)
        last = word
    return " ".join(res)
```

```{code-cell}
minor_selector = key_regions.localkey.str.islower()
```

```{code-cell}
minor_regions = key_regions[minor_selector]
major_regions = key_regions[~minor_selector]
```

#### All stepwise ascending bass progressions in minor

```{code-cell}
ascending_minor = defaultdict(list)
for bd, chord in zip(
    minor_regions.ascending_bd.sum(), minor_regions.ascending_chords.sum()
):
    ascending_minor[remove_immediate_repetitions(bd)].append(chord)
ascending_minor_counts = Counter({k: len(v) for k, v in ascending_minor.items()})
utils.prettify_counts(ascending_minor_counts)
```

```{code-cell}
show_progression = "3 4 5"
chords_3_4_5 = Counter(ascending_minor[show_progression])
utils.prettify_counts(chords_3_4_5)
```

#### All stepwise ascending bass progressions in major

```{code-cell}
ascending_major = defaultdict(list)
for bd, chord in zip(
    major_regions.ascending_bd.sum(), major_regions.ascending_chords.sum()
):
    ascending_major[remove_immediate_repetitions(bd)].append(chord)
ascending_major_counts = Counter({k: len(v) for k, v in ascending_major.items()})
utils.prettify_counts(ascending_major_counts)
```

```{code-cell}
show_progression = "6 7 1"
chords_6_7_1 = Counter(ascending_major[show_progression])
utils.prettify_counts(chords_6_7_1)
```

#### All stepwise descending bass progressions in minor

```{code-cell}
descending_minor = defaultdict(list)
for bd, chord in zip(
    minor_regions.descending_bd.sum(), minor_regions.descending_chords.sum()
):
    descending_minor[remove_immediate_repetitions(bd)].append(chord)
descending_minor_counts = Counter({k: len(v) for k, v in descending_minor.items()})
utils.prettify_counts(descending_minor_counts)
```

```{code-cell}
show_progression = "3 2 1"
chords_3_2_1 = Counter(descending_minor[show_progression])
utils.prettify_counts(chords_3_2_1)
```

#### All stepwise descending bass progressions in major

```{code-cell}
descending_major = defaultdict(list)
for bd, chord in zip(
    major_regions.descending_bd.sum(), major_regions.descending_chords.sum()
):
    descending_major[remove_immediate_repetitions(bd)].append(chord)
descending_major_counts = Counter({k: len(v) for k, v in descending_major.items()})
utils.prettify_counts(descending_major_counts)
```

```{code-cell}
show_progression = "5 4 3"
chords_5_4_3 = Counter(descending_major[show_progression])
utils.prettify_counts(chords_5_4_3)
```

### Transitions between bass degrees

```{code-cell}
full_grams = {
    i: S[(S != S.shift()).fillna(True)].to_list() for i, S in bd_series.items()
}
print(bd_series[1])
full_grams[1]
```

```{code-cell}
minor_region_selector = key_regions.localkey.str.islower()
minor_regions = key_regions[minor_region_selector].localkey.to_dict()
major_regions = key_regions[~minor_region_selector].localkey.to_dict()
full_grams_minor = [ms3.fifths2sd(full_grams[i], True) + ["∅"] for i in minor_regions]
full_grams_major = [ms3.fifths2sd(full_grams[i], False) + ["∅"] for i in major_regions]
```

```{code-cell}
utils.plot_transition_heatmaps(full_grams_major, full_grams_minor)
save_pdf_path = os.path.join(
    RESULTS_PATH, f"bass_degree_bigrams{utils.DEFAULT_OUTPUT_FORMAT}"
)
plt.savefig(save_pdf_path, dpi=400)
plt.show()
```

```{code-cell}
utils.plot_transition_heatmaps(
    full_grams_major, full_grams_minor, sort_scale_degrees=True
)
save_pdf_path = os.path.join(
    RESULTS_PATH, f"bass_degree_bigrams_scale_order{utils.DEFAULT_OUTPUT_FORMAT}"
)
plt.savefig(save_pdf_path, dpi=400)
plt.show()
```

#### Most frequent 2-grams

Select number of first k transitions to display:

```{code-cell}
k = 25
```

##### Major

```{code-cell}
utils.sorted_gram_counts(full_grams_major, 2)
```

##### Minor

```{code-cell}
utils.sorted_gram_counts(full_grams_minor, 2)
```

#### Most frequent 3-grams

##### Major

```{code-cell}
utils.sorted_gram_counts(full_grams_major, 3)
```

##### Minor

```{code-cell}
utils.sorted_gram_counts(full_grams_minor, 3)
```

#### Most frequent 4-grams

##### Major

```{code-cell}
utils.sorted_gram_counts(full_grams_major, 4)
```

##### Minor

```{code-cell}
utils.sorted_gram_counts(full_grams_minor, 4)
```

#### Most frequent 5-grams

##### Major

```{code-cell}
utils.sorted_gram_counts(full_grams_major, 5)
```

##### Minor

```{code-cell}
utils.sorted_gram_counts(full_grams_minor, 5)
```
