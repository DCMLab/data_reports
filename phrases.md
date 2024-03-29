---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
---

# Phrases

```{code-cell} ipython3
import os
from fractions import Fraction

from git import Repo
import dimcat as dc
import ms3
import pandas as pd
import plotly.express as px

from utils import STD_LAYOUT, color_background, value_count_df, get_repo_name, resolve_dir
```

```{code-cell} ipython3
CORPUS_PATH = os.environ.get('CORPUS_PATH', "~/dcml_corpora")
CORPUS_PATH = resolve_dir(CORPUS_PATH)
```

```{code-cell} ipython3
repo = Repo(CORPUS_PATH)
notebook_repo = Repo('.', search_parent_directories=True)
print(f"Notebook repository '{get_repo_name(notebook_repo)}' @ {notebook_repo.commit().hexsha[:7]}")
print(f"Data repo '{get_repo_name(CORPUS_PATH)}' @ {repo.commit().hexsha[:7]}")
print(f"dimcat version {dc.__version__}")
print(f"ms3 version {ms3.__version__}")
```

## Data loading

```{code-cell} ipython3
dataset = dc.Dataset()
dataset.load(directory=CORPUS_PATH)
dataset.data
```

```{code-cell} ipython3
dataset.data
```

### Filtering out pieces without cadence annotations

```{code-cell} ipython3
hascadence = dc.HasCadenceAnnotationsFilter().process_data(dataset)
print(f"Before: {len(dataset.indices[()])} pieces; after removing those without cadence labels: {len(hascadence.indices[()])}")
```

### Show corpora containing pieces with cadence annotations

```{code-cell} ipython3
grouped_by_dataset = dc.CorpusGrouper().process_data(hascadence)
corpora = {group[0]: f"{len(ixs)} pieces" for group, ixs in  grouped_by_dataset.indices.items()}
print(f"{len(corpora)} corpora with {sum(map(len, grouped_by_dataset.indices.values()))} pieces containing cadence annotations:")
corpora
```

### All annotation labels from the selected pieces

```{code-cell} ipython3
all_labels = hascadence.get_facet('expanded')

print(f"{len(all_labels.index)} hand-annotated harmony labels:")
all_labels.iloc[:10, 14:].style.apply(color_background, subset="chord")
```

### Metadata

```{code-cell} ipython3
dataset_metadata = hascadence.data.metadata()
hascadence_metadata = dataset_metadata.loc[hascadence.indices[()]]
hascadence_metadata.index.rename('dataset', level=0, inplace=True)
hascadence_metadata.head()
```

```{code-cell} ipython3
---
pycharm:
  is_executing: true
---
mean_composition_years = hascadence_metadata.groupby(level=0).composed_end.mean().astype(int).sort_values()
chronological_order = mean_composition_years.index.to_list()
bar_data = pd.concat([mean_composition_years.rename('year'), 
                      hascadence_metadata.groupby(level='dataset').size().rename('pieces')],
                     axis=1
                    ).reset_index()
fig = px.bar(bar_data, x='year', y='pieces', color='dataset', title='Pieces contained in the dataset')
fig.update_traces(width=5)
```

## Overview
### Presence of phrase annotation symbols per dataset:

```{code-cell} ipython3
---
pycharm:
  is_executing: true
---
all_labels.groupby(["corpus"]).phraseend.value_counts()
```

### Presence of legacy phrase endings

```{code-cell} ipython3
---
pycharm:
  is_executing: true
---
all_labels[all_labels.phraseend == r'\\'].style.apply(color_background, subset="label")
```

### A table with the extents of all annotated phrases
**Relevant columns:**
* `quarterbeats`: start position for each phrase
* `duration_qb`: duration of each phrase, measured in quarter notes 
* `phrase_slice`: time interval of each annotated phrases (for segmenting chord progressions and notes)

```{code-cell} ipython3
---
pycharm:
  is_executing: true
tags: []
---
# segmented = PhraseSlicer().process_data(hascadence)
segmented = dc.PhraseSlicer().process_data(grouped_by_dataset)
phrases = segmented.get_slice_info()
print(f"Overall number of phrases is {len(phrases.index)}")
phrases.head(10).style.apply(color_background, subset=["quarterbeats", "duration_qb"])
```

```{code-cell} ipython3
---
pycharm:
  is_executing: true
---
print(phrases.duration_qb.dtype)
phrases.duration_qb = pd.to_numeric(phrases.duration_qb)
```

### Annotation table sliced by phrase annotations

ToDo: Example for overlap / phrase beginning without new chord

```{code-cell} ipython3
---
pycharm:
  is_executing: true
---
phrase_segments = segmented.get_facet("expanded")
phrase_segments.head(10)
```

```{code-cell} ipython3
---
pycharm:
  is_executing: true
---
print(phrase_segments.duration_qb.dtype)
phrase_segments.duration_qb = pd.to_numeric(phrase_segments.duration_qb)
```

## Distribution of phrase lengths
### Histogram summarizing the lengths of all phrases measured in quarter notes

```{code-cell} ipython3
---
pycharm:
  is_executing: true
---
phrase_durations = phrases.duration_qb.value_counts() 
histogram = px.histogram(x=phrase_durations.index, y=phrase_durations, labels=dict(x='phrase lengths binned to a quarter note', y='#phrases within length bin'))
histogram.update_traces(xbins=dict( # bins used for histogram
        #start=0.0,
        end=100.0,
        size=1
    ))
histogram.update_xaxes(dtick=4)
histogram.show()
```

### Bar plot showing approximative phrase length in measures

**Simply by subtracting for the span of every phrase the first measure measure number from the last.**

```{code-cell} ipython3
---
pycharm:
  is_executing: true
---
phrase_gpb = phrase_segments.groupby(level=[0,1,2])
phrase_length_in_measures = phrase_gpb.mn.max() - phrase_gpb.mn.min()
measure_length_counts = phrase_length_in_measures.value_counts()
fig = px.bar(x=measure_length_counts.index, y=measure_length_counts, labels=dict(x="approximative size of all phrases (difference between end and start measure number)",
                                                                           y="#phrases"))
fig.update_xaxes(dtick=4)
```

### Histogram summarizing phrase lengths by precise length expressed in measures

**In order to divide the phrase length by the length of a measure, the phrases containing more than one time signature are filtered out.**

+++

**Durations computed by dividing the duration by the measure length**

```{code-cell} ipython3
---
pycharm:
  is_executing: true
---
phrase2timesigs = phrase_gpb.timesig.unique()
n_timesignatures_per_phrase = phrase2timesigs.map(len)
uniform_timesigs = phrase2timesigs[n_timesignatures_per_phrase == 1].map(lambda l: l[0])
more_than_one = n_timesignatures_per_phrase > 1
print(f"Filtered out the {more_than_one.sum()} phrases incorporating more than one time signature.")
n_timesigs = n_timesignatures_per_phrase.value_counts()
display(n_timesigs.reset_index().rename(columns=dict(index='#time signatures', timesig='#phrases')))
uniform_timesig_phrases = phrases.loc[uniform_timesigs.index]
timesig_in_quarterbeats = uniform_timesigs.map(Fraction) * 4
exact_measure_lengths = uniform_timesig_phrases.duration_qb / timesig_in_quarterbeats
uniform_timesigs = pd.concat([exact_measure_lengths.rename('duration_measures'), uniform_timesig_phrases], axis=1)
fig = px.histogram(uniform_timesigs, x='duration_measures', 
                   labels=dict(duration_measures='phrase length in measures, factoring in time signatures'))
fig.update_traces(xbins=dict( # bins used for histogram
        #start=0.0,
        #end=100.0,
        size=1
    ))
fig.update_xaxes(dtick=4)
```

```{code-cell} ipython3
---
pycharm:
  is_executing: true
---
uniform_timesigs.head(10).style.apply(color_background, subset='duration_measures')
```

### Inspecting long phrases

```{code-cell} ipython3
---
pycharm:
  is_executing: true
---
timsig_counts = uniform_timesigs.timesig.value_counts()
fig = px.bar(timsig_counts, labels=dict(index="time signature", value="#phrases"))
fig.update_layout(**STD_LAYOUT)
fig.update_yaxes(gridcolor='lightgrey')
```

```{code-cell} ipython3
---
pycharm:
  is_executing: true
---
filter_counts_smaller_than = 5
filtered_timesigs = timsig_counts[timsig_counts < filter_counts_smaller_than].index.to_list()
```

```{code-cell} ipython3
---
pycharm:
  is_executing: true
---
fig = px.histogram(uniform_timesigs[~uniform_timesigs.timesig.isin(filtered_timesigs)], 
                   x='duration_measures', facet_col='timesig', facet_col_wrap=2, height=1500)
fig.update_xaxes(matches=None, showticklabels=True, visible=True, dtick=4)
fig.update_yaxes(matches=None, showticklabels=True, visible=True)
fig.update_traces(xbins=dict( # bins used for histogram
        #start=0.0,
        end=50.0,
        size=1
    ))
```

```{code-cell} ipython3
---
pycharm:
  is_executing: true
---
see_greater_equal = 33
longest_measure_length = uniform_timesigs.loc[uniform_timesigs.duration_measures >= see_greater_equal, ["duration_measures", "timesig"]]
for timesig, long_phrases in longest_measure_length.groupby('timesig'):
    L = len(long_phrases)
    plural = 's' if L > 1 else ''
    print(f"{L} long phrase{plural} in {timesig} meter:")
    display(long_phrases.sort_values('duration_measures'))
```

## Local keys

```{code-cell} ipython3
---
pycharm:
  is_executing: true
---
local_keys_per_phrase = phrase_gpb.localkey.unique().map(tuple)
n_local_keys_per_phrase = local_keys_per_phrase.map(len)
phrases_with_keys = pd.concat([n_local_keys_per_phrase.rename('n_local_keys'),
                               local_keys_per_phrase.rename('local_keys'),
                               phrases], axis=1)
phrases_with_keys.head(10).style.apply(color_background, subset=['n_local_keys', 'local_keys'])
```

### Number of unique local keys per phrase

```{code-cell} ipython3
---
pycharm:
  is_executing: true
---
count_n_keys = phrases_with_keys.n_local_keys.value_counts().rename("#phrases").to_frame()
count_n_keys.index.rename("unique keys", inplace=True)
count_n_keys
```

### The most frequent keys for non-modulating phrases

```{code-cell} ipython3
---
pycharm:
  is_executing: true
---
unique_key_selector = phrases_with_keys.n_local_keys == 1
phrases_with_unique_key = phrases_with_keys[unique_key_selector].copy()
phrases_with_unique_key.local_keys = phrases_with_unique_key.local_keys.map(lambda t: t[0])
value_count_df(phrases_with_unique_key.local_keys, counts="#phrases")
```

### Most frequent modulations within one phrase

```{code-cell} ipython3
---
pycharm:
  is_executing: true
tags: []
---
two_keys_selector = phrases_with_keys.n_local_keys > 1
phrases_with_unique_key = phrases_with_keys[two_keys_selector].copy()
value_count_df(phrases_with_unique_key.local_keys, "modulations")
```
