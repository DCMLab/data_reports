---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: revamp
  language: python
  name: revamp
---

# Overview

This notebook gives a general overview of the features included in the dataset.

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide imports
  code_prompt_show: Show imports
tags: [hide-cell]
---
%load_ext autoreload
%autoreload 2
import os
from git import Repo
import dimcat as dc
import ms3
import pandas as pd
import plotly.express as px

from utils import (CORPUS_COLOR_SCALE, STD_LAYOUT, corpus_mean_composition_years,
                   get_corpus_display_name, get_repo_name, print_heading, resolve_dir)
```

```{code-cell}
from utils import OUTPUT_FOLDER, write_image
RESULTS_PATH = os.path.abspath(os.path.join(OUTPUT_FOLDER, "overview"))
os.makedirs(RESULTS_PATH, exist_ok=True)
def save_figure_as(fig, filename, directory=RESULTS_PATH, **kwargs):
    write_image(fig, filename, directory, **kwargs)
```

**Loading data**

```{code-cell}
package_path = resolve_dir("~/distant_listening_corpus/couperin_concerts/couperin_concerts.datapackage.json")
repo = Repo(os.path.dirname(package_path))
print_heading("Data and software versions")
print(f"Data repo '{get_repo_name(repo)}' @ {repo.commit().hexsha[:7]}")
print(f"dimcat version {dc.__version__}")
print(f"ms3 version {ms3.__version__}")
D = dc.Dataset.from_package(package_path)
D
```

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide data loading
  code_prompt_show: Show data loading
tags: [hide-cell]
---
all_metadata = D.get_metadata()
assert len(all_metadata) > 0, "No pieces selected for analysis."
all_notes = D.get_feature('notes').df
all_measures = D.get_feature('measures').df
mean_composition_years = corpus_mean_composition_years(all_metadata)
chronological_order = mean_composition_years.index.to_list()
corpus_colors = dict(zip(chronological_order, CORPUS_COLOR_SCALE))
corpus_names = {corp: get_corpus_display_name(corp) for corp in chronological_order}
chronological_corpus_names = list(corpus_names.values())
corpus_name_colors = {corpus_names[corp]: color for corp, color in corpus_colors.items()}
```

## Composition dates

This section relies on the dataset's metadata.

```{code-cell}
valid_composed_start = pd.to_numeric(all_metadata.composed_start, errors='coerce')
valid_composed_end = pd.to_numeric(all_metadata.composed_end, errors='coerce')
print(f"Composition dates range from {int(valid_composed_start.min())} {valid_composed_start.idxmin()} "
      f"to {int(valid_composed_end.max())} {valid_composed_end.idxmax()}.")
```

### Mean composition years per corpus

```{code-cell}
:tags: [hide-input]

piece_is_annotated = all_metadata.label_count > 0
summary = all_metadata[piece_is_annotated].copy()
summary.length_qb = all_measures[piece_is_annotated].groupby(level=[0,1]).act_dur.sum() * 4.0
summary = pd.concat([summary,
                     all_notes[piece_is_annotated].groupby(level=[0,1]).size().rename('notes'),
                    ], axis=1)
bar_data = pd.concat([mean_composition_years.rename('year'),
                      summary.groupby(level='corpus').size().rename('pieces')],
                     axis=1
                    ).reset_index()

N = len(summary)
fig = px.bar(
    bar_data,
    x='year',
    y='pieces',
    color='corpus',
    color_discrete_map=corpus_colors,
    title=f"Temporal coverage of the {N} annotated pieces in the Distant Listening Corpus"
)
fig.update_traces(width=5)
fig.update_layout(**STD_LAYOUT)
fig.update_traces(width=5)
save_figure_as(fig, "pieces_timeline_bars")
fig.show()
```

### Composition years histogram

```{code-cell}
:tags: [hide-input]

hist_data = summary.reset_index()
hist_data.corpus = hist_data.corpus.map(corpus_names)
fig = px.histogram(
    hist_data,
    x='composed_end',
    color='corpus',
    labels=dict(composed_end='decade',
               count='pieces',
              ),
    color_discrete_map=corpus_name_colors,
    title=f"Temporal coverage of the {N} annotated pieces in the Distant Listening Corpus"
                  )
fig.update_traces(xbins=dict(
    size=10
))
fig.update_layout(**STD_LAYOUT)
fig.update_legends(
  font=dict(size=16)
)
save_figure_as(fig, "pieces_timeline_histogram", height=1250)
fig.show()
```

## Dimensions

### Overview

```{code-cell}
:tags: [hide-input]

corpus_metadata = summary.groupby(level=0)
n_pieces = corpus_metadata.size().rename('pieces')
absolute_numbers = dict(
    measures = corpus_metadata.last_mn.sum(),
    length = corpus_metadata.length_qb.sum(),
    notes = corpus_metadata.notes.sum(),
    labels = corpus_metadata.label_count.sum(),
)
absolute = pd.DataFrame.from_dict(absolute_numbers)
absolute = pd.concat([n_pieces, absolute], axis=1)
sum_row = pd.DataFrame(absolute.sum(), columns=['sum']).T
absolute = pd.concat([absolute, sum_row])
relative = absolute.div(n_pieces, axis=0)
complete_summary = pd.concat([absolute, relative, absolute.iloc[:1,2:].div(absolute.measures, axis=0)], axis=1, keys=['absolute', 'per piece', 'per measure'])
complete_summary = complete_summary.apply(pd.to_numeric).round(2)
complete_summary.index = complete_summary.index.map(dict(corpus_names, sum='sum'))
complete_summary
```

### Measures

```{code-cell}
print(f"{len(all_measures.index)} measures over {len(all_measures.groupby(level=[0,1]))} files.")
all_measures.head()
```

```{code-cell}
print("Distribution of time signatures per XML measure (MC):")
all_measures.timesig.value_counts(dropna=False)
```

### Harmony labels

All symbols, independent of the local key (the mode of which changes their semantics).

```{code-cell}
try:
    all_annotations = D.get_feature("harmonylabels").df
except Exception:
    all_annotations = pd.DataFrame()
n_annotations = len(all_annotations.index)
includes_annotations = n_annotations > 0
if includes_annotations:
    display(all_annotations.head())
    print(f"Concatenated annotation tables contains {all_annotations.shape[0]} rows.")
    no_chord = all_annotations.root.isna()
    if no_chord.sum() > 0:
        print(f"{no_chord.sum()} of them are not chords. Their values are: {all_annotations.label[no_chord].value_counts(dropna=False).to_dict()}")
    all_chords = all_annotations[~no_chord].copy()
    print(f"Dataset contains {all_chords.shape[0]} tokens and {len(all_chords.chord.unique())} types over {len(all_chords.groupby(level=[0,1]))} documents.")
    all_annotations['corpus_name'] = all_annotations.index.get_level_values(0).map(get_corpus_display_name)
    all_chords['corpus_name'] = all_chords.index.get_level_values(0).map(get_corpus_display_name)
else:
    print(f"Dataset contains no annotations.")
```
