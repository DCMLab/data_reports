---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: corpus_docs
  language: python
  name: corpus_docs
---

# Annotations

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

from utils import CORPUS_COLOR_SCALE, color_background, corpus_mean_composition_years, \
  get_corpus_display_name, get_repo_name, print_heading, resolve_dir, make_sunburst, rectangular_sunburst

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
```

```{code-cell}
from utils import OUTPUT_FOLDER, write_image
RESULTS_PATH = os.path.abspath(os.path.join(OUTPUT_FOLDER, "scale_degrees"))
os.makedirs(RESULTS_PATH, exist_ok=True)
def save_figure_as(fig, filename, directory=RESULTS_PATH, **kwargs):
    write_image(fig, filename, directory, **kwargs)
```

```{code-cell}
:tags: [hide-input]

# CORPUS_PATH = os.path.abspath(os.path.join('..', '..')) # for running the notebook in the homepage deployment workflow
CORPUS_PATH = "~/distant_listening_corpus/couperin_concerts"                # for running the notebook locally
print_heading("Notebook settings")
print(f"CORPUS_PATH: {CORPUS_PATH!r}")
CORPUS_PATH = resolve_dir(CORPUS_PATH)
```

```{code-cell}
:tags: [hide-input]

repo = Repo(CORPUS_PATH)
print_heading("Data and software versions")
print(f"Data repo '{get_repo_name(repo)}' @ {repo.commit().hexsha[:7]}")
print(f"dimcat version {dc.__version__}")
print(f"ms3 version {ms3.__version__}")
```

```{code-cell}
:tags: [remove-output]

dataset = dc.Dataset()
dataset.load(directory=CORPUS_PATH, parse_tsv=False)
```

```{code-cell}
:tags: [remove-input]

annotated_view = dataset.data.get_view('annotated')
annotated_view.include('facets', 'measures', 'expanded')
annotated_view.pieces_with_incomplete_facets = False
dataset.data.set_view(annotated_view)
dataset.data.parse_tsv(choose='auto')
dataset.get_indices()
dataset.data
```

```{code-cell}
:tags: [remove-input]

print(f"N = {dataset.data.count_pieces()} annotated pieces, {dataset.data.count_parsed_tsvs()} parsed dataframes.")
```

```{code-cell}
all_metadata = dataset.data.metadata()
assert len(all_metadata) > 0, "No pieces selected for analysis."
print(f"Metadata covers {len(all_metadata)} of the {dataset.data.count_pieces()} scores.")
mean_composition_years = corpus_mean_composition_years(all_metadata)
chronological_order = mean_composition_years.index.to_list()
corpus_colors = dict(zip(chronological_order, CORPUS_COLOR_SCALE))
corpus_names = {corp: get_corpus_display_name(corp) for corp in chronological_order}
chronological_corpus_names = list(corpus_names.values())
corpus_name_colors = {corpus_names[corp]: color for corp, color in corpus_colors.items()}
```

## DCML harmony labels

```{code-cell}
:tags: [hide-input]

try:
    all_annotations = dataset.get_facet('expanded')
except Exception:
    all_annotations = pd.DataFrame()
n_annotations = len(all_annotations.index)
includes_annotations = n_annotations > 0
if includes_annotations:
    display(all_annotations.head())
    print(f"Concatenated annotation tables contain {all_annotations.shape[0]} rows.")
    no_chord = all_annotations.root.isna()
    if no_chord.sum() > 0:
        print(f"{no_chord.sum()} of them are not chords. Their values are: {all_annotations.label[no_chord].value_counts(dropna=False).to_dict()}")
    all_chords = all_annotations[~no_chord].copy()
    print(f"Dataset contains {all_chords.shape[0]} tokens and {len(all_chords.chord.unique())} types over {len(all_chords.groupby(level=[0,1]))} documents.")
    all_annotations['corpus_name'] = all_annotations.index.get_level_values(0).map(corpus_names)
    all_chords['corpus_name'] = all_chords.index.get_level_values(0).map(corpus_names)
else:
    print(f"Dataset contains no annotations.")
```

## Key areas

```{code-cell}
from ms3 import roman_numeral2fifths, transform

keys_segmented = dc.LocalKeySlicer().process_data(dataset)
keys = keys_segmented.get_slice_info()
print(f"Overall number of key segments is {len(keys.index)}")
keys["localkey_fifths"] = transform(keys, roman_numeral2fifths, ['localkey', 'globalkey_is_minor'])
keys.head(5).style.apply(color_background, subset="localkey")
```

```{code-cell}
mode_slices = dc.ModeGrouper().process_data(keys_segmented)
```

### Whole dataset

```{code-cell}
mode_slices.get_slice_info()
```

```{code-cell}
chords_by_localkey = mode_slices.get_facet('expanded')
chords_by_localkey
```

```{code-cell}
for is_minor, df in chords_by_localkey.groupby(level=0, group_keys=False):
    df = df.droplevel(0)
    df = df[df.bass_note.notna()]
    sd = ms3.fifths2sd(df.bass_note).rename('sd')
    sd.index = df.index
    sd_progression = df.groupby(level=[0,1,2], group_keys=False).bass_note.apply(lambda S: S.shift(-1) - S).rename('sd_progression')
    if is_minor:
        chords_by_localkey_minor = pd.concat([df, sd, sd_progression], axis=1)
    else:
        chords_by_localkey_major = pd.concat([df, sd, sd_progression], axis=1)
```

## Scale degrees

```{code-cell}
chords_by_localkey_minor
```

```{code-cell}
fig = make_sunburst(chords_by_localkey_major, 'major')
save_figure_as(fig, "bass_degree_major_sunburst")
fig.show()
```

```{code-cell}
fig = make_sunburst(chords_by_localkey_minor, 'minor')
save_figure_as(fig, "bass_degree_minor_sunburst")
fig.show()
```

```{code-cell}
fig = rectangular_sunburst(chords_by_localkey_major, path=['sd', 'figbass', 'interval'], title="MAJOR")
save_figure_as(fig, "bass_degree-figbass-progression_major_sunburst")
fig.show()
```

```{code-cell}
fig = rectangular_sunburst(chords_by_localkey_major, path=['sd', 'interval', 'figbass'], title="MAJOR")
save_figure_as(fig, "bass_degree-progression-figbass_major_sunburst")
fig.show()
```

```{code-cell}
fig = rectangular_sunburst(chords_by_localkey_minor, path=['sd', 'figbass', 'interval'], title="MINOR")
save_figure_as(fig, "bass_degree-figbass-progression_minor_sunburst")
fig.show()
```

```{code-cell}
fig = rectangular_sunburst(chords_by_localkey_minor, path=['sd', 'interval', 'figbass'], title="MINOR")
save_figure_as(fig, "bass_degree-progression-figbass_minor_sunburst")
fig.show()
```

```{code-cell}
fig = rectangular_sunburst(chords_by_localkey_major, path=['sd', 'interval', 'figbass', 'following_figbass'], title="MAJOR")
save_figure_as(fig, "bass_degree-progression-figbass-subsequent_figbass_major_sunburst")
fig.show()
```

```{code-cell}
fig = rectangular_sunburst(chords_by_localkey_minor, path=['sd', 'interval', 'figbass', 'following_figbass'], title="MINOR")
save_figure_as(fig, "bass_degree-progression-figbass-subsequent_figbass_minor_sunburst")
fig.show()
```
