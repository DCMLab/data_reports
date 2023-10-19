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

# Chromatic bass progressions

```{code-cell}
import os
import ms3
import pandas as pd
from git import Repo

from utils import cnt, print_heading, resolve_dir, get_repo_name, remove_none_labels, remove_non_chord_labels

pd.options.display.max_columns = 50
pd.options.display.max_rows = 100
```

```{code-cell}
from utils import OUTPUT_FOLDER, write_image
RESULTS_PATH = os.path.abspath(os.path.join(OUTPUT_FOLDER, "chromatic_bass"))
os.makedirs(RESULTS_PATH, exist_ok=True)
def save_figure_as(fig, filename, directory=RESULTS_PATH, **kwargs):
    write_image(fig, filename, directory, **kwargs)
```

```{code-cell}
# CORPUS_PATH = os.path.abspath(os.path.join('..', '..')) # for running the notebook in the homepage deployment workflow
CORPUS_PATH = "~/distant_listening_corpus/couperin_concerts"                # for running the notebook locally
print_heading("Notebook settings")
print(f"CORPUS_PATH: {CORPUS_PATH!r}")
CORPUS_PATH = resolve_dir(CORPUS_PATH)
```

```{code-cell}
repo = Repo(CORPUS_PATH)
print_heading("Data and software versions")
print(f"Data repo '{get_repo_name(repo)}' @ {repo.commit().hexsha[:7]}")
print("dimcat version [NOT USED]")
print(f"ms3 version {ms3.__version__}")
```

```{code-cell}
parse_obj = ms3.Parse(CORPUS_PATH)
annotated_view = parse_obj.get_view('annotated')
annotated_view.include('facets', 'expanded')
annotated_view.pieces_with_incomplete_facets = False
parse_obj.set_view(annotated_view)
parse_obj.parse_tsv(choose='auto')
parse_obj
```

```{code-cell}
labels = parse_obj.get_facet("expanded")
labels
```

#### Delete @none labels
This creates progressions between the label before and after the `@none` label that might not actually be perceived as transitions!

```{code-cell}
labels = remove_none_labels(labels)
```

#### Delete non-chord labels (typically, phrase labels)

```{code-cell}
labels = remove_non_chord_labels(labels)
```

## Transform `bass_note` column

+++

### Expressing all bass notes as scale degrees of global tonic
Since all scale degrees are expressed as fifths-intervals, this is as easy as adding the local key expressed as fifths

```{code-cell}
transpose_by = ms3.transform(labels, ms3.roman_numeral2fifths, ['localkey', 'globalkey_is_minor'])
bass = labels.bass_note + transpose_by
bass.head()
```

### Adding bass note names to DataFrame

```{code-cell}
transpose_by = ms3.transform(labels, ms3.name2fifths, ['globalkey'])
labels['bass_name'] = ms3.fifths2name(bass + transpose_by).values
labels.head()
```

### Calculating intervals between successive bass notes
Sloppy version: Include intervals across movement boundaries

#### Bass progressions expressed in fifths

```{code-cell}
bass = bass.bfill()
ivs = bass - bass.shift()
ivs.value_counts()
```

#### Bass progressions expressed in (enharmonic) semitones

```{code-cell}
pc_ivs = ms3.fifths2pc(ivs)
pc_ivs.index = ivs.index
pc_ivs = pc_ivs.where(pc_ivs <= 6, pc_ivs % -6).fillna(0)
pc_ivs.value_counts()
```

## Chromatic bass progressions

+++

### Successive descending semitones

```{code-cell}
desc = count_subsequent_occurrences(pc_ivs, -1)
desc.n.value_counts()
```

#### Storing those with three or more

```{code-cell}
three_desc = labels.loc[desc[desc.n > 2].ixs.sum()]
three_desc.to_csv(os.path.join(RESULTS_PATH, 'three_desc.tsv'), sep='\t')
three_desc.head(30)
```

#### Storing those with four or more

```{code-cell}
four_desc = labels.loc[desc[desc.n > 3].ixs.sum()]
four_desc.to_csv(os.path.join(RESULTS_PATH, 'four_desc.tsv'), sep='\t')
four_desc.head(30)
```

### Successive ascending semitones

```{code-cell}
asc = count_subsequent_occurrences(pc_ivs, 1)
asc.n.value_counts()
```

#### Storing those with three or more

```{code-cell}
three_asc = labels.loc[asc[asc.n > 2].ixs.sum()]
three_asc.to_csv(os.path.join(RESULTS_PATH, 'three_asc.tsv'), sep='\t')
three_asc.head(30)
```

#### Storing those with four or more

```{code-cell}
four_asc = labels.loc[asc[asc.n > 3].ixs.sum()]
four_asc.to_csv(os.path.join(RESULTS_PATH, 'four_asc.tsv'), sep='\t')
four_asc.head(30)
```

## Filtering for particular progressions with length >= 3
Finding only direct successors

```{code-cell}
def filtr(df, query, column='chord'):
    vals = df[column].to_list()
    n_grams = [t for t in zip(*(vals[i:] for i in range(len(query))))]
    if isinstance(query[0], str):
        lengths = [len(q) for q in query]
        try:
          n_grams = [tuple(e[:l] for e,l  in zip(t, lengths)) for t in n_grams]
        except Exception:
          print(n_grams)
          raise
    return query in n_grams

def show(df, query, column='chord'):
    selector = df.groupby(level=0).apply(filtr, query, column)
    return df[selector[df.index.get_level_values(0)].values]
```

### Descending

```{code-cell}
descending = pd.concat([labels.loc[ix_seq] for ix_seq in desc[desc.n > 2].ixs.values], keys=range((desc.n > 2).sum()))
descending
```

#### Looking for `Ger i64`

```{code-cell}
show(descending, ('Ger', 'i64'))
```

#### `i64`

```{code-cell}
show(descending, ('i64',))
```

#### `Ger V(64)`

```{code-cell}
show(descending, ('Ger', 'V(64'))
```

#### Bass degrees `b6 5 #4`

```{code-cell}
show(descending, (-4, 1, 6), 'bass_note')
```

### Ascending

```{code-cell}
ascending = pd.concat([labels.loc[ix_seq] for ix_seq in asc[asc.n > 2].ixs.values], keys=range((asc.n > 2).sum()))
ascending = ascending[ascending.label != '@none']
ascending
```

#### `i64 Ger`

```{code-cell}
show(ascending, ('i64', 'Ger'))
```

#### `i64`

```{code-cell}
show(ascending, ('i64',))
```

#### `V(64) Ger`

```{code-cell}
show(ascending, ('V(64)', 'Ger'))
```

#### Bass degrees `#4 5 b6`

```{code-cell}
show(ascending, (6, 1, -4), 'bass_note')
```
