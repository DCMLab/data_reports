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

# Chromatic bass progressions

```{code-cell} ipython3
import os
import ms3
import pandas as pd
from helpers import cnt
pd.options.display.max_columns = 50
pd.options.display.max_rows = 100
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
CORPUS_PATH = "~/all_subcorpora/couperin_concerts"
RESULTS_PATH = os.path.abspath(os.path.join("..", "results"))
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
corpus_obj = ms3.Corpus(CORPUS_PATH)
corpus_obj.parse_tsv()
corpus_obj
```

```{code-cell} ipython3
labels = corpus_obj.expanded()
labels
```

## Transform `bass_note` column

+++

### Expressing all bass notes as scale degrees of global tonic
Since all scale degrees are expressed as fifths-intervals, this is as easy as adding the local key expressed as fifths

```{code-cell} ipython3
transpose_by = ms3.transform(labels, ms3.roman_numeral2fifths, ['localkey', 'globalkey_is_minor'])
bass = labels.bass_note + transpose_by
bass.head()
```

### Adding bass note names to DataFrame

```{code-cell} ipython3
transpose_by = ms3.transform(labels, ms3.name2fifths, ['globalkey'])
labels['bass_name'] = ms3.fifths2name(bass + transpose_by).values
labels.head()
```

### Calculating intervals between successive bass notes
Sloppy version: Include intervals across movement boundaries

#### Bass progressions expressed in fifths

```{code-cell} ipython3
bass = bass.bfill()
ivs = bass - bass.shift()
ivs.value_counts()
```

#### Bass progressions expressed in (enharmonic) semitones

```{code-cell} ipython3
pc_ivs = ms3.fifths2pc(ivs)
pc_ivs.index = ivs.index
pc_ivs = pc_ivs.where(pc_ivs <= 6, pc_ivs % -6).fillna(0)
pc_ivs.value_counts()
```

## Chromatic bass progressions

+++

### Successive descending semitones

```{code-cell} ipython3
desc = cnt(pc_ivs, -1)
desc.n.value_counts()
```

#### Storing those with three or more

```{code-cell} ipython3
three_desc = labels.loc[desc[desc.n > 2].ixs.sum()]
three_desc.to_csv(os.path.join(RESULTS_PATH, 'three_desc.tsv'), sep='\t')
three_desc.head(30)
```

#### Storing those with four or more

```{code-cell} ipython3
four_desc = labels.loc[desc[desc.n > 3].ixs.sum()]
four_desc.to_csv(os.path.join(RESULTS_PATH, 'four_desc.tsv'), sep='\t')
four_desc.head(30)
```

### Successive ascending semitones

```{code-cell} ipython3
asc = cnt(pc_ivs, 1)
asc.n.value_counts()
```

#### Storing those with three or more

```{code-cell} ipython3
three_asc = labels.loc[asc[asc.n > 2].ixs.sum()]
three_asc.to_csv(os.path.join(RESULTS_PATH, 'three_asc.tsv'), sep='\t')
three_asc.head(30)
```

#### Storing those with four or more

```{code-cell} ipython3
four_asc = labels.loc[asc[asc.n > 3].ixs.sum()]
four_asc.to_csv(os.path.join(RESULTS_PATH, 'four_asc.tsv'), sep='\t')
four_asc.head(30)
```

## Filtering for particular progressions with length >= 3
Finding only direct successors

```{code-cell} ipython3
def filtr(df, query, column='chord'):
    vals = df[column].to_list()
    n_grams = [t for t in zip(*(vals[i:] for i in range(len(query))))]
    if isinstance(query[0], str):
        lengths = [len(q) for q in query]
        n_grams = [tuple(e[:l] for e,l  in zip(t, lengths)) for t in n_grams]
    return query in n_grams

def show(df, query, column='chord'):
    selector = df.groupby(level=0).apply(filtr, query, column)
    return df[selector[df.index.get_level_values(0)].values]
```

### Descending

```{code-cell} ipython3
descending = pd.concat([labels.loc[ix_seq] for ix_seq in desc[desc.n > 2].ixs.values], keys=range((desc.n > 2).sum()))
descending
```

#### Looking for `Ger i64`

```{code-cell} ipython3
show(descending, ('Ger', 'i64'))
```

#### `i64`

```{code-cell} ipython3
show(descending, ('i64',))
```

#### `Ger V(64)`

```{code-cell} ipython3
show(descending, ('Ger', 'V(64'))
```

#### Bass degrees `b6 5 #4`

```{code-cell} ipython3
show(descending, (-4, 1, 6), 'bass_note')
```

### Ascending

```{code-cell} ipython3
ascending = pd.concat([labels.loc[ix_seq] for ix_seq in asc[asc.n > 2].ixs.values], keys=range((asc.n > 2).sum()))
ascending = ascending[ascending.label != '@none']
ascending
```

#### `i64 Ger`

```{code-cell} ipython3
show(ascending, ('i64', 'Ger'))
```

#### `i64`

```{code-cell} ipython3
show(ascending, ('i64',))
```

#### `V(64) Ger`

```{code-cell} ipython3
show(ascending, ('V(64)', 'Ger'))
```

#### Bass degrees `#4 5 b6`

```{code-cell} ipython3
show(ascending, (6, 1, -4), 'bass_note')
```
