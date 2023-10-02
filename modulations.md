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

+++ {"jupyter": {"outputs_hidden": false}}

# Modulation Plans

Use for the `couperin_concerts` corpus only. Headings and function calls have been programatically generated for that
corpus.

```{code-cell} ipython3
from git import Repo

from utils import print_heading, resolve_dir, get_repo_name
%load_ext autoreload
%autoreload 2

from typing import Literal
import re
import ms3
import pandas as pd
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)

from create_gantt import create_modulation_plan, get_phraseends
```

```{code-cell} ipython3
# import os
# CORPUS_PATH = os.path.abspath(os.path.join('..', '..'))  # for running the notebook in the homepage deployment
# workflow
CORPUS_PATH = "~/all_subcorpora/couperin_concerts"         # for running the notebook locally
print_heading("Notebook settings")
print(f"CORPUS_PATH: {CORPUS_PATH!r}")
CORPUS_PATH = resolve_dir(CORPUS_PATH)
```

```{code-cell} ipython3
repo = Repo(CORPUS_PATH)
print_heading("Data and software versions")
print(f"Data repo '{get_repo_name(repo)}' @ {repo.commit().hexsha[:7]}")
print("dimcat version [NOT USED]")
print(f"ms3 version {ms3.__version__}")
```

```{code-cell} ipython3
corpus_obj = ms3.Corpus(CORPUS_PATH)
corpus_obj.view.include('facet', 'expanded')
corpus_obj.parse_tsv()
corpus_obj
```

```{code-cell} ipython3
md = corpus_obj.metadata()
md.head()
```

```{code-cell} ipython3
def make_modulation_plans(
    corpus_obj: ms3.Corpus,
    yaxis: Literal['semitones', 'fifths', 'numeral'] = 'semitones',
    regex = None
):
    for fname, piece in corpus_obj.iter_pieces():
        if regex is not None and not re.search(regex, fname):
            continue
        print(f"Creating modulation plan for {fname}...")
        at = piece.expanded()
        at = at[at.quarterbeats.notna() & (at.quarterbeats != "")]
        metadata = md.loc[fname]
        last_mn = metadata.last_mn
        try:
            globalkey = metadata.annotated_key
        except Exception:
            print('Global key is missing in the metadata.')
            globalkey = '?'
        data = ms3.make_gantt_data(at)
        if len(data) == 0:
            print(f"Could not create Gantt data for {fname}...")
            continue
        phrases = get_phraseends(at, "quarterbeats")
        data.sort_values(yaxis, ascending=False, inplace=True)
        fig = create_modulation_plan(data, title=f"{fname}", globalkey=globalkey, task_column=yaxis, phraseends=phrases)
        fig.show()
```

## c01

```{code-cell} ipython3
make_modulation_plans(corpus_obj, regex='c01')
```

## c02

```{code-cell} ipython3
make_modulation_plans(corpus_obj, regex='c02')
```

## c03

```{code-cell} ipython3
make_modulation_plans(corpus_obj, regex='c03')
```

## c04

```{code-cell} ipython3
make_modulation_plans(corpus_obj, regex='c04')
```

## c05

```{code-cell} ipython3
make_modulation_plans(corpus_obj, regex='c05')
```

## c06

```{code-cell} ipython3
make_modulation_plans(corpus_obj, regex='c06')
```

## c07

```{code-cell} ipython3
make_modulation_plans(corpus_obj, regex='c07')
```

## c08

```{code-cell} ipython3
make_modulation_plans(corpus_obj, regex='c08')
```

## c09

```{code-cell} ipython3
make_modulation_plans(corpus_obj, regex='c09')
```

## c10

```{code-cell} ipython3
make_modulation_plans(corpus_obj, regex='c10')
```

## c11

```{code-cell} ipython3
make_modulation_plans(corpus_obj, regex='c11')
```

## c14

```{code-cell} ipython3
make_modulation_plans(corpus_obj, regex='c14')
```

## parnasse

```{code-cell} ipython3
make_modulation_plans(corpus_obj, regex='parnasse')
```
