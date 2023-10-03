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

# Modulation Plans

Use for the `mozart_piano_sonatas` corpus only. Headings and function calls have been programatically generated for that
corpus.

````{raw-cell}
# To generate the (MyST) notebook cells with function calls for the `mozart_piano_sonatas` corpus

import os
folder = os.path.abspath("../../MS3")
sonatas = set(f[:4] for f in os.listdir(folder) if f.endswith('.mscx'))
for name in sorted(sonatas):
    print(f"""## {name}

```{{code-cell}} ipython3
make_modulation_plans(corpus_obj, regex='{name}')
```
""")
````

```{code-cell}
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

```{code-cell}
# import os
# CORPUS_PATH = os.path.abspath(os.path.join('..', '..'))  # for running the notebook in the homepage deployment
# workflow
CORPUS_PATH = "~/all_subcorpora/mozart_piano_sonatas"         # for running the notebook locally
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
corpus_obj = ms3.Corpus(CORPUS_PATH)
corpus_obj.view.include('facet', 'expanded')
corpus_obj.parse_tsv()
corpus_obj
```

```{code-cell}
md = corpus_obj.metadata()
md.head()
```

```{code-cell}
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

## K279

```{code-cell}
make_modulation_plans(corpus_obj, regex='K279')
```

## K280

```{code-cell}
make_modulation_plans(corpus_obj, regex='K280')
```

## K281

```{code-cell}
make_modulation_plans(corpus_obj, regex='K281')
```

## K282

```{code-cell}
make_modulation_plans(corpus_obj, regex='K282')
```

## K283

```{code-cell}
make_modulation_plans(corpus_obj, regex='K283')
```

## K284

```{code-cell}
make_modulation_plans(corpus_obj, regex='K284')
```

## K309

```{code-cell}
make_modulation_plans(corpus_obj, regex='K309')
```

## K310

```{code-cell}
make_modulation_plans(corpus_obj, regex='K310')
```

## K311

```{code-cell}
make_modulation_plans(corpus_obj, regex='K311')
```

## K330

```{code-cell}
make_modulation_plans(corpus_obj, regex='K330')
```

## K331

```{code-cell}
make_modulation_plans(corpus_obj, regex='K331')
```

## K332

```{code-cell}
make_modulation_plans(corpus_obj, regex='K332')
```

## K333

```{code-cell}
make_modulation_plans(corpus_obj, regex='K333')
```

## K457

```{code-cell}
make_modulation_plans(corpus_obj, regex='K457')
```

## K533

```{code-cell}
make_modulation_plans(corpus_obj, regex='K533')
```

## K545

```{code-cell}
make_modulation_plans(corpus_obj, regex='K545')
```

## K570

```{code-cell}
make_modulation_plans(corpus_obj, regex='K570')
```

## K576

```{code-cell}
make_modulation_plans(corpus_obj, regex='K576')
```
