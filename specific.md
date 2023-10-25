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

from utils import (CORPUS_COLOR_SCALE, corpus_mean_composition_years,
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
all_metadata = ms3.load_tsv("couperin_metadata.tsv", index_col=0)
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

```{code-cell}
gpb = all_metadata.groupby("workTitle", sort=False)
n_movements = gpb.size().rename('movements')
length_per_concert = gpb.length_qb.sum().round().astype('Int64').rename("length")
measures_per_concert = gpb.last_mn.sum().rename("measures")
notes_per_concert = gpb.n_onsets.sum().rename("notes")
labels_per_concert = gpb.label_count.sum().rename("labels")
overview_table = pd.concat([
  n_movements,
  measures_per_concert,
  length_per_concert,
  notes_per_concert,
  labels_per_concert
], axis=1)
overview_table
```

```{code-cell}
sum_row = pd.DataFrame(overview_table.sum(), columns=['sum'], dtype="Int64").T
absolute = pd.concat([overview_table, sum_row])
absolute
```

```{code-cell}
relative = absolute.div(n_movements, axis=0).round(1)
complete_overview_table = pd.concat([absolute, relative], axis=1, keys=['per concert', 'per piece'])
complete_overview_table
```

```{code-cell}
complete_overview_table.to_clipboard()
```
