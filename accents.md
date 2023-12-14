---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
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
# import os
# from git import Repo
import dimcat as dc
import pandas as pd
# import ms3
import plotly.express as px

from utils import (CORPUS_COLOR_SCALE, STD_LAYOUT, corpus_mean_composition_years,
                   get_corpus_display_name, get_repo_name, print_heading, resolve_dir)
```

```{code-cell}
D = dc.Dataset.from_package("/home/laser/distant_listening_corpus/distant_listening_corpus.datapackage.json")
D
```

```{code-cell}
all_metadata = D.get_metadata()
assert len(all_metadata) > 0, "No pieces selected for analysis."
mean_composition_years = corpus_mean_composition_years(all_metadata)
chronological_order = mean_composition_years.index.to_list()
```

```{code-cell}
controls = D.get_feature("Articulation")
controls.df
```

```{code-cell}
chords = controls[controls.event == "Chord"]
```

```{code-cell}
articulation_counts = chords.articulation.value_counts(dropna=False).reset_index()
#articulation_counts.iat[0,0] = "no articulation"
articulation_counts
```

```{code-cell}
def print_accented_ratio(articulation_column, selected_accents):
    accent_mask = articulation_column.isin(selected_accents)
    n_accented = accent_mask.sum()
    n_chords = len(articulation_column)
    print(f"{n_accented} out of {n_chords} ({n_accented / n_chords:.1%}) positions carry an accent.")


selected_accents = {
    "articAccentAbove",
    "articAccentBelow",
    "articAccentStaccatoAbove",
    "articAccentStaccatoBelow",
    "articMarcatoAbove",
    "articMarcatoBelow",
    "articMarcatoStaccatoAbove",
    "articMarcatoStaccatoBelow",
    "articTenutoAccentAbove",
    "articTenutoAccentBelow",
}

print_accented_ratio(chords.articulation, selected_accents)
```

```{code-cell}
selected_including_staccatissimo = selected_accents.union({"articStaccatissimoAbove", "articStaccatissimoBelow"})
print_accented_ratio(chords.articulation, selected_including_staccatissimo)
```

```{code-cell}
def accented_ratio(articulation_column, selected_accents):
    accent_mask = articulation_column.isin(selected_accents)
    n_accented = accent_mask.sum()
    n_chords = len(articulation_column)
    return n_accented / n_chords
```

```{code-cell}
piecewise_ratios = chords.articulation.groupby(["corpus", "piece"]).apply(accented_ratio, selected_accents=selected_accents)
```

```{code-cell}
gpb = piecewise_ratios.groupby("corpus")
corpus_wise = pd.concat([gpb.mean().rename("mean"), gpb.sem().rename("stderr")], axis=1)
corpus_wise
```

```{code-cell}
px.bar(corpus_wise, y="mean", error_y="stderr")
```
