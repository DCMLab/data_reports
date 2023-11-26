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
from dimcat.steps import groupers, analyzers

from utils import get_repo_name, print_heading, resolve_dir
```

```{code-cell}
from utils import DEFAULT_OUTPUT_FORMAT, OUTPUT_FOLDER
from dimcat.plotting import write_image
RESULTS_PATH = os.path.abspath(os.path.join(OUTPUT_FOLDER, "couperin_article"))
os.makedirs(RESULTS_PATH, exist_ok=True)
def make_output_path(filename):
    return os.path.join(RESULTS_PATH, f"{filename}{DEFAULT_OUTPUT_FORMAT}")
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
notes = D.get_feature("notes")
notes.plot()
```

```{code-cell}
result = notes.get_default_analysis()
result.make_pie_chart()
```

```{code-cell}
def make_overview_table(metadata: pd.DataFrame, groupby):
    gpb = metadata.groupby(groupby, sort=False)
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
    return overview_table

all_metadata = D.get_metadata().df
overview_table = make_overview_table(all_metadata, "workTitle")
overview_table
```

```{code-cell}
sum_row = pd.DataFrame(overview_table.sum(), columns=['sum'], dtype="Int64").T
absolute = pd.concat([overview_table, sum_row])
absolute
```

```{code-cell}
absolute.to_clipboard()
```

```{code-cell}
import logging
L = logging.getLogger("dimcat")
L.setLevel(logging.WARNING)
if not L.handlers:
  L.addHandler(logging.StreamHandler())
L.debug("TEST")
```

```{code-cell}
package_path = resolve_dir("../couperin_corelli.datapackage.json")
D = dc.Dataset.from_package(package_path)
D
```

```{code-cell}
CorpusG = groupers.CorpusGrouper()
grouped_D = dc.Pipeline([
    groupers.ModeGrouper(),
    CorpusG
]).process(D)
grouped_D
```

```{code-cell}
chord_labels = grouped_D.get_feature("HarmonyLabels")
unigram_durations = chord_labels.apply_step(analyzers.Proportions)
duration_ranking = unigram_durations.make_ranking_table(
  drop_cols=["chord_and_mode", "proportion"],
  top_k=0
)
duration_ranking
```

```{code-cell}
unigram_occurrences = chord_labels.apply_step("Counter")
occurrence_ranking = unigram_occurrences.make_ranking_table(
  drop_cols=["chord_and_mode", "proportion"],
  top_k=0
)
occurrence_ranking
```

```{code-cell}
duration_ranking.to_clipboard()
```

```{code-cell}
bass_notes = grouped_D.get_feature(dict(dtype="BassNotes", format="SCALE_DEGREE_MAJOR"))
bass_note_distribution = bass_notes.get_default_analysis()
bass_note_distribution
```

```{code-cell}
bass_note_distribution.make_bar_plot(output=make_output_path("bass_note_distribution_couperin_corelli"), height=1000)
```

```{code-cell}
print(f"Fraction covered by P1, P4, and P5:")
bass_note_distribution.combine_results().loc[pd.IndexSlice[:,:,[-1,0,1]]].groupby(level=[0,1]).proportion.sum().mul(100).round(1).astype(str).add(" %")
```

```{code-cell}
cadences = grouped_D.get_feature(dict(dtype="CadenceLabels", format="TYPE"))
cadences
```

```{code-cell}
cadences.plot_grouped(output=make_output_path("cadence_distribution_couperin_corelli"), font_size=40)
```
