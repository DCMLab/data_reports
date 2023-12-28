---
jupytext:
  formats: md:myst,ipynb,py:percent
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

# Chord Profiles

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide imports
  code_prompt_show: Show imports
tags: [hide-cell]
---
# %load_ext autoreload
# %autoreload 2
import os

import dimcat as dc
import ms3
import pandas as pd
from dimcat.data.resources import Durations
from dimcat.plotting import write_image
from git import Repo

from utils import (
    DEFAULT_OUTPUT_FORMAT,
    OUTPUT_FOLDER,
    get_repo_name,
    print_heading,
    resolve_dir,
)

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
```

```{code-cell}
RESULTS_PATH = os.path.abspath(os.path.join(OUTPUT_FOLDER, "chord_profiles"))
os.makedirs(RESULTS_PATH, exist_ok=True)


def make_output_path(filename, extension=None):
    if extension:
        extension = "." + extension.lstrip(".")
    else:
        extension = DEFAULT_OUTPUT_FORMAT
    return os.path.join(RESULTS_PATH, f"{filename}{extension}")


def save_figure_as(fig, filename, directory=RESULTS_PATH, **kwargs):
    write_image(fig, filename, directory, **kwargs)
```

```{code-cell}
:tags: [hide-input]

package_path = resolve_dir(
    "~/distant_listening_corpus/distant_listening_corpus.datapackage.json"
)
repo = Repo(os.path.dirname(package_path))
print_heading("Data and software versions")
print(f"Data repo '{get_repo_name(repo)}' @ {repo.commit().hexsha[:7]}")
print(f"dimcat version {dc.__version__}")
print(f"ms3 version {ms3.__version__}")
D = dc.Dataset.from_package(package_path)
D
```

```{code-cell}
pipeline = [
    dict(dtype="HasHarmonyLabelsFilter", keep_values=[True]),
    "CorpusGrouper",
]
analyzed_D = D.apply_step(*pipeline)
harmony_labels = analyzed_D.get_feature("HarmonyLabels")
harmony_labels
```

```{code-cell}
chord_proportions: Durations = harmony_labels.apply_step("Proportions")
corpus_profiles = chord_proportions.make_ranking_table()
corpus_profiles.iloc[:50, :50]
```

```{code-cell}
piece2profile = {
    group: profile["duration_qb"].droplevel([0, 1])
    for group, profile in chord_proportions.groupby(["corpus", "piece"])
}
piece_profiles = pd.concat(piece2profile, axis=1, names=["corpus", "piece"])
```

```{code-cell}
piece_profiles.sort_index(
    key=lambda _: pd.Index(piece_profiles.notna().sum(axis=1)), ascending=False
).iloc[:50, :50]
```

```{code-cell}
piece_profiles.sort_index(
    key=lambda _: pd.Index(piece_profiles.sum(axis=1)), ascending=False
).iloc[:50, :50]
```

```{code-cell}
corpus_proportions = chord_proportions.combine_results("corpus")
corpus_proportions
```

```{code-cell}
corpus2profile = {
    group: profile["duration_qb"].droplevel(0)
    for group, profile in corpus_proportions.groupby("corpus")
}
corpus_profiles = pd.concat(corpus2profile, axis=1, names=["corpus"])
chord_occurrence_mask = corpus_profiles.notna()
corpus_frequency = chord_occurrence_mask.sum(axis=1)
chord_occurrence_mask = chord_occurrence_mask.sort_index(
    key=lambda _: pd.Index(corpus_frequency), ascending=False
)
chord_occurrence_mask = chord_occurrence_mask.sort_index(
    key=lambda _: pd.Index(chord_occurrence_mask.sum()), ascending=False, axis=1
)
corpus_profiles.sort_index(key=lambda _: pd.Index(corpus_frequency), ascending=False)
```

```{code-cell}
mask_with_sum = pd.concat(
    [chord_occurrence_mask, chord_occurrence_mask.sum(axis=1).rename("sum")], axis=1
)
mask_with_sum.to_csv(make_output_path("chord_occurrence_mask", "tsv"), sep="\t")
```
