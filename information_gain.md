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

# Information gain of antecedents and consequents

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
from typing import Dict, Iterable, Tuple, TypeAlias

import dimcat as dc
import ms3
import pandas as pd
from dimcat import resources
from dimcat.plotting import make_bar_plot, write_image
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
RESULTS_PATH = os.path.abspath(os.path.join(OUTPUT_FOLDER, "reduction"))
os.makedirs(RESULTS_PATH, exist_ok=True)


def make_output_path(filename):
    return os.path.join(RESULTS_PATH, f"{filename}{DEFAULT_OUTPUT_FORMAT}")


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
    "KeySlicer",
    dict(dtype="BigramAnalyzer", features="BassNotes", format="FULL_WITHOUT_CONTEXT"),
]
analyzed_D = D.apply_step(*pipeline)
bigram_table = analyzed_D.get_result()
```

```{code-cell}
analyzed_D.get_feature("bassnotes")
```

## Bigrams (fast)

```{code-cell}
bigrams_df = bigram_table.make_bigram_df(
    ("bass_degree", "intervals_over_bass"),
    join_str=True,
    context_columns=("mc", "mc_onset"),
    terminal_symbols="DROP",
)
bigrams_df.to_csv(os.path.join(RESULTS_PATH, "bass_note_bigrams.tsv"), sep="\t")
```

```{code-cell}
bigram_tuples = bigram_table.make_bigram_tuples(
    ("bass_degree", "intervals_over_bass"),
    join_str=True,
    context_columns=("mc", "mc_onset"),
    terminal_symbols="DROP",
)
bass_note_bigram_counts = bigram_tuples.apply_step(
    dict(dtype="Counter", smallest_unit="GROUP")
)
bass_note_bigram_counts.to_csv(
    os.path.join(RESULTS_PATH, "bass_note_bigram_counts.tsv"), sep="\t"
)
```

```{code-cell}
antecedents_: TypeAlias = Iterable[str | Tuple[str]]


def compute_information_gains(
    bigram_table: resources.NgramTable,
    consequent: str | Tuple[str],
    compared_antecedents: antecedents_ | Dict[str | Tuple[str], antecedents_],
) -> pd.DataFrame:
    antecedents, categories, values = [], [], []
    if isinstance(compared_antecedents, dict):
        groupwise_dfs = {
            group: compute_information_gains(bigram_table, consequent, antecedents)
            for group, antecedents in compared_antecedents.items()
        }
        return pd.concat(groupwise_dfs, names=["group"]).reset_index(level=0)
    for antecedent in compared_antecedents:
        values.append(bigram_table.compute_information_gain(antecedent, consequent))
        if isinstance(antecedent, str):
            antecedents.append(antecedent)
            categories.append(antecedent)
        else:
            antecedents.append(", ".join(antecedent))
            categories.append(antecedent[0])
    return pd.DataFrame(
        {"category": categories, "antecedent": antecedents, "information_gain": values}
    )
```

```{code-cell}
antecedents = {
    "predictor": ["chord", "bass_note", "root"],
    "predictor + localkey": [
        ("bass_note", "localkey_mode"),
        ("root", "localkey_mode"),
    ],
    "predictor + intervals": [
        ("bass_note", "intervals_over_bass"),
        ("root", "intervals_over_root"),
    ],
    "predictor + localkey + intervals": [
        ("bass_note", "intervals_over_bass", "localkey_mode"),
        ("root", "intervals_over_root", "localkey_mode"),
    ],
}
ig_values = compute_information_gains(bigram_table, "chord", antecedents)
ig_values
```

```{code-cell}
make_bar_plot(
    ig_values,
    x_col="group",
    y_col="information_gain",
    color="category",
    title="Information gain of several predictors on the subsequent chord",
    barmode="group",
    labels=dict(category="predictor"),
)
```
