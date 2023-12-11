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
from typing import List

import dimcat as dc
import ms3
import pandas as pd
from dimcat.plotting import write_image
from dimcat.utils import grams
from dimcat import resources
from git import Repo
from scipy.stats import entropy

from utils import (
    DEFAULT_COLUMNS,
    DEFAULT_OUTPUT_FORMAT,
    OUTPUT_FOLDER,
    get_repo_name,
    make_key_region_summary_table,
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

def bigram_matrix(bigrams):
    """Expects columns 'a' and 'b'."""
    return (
        bigrams.groupby("antecedent")
        .consequent.value_counts()
        .unstack()
        .fillna(0)
        .astype(int)
    )


# bass_note_matrix = bigram_matrix(bigrams_df)


def normalized_entropy(matrix_or_series):
    """For matrices, compute normalized entropy for each row."""
    is_matrix = len(matrix_or_series.shape) > 1
    if is_matrix:
        result = matrix_or_series.apply(lambda x: entropy(x, base=2), axis=1)
    else:
        result = entropy(matrix_or_series, base=2)
    return result


def get_weighted_bigram_entropy(bigrams):
    """Expects columns 'antecedent' and 'consequent'."""
    unigram_frequencies = bigrams.antecedent.value_counts(normalize=True)
    matrix = bigram_matrix(bigrams)
    normalized_entropies = normalized_entropy(matrix)
    return (unigram_frequencies * normalized_entropies).sum()


def get_bigrams(column, remove_repetitions: bool = False):
    return bigram_table.make_bigram_table(column, terminal_symbols="DROP")


def compute_bigram_information_gain(
    column="bass_degree", remove_repetitions: bool = False
):
    """Compute information gain for knowing the previous token."""
    bigrams = get_bigrams(column)
    if remove_repetitions:
        bigrams = bigrams[bigrams.antecedent != bigrams.consequent]
    return bigram_information_gain(bigrams)


def bigram_information_gain(bigrams):
    target_entropy = normalized_entropy(bigrams.consequent.value_counts().astype(int))
    conditioned_entropy = get_weighted_bigram_entropy(bigrams)
    return target_entropy - conditioned_entropy
```

## DCML harmony labels

```{code-cell}
pipeline = [
    dict(dtype="HasHarmonyLabelsFilter", keep_values=[True]),
    "KeySlicer",
    "ModeGrouper",
    dict(dtype="BigramAnalyzer", features="BassNotes", format="FULL_WITHOUT_CONTEXT"),
]
analyzed_D = D.apply_step(*pipeline)
analyzed_D
bass_notes = analyzed_D.get_feature("BassNotes")
bass_notes
```

```{code-cell}
bigram_table: resources.NgramTable = analyzed_D.get_result()
bigram_table
```

## Bigrams (fast)

```{code-cell}
bigrams_df = bigram_table.make_bigram_table(
    ("bass_degree", "intervals_over_bass"),
    join_str=True,
    context_columns=("mc", "mc_onset"),
    terminal_symbols="DROP",
)
bigrams_df.to_csv(os.path.join(RESULTS_PATH, "bass_note_bigrams.tsv"), sep="\t")
```

```{code-cell}
bigrams_df.head()
```

```{code-cell}
bigram_tuples = bigram_table.make_bigram_tuples(
    ("bass_degree", "intervals_over_bass"),
    join_str=True,
    context_columns=("mc", "mc_onset"),
    terminal_symbols="DROP",
)
bigram_tuples
```

```{code-cell}
bass_note_bigram_counts = bigram_tuples.apply_step(
    dict(dtype="Counter", smallest_unit="GROUP")
)
bass_note_bigram_counts.to_csv(
    os.path.join(RESULTS_PATH, "bass_note_bigram_counts.tsv"), sep="\t"
)
```

```{code-cell}
transitions = bigram_table.get_transitions(("bass_degree", "intervals_over_bass"), join_str=True)
transitions
```

```{code-cell}
transitions.compute_information_gain(None)
```

```{code-cell}
transitions.make_ranking_table(None)
```

```{code-cell}
get_weighted_bigram_entropy(bigrams_df)
```

```{code-cell}
bass_notes.bass_note.nunique()
```

```{code-cell}
bass_notes.root.nunique()
```

```{code-cell}
compute_bigram_information_gain("bass_note")
```

```{code-cell}
compute_bigram_information_gain("root")
```

```{code-cell}
compute_bigram_information_gain("bass_note", remove_repetitions=True)
```

```{code-cell}
compute_bigram_information_gain("root", remove_repetitions=True)
```

```{code-cell}
compute_bigram_information_gain(("bass_note", "intervals_over_bass"))
```

```{code-cell}
compute_bigram_information_gain(("root", "intervals_over_root"))
```

## N-grams (slower)

```{code-cell}
def get_grams_from_segment(
    segment_df: pd.DataFrame,
    columns: str | List[str] = "bass_note",
    n: int = 2,
    fast: bool = True,
) -> pd.DataFrame:
    """Assumes that NA values occur only at the beginning. Fast means without retaining default columns."""
    if isinstance(columns, str):
        columns = [columns]
    if fast:
        selection = segment_df[columns].dropna(how="any")
    else:
        selection = segment_df[DEFAULT_COLUMNS + columns].dropna(how="any")
    value_sequence = list(selection[columns].itertuples(index=False, name=None))
    n_grams = grams(value_sequence, n=n)
    if len(n_grams) == 0:
        return pd.DataFrame()
    if n > 2:
        n_gram_iterator = list((tup[:-1], tup[-1]) for tup in n_grams)
    else:
        n_gram_iterator = n_grams
    if fast:
        return pd.DataFrame.from_records(n_gram_iterator, columns=["a", "b"])
    else:
        result = selection.iloc[: -n + 1][DEFAULT_COLUMNS]
        n_grams = pd.DataFrame.from_records(
            n_gram_iterator, columns=["a", "b"], index=result.index
        )
        return pd.concat([result, n_grams], axis=1)


def make_columns(S, n):
    list_of_tuples = S.iloc[0]
    if len(list_of_tuples) == 0:
        return pd.DataFrame()
    if n > 2:
        n_gram_iterator = list((tup[:-1], tup[-1]) for tup in list_of_tuples)
    else:
        n_gram_iterator = list_of_tuples
    return pd.DataFrame.from_records(n_gram_iterator, columns=["a", "b"])


def get_n_grams(
    df: pd.DataFrame,
    columns: str | List[str] = "bass_note",
    n: int = 2,
    fast: bool = True,
    **groupby_kwargs,
):
    if isinstance(columns, str):
        columns = [columns]
    groupby_kwargs = dict(groupby_kwargs, group_keys=fast)
    return df.groupby(**groupby_kwargs).apply(
        lambda df: get_grams_from_segment(df, columns=columns, n=n, fast=fast)
    )
    lists_of_tuples = df.groupby(**groupby_kwargs).apply(
        lambda df: list(
            df[columns].dropna(how="any").itertuples(index=False, name=None)
        )
    )
    n_grams = lists_of_tuples.map(lambda tup: grams(tup, n=n))
    return n_grams


def compute_n_gram_information_gain(
    df: pd.DataFrame,
    columns: str | List[str] = "bass_note",
    n: int = 2,
    fast: bool = True,
    **groupby_kwargs,
):
    if isinstance(columns, str):
        columns = [columns]
    groupby_kwargs = dict(groupby_kwargs, group_keys=fast)
    n_grams = get_n_grams(df, columns=columns, n=n, fast=fast, **groupby_kwargs)
    return bigram_information_gain(n_grams)


bass_notes["notna_segment"] = bass_notes.bass_note.isna().cumsum()
default_groupby = ["corpus", "fname", "localkey_slice", "notna_segment"]
```

```{code-cell}
compute_n_gram_information_gain(
    bass_notes, columns="bass_note", by=default_groupby, n=3
)
```

```{code-cell}
compute_n_gram_information_gain(bass_notes, columns="root", by=default_groupby, n=3)
```

```{code-cell}
compute_n_gram_information_gain(
    bass_notes, columns=["bass_note", "intervals_over_bass"], by=default_groupby, n=3
)
```

```{code-cell}
compute_n_gram_information_gain(
    bass_notes, columns=["root", "intervals_over_root"], by=default_groupby, n=3
)
```

```{code-cell}
key_regions = make_key_region_summary_table(
    bass_notes, level=[0, 1, 2], group_keys=False
)
```
