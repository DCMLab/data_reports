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
from typing import Iterable, Optional

import dimcat as dc
import ms3
import numpy as np
import pandas as pd
import plotly.express as px
from dimcat import resources
from dimcat.data.resources import Durations
from dimcat.data.resources.dc import UnitOfAnalysis
from dimcat.plotting import make_bar_plot, make_scatter_plot, write_image
from git import Repo

from utils import (
    DEFAULT_OUTPUT_FORMAT,
    OUTPUT_FOLDER,
    get_corpus_display_name,
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

def _compute_cross_entropies(P_column_vectors, Q_column_vectors=None):
    """Expects an NxP matrix where the index has N observations and the P columns represent probability distributions
    of P groups. If Q is not specified, the result will contain the cross-entropies between all pairs of groups in a
    way that value at (i, j) corresponds to the cross_entropy H(p_i, p_j). In other words, each row contains the
    'predictive entropy' of all groups for the respective group and when i==j, the value is the entropy of the group.
    If Q is specified, it needs to be an NxQ matrix with Q columns representing probability distributions over the
    exact same N observations. In that case, each (i, j) value corresponds to the cross-entropy H(p_i, q_j), i.e.,
    each row contains the 'predictive entropy' of all Q groups for the respective P group.
    """
    if Q_column_vectors is None:
        msg_lengths = -np.log2(P_column_vectors)
    else:
        msg_lengths = -np.log2(Q_column_vectors)
    probabilities = P_column_vectors.T
    return (
        probabilities @ msg_lengths
    )  # dot product between probability rows with message length columns


def _make_groupwise_probabilities(
    grouped_absolute_values, pivot_index, pivot_columns, pivot_values, smoothing=1e-20
):
    grouped_values = grouped_absolute_values.pivot_table(
        values=pivot_values, index=pivot_index, columns=pivot_columns, fill_value=0
    )
    if smoothing is not None:
        grouped_values = grouped_values.add(smoothing)
    group_probabilities = grouped_values.div(grouped_values.sum(axis=0), axis=1)
    return group_probabilities


def make_groupwise_probabilities(
    analysis_result: resources.Durations,
    group_cols: Optional[UnitOfAnalysis | str | Iterable[str]] = UnitOfAnalysis.GROUP,
    smoothing: Optional[float] = 1e-20,
):
    group_cols = analysis_result._resolve_group_cols_arg(group_cols)
    grouped_results = analysis_result.combine_results(group_cols=group_cols)
    pivot_index = analysis_result.x_column
    pivot_columns = group_cols
    pivot_values = analysis_result.y_column
    group_probabilities = _make_groupwise_probabilities(
        grouped_results, pivot_index, pivot_columns, pivot_values, smoothing
    )
    return group_probabilities


def compute_cross_entropies(
    analysis_result: resources.Durations,
    P_groups: UnitOfAnalysis | str | Iterable[str],
    Q_groups: Optional[UnitOfAnalysis | str | Iterable[str]] = None,
    smoothing: float = 1e-20,
):
    P_probs = make_groupwise_probabilities(analysis_result, P_groups, smoothing)
    if Q_groups is None:
        return _compute_cross_entropies(P_probs)
    Q_probs = make_groupwise_probabilities(analysis_result, Q_groups, smoothing)
    return _compute_cross_entropies(P_probs, Q_probs)


def mean_of_other_groups(df, group):
    df = df.drop(group, axis=1)  # do include corpus predicting its own pieces
    piecewise_mean = df.mean(axis=1)
    return pd.Series(
        {
            "corpus": get_corpus_display_name(group),
            "uniqueness": piecewise_mean.mean(),
            "sem": piecewise_mean.sem(),
        }
    )


def compute_corpus_uniqueness(chord_proportions):
    piece_by_corpus = compute_cross_entropies(chord_proportions, "piece", "corpus")
    corpus_uniqueness = pd.DataFrame(
        [
            mean_of_other_groups(df, corpus)
            for corpus, df in piece_by_corpus.groupby("corpus")
        ]
    )
    return corpus_uniqueness


def plot_uniqueness(chord_proportions, chronological_corpus_names):
    corpus_uniqueness = compute_corpus_uniqueness(chord_proportions)
    return make_bar_plot(
        corpus_uniqueness,
        x_col="corpus",
        y_col="uniqueness",
        error_y="sem",
        title="Uniqueness of corpus pieces as average cross-entropy relative to other corpora",
        category_orders=dict(corpus=chronological_corpus_names),
        layout=dict(autosize=False),
        height=800,
        width=1200,
    )


def compute_corpus_coherence(
    chord_proportions,
):
    corpuswise_coherence = []
    for corpus, df in chord_proportions.groupby("corpus"):
        corpus_probs = _make_groupwise_probabilities(
            df,
            pivot_index="chord_and_mode",
            pivot_columns=("corpus", "piece"),
            pivot_values="duration_qb",
        )
        piece_by_piece = _compute_cross_entropies(corpus_probs)
        np.fill_diagonal(piece_by_piece.values, np.nan)  # exclude self-predictions
        by_other_pieces = piece_by_piece.mean(axis=1)
        corpuswise_coherence.append(
            pd.Series(
                {
                    "corpus": get_corpus_display_name(corpus),
                    "coherence": by_other_pieces.mean(),
                    "sem": by_other_pieces.sem(),
                }
            )
        )
    return pd.DataFrame(corpuswise_coherence)


def plot_coherence(chord_proportions, chronological_corpus_names):
    corpus_coherence = compute_corpus_coherence(chord_proportions)
    return make_bar_plot(
        corpus_coherence,
        x_col="corpus",
        y_col="coherence",
        error_y="sem",
        title="Coherence of corpus pieces as average cross-entropy relative to other pieces",
        category_orders=dict(corpus=chronological_corpus_names),
        layout=dict(autosize=False),
        height=800,
        width=1200,
    )
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
all_metadata = analyzed_D.get_metadata()
chronological_corpus_names = all_metadata.get_corpus_names()
```

```{code-cell}
chord_proportions: Durations = harmony_labels.apply_step("Proportions")
chord_proportions.make_ranking_table()
```

```{code-cell}
corpus_by_corpus = compute_cross_entropies(chord_proportions, "corpus")
px.imshow(corpus_by_corpus, color_continuous_scale="RdBu_r", width=1000, height=1000)
```

```{code-cell}
plot_uniqueness(chord_proportions, chronological_corpus_names)
```

```{code-cell}
plot_coherence(chord_proportions, chronological_corpus_names)
```

```{code-cell}
corpus_coherence = compute_corpus_coherence(chord_proportions)
corpus_uniqueness = compute_corpus_uniqueness(chord_proportions)
uniqueness_coherence = corpus_uniqueness.merge(corpus_coherence, on="corpus")
make_scatter_plot(
    uniqueness_coherence,
    x_col="uniqueness",
    y_col="coherence",
    error_x="sem_x",
    error_y="sem_y",
    hover_data=["corpus"],
    title="Uniqueness vs. coherence of corpus pieces",
    layout=dict(autosize=False),
    height=800,
    width=1200,
)
```
