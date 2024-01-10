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

# Cross entropies between corpora

ToDo:
* Uniqueness in Bezug auf das vorher Dagewesene
* Bigrams

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
from typing import Iterable, List, Optional

import dimcat as dc
import ms3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dimcat import resources
from dimcat.data.resources import Durations
from dimcat.data.resources.dc import UnitOfAnalysis
from dimcat.plotting import make_bar_plot, make_scatter_plot, write_image
from dimcat.utils import get_middle_composition_year
from git import Repo

import utils

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
```

```{code-cell}
RESULTS_PATH = os.path.abspath(os.path.join(utils.OUTPUT_FOLDER, "reduction"))
os.makedirs(RESULTS_PATH, exist_ok=True)


def make_output_path(
    filename: str,
    extension=None,
    path=RESULTS_PATH,
) -> str:
    return utils.make_output_path(
        filename,
        extension=extension,
        path=path,
    )


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
    grouped_absolute_values: resources.Durations,
    pivot_index,
    pivot_columns,
    pivot_values,
    smoothing=1e-20,
) -> pd.DataFrame:
    """Pivots the table, turning group chunks into columns, adds the smoothing value to each cell, and normalizes each
    column.
    """
    grouped_values = grouped_absolute_values.pivot_table(
        values=pivot_values, index=pivot_index, columns=pivot_columns, fill_value=0
    )
    if smoothing is not None:
        grouped_values = grouped_values.add(smoothing)
    group_probabilities = grouped_values.div(grouped_values.sum(axis=0), axis=1)
    return group_probabilities


def make_groupwise_probabilities(
    analysis_result: resources.Result,
    group_cols: Optional[UnitOfAnalysis | str | Iterable[str]] = UnitOfAnalysis.GROUP,
    smoothing: Optional[float] = 1e-20,
) -> pd.DataFrame:
    """Turns a Durations result (long format, absolute durations in quarter notes) into an NxM matrix of M probability
    distributions where N is the vocabulary size of the value_column (x_column) and M is the number of groups according
    to the group_cols argument. The columns are normalized after adding the smoothing value to each cell, to avoid
    log(0) errors.
    """
    group_cols = analysis_result._resolve_group_cols_arg(group_cols)
    grouped_results = analysis_result.combine_results(group_cols=group_cols)
    pivot_index = grouped_results.x_column
    pivot_columns = group_cols
    pivot_values = grouped_results.y_column
    group_probabilities = _make_groupwise_probabilities(
        grouped_results, pivot_index, pivot_columns, pivot_values, smoothing
    )
    return group_probabilities


def compute_cross_entropies(
    analysis_result: resources.Result,
    P_groups: UnitOfAnalysis | str | Iterable[str],
    Q_groups: Optional[UnitOfAnalysis | str | Iterable[str]] = None,
    smoothing: float = 1e-20,
):
    """If Q_groups is None, returns an PxP matrix of cross-entropies between all pairs of groups according to the
    P_groups argument. If Q_groups is specified, returns an PxQ matrix of cross-entropies between all pairs of groups
    according to the P_groups and Q_groups arguments. This is often interpreted as the information surplus that it takes
    to encode q using a code optimized for encoding p.
    """
    P_probs = make_groupwise_probabilities(analysis_result, P_groups, smoothing)
    if Q_groups is None:
        return _compute_cross_entropies(P_probs)
    Q_probs = make_groupwise_probabilities(analysis_result, Q_groups, smoothing)
    return _compute_cross_entropies(P_probs, Q_probs)
```

![uniqueness](img/uniqueness_white_fig2.6_p68.png)
Fig. 2.6 from White, C. (2022). The music in the data: Corpus analysis, music analysis, and tonal traditions
(1st ed.). Routledge. https://doi.org/10.4324/9781003285663 p. 68

```{code-cell}
def mean_of_other_groups(df, excluded_group: str) -> pd.Series:
    """Computes the mean (of cross-entropies or whatever) for each row but only after dropping the column named
    ``group``. In the context of :func:`compute_corpus_uniqueness`, this function is applied corpus-group-wise to
    compute means of all other corpora's cross-entropies, exluding the piece's corpus itself.
    """
    if excluded_group in df.columns:
        df = df.drop(
            excluded_group, axis=1
        )  # do not include corpus predicting its own pieces
    # piecewise_mean = df.mean(axis=1)
    all_values = df.melt()["value"]
    return pd.Series(
        {
            "corpus": utils.get_corpus_display_name(excluded_group),
            "uniqueness": all_values.mean(),
            "sem": all_values.sem(),
        }
    )


def compute_corpus_uniqueness(chord_proportions):
    """Computes the cross entropies between each piece relative to every corpus (i.e., corpus as model predicting the
    piece) and averages for the pieces of each corpus the cross-entropies relative to all other corpora except itself.
    """
    piece_by_corpus = compute_cross_entropies(chord_proportions, "piece", "corpus")
    corpus_uniqueness = pd.DataFrame(
        [
            mean_of_other_groups(df, corpus)
            for corpus, df in piece_by_corpus.groupby("corpus")
        ]
    )
    return corpus_uniqueness


def compute_chronological_corpus_uniqueness(
    chord_proportions,
    chronological_corpus_names: List[str],
):
    """Computes the corpus uniqueness as the mean predictability (cross entropy) of the historically older corpora.
    To
    """
    piece_by_corpus = compute_cross_entropies(chord_proportions, "piece", "corpus")
    piece_by_corpus = piece_by_corpus.loc[
        chronological_corpus_names, chronological_corpus_names
    ]
    groupby = piece_by_corpus.groupby("corpus")
    corpus_uniqueness = []
    for corpus in chronological_corpus_names:
        df = groupby.get_group(corpus)
        corpus_col_position = df.columns.get_loc(corpus)
        if corpus_col_position == 0:
            continue
        corpus_uniqueness.append(
            mean_of_other_groups(df.iloc(axis=1)[:corpus_col_position], corpus)
        )
    return pd.DataFrame(corpus_uniqueness)


def plot_uniqueness(
    chord_proportions,
    chronological_corpus_names: List[str],
    only_among_historically_older: bool = False,
):
    corpus_uniqueness = compute_corpus_uniqueness(chord_proportions)
    if only_among_historically_older:
        corpus_uniqueness_amongst_older = compute_chronological_corpus_uniqueness(
            chord_proportions, chronological_corpus_names
        )
        corpus_uniqueness = pd.concat(
            [corpus_uniqueness, corpus_uniqueness_amongst_older],
            keys=["all other", "historically older"],
            names=["relative to"],
        ).reset_index()
        color = "relative to"
    else:
        color = None
    display_names = [
        utils.get_corpus_display_name(c) for c in chronological_corpus_names
    ]
    return make_bar_plot(
        corpus_uniqueness,
        x_col="corpus",
        y_col="uniqueness",
        error_y="sem",
        color=color,
        title="Uniqueness of corpus pieces as average cross-entropy relative to other corpora",
        category_orders=dict(corpus=display_names),
        layout=dict(autosize=False),
        height=800,
        width=1200,
    )
```

![coherence](img/coherence_white_fig.2.6_p68.png)
Fig. 2.6 from White, C. (2022). The music in the data: Corpus analysis, music analysis, and tonal traditions
(1st ed.). Routledge. https://doi.org/10.4324/9781003285663 p. 68

```{code-cell}
def compute_corpus_incoherence(
    analysis_result,
):
    corpuswise_incoherence = []
    grouped_results = analysis_result.combine_results(group_cols=["corpus", "piece"])
    pivot_index = grouped_results.x_column
    pivot_values = grouped_results.y_column
    for corpus, df in grouped_results.groupby("corpus"):
        corpus_probs = _make_groupwise_probabilities(
            df,
            pivot_index=pivot_index,
            pivot_columns=("corpus", "piece"),
            pivot_values=pivot_values,
        )  # VxP (vocab size x pieces)
        piece_by_piece = _compute_cross_entropies(
            corpus_probs
        )  # PxP (each containing all other pieces "predicting"
        # the respective piece, i.e., the cross-entropy of a given piece's distribution relative to all other pieces)
        np.fill_diagonal(piece_by_piece.values, np.nan)  # exclude self-predictions
        # by_other_pieces = piece_by_piece.mean(axis=1)
        all_values = piece_by_piece.melt()["value"]
        corpuswise_incoherence.append(
            pd.Series(
                {
                    "corpus": utils.get_corpus_display_name(corpus),
                    "incoherence": all_values.mean(),
                    "sem": all_values.sem(),
                }
            )
        )
    return pd.DataFrame(corpuswise_incoherence)


def plot_incoherence(chord_proportions, chronological_corpus_names):
    corpus_incoherence = compute_corpus_incoherence(chord_proportions)
    display_names = [
        utils.get_corpus_display_name(c) for c in chronological_corpus_names
    ]
    return make_bar_plot(
        corpus_incoherence,
        x_col="corpus",
        y_col="incoherence",
        error_y="sem",
        title="Incoherence of corpus pieces as average cross-entropy relative to other pieces",
        category_orders=dict(corpus=display_names),
        layout=dict(autosize=False),
        height=800,
        width=1200,
    )
```

```{code-cell}
:tags: [hide-input]

package_path = utils.resolve_dir(
    "~/distant_listening_corpus/distant_listening_corpus.datapackage.json"
)
repo = Repo(os.path.dirname(package_path))
utils.print_heading("Data and software versions")
print(f"Data repo '{utils.get_repo_name(repo)}' @ {repo.commit().hexsha[:7]}")
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
chord_proportions.make_ranking_table()
```

```{code-cell}
def plot_corpus_by_corpus(
    result: resources.Result,
    chronological_corpus_names: List[str] = None,
) -> go.Figure:
    corpus_by_corpus = compute_cross_entropies(result, "corpus")
    if chronological_corpus_names:
        corpus_by_corpus_chronological = corpus_by_corpus.loc[
            chronological_corpus_names, chronological_corpus_names
        ]
    return px.imshow(
        corpus_by_corpus_chronological,
        color_continuous_scale="RdBu_r",
        title=f"Cross-entropy between corpora of distributions over chord {result.name.lower()}",
        width=1000,
        height=1000,
    )


chronological_corpus_names = analyzed_D.get_metadata().get_corpus_names(func=None)
plot_corpus_by_corpus(chord_proportions, chronological_corpus_names)
```

```{code-cell}
plot_uniqueness(chord_proportions, chronological_corpus_names, True)
```

```{code-cell}
plot_incoherence(chord_proportions, chronological_corpus_names)
```

```{code-cell}
def plot_uniqueness_incoherence(
    analysis_result: resources.Result, metadata: Optional[resources.Metadata] = None
) -> go.Figure:
    corpus_incoherence = compute_corpus_incoherence(analysis_result)
    corpus_uniqueness = compute_corpus_uniqueness(analysis_result)
    uniqueness_incoherence = corpus_uniqueness.merge(corpus_incoherence, on="corpus")
    if metadata is not None:
        mean_comp_years = get_middle_composition_year(metadata)
        corpus_features = mean_comp_years.groupby("corpus").agg(["mean", "size"])
        corpus_features.columns = ["mean_composition_year", "pieces"]
        corpus_features.index = corpus_features.index.map(utils.get_corpus_display_name)
        uniqueness_incoherence = uniqueness_incoherence.merge(
            corpus_features, on="corpus"
        )
        color = "mean_composition_year"
        size = "pieces"
    else:
        color = None
        size = None
    return make_scatter_plot(
        uniqueness_incoherence,
        x_col="uniqueness",
        y_col="incoherence",
        error_x="sem_x",
        error_y="sem_y",
        hover_data=["corpus"],
        color=color,
        color_continuous_scale="bluered",  # "sunsetdark", # "blackbody"
        size=size,
        title="Uniqueness vs. incoherence of corpus pieces",
        layout=dict(autosize=False),
        height=800,
        width=1200,
    )


plot_uniqueness_incoherence(chord_proportions, analyzed_D.get_metadata())
```

```{code-cell}
chord_bgt: resources.NgramTable = harmony_labels.apply_step("BigramAnalyzer")
chord_bigrams = chord_bgt.make_bigram_tuples()
chord_bigrams.make_ranking_table()
```

## Same evaluations based on chord bigrams

```{code-cell}
plot_corpus_by_corpus(chord_bigrams, chronological_corpus_names)
```

```{code-cell}
plot_uniqueness(chord_bigrams, chronological_corpus_names, True)
```

```{code-cell}
chord_bigrams.combine_results("corpus")
```

```{code-cell}
plot_incoherence(chord_bigrams, chronological_corpus_names)
```

```{code-cell}
plot_uniqueness_incoherence(chord_bigrams, analyzed_D.get_metadata())
```
