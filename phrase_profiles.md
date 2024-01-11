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

# Chord profiles for phrases in the DLC

ToDo: Inspect the 7 phrases that have 0-duration when drop_duplicated_ultima_rows=True:

```{raw-cell}
phrase_data = phrase_annotations.get_phrase_data(["chord_and_mode", "duration_qb"], components="phrase")
phrase_data.groupby(["corpus", "piece", "phrase_id"]).duration_qb.filter(lambda S: S.sum() == 0)
```

```{code-cell} ipython3
---
mystnb:
  code_prompt_hide: Hide imports
  code_prompt_show: Show imports
tags: [hide-cell]
---
%load_ext autoreload
%autoreload 2

import os
from typing import List, Optional

import dimcat as dc
import ms3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dimcat import resources
from dimcat.plotting import update_figure_layout, write_image
from dimcat.utils import get_middle_composition_year
from git import Repo
from sklearn.decomposition import PCA

import utils

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
```

```{code-cell} ipython3
RESULTS_PATH = os.path.abspath(os.path.join(utils.OUTPUT_FOLDER, "phrases"))
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

```{code-cell} ipython3
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
chronological_corpus_names = D.get_metadata().get_corpus_names(func=None)
D
```

```{code-cell} ipython3
phrase_annotations: resources.PhraseAnnotations = D.get_feature("PhraseAnnotations")
phrase_annotations
```

```{code-cell} ipython3
def make_phrase_data(
    phrase_annotations, columns, components=("body", "codetta"), **kwargs
):
    phrase_data = phrase_annotations.get_phrase_data(
        columns, components=components, **kwargs
    )
    return phrase_data


def make_phrase_bigram_table(
    phrase_annotations: resources.PhraseAnnotations, columns: str | List[str]
) -> resources.NgramTable:
    phrase_data = make_phrase_data(
        phrase_annotations,
        columns,
        components=(
            "body",
            "codetta",
        ),  # ideally using "phrase" but see 0-duration-ToDo at the top
        drop_levels="phrase_component",  # otherwise, no bigrams spanning body and codetta
    )
    phrase_bgt = phrase_data.apply_step("BigramAnalyzer")
    return phrase_bgt


def prepare_data(phrase_annotations, features, smooth=1e-20):
    if isinstance(features, str):
        features = [features]
    if "duration_qb" in features:
        columns = features
    else:
        columns = features + ["duration_qb"]
    phrase_data = make_phrase_data(phrase_annotations, columns)
    phrase_data_df = phrase_data.df.dropna()
    unigram_distribution = (
        phrase_data_df.groupby(features).duration_qb.sum().sort_values(ascending=False)
    )
    f = phrase_data_df.pivot_table(
        index=["corpus", "piece", "phrase_id"],
        columns=features,
        values="duration_qb",
        aggfunc="sum",
    )  # so-called "count frequency", here weighted by chord duration
    tf = f.fillna(0.0).add(smooth).div(f.sum(axis=1), axis=0)  # term frequency
    D, V = f.shape  # D = number of documents, V = vocabulary size
    df = f.notna().sum().sort_values(ascending=False)  # absolute document frequency
    f = f.fillna(0.0)
    idf = pd.Series(np.log(D / df), index=df.index)  # inverse document frequency
    return unigram_distribution, f, tf, df, idf


def plot_pca(
    data,
    info="data",
    show_features=20,
    color="corpus",
    symbol=None,
    size=None,
    **kwargs,
) -> Optional[go.Figure]:
    phrase_pca = PCA(3)
    decomposed_phrases = pd.DataFrame(
        phrase_pca.fit_transform(data), index=data.index, columns=["c1", "c2", "c3"]
    )
    print(
        f"Explained variance ratio: {phrase_pca.explained_variance_ratio_} "
        f"({phrase_pca.explained_variance_ratio_.sum():.1%})"
    )
    concatenate_this = [decomposed_phrases]
    hover_data = ["corpus", "piece"]
    if color is not None:
        if isinstance(color, pd.Series):
            concatenate_this.append(color)
            color = color.name
        hover_data.append(color)
    if symbol is not None:
        if isinstance(symbol, pd.Series):
            concatenate_this.append(symbol)
            symbol = symbol.name
        hover_data.append(symbol)
    if size is not None:
        if isinstance(size, pd.Series):
            concatenate_this.append(size)
            size = size.name
        hover_data.append(size)
    if len(concatenate_this) > 1:
        scatter_data = pd.concat(concatenate_this, axis=1).reset_index()
    else:
        scatter_data = decomposed_phrases
    fig = px.scatter_3d(
        scatter_data.reset_index(),
        x="c1",
        y="c2",
        z="c3",
        color=color,
        symbol=symbol,
        hover_data=hover_data,
        hover_name="phrase_id",
        title=f"3 principal components of the {info}",
        height=800,
        **kwargs,
    )
    marker_settings = dict(opacity=0.3)
    if size is None:
        marker_settings["size"] = 3
    update_figure_layout(
        fig,
        legend={"itemsizing": "constant"},
        traces_settings=dict(marker=marker_settings),
    )
    if show_features < 1:
        return fig
    fig.show()
    for i in range(3):
        component = pd.Series(
            phrase_pca.components_[i], index=data.columns, name="coefficient"
        ).sort_values(ascending=False, key=abs)
        fig = px.bar(
            component.iloc[:show_features],
            labels=dict(index="feature", value="coefficient"),
            title=f"{show_features} most weighted features of component {i+1}",
        )
        fig.show()
```

```{code-cell} ipython3
phrase_data = make_phrase_data(phrase_annotations, ["localkey_mode", "duration_qb"])
PHRASE_LENGTH = (
    phrase_data.groupby(["corpus", "piece", "phrase_id"])
    .duration_qb.sum()
    .rename("phrase_duration")
)
mode_proportions = phrase_data.groupby(
    ["corpus", "piece", "phrase_id", "localkey_mode"]
).duration_qb.sum()
PHRASE_MODE_BINARY = (
    mode_proportions.groupby(["corpus", "piece", "phrase_id"])
    .apply(lambda S: S.idxmax()[-1])
    .rename("localkey_mode")
)
PHRASE_MODE_TERNARY = (
    phrase_data.groupby(["corpus", "piece", "phrase_id"])
    .localkey_mode.unique()
    .map(lambda arr: "both" if len(arr) > 1 else arr[0])
)
PHRASE_COMPOSITION_YEAR = (
    get_middle_composition_year(D.get_metadata())
    .reindex(PHRASE_MODE_BINARY.index)
    .rename("year")
)
SCATTER_PLOT_SETTINGS = dict(
    color=PHRASE_MODE_TERNARY,
    color_discrete_map=dict(major="blue", minor="red", both="green"),
)
```

## Full chord symbols

```{code-cell} ipython3
unigram_distribution, f, tf, df, idf = prepare_data(
    phrase_annotations, "chord_and_mode"
)
unigram_distribution
```

### Full chord frequency matrix

Chord symbols carry their mode information, so it is to expected that modes be clearly separated.

```{code-cell} ipython3
plot_pca(tf, "chord frequency matrix", **SCATTER_PLOT_SETTINGS)
```

### Phrases entirely in major

```{code-cell} ipython3
px.histogram(PHRASE_LENGTH)
```

```{code-cell} ipython3
mode_tf = {group: df for group, df in tf.groupby(PHRASE_MODE_TERNARY)}
plot_pca(
    mode_tf["major"],
    "chord frequency matrix for phrases in major",
    color=PHRASE_COMPOSITION_YEAR,
    size=PHRASE_LENGTH / PHRASE_LENGTH.sum(),
)
```

```{code-cell} ipython3
plot_pca(
    mode_tf["minor"],
    "chord frequency matrix for phrases in minor",
    color=PHRASE_COMPOSITION_YEAR,
)
```

### PCA of tf-idf

```{code-cell} ipython3
plot_pca(tf.mul(idf), "tf-idf matrix", **SCATTER_PLOT_SETTINGS)
```

### For comparison: PCA of t-idf (absolute chord durations weighted by idf)

PCA consistently explains a multiple of the variance for f-idf compared to tf-idf (normalized chord weights)

```{code-cell} ipython3
plot_pca(f.fillna(0.0).mul(idf), "f-idf matrix", **SCATTER_PLOT_SETTINGS)
```

## Reduced chords (without suspensions, additions, alterations)

```{code-cell} ipython3
unigram_distribution, f, tf, df, idf = prepare_data(
    phrase_annotations, "chord_reduced_and_mode"
)
unigram_distribution
```

```{code-cell} ipython3
plot_pca(tf, "(reduced) chord frequency matrix", **SCATTER_PLOT_SETTINGS)
```

```{code-cell} ipython3
plot_pca(f.mul(idf), "f-idf matrix (reduced chords)", **SCATTER_PLOT_SETTINGS)
```

```{code-cell} ipython3
def plot_pca(
    data, info="data", show_features=20, color="corpus", symbol=None, **kwargs
) -> Optional[go.Figure]:
    phrase_pca = PCA(3)
    decomposed_phrases = pd.DataFrame(
        phrase_pca.fit_transform(data), index=data.index, columns=["c1", "c2", "c3"]
    )
    print(
        f"Explained variance ratio: {phrase_pca.explained_variance_ratio_} "
        f"({phrase_pca.explained_variance_ratio_.sum():.1%})"
    )
    concatenate_this = [decomposed_phrases]
    if color is not None:
        if isinstance(color, pd.Series):
            concatenate_this.append(color)
            color = color.name
    if symbol is not None:
        if isinstance(symbol, pd.Series):
            concatenate_this.append(symbol)
            symbol = symbol.name
    if len(concatenate_this) > 1:
        scatter_data = pd.concat(concatenate_this, axis=1).reset_index()
    else:
        scatter_data = decomposed_phrases
    fig = px.scatter_3d(
        scatter_data.reset_index(),
        x="c1",
        y="c2",
        z="c3",
        color=color,
        symbol=symbol,
        hover_data=["piece", "corpus"],
        hover_name="phrase_id",
        title=f"3 principal components of the {info}",
        height=800,
        **kwargs,
    )
    update_figure_layout(
        fig,
        legend={"itemsizing": "constant"},
        traces_settings=dict(marker_size=3, marker_opacity=0.3),
    )
    if show_features < 1:
        return fig
    fig.show()
    for i in range(3):
        component = pd.Series(
            phrase_pca.components_[i], index=data.columns, name="coefficient"
        ).sort_values(ascending=False, key=abs)
        fig = px.bar(
            component.iloc[:show_features],
            labels=dict(index="feature", value="coefficient"),
            title=f"{show_features} most weighted features of component {i+1}",
        )
        fig.show()
```

## Only root, regardless of chord type or inversion

PCA plot has straight lines. Th

```{code-cell} ipython3
unigram_distribution, f, tf, df, idf = prepare_data(phrase_annotations, "root")
unigram_distribution
```

```{code-cell} ipython3
plot_pca(tf, "root frequency matrix", **SCATTER_PLOT_SETTINGS)
```

## Grid search on variance explained by PCA components

```{raw-cell}
def do_pca_grid_search(
    data: pd.DataFrame,
    features: npt.ArrayLike,
    max_components: int = 10,
):
    max_features = len(features)
    n_columns = max_components if max_components > 0 else ceil(max_features / 2)
    grid_search = np.zeros((max_features, n_columns))
    for n_features in range(1, max_features + 1):
        print(n_features, end=" ")
        selected_features = features[:n_features]
        selected_data = data.loc(axis=1)[selected_features]
        if max_components > 0:
            up_to = min(max_components, n_features)
        else:
            up_to = ceil(n_features / 2)
        for n_components in range(1, up_to + 1):
            pca = PCA(n_components)
            _ = pca.fit_transform(selected_data)
            variance = pca.explained_variance_ratio_.sum()
            grid_search[n_features - 1, n_components - 1] = variance
            print(f"{variance:.1%}", end=" ")
        print()
    result = pd.DataFrame(
        grid_search,
        index=pd.RangeIndex(1, max_features + 1, name="features"),
        columns=pd.RangeIndex(1, n_columns + 1, name="components"),
    )
    return result


grid_search_by_occurrence = do_pca_grid_search(tf, df.index[:100])
```

```{raw-cell}
grid_search_by_duration = do_pca_grid_search(tf, unigram_distribution.index[:100])
```

```{raw-cell}
grid_search_by_duration - grid_search_by_occurrence
```
