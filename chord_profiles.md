---
jupytext:
  formats: md:myst,ipynb,py:percent
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: revamp
  language: python
  name: revamp
---

# Chord profiles for pieces in the DLC

Initial exploration and experiments with chord profiles, PCA, RobustSclaer, and KMeans clustering.

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide imports
  code_prompt_show: Show imports
tags: [hide-cell]
---

import os

import dimcat as dc
import ms3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dimcat import analyzers, resources
from dimcat.plotting import write_image
from dimcat.utils import get_middle_composition_year
from git import Repo

from sklearn import set_config
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

import utils

%load_ext autoreload
%autoreload 2


set_config(transform_output="pandas")

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
```

```{code-cell}
RESULTS_PATH = os.path.abspath(os.path.join(utils.OUTPUT_FOLDER, "pieces"))
# os.makedirs(RESULTS_PATH, exist_ok=True)


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

```{code-cell}
harmony_labels: resources.PhraseAnnotations = D.get_feature("HarmonyLabels")
harmony_labels
```

```{code-cell}
chord_and_mode: resources.PrevalenceMatrix = analyzers.PrevalenceAnalyzer().process(
    harmony_labels
)
print(f"Shape: {chord_and_mode.shape}")
chord_and_mode.iloc[:10, :10]
```

```{code-cell}
chord_and_mode.metadata
```

```{code-cell}
PIECE_LENGTH = (
    harmony_labels.groupby(["corpus", "piece"])
    .duration_qb.sum()
    .rename("piece_duration")
)
```

```{code-cell}
pl = chord_and_mode.metadata.length_qb.rename("piece_duration")
pl
```

```{code-cell}
PIECE_LENGTH = (
    harmony_labels.groupby(["corpus", "piece"])
    .duration_qb.sum()
    .rename("piece_duration")
)
PIECE_MODE = harmony_labels.groupby(["corpus", "piece"]).globalkey_mode.first()
PIECE_COMPOSITION_YEAR = get_middle_composition_year(chord_and_mode.metadata).rename(
    "year"
)
SCATTER_PLOT_SETTINGS = dict(
    color=PIECE_MODE,
    color_discrete_map=dict(major="blue", minor="red", both="green"),
)
```

### Full chord frequency matrix

Chord symbols carry their mode information, so it is to expected that modes be clearly separated.

```{code-cell}
utils.plot_pca(
    data=chord_and_mode.relative, info="chord frequency matrix", **SCATTER_PLOT_SETTINGS
)
```

```{code-cell}
scaler = RobustScaler()
robust = scaler.fit_transform(chord_and_mode.relative)
utils.plot_pca(data=robust, info="chord frequency matrix", **SCATTER_PLOT_SETTINGS)
```

```{code-cell}
scaler = RobustScaler(quantile_range=(5, 95))
robust = scaler.fit_transform(chord_and_mode.relative)
utils.plot_pca(data=robust, info="chord frequency matrix", **SCATTER_PLOT_SETTINGS)
```

```{code-cell}
z = (
    chord_and_mode.absolute - chord_and_mode.absolute.mean()
) / chord_and_mode.absolute.std()
utils.plot_pca(data=z, info="chord frequency matrix", **SCATTER_PLOT_SETTINGS)
```

```{code-cell}
def plot_kmeans(data, n_clusters, cluster_data_itself: bool = False, **kwargs):
    pca_coordinates, pca = utils.get_component_analysis_coordinates(PCA(3), data)
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    if cluster_data_itself:
        kmeans.fit(data)
    else:
        kmeans.fit(pca_coordinates)
    fig = utils.plot_pca(pca_coordinates=pca_coordinates, show_features=0, **kwargs)
    cluster_labels = "cluster" + pd.Series(
        kmeans.labels_, index=data.index, name="cluster"
    ).astype(str)
    cluster_hulls = utils.get_hull_coordinates(pca_coordinates, cluster_labels)
    for clust, coordinates in cluster_hulls.items():
        fig.add_trace(
            go.Mesh3d(
                alphahull=0,
                opacity=0.1,
                x=coordinates.pca0,
                y=coordinates.pca1,
                z=coordinates.pca2,
                hoverinfo="skip",
            )
        )
    return fig


plot_kmeans(chord_and_mode.relative, 22, cluster_data_itself=False)
```

### Pieces in global major vs. minor

```{code-cell}
pl_log = np.log2(PIECE_LENGTH)
PL_NORM = pl_log.add(-pl_log.min()).div(pl_log.max() - pl_log.min())
px.histogram(PL_NORM, title="log-normalized phrase lengths")
```

```{code-cell}
mode_tf = {group: df for group, df in chord_and_mode.relative.groupby(PIECE_MODE)}
utils.plot_pca(
    mode_tf["major"],
    info="chord frequency matrix for pieces in major",
    # color=PIECE_COMPOSITION_YEAR,
    size=PL_NORM,
)
```

```{code-cell}
mode_f = {group: df for group, df in chord_and_mode.absolute.groupby(PIECE_MODE)}
utils.plot_pca(
    mode_f["major"],
    info="chord proportion matrix for pieces in major",
    # color=PIECE_COMPOSITION_YEAR,
    size=None,
)
```

```{code-cell}
utils.plot_pca(
    mode_tf["minor"],
    info="chord frequency matrix for pieces in minor",
    # color=PIECE_COMPOSITION_YEAR,
)
```

```{code-cell}
utils.plot_pca(
    mode_f["minor"],
    info="chord proportions matrix for pieces in minor",
    # color=PIECE_COMPOSITION_YEAR,
)
```

### PCA of tf-idf

```{code-cell}
utils.plot_pca(chord_and_mode.tf_idf(), info="tf-idf matrix", **SCATTER_PLOT_SETTINGS)
```

### For comparison: PCA of t-idf (absolute chord durations weighted by idf)

PCA consistently explains a multiple of the variance for f-idf compared to tf-idf (normalized chord weights)

```{code-cell}
utils.plot_pca(
    chord_and_mode.absolute.mul(chord_and_mode.inverse_document_frequencies()),
    info="tf-idf matrix",
    **SCATTER_PLOT_SETTINGS,
)
```

## Reduced chords (without suspensions, additions, alterations)

```{code-cell}
chord_reduced: resources.PrevalenceMatrix = harmony_labels.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns="chord_reduced_and_mode",
        index=["corpus", "piece"],
    )
)
chord_reduced.iloc[:10, :10]
```

```{code-cell}
chord_reduced.type_prevalence()
```

```{code-cell}
utils.plot_pca(
    chord_reduced.absolute.mul(chord_reduced.inverse_document_frequencies()),
    info="f-idf matrix (reduced chords)",
    **SCATTER_PLOT_SETTINGS,
)
```

## Only root, regardless of chord type or inversion

PCA plot has straight lines. Th

```{code-cell}
root: resources.PrevalenceMatrix = harmony_labels.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns="root",
        index=["corpus", "piece"],
    )
)
```

```{code-cell}
utils.plot_pca(root.relative, info="root frequency matrix")
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
