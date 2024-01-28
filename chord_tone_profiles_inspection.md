---
jupytext:
  formats: md:myst,ipynb,py:percent
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: pydelta
  language: python
  name: pydelta
---

# Inspection of Chord-Tone Profiles

PCA, LDA etc.

```{code-cell} ipython3
%load_ext autoreload
%autoreload 2

import os

import pandas as pd
from dimcat.plotting import make_scatter_plot, write_image
from matplotlib import pyplot as plt
from sklearn import set_config
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.manifold import (
    MDS,
    TSNE,
    Isomap,
    LocallyLinearEmbedding,
    SpectralEmbedding,
)
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection

import utils

plt.rcParams["figure.dpi"] = 300

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
```

```{code-cell} ipython3
RESULTS_PATH = os.path.expanduser("~/git/diss/31_profiles/")
os.makedirs(RESULTS_PATH, exist_ok=True)


def make_output_path(filename, extension=None):
    return utils.make_output_path(
        filename, extension, RESULTS_PATH, use_subfolders=True
    )


def save_figure_as(fig, filename, directory=RESULTS_PATH, **kwargs):
    if not any(key in kwargs for key in ("height", "width")):
        kwargs["width"] = 1280
        kwargs["height"] = 720
    write_image(fig, filename, directory, **kwargs)
```

```{code-cell} ipython3
def make_info(corpus, name) -> str:
    info = f"{corpus.metadata.features}, {corpus.metadata.norm}"
    if name:
        info = f"{name} of the {info}"
    return info


data, metadata = utils.load_profiles()
PIECE_YEARS = metadata.get_composition_years().rename("mean_composition_year")
PIECE_MODE = metadata.annotated_key.str.islower().map({True: "minor", False: "major"})
```

Adapted from https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html

```{code-cell} ipython3
set_config(transform_output="default")


def get_embedding_objects(n_components=2, n_neighbors=30):
    return {
        "Random projection embedding": SparseRandomProjection(
            n_components=n_components, random_state=42
        ),
        "Truncated SVD embedding": TruncatedSVD(n_components=n_components),
        "Linear Discriminant Analysis embedding": LinearDiscriminantAnalysis(
            n_components=n_components
        ),
        "Isomap embedding": Isomap(n_neighbors=n_neighbors, n_components=n_components),
        "Standard LLE embedding": LocallyLinearEmbedding(
            n_neighbors=n_neighbors, n_components=n_components, method="standard"
        ),
        "Modified LLE embedding": LocallyLinearEmbedding(
            n_neighbors=n_neighbors, n_components=n_components, method="modified"
        ),
        "Hessian LLE embedding": LocallyLinearEmbedding(
            n_neighbors=n_neighbors,
            n_components=n_components,
            method="hessian",
            eigen_solver="dense",
        ),
        # "LTSA LLE embedding": LocallyLinearEmbedding(
        #     n_neighbors=n_neighbors, n_components=n_components, method="ltsa"
        # ),
        "MDS embedding": MDS(
            n_components=n_components, n_init=1, max_iter=120, n_jobs=2
        ),
        "Random Trees embedding": make_pipeline(
            RandomTreesEmbedding(n_estimators=200, max_depth=5, random_state=0),
            TruncatedSVD(n_components=n_components),
        ),
        "Spectral embedding": SpectralEmbedding(
            n_components=n_components, random_state=0, eigen_solver="arpack"
        ),
        "t-SNE embedding": TSNE(
            n_components=n_components,
            n_iter=500,
            n_iter_without_progress=150,
            n_jobs=2,
            random_state=0,
        ),
        "NCA embedding": NeighborhoodComponentsAnalysis(
            n_components=n_components, init="pca", random_state=0
        ),
    }


def make_embeddings(X, y, n_components=2, n_neighbors=30) -> pd.DataFrame:
    projections = {}
    embeddings = get_embedding_objects(n_components, n_neighbors)
    for name, transformer in embeddings.items():
        # if name.startswith("Linear Discriminant Analysis"):
        #     data = X.copy()
        #     data.flat[:: X.shape[1] + 1] += 0.01  # Make X invertible
        # else:
        #     data = X

        print(f"Computing {name}...")
        projections[name] = transformer.fit_transform(X, y)
    concatenate_this = {}
    marker_info = y.reset_index()  # columns "corpus, piece" and "group"
    for name, coordinates in projections.items():
        columns = ["x", "y"] if n_components == 2 else ["x", "y", "z"]
        df = pd.DataFrame(coordinates, columns=columns)
        concatenate_this[name] = pd.concat([df, marker_info], axis=1)
    scatter_data = pd.concat(concatenate_this, names=["embedding", "i"]).reset_index(0)
    return scatter_data


def show_projections(scatter_data, **kwargs):
    plot_settings = dict(
        df=scatter_data,
        x_col="x",
        y_col="y",
        color="group",
        hover_data=["corpus, piece", "group"],
        facet_col="embedding",
        facet_col_wrap=3,
        y_axis=dict(matches=None),
        x_axis=dict(matches=None),
        height=2000,
        **kwargs,
    )
    return make_scatter_plot(
        **plot_settings,
    )
```

```{code-cell} ipython3
corpus = data[("local_root_ct", "rootnorm")]
X, _, y, _ = utils.make_split(corpus, test_size=0)
scatter_data = make_embeddings(X, y, n_neighbors=18)
show_projections(
    scatter_data, title="Embeddings of local-root chord-tone profiles, rootnorm"
)
```

```{code-cell} ipython3
scaled_corpus = StandardScaler().fit_transform(corpus)
scatter_data = make_embeddings(scaled_corpus, y, n_neighbors=10)
show_projections(
    scatter_data,
    title="Embeddings of standardized local-root chord-tone profiles, rootnorm",
)
```

## Principal Component Analyses
### Chord profiles

```{code-cell} ipython3
data.keys()
```

```{code-cell} ipython3
def show_pca(feature, norm="piecenorm", **kwargs):
    global data
    corpus = data[(feature, norm)]
    info = make_info(corpus, "PCA")
    return utils.plot_pca(corpus, info=info, **kwargs)


show_pca("root_per_globalkey", color=PIECE_YEARS, symbol=PIECE_MODE)
```

```{code-cell} ipython3
show_pca("root_per_localkey", color=PIECE_YEARS, symbol=PIECE_MODE)
```

```{code-cell} ipython3
show_pca("root_per_tonicization", color=PIECE_YEARS, symbol=PIECE_MODE)
```

```{raw-cell}
## Pitch-class profiles
```

```{raw-cell}
show_pca("globalkey_profiles", color=PIECE_YEARS, symbol=PIECE_MODE)
```

```{raw-cell}
show_pca("localkey_profiles", color=PIECE_YEARS, symbol=PIECE_MODE)
```

```{raw-cell}
show_pca("tonicization_profiles", color=PIECE_YEARS, symbol=PIECE_MODE)
```

### Chord-tone profiles

```{code-cell} ipython3
show_pca("global_root_ct", color=PIECE_YEARS, symbol=PIECE_MODE)
```

```{code-cell} ipython3
show_pca("local_root_ct", color=PIECE_YEARS, symbol=PIECE_MODE)
```

```{code-cell} ipython3
show_pca("tonicization_root_ct", color=PIECE_YEARS, symbol=PIECE_MODE)
```

```{raw-cell}
## Create chord-tone profiles for multiple chord features

Tokens are `(feature, ..., chord_tone)` tuples.
```

```{raw-cell}
chord_reduced: resources.PrevalenceMatrix = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["chord_reduced_and_mode", "fifths_over_local_tonic"],
        index=["corpus", "piece"],
    )
)
print(f"Shape: {chord_reduced.shape}")
utils.replace_boolean_column_level_with_mode(chord_reduced)
```

```{raw-cell}
numerals: resources.PrevalenceMatrix = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["effective_localkey_is_minor", "numeral", "fifths_over_local_tonic"],
        index=["corpus", "piece"],
    )
)
print(f"Shape: {numerals.shape}")
utils.replace_boolean_column_level_with_mode(numerals)
```

```{raw-cell}
roots: resources.PrevalenceMatrix = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["root", "fifths_over_local_tonic"],
        index=["corpus", "piece"],
    )
)
print(f"Shape: {roots.shape}")
```

```{raw-cell}
root_per_globalkey = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["root_per_globalkey", "fifths_over_local_tonic"],
        index=["corpus", "piece"],
    )
)
print(f"Shape: {root_per_globalkey.shape}")
```

```{raw-cell}
fig = utils.plot_document_frequency(chord_reduced)
save_figure_as(fig, "document_frequency_of_chord_tones")
fig
```

```{raw-cell}
utils.plot_document_frequency(numerals, info="numerals")
```

```{raw-cell}
utils.plot_document_frequency(roots, info="roots")
```

```{raw-cell}
utils.plot_document_frequency(
    root_per_globalkey, info="root relative to global tonic"
)
```

```{raw-cell}
## Principal Component Analyses
```

```{raw-cell}
# chord_reduced.query("piece in ['op03n12a', 'op03n12b']").dropna(axis=1, how='all')
```

```{raw-cell}
metadata = D.get_metadata()
CORPUS_YEARS = utils.corpus_mean_composition_years(metadata)
PIECE_YEARS = metadata.get_composition_years().rename("mean_composition_year")
utils.plot_pca(
    chord_reduced.relative,
    info="chord-tone profiles of reduced chords",
    color=PIECE_YEARS,
)
```

```{raw-cell}
utils.plot_pca(
    chord_reduced.combine_results("corpus").relative,
    info="chord-tone profiles of reduced chords",
    color=CORPUS_YEARS,
    size=5,
)
```

```{raw-cell}
utils.plot_pca(
    numerals.relative, info="numeral profiles of numerals", color=PIECE_YEARS
)
```

```{raw-cell}
utils.plot_pca(
    numerals.combine_results("corpus").relative,
    info="chord-tone profiles of numerals",
    color=CORPUS_YEARS,
    size=5,
)
```

```{raw-cell}
utils.plot_pca(
    roots.relative, info="root profiles of chord roots (local)", color=PIECE_YEARS
)
```

```{raw-cell}
utils.plot_pca(
    roots.combine_results("corpus").relative,
    info="chord-tone profiles of chord roots (local)",
    color=CORPUS_YEARS,
    size=5,
)
```

```{raw-cell}
utils.plot_pca(
    root_per_globalkey.relative,
    info="root profiles of chord roots (global)",
    color=PIECE_YEARS,
)
```

```{raw-cell}
utils.plot_pca(
    root_per_globalkey.combine_results("corpus").relative,
    info="chord-tone profiles of chord roots (global)",
    color=CORPUS_YEARS,
    size=5,
)
```
