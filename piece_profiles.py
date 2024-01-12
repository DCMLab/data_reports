# ---
# jupyter:
#   jupytext:
#     formats: md:myst,ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: revamp
#     language: python
#     name: revamp
# ---

# %% [markdown]
# # Chord profiles for pieces in the DLC

# %% mystnb={"code_prompt_hide": "Hide imports", "code_prompt_show": "Show imports"} tags=["hide-cell"]
# # %load_ext autoreload
# # %autoreload 2

import os
from typing import Optional

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

# %%
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


# %% tags=["hide-input"]
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

# %%
harmony_labels: resources.PhraseAnnotations = D.get_feature("HarmonyLabels")
harmony_labels


# %%
def prepare_data(harmony_labels, feature, smooth=1e-20):
    unigram_distribution = (
        harmony_labels.groupby(feature).duration_qb.sum().sort_values(ascending=False)
    )
    f = harmony_labels.pivot_table(
        index=["corpus", "piece"],
        columns=feature,
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
    piece_pca = PCA(3)
    decomposed_pieces = pd.DataFrame(
        piece_pca.fit_transform(data), index=data.index, columns=["c1", "c2", "c3"]
    )
    print(
        f"Explained variance ratio: {piece_pca.explained_variance_ratio_} "
        f"({piece_pca.explained_variance_ratio_.sum():.1%})"
    )
    concatenate_this = [decomposed_pieces]
    hover_data = ["corpus"]
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
        scatter_data = decomposed_pieces
    fig = px.scatter_3d(
        scatter_data.reset_index(),
        x="c1",
        y="c2",
        z="c3",
        color=color,
        symbol=symbol,
        hover_data=hover_data,
        hover_name="piece",
        title=f"3 principal components of the {info}",
        height=800,
        **kwargs,
    )
    marker_settings = dict(opacity=0.3)
    if size is None:
        marker_settings["size"] = 3
    update_figure_layout(
        fig,
        scene_dragmode="orbit",
        traces_settings=dict(marker=marker_settings),
    )
    if show_features < 1:
        return fig
    fig.show()
    for i in range(3):
        component = pd.Series(
            piece_pca.components_[i], index=data.columns, name="coefficient"
        ).sort_values(ascending=False, key=abs)
        fig = px.bar(
            component.iloc[:show_features],
            labels=dict(index="feature", value="coefficient"),
            title=f"{show_features} most weighted features of component {i+1}",
        )
        fig.show()


# %%
PIECE_LENGTH = (
    harmony_labels.groupby(["corpus", "piece"])
    .duration_qb.sum()
    .rename("piece_duration")
)
PIECE_MODE = harmony_labels.groupby(["corpus", "piece"]).globalkey_mode.first()
PIECE_COMPOSITION_YEAR = get_middle_composition_year(D.get_metadata()).rename("year")
SCATTER_PLOT_SETTINGS = dict(
    color=PIECE_MODE,
    color_discrete_map=dict(major="blue", minor="red", both="green"),
)

# %% [markdown]
# ## Full chord symbols

# %%
unigram_distribution, f, tf, df, idf = prepare_data(harmony_labels, "chord_and_mode")
unigram_distribution

# %% [markdown]
# ### Full chord frequency matrix
#
# Chord symbols carry their mode information, so it is to expected that modes be clearly separated.

# %%
plot_pca(tf, "chord frequency matrix", **SCATTER_PLOT_SETTINGS)

# %% [markdown]
# ### Phrases entirely in major

# %%
pl_log = np.log2(PIECE_LENGTH)
PL_NORM = pl_log.add(-pl_log.min()).div(pl_log.max() - pl_log.min())
px.histogram(PL_NORM)

# %%
mode_tf = {group: df for group, df in tf.groupby(PIECE_MODE)}
plot_pca(
    mode_tf["major"],
    "chord frequency matrix for pieces in major",
    color=PIECE_COMPOSITION_YEAR,
    size=PIECE_LENGTH,
)

# %%
mode_f = {group: df for group, df in f.groupby(PIECE_MODE)}
plot_pca(
    mode_f["major"],
    "chord proportion matrix for pieces in major",
    color=PIECE_COMPOSITION_YEAR,
    size=None,
)

# %%
plot_pca(
    mode_tf["minor"],
    "chord frequency matrix for pieces in minor",
    color=PIECE_COMPOSITION_YEAR,
)

# %%
plot_pca(
    mode_f["minor"],
    "chord proportions matrix for pieces in minor",
    color=PIECE_COMPOSITION_YEAR,
)

# %% [markdown]
# ### PCA of tf-idf

# %%
plot_pca(tf.mul(idf), "tf-idf matrix", **SCATTER_PLOT_SETTINGS)

# %% [markdown]
# ### For comparison: PCA of t-idf (absolute chord durations weighted by idf)
#
# PCA consistently explains a multiple of the variance for f-idf compared to tf-idf (normalized chord weights)

# %%
plot_pca(f.fillna(0.0).mul(idf), "f-idf matrix")

# %% [markdown]
# ## Reduced chords (without suspensions, additions, alterations)

# %%
unigram_distribution, f, tf, df, idf = prepare_data(
    harmony_labels, "chord_reduced_and_mode"
)
unigram_distribution

# %%
plot_pca(f.mul(idf), "f-idf matrix (reduced chords)", **SCATTER_PLOT_SETTINGS)

# %% [markdown]
# ## Only root, regardless of chord type or inversion
#
# PCA plot has straight lines. Th

# %%
unigram_distribution, f, tf, df, idf = prepare_data(harmony_labels, "root")
unigram_distribution

# %%
plot_pca(tf, "root frequency matrix")

# %% [markdown]
# ## Grid search on variance explained by PCA components

# %% [raw]
# def do_pca_grid_search(
#     data: pd.DataFrame,
#     features: npt.ArrayLike,
#     max_components: int = 10,
# ):
#     max_features = len(features)
#     n_columns = max_components if max_components > 0 else ceil(max_features / 2)
#     grid_search = np.zeros((max_features, n_columns))
#     for n_features in range(1, max_features + 1):
#         print(n_features, end=" ")
#         selected_features = features[:n_features]
#         selected_data = data.loc(axis=1)[selected_features]
#         if max_components > 0:
#             up_to = min(max_components, n_features)
#         else:
#             up_to = ceil(n_features / 2)
#         for n_components in range(1, up_to + 1):
#             pca = PCA(n_components)
#             _ = pca.fit_transform(selected_data)
#             variance = pca.explained_variance_ratio_.sum()
#             grid_search[n_features - 1, n_components - 1] = variance
#             print(f"{variance:.1%}", end=" ")
#         print()
#     result = pd.DataFrame(
#         grid_search,
#         index=pd.RangeIndex(1, max_features + 1, name="features"),
#         columns=pd.RangeIndex(1, n_columns + 1, name="components"),
#     )
#     return result
#
#
# grid_search_by_occurrence = do_pca_grid_search(tf, df.index[:100])

# %% [raw]
# grid_search_by_duration = do_pca_grid_search(tf, unigram_distribution.index[:100])

# %% [raw]
# grid_search_by_duration - grid_search_by_occurrence
