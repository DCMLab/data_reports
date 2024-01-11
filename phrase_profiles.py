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
# # Chord profiles for phrases in the DLC

# %% mystnb={"code_prompt_hide": "Hide imports", "code_prompt_show": "Show imports"} tags=["hide-cell"]
# %load_ext autoreload
# %autoreload 2

import os
from math import ceil
from typing import List

import dimcat as dc
import ms3
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px
from dimcat import resources
from dimcat.plotting import write_image
from git import Repo
from sklearn.decomposition import PCA

import utils

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)

# %%
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
phrase_annotations: resources.PhraseAnnotations = D.get_feature("PhraseAnnotations")
phrase_annotations


# %%
def make_phrase_data(phrase_annotations, columns, components="phrase", **kwargs):
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
        components="phrase",
        drop_levels="phrase_component",  # otherwise, no bigrams spanning body and codetta
    )
    phrase_bgt = phrase_data.apply_step("BigramAnalyzer")
    return phrase_bgt


# %%
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
    idf = pd.Series(np.log(D / df), index=df.index)  # inverse document frequency
    return unigram_distribution, f, tf, df, idf


unigram_distribution, f, tf, df, idf = prepare_data(
    phrase_annotations, "chord_and_mode"
)
unigram_distribution


# %%
def plot_pca(data, info="data"):
    phrase_pca = PCA(3)
    decomposed_phrases = pd.DataFrame(phrase_pca.fit_transform(data), index=data.index)
    print(
        f"Explained variance: {phrase_pca.explained_variance_} ({phrase_pca.explained_variance_.sum():.1%})"
    )
    fig = px.scatter_3d(
        decomposed_phrases.reset_index(),
        x=0,
        y=1,
        z=2,
        color="corpus",
        hover_name="piece",
        title=f"3 principal components of the {info}",
    )
    return fig


plot_pca(tf, "chord frequency matrix")

# %%
plot_pca(tf.mul(idf), "tf-idf matrix")

# %%
unigram_distribution, f, tf, df, idf = prepare_data(
    phrase_annotations, "chord_reduced_and_mode"
)
unigram_distribution

# %%
plot_pca(tf, "chord frequency matrix")

# %%
plot_pca(tf.mul(idf), "tf-idf matrix")


# %%
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
            variance = pca.explained_variance_.sum()
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

# %%
grid_search_by_duration = do_pca_grid_search(tf, unigram_distribution.index[:100])

# %%
grid_search_by_duration - grid_search_by_occurrence
