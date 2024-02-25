# ---
# jupyter:
#   jupytext:
#     formats: md:myst,ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: revamp
#     language: python
#     name: revamp
# ---

# %% [markdown]
# # Chord Profiles

# %% mystnb={"code_prompt_hide": "Hide imports", "code_prompt_show": "Show imports"} tags=["hide-cell"]
# %load_ext autoreload
# %autoreload 2
import os
from typing import Tuple, Union

import dimcat as dc
import ms3
import numpy as np
import pandas as pd
from dimcat import resources
from dimcat.plotting import make_bar_plot, write_image
from git import Repo
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram  # , linkage
from sklearn.cluster import AgglomerativeClustering

import utils
from dendrograms import Dendrogram, TableDocumentDescriber

plt.rcParams["figure.dpi"] = 300

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)

# %%
RESULTS_PATH = os.path.expanduser("~/git/diss/32_profiles/figs")
os.makedirs(RESULTS_PATH, exist_ok=True)


def make_output_path(filename, extension=None):
    if extension:
        extension = "." + extension.lstrip(".")
    else:
        extension = utils.DEFAULT_OUTPUT_FORMAT
    return os.path.join(RESULTS_PATH, f"{filename}{extension}")


def save_figure_as(fig, filename, directory=RESULTS_PATH, **kwargs):
    if not any(key in kwargs for key in ("height", "width")):
        kwargs["width"] = 1280
        kwargs["height"] = 720
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
D

# %%
chord_slices = utils.get_sliced_notes(D)
chord_slices.head(5)

# %% [markdown]
# ## Document frequencies of chord features

# %%
utils.compare_corpus_frequencies(
    chord_slices,
    [
        "chord_reduced_and_mode",
        ["effective_localkey_is_minor", "numeral"],
        "root",
        "root_fifths_over_global_tonic",
    ],
)

# %% [markdown]
# ## Create chord-tone profiles for multiple chord features
#
# Tokens are `(feature, ..., chord_tone)` tuples.

# %%
chord_reduced: resources.PrevalenceMatrix = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["chord_reduced_and_mode", "fifths_over_local_tonic"],
        index=["corpus", "piece"],
    )
)
print(f"Shape: {chord_reduced.shape}")

# %%
numerals: resources.PrevalenceMatrix = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["effective_localkey_is_minor", "numeral", "fifths_over_local_tonic"],
        index=["corpus", "piece"],
    )
)
print(f"Shape: {numerals.shape}")
utils.replace_boolean_column_level_with_mode(numerals)

# %%
roots: resources.PrevalenceMatrix = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["root", "fifths_over_local_tonic"],
        index=["corpus", "piece"],
    )
)
print(f"Shape: {roots.shape}")

# %%
root_fifths_over_global_tonic = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["root_fifths_over_global_tonic", "fifths_over_local_tonic"],
        index=["corpus", "piece"],
    )
)
print(f"Shape: {root_fifths_over_global_tonic.shape}")


# %%
def get_lower_triangle_values(data: Union[pd.DataFrame, np.array], offset: int = 0):
    is_dataframe = isinstance(data, pd.DataFrame)
    if is_dataframe:
        matrix = data.values
    else:
        matrix = data
    i, j = np.tril_indices_from(matrix, offset)
    values = matrix[i, j]
    if not is_dataframe:
        return values
    try:
        level_0 = utils.merge_index_levels(data.index[i])
        level_1 = utils.merge_index_levels(data.columns[j])
        index = pd.MultiIndex.from_arrays([level_0, level_1])
    except Exception:
        print(data.index[i], data.columns[j])
    return pd.Series(values, index=index)


# %%
cos_dist_chord_tones = utils.make_cosine_distances(
    chord_reduced.relative, standardize=False, flat_index=False
)
# np.fill_diagonal(cos_dist_chord_tones.values, np.nan)
cos_dist_chord_tones.iloc[:10, :10]

# %%
ABC = cos_dist_chord_tones.loc(axis=1)[["ABC"]]
ABC.shape


# %%
def cross_corpus_distances(
    group_of_columns: pd.DataFrame, group_name: str, group_level: int | str = 0
):
    rows = []
    for group, group_distances in group_of_columns.groupby(level=group_level):
        if group == group_name:
            i, j = np.tril_indices_from(group_distances, -1)
            distances = group_distances.values[i, j]
        else:
            distances = group_distances.values.flatten()
        mean_distance = np.mean(distances)
        sem = np.std(distances) / np.sqrt(distances.shape[0] - 1)
        row = pd.Series(
            {
                "corpus": utils.get_corpus_display_name(group),
                "mean_distance": mean_distance,
                "sem": sem,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


ABC_corpus_distances = cross_corpus_distances(ABC, "ABC")
make_bar_plot(
    ABC_corpus_distances.sort_values("mean_distance"),
    x_col="corpus",
    y_col="mean_distance",
    error_y="sem",
    title="Mean cosine distances between pieces of all corpora and ABC",
)

# %%
corpus_numerals: resources.PrevalenceMatrix = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["effective_localkey_is_minor", "numeral"],
        index="corpus",
    )
)
utils.replace_boolean_column_level_with_mode(corpus_numerals)
corpus_numerals.document_frequencies()

# %%
culled_chord_tones = chord_reduced.get_culled_matrix(1 / 3)
culled_chord_tones.shape

# %%
utils.plot_cosine_distances(culled_chord_tones.relative, standardize=True)


# %%
def linkage_matrix(model):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    return linkage_matrix


def plot_dendrogram(model, **kwargs):
    lm = linkage_matrix(model)
    dendrogram(lm, **kwargs)


cos_distance_matrix = utils.make_cosine_distances(culled_chord_tones.relative)
cos_distance_matrix

# %%
labels = cos_distance_matrix.index.to_list()
ac = AgglomerativeClustering(
    metric="precomputed", linkage="complete", distance_threshold=0, n_clusters=None
)
ac.fit_predict(cos_distance_matrix)
plot_dendrogram(ac, truncate_mode="level", p=0)
plt.title("Hierarchical Clustering using maximum cosine distances")
# plt.savefig('aggl_mvt_max_cos.png', bbox_inches='tight')

# %% [raw]
# sliced_notes.store_resource(
#     basepath="~/dimcat_data",
#     name="sliced_notes"
# )

# %% [raw]
# restored = dc.deserialize_json_file("/home/laser/dimcat_data/sliced_notes.resource.json")
# restored.df

# %%
ac.fit_predict(cos_distance_matrix)
lm = linkage_matrix(ac)  # probably want to use this to have better control
# lm = linkage(cos_distance_matrix)
describer = TableDocumentDescriber(D.get_metadata().reset_index())
plt.figure(figsize=(10, 60))
ddg = Dendrogram(lm, describer, labels)


# %%
def find_index_of_r1_r2(C: pd.Series) -> Tuple[int, int]:
    """Takes a Series representing C = 1 / (frequency(rank) - rank) and returns the indices of r1 and r2, left and
    right of the discontinuity."""
    r1_i = C.idxmax()
    r2_i = C.lt(0).idxmax()
    assert (
        r2_i == r1_i + 1
    ), f"Expected r1 and r2 to be one apart, but got r1_i = {r1_i}, r2_i = {r2_i}"
    return r1_i, r2_i


def compute_h(df) -> int | float:
    """Computes the h-point of a DataFrame with columns "rank" and "frequency" and returns the rank of the h-point.
    Returns a rank integer if a value with r = f(r) exists, otherwise rank float.
    """
    if (mask := df.frequency.eq(df["rank"])).any():
        h_ix = df.index[mask][0]
        return df.at[h_ix, "rank"]
    C = 1 / (df.frequency - df["rank"])
    r1_i, r2_i = find_index_of_r1_r2(C)
    (r1, f_r1), (r2, f_r2) = df.loc[[r1_i, r2_i], ["rank", "frequency"]].values
    return (f_r1 * r2 - f_r2 * r1) / (r2 - r1 + f_r1 - f_r2)
