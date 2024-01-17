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
# # Chord Profiles

# %% mystnb={"code_prompt_hide": "Hide imports", "code_prompt_show": "Show imports"} tags=["hide-cell"]
# # %load_ext autoreload
# # %autoreload 0
import os
from importlib import reload
from typing import Iterable, List, Literal, Optional, Tuple

import dimcat as dc
import ms3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dimcat import resources, slicers
from dimcat.data.resources.features import BassNotesFormat
from dimcat.data.resources.utils import (
    join_df_on_index,
    merge_columns_into_one,
    transpose_notes_to_c,
)
from dimcat.plotting import make_bar_plot, make_scatter_plot, write_image
from git import Repo
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram  # , linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import Normalizer, StandardScaler

import utils
from dendrograms import Dendrogram, TableDocumentDescriber

plt.rcParams["figure.dpi"] = 300

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)

# %%
RESULTS_PATH = os.path.expanduser("~/git/diss/31_profiles/figs")
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


# %%
def sort_tpcs(
    tpcs: Iterable[int], ascending: bool = True, start: Optional[int] = None
) -> list[int]:
    """Sort tonal pitch classes by order on the piano.

    Args:
        tpcs: Tonal pitch classes to sort.
        ascending: Pass False to sort by descending order.
        start: Start on or above this TPC.
    """
    result = sorted(tpcs, key=lambda x: (ms3.fifths2pc(x), -x))
    if start is not None:
        pcs = ms3.fifths2pc(result)
        start = ms3.fifths2pc(start)
        i = 0
        while i < len(pcs) - 1 and pcs[i] < start:
            i += 1
        result = result[i:] + result[:i]
    return result if ascending else list(reversed(result))


def plot_chord_profiles(
    chord_profiles,
    chord_and_mode,
    xaxis_format: Optional[BassNotesFormat] = BassNotesFormat.SCALE_DEGREE,
    **kwargs,
):
    chord, mode = chord_and_mode.split(", ")
    minor = mode == "minor"
    chord_tones = ms3.chord2tpcs(chord, minor=minor)
    bass_note = chord_tones[0]
    profiles = chord_profiles.query(
        f"chord_and_mode == '{chord_and_mode}'"
    ).reset_index()
    tpc_order = sort_tpcs(profiles.fifths_over_local_tonic.unique(), start=bass_note)

    if xaxis_format is None or xaxis_format == BassNotesFormat.FIFTHS:
        x_col = "fifths_over_local_tonic"
        category_orders = None
    else:
        x_values = profiles.fifths_over_local_tonic
        format = BassNotesFormat(xaxis_format)
        if format == BassNotesFormat.SCALE_DEGREE:
            profiles["Scale degree"] = ms3.transform(
                x_values, ms3.fifths2sd, minor=minor
            )
            x_col = "Scale degree"
            category_orders = {"Scale degree": ms3.fifths2sd(tpc_order, minor=minor)}

    fig = make_bar_plot(
        profiles,
        x_col=x_col,
        y_col="proportion",
        color="corpus",
        title=f"Chord profiles of {chord_and_mode}",
        category_orders=category_orders,
        **kwargs,
    )
    return fig


def prepare_chord_tone_data(
    chord_slices: pd.DataFrame,
    groupby: str | List[str],
    chord_and_mode: Optional[str | Iterable[str]] = None,
    smooth=1e-20,
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if isinstance(groupby, str):
        groupby = [groupby]
    if chord_and_mode is None:
        return utils.prepare_tf_idf_data(
            chord_slices,
            index=groupby,
            columns=["chord_and_mode", "fifths_over_local_tonic"],
            smooth=smooth,
        )
    if isinstance(chord_and_mode, str):
        absolute_data = chord_slices.query(f"chord_and_mode == '{chord_and_mode}'")
        return utils.prepare_tf_idf_data(
            absolute_data,
            index=groupby,
            columns=["fifths_over_local_tonic"],
            smooth=smooth,
        )
    results = [
        prepare_chord_tone_data(
            chord_slices, groupby=groupby, chord_and_mode=cm, smooth=smooth
        )
        for cm in chord_and_mode
    ]
    concatenated_results = []
    for tup in zip(*results):
        concatenated_results.append(pd.concat(tup, axis=1, keys=chord_and_mode))
    return tuple(concatenated_results)


def prepare_numeral_chord_tone_data(
    chord_slices: pd.DataFrame,
    groupby: str | List[str],
    numeral: Optional[str | Iterable[str]] = None,
    smooth=1e-20,
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if isinstance(groupby, str):
        groupby = [groupby]
    if numeral is None:
        return utils.prepare_tf_idf_data(
            chord_slices,
            index=groupby,
            columns=["numeral", "fifths_over_local_tonic"],
            smooth=smooth,
        )
    if isinstance(numeral, str):
        absolute_data = chord_slices.query(f"numeral == '{numeral}'")
        return utils.prepare_tf_idf_data(
            absolute_data,
            index=groupby,
            columns=["numeral", "fifths_over_local_tonic"],
            smooth=smooth,
        )
    results = [
        prepare_numeral_chord_tone_data(
            chord_slices, groupby=groupby, numeral=cm, smooth=smooth
        )
        for cm in numeral
    ]
    concatenated_results = []
    for tup in zip(*results):
        concatenated_results.append(pd.concat(tup, axis=1, keys=numeral))
    return tuple(concatenated_results)


def make_chord_tone_profile(chord_slices: pd.DataFrame) -> pd.DataFrame:
    """Chord tone profiles in long format. Come with the absolute column 'duration_qb' and the
    relative column 'proportion', normalized per chord per corpus.
    """
    chord_tone_profiles = chord_slices.groupby(
        ["corpus", "chord_and_mode", "fifths_over_local_tonic"]
    ).duration_qb.sum()
    normalization = chord_tone_profiles.groupby(["corpus", "chord_and_mode"]).sum()
    chord_tone_profiles = pd.concat(
        [
            chord_tone_profiles,
            chord_tone_profiles.div(normalization).rename("proportion"),
        ],
        axis=1,
    )
    return chord_tone_profiles


def compare_corpus_frequencies(chord_slices, features: str | Iterable[str]):
    if isinstance(features, str):
        features = [features]
    doc_freqs = []
    for feature in features:
        _, _, _, df, _ = utils.prepare_tf_idf_data(
            chord_slices,
            index=["corpus"],
            columns=feature,
        )
        doc_freqs.append(df.astype("Int64").rename("corpus_frequency"))
    if len(doc_freqs) == 1:
        return doc_freqs[0]
    return pd.concat([series.reset_index() for series in doc_freqs], axis=1)


def make_cosine_distances(
    tf, standardize=True, norm: Optional[Literal["l1", "l2", "max"]] = None
):
    if standardize:
        scaler = StandardScaler()
        scaler.set_output(transform="pandas")
        tf = scaler.fit_transform(tf)
    if norm:
        scaler = Normalizer(norm=norm)
        scaler.set_output(transform="pandas")
        tf = scaler.fit_transform(tf)
    index = utils.merge_index_levels(tf.index)
    distance_matrix = pd.DataFrame(cosine_distances(tf), index=index, columns=index)
    return distance_matrix


def plot_cosine_distances(tf: pd.DataFrame, standardize=True):
    cos_distance_matrix = make_cosine_distances(tf, standardize=standardize)
    fig = go.Figure(
        data=go.Heatmap(
            z=cos_distance_matrix,
            x=cos_distance_matrix.columns,
            y=cos_distance_matrix.index,
            colorscale="Blues",
            colorbar=dict(title="Cosine distance"),
            zmax=1.0,
        ),
        layout=dict(
            title="Piece-wise cosine distances between chord-tone profiles",
            width=1000,
            height=1000,
        ),
    )
    return fig


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
metadata = D.get_metadata()
CORPUS_YEARS = utils.corpus_mean_composition_years(metadata)
CORPUS_YEARS
D

# %%
label_slicer = slicers.HarmonyLabelSlicer()
sliced_D = label_slicer.process(D)

# %%
sliced_notes = sliced_D.get_feature(resources.Notes)
sliced_notes

# %%
slice_info = label_slicer.slice_metadata.droplevel(-1)
merge_columns = [col for col in slice_info.columns if col not in sliced_notes.columns]
slice_info = join_df_on_index(
    slice_info[merge_columns], sliced_notes.index, how="right"
)
chord_slices = pd.concat([sliced_notes, slice_info], axis=1)
chord_slices = pd.concat([chord_slices, transpose_notes_to_c(chord_slices)], axis=1)
chord_slices

# %% [markdown]
# ### Corpus frequencies of chord labels

# %%
corpus_frequencies = compare_corpus_frequencies(
    chord_slices, ["chord_and_mode", "chord_reduced_and_mode", "numeral"]
)
corpus_frequencies

# %%
corpus_frequencies.iloc[:30].to_clipboard()

# %% [markdown]
# ### Vocabulary
#
# Features are `(chord, chord_tone)` tuples.

# %%
full_unigrams, f, tf, df, idf = prepare_chord_tone_data(
    chord_slices,
    groupby=["corpus", "piece"],
)
print(f"Shape: {tf.shape}")


# %%
full_unigrams

# %%
vocabulary = merge_columns_into_one(df.index.to_frame(index=False), join_str=True)
doc_freq_data = pd.DataFrame(
    dict(
        chord_tones=vocabulary,
        document_frequency=df.values,
        rank=range(1, len(vocabulary) + 1),
    )
)
D, V = f.shape
fig = make_scatter_plot(
    doc_freq_data,
    x_col="rank",
    y_col="document_frequency",
    hover_data="chord_tones",
    log_x=True,
    log_y=True,
    title=f"Document frequency of chord tones (D = {D}), V = {V}",
)
save_figure_as(fig, "document_frequency_of_chord_tones")
fig

# %%
selected_numerals = corpus_frequencies.loc[
    corpus_frequencies.iloc[:, -1] == 39, "numeral"
]
absolute_numeral_data = chord_slices[chord_slices.numeral.isin(selected_numerals)]
absolute_numeral_data.head(5)

# %%
chord_slices.query("numeral in @selected_numerals").pivot_table(
    index=["corpus", "piece"],
    columns=["numeral", "fifths_over_local_tonic"],
    values="duration_qb",
    aggfunc="sum",
)

# %%
utils.plot_pca(
    tf,
    "concatenated chord-tone-profile vectors of I and V7",
    color=CORPUS_YEARS,
    size=5,
)

# %%

# %%
culled_vocabulary = df[df.ge(D / 3)]
culled_tf = tf.loc[:, culled_vocabulary.index]
culled_tf.shape

# %%
reload(utils)
plot_cosine_distances(culled_tf, standardize=True)


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


cos_distance_matrix = make_cosine_distances(culled_tf)
cos_distance_matrix

# %%
labels = utils.merge_index_levels(tf.index).to_list()
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
describer = TableDocumentDescriber(metadata.reset_index())
plt.figure(figsize=(10, 60))
ddg = Dendrogram(lm, describer, labels)


# %%
def test_equivalence(arr, metric="cosine"):
    scipy_result = squareform(pdist(arr, metric=metric))
    sklearn_result = pairwise_distances(arr, metric=metric)
    return np.isclose(scipy_result, sklearn_result).all()


# np.savez_compressed("tf_matrix.npz", tf.values, allow_pickle=False)
# npz = np.load("tf_matrix.npz")
# Arr = npz["arr_0"]
# Arr.shape
# metrics = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean",
#           "hamming", "jaccard", "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean",
#          "sokalmichener", "sokalsneath", "sqeuclidean", "yule"]
# for metric in metrics:
#     print(metric, test_equivalence(Arr[:, :-25670], metric))


# %%
chord_tone_profiles = make_chord_tone_profile(chord_slices)

# %%
plot_chord_profiles(chord_tone_profiles, "i, minor", log_y=True)

# %%
plot_chord_profiles(chord_tone_profiles, "V7, minor")

# %%
pipeline = [
    "HasHarmonyLabelsFilter",
    "ModeGrouper",
    "CorpusGrouper",
]
analyzed_D = D.apply_step(*pipeline)
harmony_labels = analyzed_D.get_feature("HarmonyLabels")
harmony_labels

# %%
sliced_D = analyzed_D.apply_step(slicers.KeySlicer)
sliced_D

# %%
key_slices = sliced_D.get_feature("HarmonyLabels")

# %%
sliced_notes = sliced_D.get_feature(resources.Notes)
sliced_notes

# %%
slicer = sliced_D.get_last_step("Slicer")
type(slicer)

# %%
normal_notes = analyzed_D.get_feature("Notes")
normal_notes

# %%
normal_notes

# %%
sn = slicer.process_resource(normal_notes)
sn

# %%
normal_notes = analyzed_D.get_feature(resources.Notes)
normal_notes

# %%
harmony_labels.loc[
    harmony_labels["scale_degrees_major"] == ("1", "3", "5", "b7"), "chord"
].value_counts()

# %%
sd_maj_sonorities = (
    harmony_labels.groupby("scale_degrees_major")
    .duration_qb.sum()
    .sort_values(ascending=False)
)
pd.concat(
    [
        sd_maj_sonorities,
        (sd_maj_sonorities / sd_maj_sonorities.sum()).rename("proportion"),
    ],
    axis=1,
)

# %%
sd_major_occurrences = (
    harmony_labels.groupby("scale_degrees_major")
    .size()
    .rename("frequency")
    .sort_values(ascending=False)
    .reset_index()
)
sd_major_occurrences.index = sd_major_occurrences.index.rename("rank") + 1
sd_major_occurrences = sd_major_occurrences.reset_index()
px.scatter(sd_major_occurrences, x="rank", y="frequency", log_y=True)

# %%
px.scatter(
    sd_major_occurrences,
    x="rank",
    y="frequency",
    log_x=True,
    log_y=True,
)


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
    C = 1 / (sd_major_occurrences.frequency - sd_major_occurrences["rank"])
    r1_i, r2_i = find_index_of_r1_r2(C)
    (r1, f_r1), (r2, f_r2) = df.loc[[r1_i, r2_i], ["rank", "frequency"]].values
    return (f_r1 * r2 - f_r2 * r1) / (r2 - r1 + f_r1 - f_r2)


compute_h(sd_major_occurrences)

# %%
sd_major_occurrences.iloc[130:150]

# %%
sd_sonorities = (
    harmony_labels.groupby("scale_degrees")
    .duration_qb.sum()
    .sort_values(ascending=False)
)
pd.concat(
    [sd_sonorities, (sd_sonorities / sd_sonorities.sum()).rename("proportion")], axis=1
)

# %%
chord_proportions: resources.Durations = harmony_labels.apply_step("Proportions")
chord_proportions.make_ranking_table(drop_cols="chord_and_mode").iloc[:50, :50]

# %%
chord_proportions.make_ranking_table(["mode"], drop_cols="chord_and_mode")

# %%
piece2profile = {
    group: profile["duration_qb"].droplevel(
        ["corpus", "piece", "mode", "chord_and_mode"]
    )
    for group, profile in chord_proportions.groupby(["corpus", "piece", "mode"])
}
piece_profiles = pd.concat(piece2profile, axis=1, names=["corpus", "piece", "mode"])
piece_profiles.sort_index(
    key=lambda _: pd.Index(piece_profiles.notna().sum(axis=1)), ascending=False
).iloc[:50, :50]

# %%
piece_profiles.sort_index(
    key=lambda _: pd.Index(piece_profiles.sum(axis=1)), ascending=False
).iloc[:50, :50]

# %%
corpus_proportions = chord_proportions.combine_results().droplevel("chord_and_mode")
corpus_proportions

# %%
corpus2profile = {
    group: profile["duration_qb"].droplevel(["corpus", "mode"])
    for group, profile in corpus_proportions.groupby(["corpus", "mode"])
}
corpus_profiles = pd.concat(corpus2profile, axis=1, names=["corpus", "mode"])
chord_occurrence_mask = corpus_profiles.notna()
corpus_frequency = chord_occurrence_mask.sum(axis=1)
chord_occurrence_mask = chord_occurrence_mask.sort_index(
    key=lambda _: pd.Index(corpus_frequency), ascending=False
)
chord_occurrence_mask = chord_occurrence_mask.sort_index(
    key=lambda _: pd.Index(chord_occurrence_mask.sum()), ascending=False, axis=1
)
corpus_profiles.sort_index(key=lambda _: pd.Index(corpus_frequency), ascending=False)

# %%
mask_with_sum = pd.concat(
    [chord_occurrence_mask, chord_occurrence_mask.sum(axis=1).rename("sum")], axis=1
)
mask_with_sum.to_csv(make_output_path("chord_occurrence_mask", "tsv"), sep="\t")

# %%
analyzed_D
