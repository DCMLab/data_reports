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

# Chord Profiles

```{code-cell} ipython3
---
mystnb:
  code_prompt_hide: Hide imports
  code_prompt_show: Show imports
tags: [hide-cell]
---
# %load_ext autoreload
# %autoreload 0
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
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import Normalizer, StandardScaler

import utils

plt.rcParams["figure.dpi"] = 300

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
```

```{code-cell} ipython3
RESULTS_PATH = os.path.abspath(os.path.join(utils.OUTPUT_FOLDER, "chord_profiles"))
os.makedirs(RESULTS_PATH, exist_ok=True)


def make_output_path(filename, extension=None):
    if extension:
        extension = "." + extension.lstrip(".")
    else:
        extension = utils.DEFAULT_OUTPUT_FORMAT
    return os.path.join(RESULTS_PATH, f"{filename}{extension}")


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
D
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
label_slicer = slicers.HarmonyLabelSlicer()
sliced_D = label_slicer.process(D)
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
sliced_notes = sliced_D.get_feature(resources.Notes)
sliced_notes
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
slice_info = label_slicer.slice_metadata.droplevel(-1)
merge_columns = [col for col in slice_info.columns if col not in sliced_notes.columns]
slice_info = join_df_on_index(
    slice_info[merge_columns], sliced_notes.index, how="right"
)
chord_slices = pd.concat([sliced_notes, slice_info], axis=1)
chord_slices = pd.concat([chord_slices, transpose_notes_to_c(chord_slices)], axis=1)
chord_slices
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
chord_profiles = chord_slices.groupby(
    ["corpus", "chord_and_mode", "fifths_over_local_tonic"]
).duration_qb.sum()
normalization = chord_profiles.groupby(["corpus", "chord_and_mode"]).sum()
chord_profiles = pd.concat(
    [chord_profiles, chord_profiles.div(normalization).rename("proportion")], axis=1
)
chord_profiles
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
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


plot_chord_profiles(chord_profiles, "i, minor", log_y=True)
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
plot_chord_profiles(chord_profiles, "V7, minor")
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
chord_slices.head()
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
_, _, _, chord_df, _ = utils.prepare_tf_idf_data(
    chord_slices,
    index=["corpus"],
    columns="chord_and_mode",
)
chord_df
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
reload(utils)


def prepare_data(
    chord_slices: pd.DataFrame,
    chord_and_mode: str | Iterable[str],
    groupby: str | List[str],
    smooth=1e-20,
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features = ["fifths_over_local_tonic"]
    if isinstance(groupby, str):
        groupby = [groupby]
    if isinstance(chord_and_mode, str):
        absolute_data = chord_slices.query(f"chord_and_mode == '{chord_and_mode}'")
        return utils.prepare_tf_idf_data(
            absolute_data,
            index=groupby,
            columns=features,
            smooth=smooth,
        )
    results = [prepare_data(chord_slices, cm, groupby, smooth) for cm in chord_and_mode]
    concatenated_results = []
    for tup in zip(*results):
        concatenated_results.append(pd.concat(tup, axis=1, keys=chord_and_mode))
    return tuple(concatenated_results)


unigram_distribution, f, tf, df, idf = prepare_data(
    chord_slices, chord_and_mode=["I, major", "V7, major"], groupby="corpus"
)
unigram_distribution
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
reload(utils)
metadata = D.get_metadata()
CORPUS_YEARS = utils.corpus_mean_composition_years(metadata)
CORPUS_YEARS
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
reload(utils)
utils.plot_pca(
    tf,
    "concatenated chord-tone-profile vectors of I and V7",
    color=CORPUS_YEARS,
    size=5,
)
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
full_unigrams, f, tf, df, idf = utils.prepare_tf_idf_data(
    chord_slices,
    index=["corpus", "piece"],
    columns=["chord_and_mode", "fifths_over_local_tonic"],
)
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
full_unigrams
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
vocabulary = merge_columns_into_one(df.index.to_frame(index=False), join_str=True)
doc_freq_data = pd.DataFrame(
    dict(
        chord_tones=vocabulary,
        document_frequency=df.values,
        rank=range(1, len(vocabulary) + 1),
    )
)
N = f.shape[0]
make_scatter_plot(
    doc_freq_data,
    x_col="rank",
    y_col="document_frequency",
    hover_data="chord_tones",
    log_x=True,
    log_y=True,
    title=f"Document frequency of chord tones (N = {N})",
)
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
culled_vocabulary = df[df.ge(N / 3)]
culled_tf = tf.loc[:, culled_vocabulary.index]
culled_tf.shape
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
reload(utils)


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


plot_cosine_distances(culled_tf, standardize=True)
```

```{code-cell} ipython3
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
```

```{code-cell} ipython3
labels = utils.merge_index_levels(tf.index).to_list()
ac = AgglomerativeClustering(
    metric="precomputed", linkage="complete", distance_threshold=0, n_clusters=None
)
ac.fit_predict(cos_distance_matrix)
plot_dendrogram(ac, truncate_mode="level", p=0)
plt.title("Hierarchical Clustering using maximum cosine distances")
# plt.savefig('aggl_mvt_max_cos.png', bbox_inches='tight')
```

```{raw-cell}
---
jupyter:
  outputs_hidden: false
---
sliced_notes.store_resource(
    basepath="~/dimcat_data",
    name="sliced_notes"
)
```

```{raw-cell}
---
jupyter:
  outputs_hidden: false
---
restored = dc.deserialize_json_file("/home/laser/dimcat_data/sliced_notes.resource.json")
restored.df
```

```{code-cell} ipython3
pipeline = [
    "HasHarmonyLabelsFilter",
    "ModeGrouper",
    "CorpusGrouper",
]
analyzed_D = D.apply_step(*pipeline)
harmony_labels = analyzed_D.get_feature("HarmonyLabels")
harmony_labels
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
sliced_D = analyzed_D.apply_step(slicers.KeySlicer)
sliced_D
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
key_slices = sliced_D.get_feature("HarmonyLabels")
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
sliced_notes = sliced_D.get_feature(resources.Notes)
sliced_notes
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
slicer = sliced_D.get_last_step("Slicer")
type(slicer)
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
normal_notes = analyzed_D.get_feature("Notes")
normal_notes
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
normal_notes
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
sn = slicer.process_resource(normal_notes)
sn
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
normal_notes = analyzed_D.get_feature(resources.Notes)
normal_notes
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
harmony_labels.loc[
    harmony_labels["scale_degrees_major"] == ("1", "3", "5", "b7"), "chord"
].value_counts()
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
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
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
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
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
px.scatter(
    sd_major_occurrences,
    x="rank",
    y="frequency",
    log_x=True,
    log_y=True,
)
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---


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
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
sd_major_occurrences.iloc[130:150]
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
sd_sonorities = (
    harmony_labels.groupby("scale_degrees")
    .duration_qb.sum()
    .sort_values(ascending=False)
)
pd.concat(
    [sd_sonorities, (sd_sonorities / sd_sonorities.sum()).rename("proportion")], axis=1
)
```

```{code-cell} ipython3
chord_proportions: resources.Durations = harmony_labels.apply_step("Proportions")
chord_proportions.make_ranking_table(drop_cols="chord_and_mode").iloc[:50, :50]
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
chord_proportions.make_ranking_table(["mode"], drop_cols="chord_and_mode")
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
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
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
piece_profiles.sort_index(
    key=lambda _: pd.Index(piece_profiles.sum(axis=1)), ascending=False
).iloc[:50, :50]
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
corpus_proportions = chord_proportions.combine_results().droplevel("chord_and_mode")
corpus_proportions
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
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
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
mask_with_sum = pd.concat(
    [chord_occurrence_mask, chord_occurrence_mask.sum(axis=1).rename("sum")], axis=1
)
mask_with_sum.to_csv(make_output_path("chord_occurrence_mask", "tsv"), sep="\t")
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
analyzed_D
```
