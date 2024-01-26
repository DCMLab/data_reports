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
#     display_name: pydelta
#     language: python
#     name: pydelta
# ---

# %% [markdown]
# # Chord Profiles

# %% mystnb={"code_prompt_hide": "Hide imports", "code_prompt_show": "Show imports"} tags=["hide-cell"]
# %load_ext autoreload
# %autoreload 2

import math
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple
from zipfile import ZipFile

import delta
import dimcat as dc
import joblib
import ms3
import pandas as pd
from dimcat import resources
from dimcat.data.resources.utils import join_df_on_index
from dimcat.plotting import make_scatter_plot, write_image
from git import Repo
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

import utils


class DataTuple(NamedTuple):
    corpus: delta.Corpus
    groupwise: delta.Corpus = None
    prevalence_matrix: Optional[resources.PrevalenceMatrix] = None


plt.rcParams["figure.dpi"] = 300

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)

EVALUATIONS_ONLY = True

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
if not EVALUATIONS_ONLY:
    chord_slices = utils.get_sliced_notes(D)
    chord_slices.head(5)

# %% [markdown]
# ## pydelta Corpus objects

# %%
describer = delta.TsvDocumentDescriber(D.get_metadata().reset_index())


def make_pydelta_corpus(
    matrix: pd.DataFrame, info: str = "chords", absolute: bool = True, **kwargs
) -> delta.Corpus:
    metadata = delta.Metadata(
        features=info,
        ordered=False,
        words=None,
        corpus="Distant Listening Corpus",
        complete=True,
        frequencies=not absolute,
        **kwargs,
    )
    corpus = delta.Corpus(matrix, document_describer=describer, metadata=metadata)
    corpus.index = utils.merge_index_levels(corpus.index)
    corpus.columns = utils.merge_index_levels(corpus.columns)
    return corpus


# %%
features = {
    "root_per_globalkey": (  # baseline globalkey-roots without any note information
        ["root_per_globalkey", "intervals_over_root"],
        "chord symbols (root per globalkey + intervals)",
    ),
    "root_per_localkey": (  # baseline localkey-roots without any note information
        ["root", "intervals_over_root"],
        "chord symbols (root per localkey + intervals)",
    ),
    "root_per_tonicization": (  # baseline root over tonicized key without any note information
        ["root_per_tonicization", "intervals_over_root"],
        "chord symbols (root per tonicization + intervals)",
    ),
    "globalkey_profiles": (  # baseline notes - globalkey
        ["fifths_over_global_tonic"],
        "tone profiles as per global key",
    ),
    "localkey_profiles": (  # baseline notes - localkey
        ["fifths_over_local_tonic"],
        "tone profiles as per local key",
    ),
    "tonicization_profiles": (  # baseline notes - tonicized key
        ["fifths_over_tonicization"],
        "tone profiles as per tonicized key",
    ),
    "global_root_ct": (
        ["root_per_globalkey", "fifths_over_root"],
        "chord-tone profiles over root-per-globalkey",
    ),
    "local_root_ct": (
        ["root", "fifths_over_root"],
        "chord-tone profiles over root-per-localkey",
    ),
    "tonicization_root_ct": (
        [
            "root_per_tonicization",
            "fifths_over_root",
        ],
        "chord-tone profiles over root-per-tonicization",
    ),
}


analyzer_config = dc.DimcatConfig(
    "PrevalenceAnalyzer",
    index=["corpus", "piece"],
)


def make_features(
    chord_slices: resources.DimcatResource,
    features: Dict[str, Tuple[str | List[str], str]],
) -> Dict[str, DataTuple]:
    data = {}
    for feature_name, (feature_columns, info) in features.items():
        print(f"Computing prevalence matrix and pydelta corpus for {info}")
        analyzer_config.update(columns=feature_columns)
        prevalence_matrix = chord_slices.apply_step(analyzer_config)
        corpus = make_pydelta_corpus(
            prevalence_matrix.relative, info=info, absolute=False, norm="piecenorm"
        )
        if prevalence_matrix.columns.nlevels > 1:
            groupwise_prevalence = prevalence_matrix.get_groupwise_prevalence(
                column_levels=feature_columns[:-1]
            )
            groupwise = make_pydelta_corpus(
                groupwise_prevalence.relative,
                info=info,
                absolute=False,
                norm="rootnorm",
            )
        else:
            groupwise = None
        data[feature_name] = DataTuple(corpus, groupwise, prevalence_matrix)
    return data


data = None if EVALUATIONS_ONLY else make_features(chord_slices, features)


# %%
def make_rankings(
    data: Dict[str, DataTuple],
    long_names: bool = False,
) -> Dict[str, pd.DataFrame]:
    rankings = {}
    for feature_name, data_tuple in data.items():
        type_prevalence = data_tuple.prevalence_matrix.type_prevalence()
        document_frequency = data_tuple.prevalence_matrix.document_frequencies()
        ranking = (
            pd.concat([type_prevalence, document_frequency], axis=1)
            .sort_values(["document_frequency", "type_prevalence"], ascending=False)
            .reset_index()
        )
        name = data_tuple.corpus.metadata.features if long_names else feature_name
        rankings[name] = ranking
    result = pd.concat(rankings, axis=1)
    result.index = (result.index + 1).rename("rank")
    return result


def show_rankings(data: Dict[str, DataTuple], top_n: int = 30):
    rankings = make_rankings(data)
    return rankings.iloc[:top_n]


if not EVALUATIONS_ONLY:
    show_rankings(data)

# %% [markdown]
# ## Compute deltas

# %%
delta.functions.deltas.keys()

# %%
selected_deltas = {
    name: func
    for name, func in delta.functions.deltas.items()
    if name
    in [
        "cosine",
        "cosine-z_score",  # Cosine Delta
        "manhattan",
        "manhattan-z_score",  # Burrow's Delta
        "manhattan-z_score-eder_std",  # Eder's Delta
        "sqeuclidean",
        "sqeuclidean-z_score",  # Quadratic Delta
    ]
}
parallel_processor = Parallel(-1, prefer="threads")


def make_data_tuple(func, *corpus):
    return DataTuple(*(func(c) for c in corpus))


def apply_deltas(*corpus):
    results = parallel_processor(
        delayed(make_data_tuple)(func, *corpus) for func in selected_deltas.values()
    )
    return dict(zip(selected_deltas.keys(), results))


def compute_deltas(
    data: Dict[str, DataTuple],
    vocab_sizes: int | Iterable[Optional[int | float]] = 5,
    test: bool = False,
    first_n: Optional[int] = None,
) -> Dict[str, Dict[str, List[DataTuple]]]:
    distance_matrices = defaultdict(
        lambda: defaultdict(list)
    )  # feature -> delta_name -> [DataTuple]
    for i, (feature_name, (corpus, groupwise, prevalence_matrix)) in enumerate(
        data.items()
    ):
        if first_n and not i < first_n:
            break
        print(f"Computing deltas for {feature_name}", end=": ")
        vocabulary_size = prevalence_matrix.n_types
        if isinstance(vocab_sizes, int):
            stride = math.ceil(vocabulary_size / vocab_sizes)
            category2top_n = {i: i * stride for i in range(1, vocabulary_size + 1)}
        else:
            category2top_n = {}
            filter_too_large = (
                None in vocab_sizes
            )  # since max-vocab is requested, skip other values that are larger
            for size_category in vocab_sizes:
                if size_category is None:
                    category2top_n[None] = vocabulary_size
                elif size_category < 1:
                    category2top_n[size_category] = math.ceil(
                        vocabulary_size * size_category
                    )
                elif filter_too_large and size_category > vocabulary_size:
                    continue
                else:
                    category2top_n[size_category] = math.ceil(size_category)
        groupwise_normalization = groupwise is not None
        for size_category, top_n in category2top_n.items():
            print(f"({size_category}, {top_n})", end=" ")
            if size_category is not None:
                corpus_top_n = corpus.top_n(top_n)
                if groupwise_normalization:
                    groupwise_top_n = groupwise.top_n(top_n)
            else:
                corpus_top_n = corpus
                if groupwise_normalization:
                    groupwise_top_n = groupwise
            if test:
                corpus_top_n = delta.Corpus(
                    corpus=corpus_top_n.sample(10),
                    document_describer=corpus_top_n.document_describer,
                    metadata=corpus_top_n.metadata,
                    complete=False,
                    frequencies=True,
                )
                if groupwise_normalization:
                    groupwise_top_n = delta.Corpus(
                        corpus=groupwise_top_n.sample(10),
                        document_describer=groupwise_top_n.document_describer,
                        metadata=groupwise_top_n.metadata,
                        complete=False,
                        frequencies=True,
                    )
            n_types = corpus_top_n.shape[1]
            # size_category: the piece of information for computing the top_n; used as dict keys
            # top_n: The information computed from that passed to Corpus.top_n() (stored in Corpus.metadata.words)
            # n_types: The actual number of (most frequent) types in this corpus, which may be less than top_n
            corpus_top_n.metadata.top_n = top_n
            corpus_top_n.metadata.n_types = n_types
            if groupwise_normalization:
                groupwise_top_n.metadata.top_n = top_n
                groupwise_top_n.metadata.n_types = n_types
                results = apply_deltas(corpus_top_n, groupwise_top_n)
            else:
                results = apply_deltas(corpus_top_n)
            for delta_name, delta_results in results.items():
                distance_matrices[feature_name][delta_name].append(delta_results)
        print()
    return {k: {kk: vv for kk, vv in v.items()} for k, v in distance_matrices.items()}


def store_distance_matrices(
    distance_matrices: Dict[str, Dict[str, List[DataTuple]]], filepath: str
):
    for feature_name, feature_data in distance_matrices.items():
        for delta_name, delta_results in feature_data.items():
            for data_tuple in delta_results:  # one per vocab_size
                vocab_size = data_tuple.corpus.metadata.top_n
                name = f"{feature_name}_{vocab_size}_{delta_name}"
                data_tuple.corpus.save_to_zip(name + ".tsv", filepath)
                if data_tuple.groupwise is not None:
                    data_tuple.groupwise.save_to_zip(name + "_groupwise.tsv", filepath)
                print(".", end="")


def pickle_distance_matrices(
    distance_matrices: Dict[str, Dict[str, List[DataTuple]]], directory: str
):
    for feature_name, feature_data in distance_matrices.items():
        for delta_name, delta_results in feature_data.items():
            for data_tuple in delta_results:  # one per vocab_size
                size = data_tuple.corpus.metadata.n_types
                name = f"{feature_name}_piecenorm"
                filename = f"{name}-{delta_name}-{size:04d}"
                joblib.dump(data_tuple.corpus, os.path.join(directory, filename))
                if data_tuple.groupwise is not None:
                    name = f"{feature_name}_rootnorm"
                    filename = f"{name}-{delta_name}-{size:04d}"
                    joblib.dump(data_tuple.groupwise, os.path.join(directory, filename))
                print(".", end="")


def load_distance_matrices(
    filepath,
    first_n: Optional[int] = None,
):
    distance_matrices = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: [None, None]))
    )
    print(f"Loading {filepath}", end="")
    with ZipFile(filepath, "r") as zip_handler:
        for i, file in enumerate(zip_handler.namelist()):
            if first_n and not i < first_n:
                break
            match = re.match(r"(.+)_(\d+)_(.+?)(_groupwise)?\.tsv$", file)
            if not match:
                continue
            with zip_handler.open(file) as f:
                matrix = pd.read_csv(f, sep="\t", index_col=0)
            try:
                metadata = delta.Metadata.from_zip_file(file, zip_handler)
            except KeyError:
                metadata = None
            dm = delta.DistanceMatrix(
                matrix, metadata=metadata, document_describer=describer
            )
            feature_name, vocab_size, delta_name, groupwise = match.groups()
            position = int(bool(groupwise))
            distance_matrices[feature_name][delta_name][int(vocab_size)][position] = dm
            print(".", end="")
    result = {}
    for feature_name, feature_data in distance_matrices.items():
        feature_results = {}
        for delta_name, delta_data in feature_data.items():
            list_of_tuples = []
            for vocab_size, delta_data in delta_data.items():
                list_of_tuples.append(DataTuple(*delta_data))
            feature_results[delta_name] = list_of_tuples
        result[feature_name] = feature_results
    return result


def get_distance_matrices(
    data: Optional[Dict[str, DataTuple]] = None,
    vocab_sizes: int | Iterable[int] = 5,
    name: Optional[str] = "chord_tone_profile_deltas",
    basepath: Optional[str] = None,
    first_n: Optional[int] = None,
    pickle: bool = True,
) -> Dict[str, Dict[str, List[DataTuple]]]:
    """

    Args:
        pickle:
            If True (default), the distance matrices are pickled to individual files in basepath/name.
            Otherwise, they will be stored as TSV files in a ZIP archive (slow). To disable storing,
            pass an empty string or None as ``name``. Please note that loading existing data instead
            of re-computing is currently implemented for ZIP files only. The purpose of the pickle
            files is to be evaluated using the `chord_profile_search` repository.
    """
    filepath = None
    if name:
        if basepath is None:
            basepath = dc.get_setting("default_basepath")
        basepath = utils.resolve_dir(basepath)
        if isinstance(vocab_sizes, int):
            zip_file = f"{name}_{vocab_sizes}.zip"
        else:
            zip_file = f"{name}_{'-'.join(map(str, vocab_sizes))}.zip"
        filepath = os.path.join(basepath, zip_file)
        if os.path.isfile(filepath):
            return load_distance_matrices(filepath, first_n=first_n)
    assert data is not None, "data is None"
    distance_matrices = compute_deltas(data, vocab_sizes)
    if name:
        if pickle:
            directory = os.path.join(basepath, name)
            os.makedires(directory, exists_ok=True)
            print(f"Pickeling distance matrices to {directory}", end="")
            pickle_distance_matrices(distance_matrices, directory)
        else:
            print(f"Storing distance matrices to {filepath}", end="")
            store_distance_matrices(distance_matrices, filepath)
    return distance_matrices


if not EVALUATIONS_ONLY:
    distance_matrices = get_distance_matrices(
        data=data, vocab_sizes=[4, 100, 2 / 3, None]
    )


# %%
def evaluate(distance_matrix: delta.DistanceMatrix, **metadata):
    row = dict(distance_matrix.metadata, **metadata)
    # row.update(distance_matrix.metadata)
    for metric, value in distance_matrix.evaluate().items():
        yield dict(row, metric=metric, value=value)
    clustering_ward = delta.Clustering(distance_matrix)
    for metric, value in clustering_ward.evaluate().items():
        yield dict(row, metric=f"{metric} (Ward)", value=value)
    # ToDo: include if original data is given to initialize with corpus medoids
    clustering_kmedoids = delta.KMedoidsClustering_distances(distance_matrix)
    for metric, value in clustering_kmedoids.evaluate().items():
        yield dict(row, metric=f"{metric} (k-medoids)", value=value)


def compute_discriminant_metrics(
    distance_matrices, cache_name: str = "chord_tone_profiles_evaluation"
) -> pd.DataFrame:
    print("Computing discriminant metrics")
    distance_metrics_rows = []
    try:
        for feature_name, feature_data in distance_matrices.items():
            print(feature_name, end=": ")
            for delta_name, delta_data in feature_data.items():
                print(delta_name, end="")
                for delta_results in delta_data:
                    for norm, distance_matrix in delta_results._asdict().items():
                        if distance_matrix is None:
                            continue
                        rows = evaluate(
                            distance_matrix,
                            feature_name=feature_name,
                            delta=delta_name,
                            norm=norm,
                        )
                        distance_metrics_rows.extend(rows)
                        print(".", end="")
            print()
    except KeyboardInterrupt:
        return pd.DataFrame(distance_metrics_rows)
    distance_evaluations = pd.DataFrame(distance_metrics_rows)
    distance_evaluations.index.rename("i", inplace=True)
    distance_evaluations.to_csv(
        make_output_path(cache_name, "tsv"), sep="\t", index=False
    )
    return distance_evaluations


def melt_evaluation_metrics(df):
    variable_names = {
        "adjusted rand",
        "cluster errors",
        "completeness",
        "entropy",
        "f-ratio",
        "fisher",
        "homogeneity",
        "purity",
        "simple score",
        "v-measure",
    }
    id_vars = [
        col
        for col in df.columns
        if not any(col.lower().startswith(v) for v in variable_names)
    ]
    return df.melt(id_vars=id_vars, var_name="metric")


def load_discriminant_metrics(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep="\t")
    if "metric" not in df.columns:
        df = melt_evaluation_metrics(df)
    return df


def get_discriminant_metrics(
    distance_matrices: Dict[str, Dict[str, List[DataTuple]]],
    cache_name: str = "chord_tone_profiles_evaluation",
):
    try:
        filepath = make_output_path(cache_name, "tsv")
        result = load_discriminant_metrics(filepath)
    except FileNotFoundError:
        result = compute_discriminant_metrics(distance_matrices, cache_name=cache_name)
    result.loc[:, "features"] = result.features.where(
        ~result.features.str.contains("groupwise"), result.features.str[10:]
    )
    result.index.rename("i", inplace=True)
    return result


# distance_evaluations = get_discriminant_metrics(distance_matrices)
distance_evaluations = load_discriminant_metrics(
    "/home/laser/git/chord_profile_search/results.tsv"
)
distance_evaluations

# %%
features

# %%
color_palette = dict(
    globalkey_profiles="AMBER_800",
    localkey_profiles="AMBER_500",
    tonicization_profiles="ORANGE_300",
    root_per_globalkey="GREEN_800",
    root_per_localkey="GREEN_500",
    root_per_tonicization="LIME_300",
    global_root_ct="INDIGO_800",
    local_root_ct="VIOLET_600",
    tonicization_root_ct="PURPLE_400",
)
feature_names = {f: name for f, (_, name) in features.items()}
color_palette = {
    feature_names[k]: utils.TailwindColorsHex.get_color(v)
    for k, v in color_palette.items()
}

# %%
fig = make_scatter_plot(
    distance_evaluations,
    x_col="top_n",
    y_col="value",
    symbol="norm",
    color="features",
    color_discrete_map=color_palette,
    facet_col="delta_title",
    facet_row="metric",
    y_axis=dict(matches=None),
    traces_settings=dict(marker_size=10),
    layout=dict(legend=dict(y=-0.05, orientation="h")),
    height=4000,
)
# fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
# fig.update_yaxes(matches="y")
for row_idx, row_figs in enumerate(fig._grid_ref):
    for col_idx, col_fig in enumerate(row_figs):
        fig.update_yaxes(
            row=row_idx + 1,
            col=col_idx + 1,
            matches="y" + str(len(row_figs) * row_idx + 1),
        )
save_figure_as(fig, "chord_tone_profiles_evaluation", height=4000, width=1300)
fig

# %%
delta.functions


# %%
def get_n_best(distance_evaluations, n=10):
    distance_evaluations.index.rename("i", inplace=True)
    smaller_is_better = ["Entropy", "Cluster Errors", "F-Ratio"]  # noqa: F841
    nlargest = distance_evaluations.groupby("metric").value.nlargest(n)
    best_largest = join_df_on_index(distance_evaluations, nlargest.index).query(
        "metric not in @smaller_is_better"
    )
    nsmallest = distance_evaluations.groupby("metric").value.nsmallest(n)
    best_smallest = join_df_on_index(distance_evaluations, nsmallest.index).query(
        "metric in @smaller_is_better"
    )
    return pd.concat([best_largest, best_smallest])


n_best = get_n_best(distance_evaluations)
n_best

# %%
winners = get_n_best(distance_evaluations, n=1)
winners[["features", "norm", "delta_descriptor"]].value_counts()

# %%
winners

# %%
n_best["delta_descriptor"].value_counts()

# %%
n_best[["features", "norm"]].value_counts()

# %%
winners[["features", "norm"]].value_counts()

# %%
winners.delta_descriptor.value_counts()

# %%
n_best[["features", "norm", "delta_descriptor"]].value_counts()

# %%
close_inspection_feature_names = [  # both corpus- and groupwise-normalized
    feature_names[f]
    for f in (
        "global_root_ct",
        "local_root_ct",
        "tonicization_root_ct",
        "root_per_globalkey",
        "root_per_localkey",
        "root_per_tonicization",
    )
]

# %%
n_best[n_best.features.isin(close_inspection_feature_names)].groupby(
    ["features", "norm"]
).delta_descriptor.value_counts()

# %%
close_inspection_deltas = [
    "sqeuclidean-z_score",  # especially for all 3 types of chord profiles
    "manhattan",  # especially for all 3 types of chord-tone profiles
    "manhattan-z_score-eder_std",
    "cosine-z_score",
]

# %% [raw]
# def get_distance_matrix(distance_matrices, feature_name, delta_name, norm, n_types):
#     norm_index = 1 if norm == "groupwise" else 0
#     results = distance_matrices[feature_name][delta_name]
#     for data_tuple in results:
#         dm = data_tuple[norm_index]
#         if dm.metadata.n_types == n_types:
#             break
#     else:
#         raise ValueError(f"No {norm} with {n_types} types found")
#     return dm
#
#
# def individual_evaluation(
#     distance_matrices,
#     feature_name,
#     delta_name,
#     norm,
#     n_types,
# ):
#     dm = get_distance_matrix(distance_matrices, feature_name, delta_name, norm, n_types)
#     clustering = delta.Clustering(dm)
#     print(clustering.describe())
#     print(clustering.evaluate())
#     plt.figure(figsize=(10, 60))
#     delta.Dendrogram(clustering)
#     # store matplotlib as PDF
#     name = f"{feature_name}_{norm}-{n_types}-{delta_name}-{norm}"
#     plt.savefig(
#         make_output_path(name, "pdf"),
#         bbox_inches="tight",
#     )
#
# def make_confusion_matrix(
#     clustering_df, y_true="GroupID", y_pred="Cluster", labels="Group"
# ):
#     id2label = set(clustering_df[[y_true, labels]].itertuples(index=False, name=None))
#     id2label = dict(sorted(id2label))
#     matrix = confusion_matrix(
#         y_true=clustering_df[y_true], y_pred=clustering_df[y_pred]
#     )
#     df = pd.DataFrame(matrix)
#     df.index = df.index.map(id2label)
#     df.columns = df.columns.map(id2label)
#     return df

# %% [markdown]
# ## Chord Profiles
# ### Globalkey

# %%
METRICS_PATH = os.path.abspath(os.path.join(RESULTS_PATH, "..", "data"))


def load_feature_metrics(
    feature_name: str,
    directory: str = METRICS_PATH,
) -> pd.DataFrame:
    candidates = [f for f in os.listdir(directory) if feature_name in f]
    if not candidates:
        raise FileNotFoundError(
            f"No files containing {feature_name} found in {directory}"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple files containing {feature_name} found in {directory}: {candidates}"
        )
    filepath = os.path.join(directory, candidates[0])
    df = pd.read_csv(filepath, sep="\t")
    if "metric" not in df.columns:
        df = melt_evaluation_metrics(df)
    if "top_n" not in df.columns:
        df["top_n"] = df.filepath.str[-4:].astype(int)
    return df


# %%
def show_gridsearch(df, save_as: Optional[str], height=4000, width=1300, **kwargs):
    fig = make_scatter_plot(
        df,
        x_col="top_n",
        y_col="value",
        symbol="norm",
        color="norm",
        facet_col="delta_title",
        facet_row="metric",
        y_axis=dict(matches=None),
        traces_settings=dict(marker_size=2),
        layout=dict(legend=dict(y=-0.05, orientation="h")),
        height=height,
        width=width,
    )
    # fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
    # fig.update_yaxes(matches="y")
    for row_idx, row_figs in enumerate(fig._grid_ref):
        for col_idx, col_fig in enumerate(row_figs):
            fig.update_yaxes(
                row=row_idx + 1,
                col=col_idx + 1,
                matches="y" + str(len(row_figs) * row_idx + 1),
            )

    if save_as:
        save_figure_as(fig, save_as, height=height, width=width)
    fig


# %%
root_per_globalkey = load_feature_metrics("root_per_globalkey")
show_gridsearch(root_per_globalkey, "root_per_globalkey_gridsearch")

# %% [markdown]
# ### Localkey

# %%
root_per_globalkey = load_feature_metrics("root_per_localkey")
show_gridsearch(root_per_globalkey, "root_per_localkey_gridsearch")

# %%
