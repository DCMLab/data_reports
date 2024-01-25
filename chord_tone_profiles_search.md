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

# Chord Profiles

```{code-cell} ipython3
---
mystnb:
  code_prompt_hide: Hide imports
  code_prompt_show: Show imports
tags: [hide-cell]
---
%load_ext autoreload
%autoreload 2

import math
import os
import re
from collections import defaultdict
from itertools import islice
from numbers import Number
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple
from zipfile import ZipFile

import delta
import dimcat as dc
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

EVALUATIONS_ONLY = False
```

```{code-cell} ipython3
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
describer = delta.TsvDocumentDescriber(D.get_metadata().reset_index())
D
```

```{code-cell} ipython3
if not EVALUATIONS_ONLY:
    chord_slices = utils.get_sliced_notes(D)
    chord_slices.head(5)
```

## pydelta Corpus objects

```{code-cell} ipython3
features = {
    "root_per_globalkey": (  # baseline globalkey-roots without any note information
        ["root_per_globalkey", "intervals_over_root"],
        "chord symbols (root per globalkey + intervals)",
    ),
    "root_per_localkey": (  # baseline localkey-roots without any note information
        ["root", "intervals_over_root"],
        "chord symbols (root per localkey + intervals)",
    ),
    # "root_per_tonicization": (  # baseline root over tonicized key without any note information
    #     ["root_per_tonicization", "intervals_over_root"],
    #     "chord symbols (root per tonicization + intervals)",
    # ),
    # "globalkey_profiles": (  # baseline notes - globalkey
    #     ["fifths_over_global_tonic"],
    #     "tone profiles as per global key",
    # ),
    # "localkey_profiles": (  # baseline notes - localkey
    #     ["fifths_over_local_tonic"],
    #     "tone profiles as per local key",
    # ),
    # "tonicization_profiles": (  # baseline notes - tonicized key
    #     ["fifths_over_tonicization"],
    #     "tone profiles as per tonicized key",
    # ),
    # "global_root_ct": (
    #     ["root_per_globalkey", "fifths_over_root"],
    #     "chord-tone profiles over root-per-globalkey",
    # ),
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
            prevalence_matrix.relative, info=info, absolute=False, norm="piecewise"
        )
        if prevalence_matrix.columns.nlevels > 1:
            groupwise_prevalence = prevalence_matrix.get_groupwise_prevalence(
                column_levels=feature_columns[:-1]
            )
            groupwise = make_pydelta_corpus(
                groupwise_prevalence.relative,
                info=info,
                absolute=False,
                norm="groupwise",
            )
        else:
            groupwise = None
        data[feature_name] = DataTuple(corpus, groupwise, prevalence_matrix)
    return data


data = None if EVALUATIONS_ONLY else make_features(chord_slices, features)
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
def store_data(data: Dict[str, DataTuple], filepath: str, overwrite: bool = True):
    if os.path.isfile(filepath):
        if overwrite:
            os.remove(filepath)
        else:
            raise FileExistsError(f"{filepath} already exists.")
    for feature_name, (corpus, groupwise, _) in data.items():
        corpus.save_to_zip(feature_name + "_piecenorm.tsv", filepath)
        if groupwise is not None:
            groupwise.save_to_zip(feature_name + "_rootnorm.tsv", filepath)
        print(".", end="")


used_pieces = data["root_per_globalkey"].prevalence_matrix.relative.index
metadata_subset = D.get_metadata().join_on_index(used_pieces)
metadata_subset.to_csv("/home/laser/git/chord_profile_search/metadata.tsv", sep="\t")
store_data(data, "/home/laser/git/chord_profile_search/data.zip")
```

```{code-cell} ipython3
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


# if not EVALUATIONS_ONLY:
#     show_rankings(data)
```

## Compute deltas

```{code-cell} ipython3
delta.functions.deltas.keys()
```

```{code-cell} ipython3
def batched(iterable, n):
    # from https://docs.python.org/3/library/itertools.html#itertools.batched introduced only with Python 3.12
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


selected_deltas = {
    name: func
    for name, func in delta.functions.deltas.items()
    if name
    in [
        # "cosine",
        "cosine-z_score",  # Cosine Delta
        "manhattan",
        "manhattan-z_score",  # Burrow's Delta
        "manhattan-z_score-eder_std",  # Eder's Delta
        "sqeuclidean",
        "sqeuclidean-z_score",  # Quadratic Delta
    ]
}
parallel_processor = Parallel(-1)


def apply_deltas(
    selected_deltas: Dict[str, Callable],
    corpus: delta.Corpus,
    groupwise: Optional[delta.Corpus] = None,
):
    """
    Computes the distance matrices for the given Corpus objects and delta functions, using all CPU cores in parallel.
    """
    if groupwise is None:
        iterator = ((func, corpus) for func in selected_deltas.values())
    else:
        iterator = (
            (func, crps)
            for func in selected_deltas.values()
            for crps in (corpus, groupwise)
        )
    results = parallel_processor(delayed(func)(corpus) for func, corpus in iterator)
    if groupwise is None:
        return {
            delta_name: DataTuple(result)
            for delta_name, result in zip(selected_deltas.keys(), results)
        }
    return {
        delta_name: DataTuple(*result)
        for delta_name, result in zip(selected_deltas.keys(), batched(results, 2))
    }


def compute_deltas(
    data: Dict[str, DataTuple],
    selected_deltas: Dict[str, Callable],
    vocab_sizes: int | Iterable[Optional[int | float]] = 5,
    test: bool = False,
    first_n: Optional[int] = None,
    basepath: Optional[str] = None,
) -> Dict[str, Dict[str, List[DataTuple]]]:
    """Iterate over the given Corpus objects and compute selected deltas for selected vocabulary sizes.

    Args:
        data: A dictionary of DataTuples expected to have as first item a Corpus, and as second item a Corpus or None.
        vocab_sizes:
            Can be an integer or an iterable. If an integer k, the vocabulary of size v will be divided in k increasing
            k/v, 2k/v, etc. That is, if k==1, all sizes will be used.
            In the case of an iterable, each integer indicates the actual (maximum) vocabulary size, and each fraction
            < 1 indicates the fraction of the actual vocabulary size. That is, an integer 1 corresponds to the most
            frequent type only (1 column), but 1/3 to the upper third of the vocabulary (1/3 * v columns).
        test: Pass True to sample 10 documents from each corpus to speed up the computations for testing.
        first_n: For testing purposes, you can pass an integer to compute only deltas for the n first features.

    Returns:

    """
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
            if vocab_sizes < 0:
                raise ValueError("vocab_sizes must be positive")
            if vocab_sizes == 1:
                category2top_n = {i: i for i in range(1, vocabulary_size + 1)}
            else:
                stride = math.ceil(vocabulary_size / vocab_sizes)
                category2top_n = {i: i * stride for i in range(1, vocabulary_size + 1)}
        elif isinstance(vocab_sizes, float):
            if not 0 < vocab_sizes < 1:
                raise ValueError("vocab_sizes float must be between 0 and 1")
            category2top_n = {vocab_sizes: math.ceil(vocabulary_size * vocab_sizes)}
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
        try:
            for size_category, top_n in category2top_n.items():
                print(f"({size_category}, {top_n})", end=" ")
                if basepath is not None:
                    zip_file = f"{feature_name}_{top_n}.zip"
                    filepath = os.path.join(basepath, zip_file)
                    if os.path.isfile(filepath):
                        print(f" -> {filepath} (skipped)")
                        continue
                if size_category is None or top_n == vocabulary_size:
                    corpus_top_n = corpus
                    if groupwise_normalization:
                        groupwise_top_n = groupwise
                else:
                    corpus_top_n = corpus.top_n(top_n)
                    if groupwise_normalization:
                        groupwise_top_n = groupwise.top_n(top_n)
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
                    results = apply_deltas(
                        selected_deltas, corpus_top_n, groupwise_top_n
                    )
                else:
                    results = apply_deltas(selected_deltas, corpus_top_n)
                for delta_name, delta_results in results.items():
                    distance_matrices[feature_name][delta_name].append(delta_results)
                if basepath is not None:
                    store_distance_matrices(distance_matrices, filepath)
                    distance_matrices = defaultdict(
                        lambda: defaultdict(list)
                    )  # feature -> delta_name -> [DataTuple]
            print()
        except KeyboardInterrupt:
            return {
                k: {kk: vv for kk, vv in v.items()}
                for k, v in distance_matrices.items()
            }
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


def iter_zipped_matrices(filepath, first_n: Optional[int] = None):
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
            yield dm, feature_name, delta_name, int(vocab_size), bool(groupwise)


def load_distance_matrices(
    filepath,
    first_n: Optional[int] = None,
):
    distance_matrices = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: [None, None]))
    )
    print(f"Loading {filepath}", end="")
    for dm, feature_name, delta_name, vocab_size, groupwise in iter_zipped_matrices(
        filepath, first_n=first_n
    ):
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
    selected_deltas: Dict[str, Callable] = selected_deltas,
    vocab_sizes: int | Iterable[int] = 5,
    name: Optional[str] = "chord_tone_profile_deltas",
    basepath: Optional[str] = None,
    first_n: Optional[int] = None,
) -> Dict[str, Dict[str, List[DataTuple]]]:
    filepath = None
    if name:
        filepath = make_filepath(
            name,
            vocab_sizes,
            extension="zip",
            basepath=basepath,
        )
        if os.path.isfile(filepath):
            return load_distance_matrices(filepath, first_n=first_n)
    assert data is not None, "data is None"
    distance_matrices = compute_deltas(
        data=data,
        selected_deltas=selected_deltas,
        vocab_sizes=vocab_sizes,
    )
    if filepath:
        print(f"Storing distance matrices to {filepath}", end="")
        store_distance_matrices(distance_matrices, filepath)
    return distance_matrices


def make_filepath(
    name,
    *specification: str | Number | Iterable[str | Number],
    extension: str = "zip",
    basepath=None,
):
    if basepath is None:
        basepath = dc.get_setting("default_basepath")
    basepath = utils.resolve_dir(basepath)
    specs = ""
    for spec in specification:
        if isinstance(spec, (str, Number)):
            specs += f"_{spec}"
        else:
            specs += "_" + "-".join(map(str, spec))
    filename = f"{name}{specs}.{extension.lstrip('.')}"
    filepath = os.path.join(basepath, filename)
    return filepath
```

```{code-cell} ipython3
# distance_matrices = get_distance_matrices(data=data, vocab_sizes=vocab_sizes=[4, 100, 2 / 3, None])
for feature in features.keys():
    distance_matrices = compute_deltas(
        data={feature: data[feature]},
        selected_deltas=selected_deltas,
        vocab_sizes=1,
        basepath=os.path.expanduser(dc.get_setting("default_basepath")),
    )
```

```{code-cell} ipython3
:is_executing: true

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


def load_discriminant_metrics(
    cache_name: str = "chord_tone_profiles_evaluation",
) -> pd.DataFrame:
    return pd.read_csv(make_output_path(cache_name, "tsv"), sep="\t")


def get_discriminant_metrics(
    distance_matrices: Dict[str, Dict[str, List[DataTuple]]],
    cache_name: str = "chord_tone_profiles_evaluation",
):
    try:
        result = load_discriminant_metrics(cache_name)
    except FileNotFoundError:
        result = compute_discriminant_metrics(distance_matrices, cache_name=cache_name)
    result.loc[:, "features"] = result.features.where(
        ~result.features.str.contains("groupwise"), result.features.str[10:]
    )
    result.index.rename("i", inplace=True)
    return result


def evaluate_on_the_fly(
    feature_name: str,
    basepath: Optional[str] = None,
):
    if basepath is None:
        basepath = utils.resolve_dir(dc.get_setting("default_basepath"))
    distance_metrics_rows = []
    try:
        for file in sorted(os.listdir(basepath)):
            match = re.match(r"(.+)_(\d+)\.zip$", file)
            if not match:
                continue
            file_feature_name = match.group(1)
            if not file_feature_name or match.group(1) == feature_name:
                continue
            zip_path = os.path.join(basepath, file)
            print(f"Evaluating {zip_path}", end=": ")
            for dm, feature, delta_name, vocab_size, groupwise in iter_zipped_matrices(
                zip_path
            ):
                norm = "groupwise" if groupwise else "corpus"
                print(f"{delta_name} ({groupwise})", end="")
                for row in evaluate(
                    dm, feature_name=feature, delta=delta_name, norm=norm
                ):
                    distance_metrics_rows.append(row)
                    print(".", end="")
            print()
    except KeyboardInterrupt:
        return pd.DataFrame(distance_metrics_rows)
    distance_evaluations = pd.DataFrame(distance_metrics_rows)
    distance_evaluations.index.rename("i", inplace=True)
    tsv_path = make_output_path(feature_name, "tsv")
    print(f"Storing evaluations to {tsv_path}")
    distance_evaluations.to_csv(tsv_path, sep="\t", index=False)
    return distance_evaluations


# distance_evaluations = get_discriminant_metrics(
#     distance_matrices, cache_name="close_inspection_metrics"
# )
distance_evaluations = evaluate_on_the_fly("root_per_globalkey")
distance_evaluations
```

```{code-cell} ipython3

```

```{code-cell} ipython3
fig = make_scatter_plot(
    distance_evaluations,
    x_col="top_n",
    y_col="value",
    symbol="norm",
    color="features",
    facet_col="delta_title",
    facet_row="metric",
    y_axis=dict(matches=None),
    traces_settings=dict(marker_size=10),
    layout=dict(legend=dict(y=-0.05, orientation="h")),
    height=2000,
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
# save_figure_as(fig, "chord_tone_profiles_evaluation", height=2500, width=1300)
fig
```

```{code-cell} ipython3
def get_n_best(distance_evaluations, n=10):
    smaller_is_better = ["Entropy", "Cluster Errors"]  # noqa: F841
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
```

```{code-cell} ipython3
winners = get_n_best(distance_evaluations, n=1)
winners[["feature_name", "norm", "n_types", "delta"]].value_counts()
```

```{code-cell} ipython3
n_best[["feature_name", "norm", "features"]].value_counts()
```

```{code-cell} ipython3
close_inspection_feature_names = [  # both corpus- and groupwise-normalized
    "local_root_ct",
    "tonicization_root_ct",
    "root_per_localkey",
    "root_per_globalkey",
]
```

```{code-cell} ipython3
n_best[n_best.feature_name.isin(close_inspection_feature_names)].groupby(
    "feature_name"
).delta.value_counts()
```

```{code-cell} ipython3
close_inspection_deltas = [
    "manhattan",
    "manhattan-z_score",
    "cosine-z_score",
    "sqeuclidean-z_score",
]
```

```{code-cell} ipython3
def get_distance_matrix(distance_matrices, feature_name, delta_name, norm, n_types):
    norm_index = 1 if norm == "groupwise" else 0
    results = distance_matrices[feature_name][delta_name]
    for data_tuple in results:
        dm = data_tuple[norm_index]
        if dm.metadata.n_types == n_types:
            break
    else:
        raise ValueError(f"No {norm} with {n_types} types found")
    return dm


def individual_evaluation(
    distance_matrices,
    feature_name,
    delta_name,
    norm,
    n_types,
):
    dm = get_distance_matrix(distance_matrices, feature_name, delta_name, norm, n_types)
    clustering = delta.Clustering(dm)
    print(clustering.describe())
    print(clustering.evaluate())
    plt.figure(figsize=(10, 60))
    delta.Dendrogram(clustering)
    # store matplotlib as PDF
    name = f"{feature_name}_{norm}-{n_types}-{delta_name}-{norm}"
    plt.savefig(
        make_output_path(name, "pdf"),
        bbox_inches="tight",
    )
```

```{raw-cell}
individual_evaluation(  # winner of Homogeneity, Purity, V-Measure, Entropy
    distance_matrices,
    feature_name="tonicization_root_ct",
    delta_name="manhattan",
    norm="corpus",
    n_types=535,
)
```

```{raw-cell}
individual_evaluation(  # winner of Fisher's LD
    distance_matrices,
    feature_name="local_root_ct",
    delta_name="cosine-z_score",
    norm="corpus",
    n_types=679,
)
```

```{raw-cell}
individual_evaluation(  # winner of F-Ratio
    distance_matrices,
    feature_name="root_per_globalkey",
    delta_name="cosine-z_score",
    norm="groupwise",
    n_types=1359,
)
```

```{raw-cell}
def make_confusion_matrix(
    clustering_df, y_true="GroupID", y_pred="Cluster", labels="Group"
):
    id2label = set(clustering_df[[y_true, labels]].itertuples(index=False, name=None))
    id2label = dict(sorted(id2label))
    matrix = confusion_matrix(
        y_true=clustering_df[y_true], y_pred=clustering_df[y_pred]
    )
    df = pd.DataFrame(matrix)
    df.index = df.index.map(id2label)
    df.columns = df.columns.map(id2label)
    return df
```

```{code-cell} ipython3

```
