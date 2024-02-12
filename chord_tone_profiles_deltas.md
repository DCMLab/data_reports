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
from itertools import count
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple
from zipfile import ZipFile

import delta
import dimcat as dc
import joblib
import ms3
import pandas as pd
import plotly.graph_objs as go
from dimcat import resources
from dimcat.data.resources.utils import join_df_on_index
from dimcat.plotting import (
    make_box_plot,
    make_line_plot,
    make_scatter_plot,
    write_image,
)
from git import Repo
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr

import utils


class DataTuple(NamedTuple):
    corpus: delta.Corpus
    groupwise: delta.Corpus = None
    prevalence_matrix: Optional[resources.PrevalenceMatrix] = None


plt.rcParams["figure.dpi"] = 300

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)

EVALUATIONS_ONLY = True
ONLY_PAPER_METRICS = True  # to plot only the metrics reported in Everts et al. (2017)
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
D
```

```{code-cell} ipython3
if not EVALUATIONS_ONLY:
    chord_slices = utils.get_sliced_notes(D)
    chord_slices.head(5)
```

## pydelta Corpus objects for selected features

Skipped if EVALUATIONS_ONLY

```{code-cell} ipython3
describer = delta.ComposerDescriber(D.get_metadata().reset_index())
```

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
        corpus = utils.make_pydelta_corpus(
            prevalence_matrix.relative, info=info, absolute=False, norm="piecenorm"
        )
        if prevalence_matrix.columns.nlevels > 1:
            groupwise_prevalence = prevalence_matrix.get_groupwise_prevalence(
                column_levels=feature_columns[:-1]
            )
            groupwise = utils.make_pydelta_corpus(
                groupwise_prevalence.relative,
                info=info,
                absolute=False,
                norm="rootnorm",
            )
        else:
            groupwise = None
        data[feature_name] = DataTuple(corpus, groupwise, prevalence_matrix)
    return data


if EVALUATIONS_ONLY:
    data = None
else:
    data = utils.make_profiles(
        chord_slices,
        describer=describer,
        profiles=features,
        save_as="/home/laser/git/chord_profile_search/data.zip",
    )
    used_pieces = data["root_per_globalkey"].prevalence_matrix.relative.index
    metadata_subset = D.get_metadata().join_on_index(used_pieces)
    metadata_subset.to_csv(
        "/home/laser/git/chord_profile_search/metadata.tsv", sep="\t"
    )
```

### Show type rankings for selected features

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


if not EVALUATIONS_ONLY:
    show_rankings(data)
```

## Compute distance matrices for selected deltas

Skipped if EVALUATIONS_ONLY

```{code-cell} ipython3
delta.functions.deltas.keys()
```

```{code-cell} ipython3
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


def resolve_vocab_sizes(
    vocab_sizes: Optional[int | Iterable[Optional[int | float]]],
    vocabulary_size: int,
) -> Dict[Optional[int | float], int]:
    """Resolves decimal fractions < 1 to the corresponding number of types based on the vocabulary size."""
    if vocab_sizes is None:
        return {None: vocabulary_size}
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
    return category2top_n


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
        category2top_n = resolve_vocab_sizes(vocab_sizes, vocabulary_size)
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
    normalize: bool = False,
    name: Optional[str] = "chord_tone_profile_deltas",
    basepath: Optional[str] = None,
    first_n: Optional[int] = None,
    pickle: bool = True,
) -> Dict[str, Dict[str, List[DataTuple]]]:
    """

    Args:
        first_n: Only for the first n pieces (e.g. for testing).
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
            if normalize:
                raise NotImplementedError(
                    "Loading distance matrices after normalization not yet implemented."
                )
            return load_distance_matrices(filepath, first_n=first_n)
    assert data is not None, "data is None"
    distance_matrices = compute_deltas(data, vocab_sizes, first_n=first_n)
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
```

## Compute discriminant metrics for distance matrices

Or load them from a TSV file.

### Pre-study

A few hand-picked values only.

This was originally computed on a laptop based on distance matrices that had been saved as a ZIP archive of TSV files
(`store_distance_matrices()`). Once the gridsearch had been implemented for running on a HPC, the distance matrices
were instead pickled individually (`pickle_distance_matrices()`).

```{code-cell} ipython3
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
    df
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


def keep_only_paper_metrics(df):
    """Keep only those evaluation metrics that were actually reported in Evert et al. (2017)."""
    starts_with = ("Adjusted Rand", "Simple Score", "Cluster Errors")
    mask = df.metric.str.startswith(starts_with)
    return df[mask].sort_values(["delta", "metric"])


distance_evaluations = get_discriminant_metrics(distance_matrices)
distance_evaluations = load_discriminant_metrics(
    "/home/laser/git/chord_profile_search/probe_measurements.tsv"
)  # these were created by having the chord_profile_search/gridsearch.py script process the folder containing
# pickled distance matrices created via this notebook
if ONLY_PAPER_METRICS:
    distance_evaluations = keep_only_paper_metrics(distance_evaluations)
distance_evaluations
```

**Define colors**

Assign a Tailwind color to each feature. To see the full palette, check `tailwindcss_v3.3.3.(png|svg)`

```{code-cell} ipython3
height = 1946 if ONLY_PAPER_METRICS else 4000

fig = make_scatter_plot(
    distance_evaluations,
    x_col="top_n",
    y_col="value",
    symbol="norm",
    color="features",
    color_discrete_map=utils.CHORD_PROFILE_COLORS,
    facet_col="delta_title",
    facet_row="metric",
    y_axis=dict(matches=None),
    traces_settings=dict(marker_size=10),
    layout=dict(legend=dict(y=-0.05, orientation="h")),
    height=height,
)
# fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
# fig.update_yaxes(matches="y")
utils.realign_subplot_axes(fig)
fig
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
distance_evaluations_norm = load_discriminant_metrics(
    "/home/laser/dimcat_data/1-0.333/results.tsv"
)  # these were created by having the chord_profile_search/gridsearch.py script process the folder containing
# pickled distance matrices created via this notebook
if ONLY_PAPER_METRICS:
    distance_evaluations_norm = keep_only_paper_metrics(distance_evaluations_norm)
fig = make_scatter_plot(
    distance_evaluations_norm,
    x_col="words",
    y_col="value",
    symbol="norm",
    color="features",
    color_discrete_map=utils.CHORD_PROFILE_COLORS,
    facet_col="delta_title",
    facet_row="metric",
    y_axis=dict(matches=None),
    traces_settings=dict(marker_size=10),
    layout=dict(legend=dict(y=-0.05, orientation="h")),
    title="After L2-normalization",
    height=height,
)
# fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
# fig.update_yaxes(matches="y")
utils.realign_subplot_axes(fig)
fig
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
delta.functions  # Best-performing: Burrow's, Cosine Delta, Eder's Delta, Eder's Simple, Manhattan
```

#### Inspection

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
def get_n_best(distance_evaluations, n=10):
    if distance_evaluations.index.nlevels == 1:
        distance_evaluations.index.rename("i", inplace=True)
    smaller_is_better = ("Entropy", "Cluster Errors", "F-Ratio")  # noqa: F841
    metric_str = distance_evaluations.metric.str
    prefer_smaller_mask = metric_str.startswith(smaller_is_better)
    nlargest = (
        distance_evaluations[~prefer_smaller_mask].groupby("metric").value.nlargest(n)
    )
    best_largest = join_df_on_index(distance_evaluations, nlargest.index).query(
        "not metric.str.startswith(@smaller_is_better)"
    )
    nsmallest = (
        distance_evaluations[prefer_smaller_mask].groupby("metric").value.nsmallest(n)
    )
    best_smallest = join_df_on_index(distance_evaluations, nsmallest.index).query(
        "metric.str.startswith(@smaller_is_better)"
    )
    return pd.concat([best_largest, best_smallest])


n = 10
n_best = get_n_best(distance_evaluations, n=n)
print(f"Selected {len(n_best)} best-performing values, {n} per metric.")
n_best.head()
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
winners = get_n_best(distance_evaluations, n=1)
winners[["features", "norm", "delta_descriptor"]].value_counts()
```

+++ {"jupyter": {"outputs_hidden": false}}

**Selected for further exploration:**

* local_root_ct_rootnorm-cosine-0679 (winner simple score)
* root per globalkey (piecenorm), Quadratic (sqeuclidean z-score), full (1359) (winner

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
winners
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
n_best["delta_descriptor"].value_counts()
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
n_best[["features", "norm"]].value_counts()
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
winners[["features", "norm"]].value_counts()
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
winners.delta_descriptor.value_counts()
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
n_best[["features", "norm", "delta_descriptor"]].value_counts()
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
close_inspection_feature_names = [  # both corpus- and groupwise-normalized
    feature_name for _, feature_name in utils.CHORD_PROFILES.values()
]
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
n_best[n_best.features.isin(close_inspection_feature_names)].groupby(
    ["features", "norm"]
).delta_descriptor.value_counts()
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
close_inspection_deltas = [
    "sqeuclidean-z_score",  # especially for all 3 types of chord profiles
    "manhattan",  # especially for all 3 types of chord-tone profiles
    "manhattan-z_score-eder_std",
    "cosine-z_score",
]
```

+++ {"jupyter": {"outputs_hidden": false}}

## Chord Profiles
### Globalkey

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
METRICS_PATH = os.path.abspath(
    os.path.join(RESULTS_PATH, "..", "data", "delta_gridsearch_norm")
)


def reorder_data(
    df,
    index_groups: Optional[str | Iterable[str]] = (
        "feature_name",
        "norm",
        "delta_descriptor",
        "top_n",
    ),
    iterative_approach: bool = False,
):
    if isinstance(index_groups, str):
        index_groups = [index_groups]
    else:
        index_groups = list(index_groups)
    df = df.set_index(index_groups).sort_index()
    group_ids = (
        df.groupby(level=index_groups)
        .ngroup()
        .reset_index(drop=True)
        .rename("group_id")
    )
    if iterative_approach:
        if len(index_groups) == 1:
            tuples_generator = (
                (group_id, idx, i)
                for group_id, idx, i in zip(group_ids, df.index, count(start=0, step=1))
            )
        else:
            tuples_generator = (
                (group_id,) + tup + (i,)
                for group_id, tup, i in zip(group_ids, df.index, count(start=0, step=1))
            )
        names = ["id"] + df.index.names + ["i"]
        new_ix = pd.MultiIndex.from_tuples(tuples_generator, names=names)
    else:
        idx = df.index.to_frame(index=False)
        i_level = pd.RangeIndex(len(df)).to_series().rename("i")
        new_ix = pd.concat([group_ids, idx, i_level], axis=1)
        new_ix = pd.MultiIndex.from_frame(new_ix)
    df.index = new_ix
    return df


def load_feature_metrics(
    feature_name: str,
    directory: str = METRICS_PATH,
    index_groups: Optional[str | Iterable[str]] = (
        "norm",
        "delta_descriptor",
        "top_n",
    ),
) -> pd.DataFrame:
    candidates = [
        f for f in os.listdir(directory) if f.endswith(".tsv") and feature_name in f
    ]
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
    if index_groups:
        df = reorder_data(df, index_groups=index_groups)
    return df


def concatenate_feature_metrics(
    features=(
        "root_per_globalkey",
        "root_per_localkey",
        "root_per_tonicization",
        "global_root_ct",
        "local_root_ct",
        "tonicization_root_ct",
    ),
    directory: str = METRICS_PATH,
    index_groups: Optional[str | Iterable[str]] = (
        "feature_name",
        "norm",
        "delta_descriptor",
        "top_n",
    ),
):
    df = pd.concat(
        [load_feature_metrics(f, directory, index_groups=None) for f in features],
        keys=list(features),
        names=["feature_name"],
    )
    try:
        df = df.reset_index(0)
    except Exception:
        df = df.reset_index(0, drop=True)
    if index_groups:
        df = reorder_data(df, index_groups=index_groups)
    return df
```

```{raw-cell}
metrics_complete = concatenate_feature_metrics(os.path.abspath(os.path.join(RESULTS_PATH, "..", "data",
"delta_gridsearch_norm")))
metrics_complete = keep_only_paper_metrics(metrics_complete)
excluded = metrics_complete.index.get_level_values("delta_descriptor").str.startswith(('sqeuc', 'euc'))
metrics_complete = metrics_complete[~excluded]
metrics_complete
fig = make_scatter_plot(
    metrics_complete.reset_index(),
    x_col="top_n",
    y_col="value",
    symbol="norm",
    color="features",
    color_discrete_map=utils.CHORD_PROFILE_COLORS,
    facet_col="delta_title",
    facet_row="metric",
    y_axis=dict(matches=None),
    traces_settings=dict(marker_size=2, opacity=0.5),
    layout=dict(legend=dict(y=-0.05, orientation="h")),
    height=height,
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
#save_figure_as(fig, "discriminant_metrics_gridsearch", height=height, width=1300)
fig
```

```{code-cell} ipython3
metrics_complete = concatenate_feature_metrics(
    features=list(utils.CHORD_PROFILES_WITH_BASELINE.keys()),
)
# metrics_complete = concatenate_feature_metrics(
#     features="local_root_ct",
#     directory="/home/laser/git/chord_profile_search/balanced_test/"
# )
# metrics_complete = concatenate_feature_metrics(
#     features=list(utils.CHORD_PROFILES_WITH_BASELINE.keys()),
#     directory = os.path.abspath(
#         os.path.join(RESULTS_PATH, "..", "data", "delta_gridsearch_epoch")
#     )
# )
vocab_proportion = pd.Series(
    metrics_complete.index.get_level_values("top_n"),
    index=metrics_complete.index,
    name="vocab_proportion",
)
vocab_proportion = vocab_proportion.groupby("feature_name").transform(
    lambda S: S / S.max()
)
metrics_complete = pd.concat([metrics_complete, vocab_proportion], axis=1)
if ONLY_PAPER_METRICS:
    metrics_complete = keep_only_paper_metrics(metrics_complete)
profile_names = {
    ("globalkey_profiles", "piecenorm"): "<i>P<sup>tpcG, ||piece||</sup></i>",
    ("localkey_profiles", "piecenorm"): "<i>P<sup>tpcL, ||piece||</sup></i>",
    ("tonicization_profiles", "piecenorm"): "<i>P<sup>tpcT, ||piece||</sup></i>",
    ("root_per_globalkey", "piecenorm"): "<i>P<sup>G, son, ||piece||</sup></i>",
    ("root_per_globalkey", "rootnorm"): "<i>P<sup>G, son, ||root||</sup></i>",
    ("root_per_localkey", "piecenorm"): "<i>P<sup>L, son, ||piece||</sup></i>",
    ("root_per_localkey", "rootnorm"): "<i>P<sup>L, son, ||root||</sup></i>",
    ("root_per_tonicization", "piecenorm"): "<i>P<sup>T, son, ||piece||</sup></i>",
    ("root_per_tonicization", "rootnorm"): "<i>P<sup>T, son, ||root||</sup></i>",
    ("global_root_ct", "piecenorm"): "<i>P<sup>G, ct, ||piece||</sup></i>",
    ("global_root_ct", "rootnorm"): "<i>P<sup>G, ct, ||root||</sup></i>",
    ("local_root_ct", "piecenorm"): "<i>P<sup>L, ct, ||piece||</sup></i>",
    ("local_root_ct", "rootnorm"): "<i>P<sup>L, ct, ||root||</sup></i>",
    ("tonicization_root_ct", "piecenorm"): "<i>P<sup>T, ct, ||piece||</sup></i>",
    ("tonicization_root_ct", "rootnorm"): "<i>P<sup>T, ct, ||root||</sup></i>",
}
color_map = {
    "<i>P<sup>G, ct, ||piece||</sup></i>": "FUCHSIA_700",
    "<i>P<sup>G, ct, ||root||</sup></i>": "FUCHSIA_400",
    "<i>P<sup>L, ct, ||piece||</sup></i>": "INDIGO_700",
    "<i>P<sup>L, ct, ||root||</sup></i>": "INDIGO_400",
    "<i>P<sup>T, ct, ||piece||</sup></i>": "PURPLE_700",
    "<i>P<sup>T, ct, ||root||</sup></i>": "PURPLE_400",
    "<i>P<sup>G, son, ||piece||</sup></i>": "LIME_600",
    "<i>P<sup>G, son, ||root||</sup></i>": "LIME_400",
    "<i>P<sup>L, son, ||piece||</sup></i>": "EMERALD_600",
    "<i>P<sup>L, son, ||root||</sup></i>": "EMERALD_400",
    "<i>P<sup>T, son, ||piece||</sup></i>": "CYAN_600",
    "<i>P<sup>T, son, ||root||</sup></i>": "CYAN_400",
    "<i>P<sup>tpcG, ||piece||</sup></i>": "RED_700",
    "<i>P<sup>tpcL, ||piece||</sup></i>": "RED_600",
    "<i>P<sup>tpcT, ||piece||</sup></i>": "ROSE_500",
}
color_map = {
    trace: utils.TailwindColorsHex.get_color(c) for trace, c in color_map.items()
}
trace_heading = "profiles"
trace_names = pd.Series(
    metrics_complete.index.droplevel((0, 3, 4, 5)).map(profile_names),
    index=metrics_complete.index,
    name=trace_heading,
)
metrics_complete = pd.concat([metrics_complete, trace_names], axis=1)
trace_order = list(color_map.keys())
metrics_complete.delta_title = metrics_complete.delta_title.replace(
    {
        "Cosine Distance": "Cosine",
        "Manhattan Distance": "Manhattan",
        "Burrows' Delta": "Burrows's Delta",
    }
)
delta_order = [
    "Cosine",
    "Cosine Delta",
    "Manhattan",
    "Burrows's Delta",
    "Eder's Delta",
    "Eder's Simple",
]


def subplot_order(S):
    category_order = trace_order if S.name == trace_heading else delta_order
    return S.map({val: i for i, val in enumerate(category_order)})


metrics_complete.sort_values(
    [trace_heading, "delta_title"], key=subplot_order, inplace=True
)
fig = make_scatter_plot(
    metrics_complete,
    x_col="top_n",
    y_col="value",
    color=trace_heading,
    color_discrete_map=color_map,
    facet_col="delta_title",
    facet_row="metric",
    x_axis=dict(tickangle=45),
    y_axis=dict(matches=None),
    traces_settings=dict(marker_size=2, opacity=0.7),
    layout=dict(legend=dict(y=1.1, orientation="h", itemsizing="constant")),
    height=height,
    log_x=True,
)
# replace repeated y-axis labels by a single one. Solution by MichaG via
# https://stackoverflow.com/questions/58167028/single-axis-caption-in-plotly-express-facet-plot
fig.for_each_xaxis(lambda y: y.update(title=""))
fig.update_layout(
    # keep the original annotations and add a list of new annotations:
    annotations=list(fig.layout.annotations)
    + [
        go.layout.Annotation(
            x=0.5,
            y=-0.06,
            font=dict(size=20),
            showarrow=False,
            text="Top n tokens",
            # textangle=-90,
            xref="paper",
            yref="paper",
        )
    ]
)


# fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
# fig.update_yaxes(matches="y")
utils.realign_subplot_axes(fig)
# save_figure_as(fig, "clustering_quality_gridsearch.pdf", height=1479, width=1000)
fig
```

```{code-cell} ipython3
color_map
```

```{code-cell} ipython3
len(metrics_complete)
```

```{code-cell} ipython3
metrics_complete.reset_index().groupby("profiles").top_n.max()
```

```{code-cell} ipython3
correlated_values = (
    metrics_complete.droplevel(-1)
    .set_index("metric", append=True)
    .value.unstack()
    .dropna(axis=0)
)
correlated_values
```

```{code-cell} ipython3
def make_correlation_matrix(correlated_values):
    print(f"{len(correlated_values)=}")
    spearman_correlations, p_values = spearmanr(correlated_values)
    return pd.DataFrame(
        spearman_correlations,
        index=correlated_values.columns,
        columns=correlated_values.columns,
    )


correlation_matrix = make_correlation_matrix(correlated_values)
correlation_matrix
```

```{code-cell} ipython3
correlation_table = (
    delta.get_triangle_values(correlation_matrix, offset=1, lower=True)
    .to_frame()
    .reset_index(allow_duplicates=True)
)
correlation_table.columns = ["", "", "Spearman correlation"]
correlation_table
```

```{code-cell} ipython3
delta_correlated_values = pd.concat(
    {
        delta: df.droplevel("delta_descriptor")
        for delta, df in correlated_values.reset_index("group_id", drop=True).groupby(
            "delta_descriptor"
        )
    },
    axis=1,
)
make_correlation_matrix((delta_correlated_values.swaplevel(axis=1).sort_index(axis=1)))
```

```{code-cell} ipython3
def get_correlations(three_metrics, spearman=False, sort=False):
    """Correlates Simple Score against both ARI values for all deltas"""
    correlated_values = (
        three_metrics.droplevel(-1)
        .set_index("metric", append=True)
        .value.unstack()
        .dropna(axis=0)
    )
    correlations = {}
    if spearman:
        corr_func = spearmanr
        corr_name = "Spearman"
    else:
        corr_func = pearsonr
        corr_name = "Pearson"
    for delta_name, values in correlated_values.groupby("delta_descriptor"):
        pam, ward, simple = values.values.T
        pam_s, pam_p = corr_func(pam, simple)
        ward_s, ward_p = corr_func(ward, simple)
        correlations[(delta_name, "ward")] = {corr_name: ward_s, "p_value": ward_p}
        correlations[(delta_name, "pam")] = {corr_name: pam_s, "p_value": pam_p}
    result = pd.DataFrame.from_dict(correlations, orient="index")
    result.index.names = ["delta", "clustering"]
    if sort:
        return result.sort_values(corr_name, ascending=False)
    return result


three_metrics = metrics_complete.query("~metric.str.startswith('Cluster')")
get_correlations(three_metrics, spearman=True)
```

```{code-cell} ipython3
get_correlations(three_metrics)
```

Means of the upper xth percentile for each of the distance metrics, per evaluation measure:

```{code-cell} ipython3
upper_quantile = 0.005
three_metrics.groupby("metric").apply(
    lambda df: df.groupby("delta_title")
    .value.apply(lambda S: S[S >= S.quantile(1 - upper_quantile)])
    .agg(["median", "mean", "sem"])
)
```

```{code-cell} ipython3
def sort_delta_level(idx):
    if not idx.name == "delta_title":
        return idx
    return idx.map({name: i for i, name in enumerate(delta_order)})


# Die Rankstabelle habe ich dann doch rausgenommen, weil sie zu viel Platz für zu wenig Information braucht
# ranks = (
#     three_metrics.groupby(["metric", "delta_title"])
#     .apply(
#         lambda df: df.groupby("profiles")
#         .value.max()
#         .rank(method="max", ascending=False)
#     )
#     .astype(int)
# )
# ranks.iloc(axis=1)[ranks.sum().argsort()].sort_index(key=sort_delta_level)
fig = make_box_plot(
    three_metrics.reset_index(),
    x_col="profiles",
    y_col="value",
    color="profiles",
    color_discrete_map=color_map,
    # facet_col="delta_title",
    facet_row="metric",
    category_orders=dict(
        delta_title=delta_order,
        profiles=three_metrics.query("metric.str.startswith('Adjust')")
        .groupby("profiles")
        .value.mean()
        .sort_values(ascending=False)
        .index.to_list(),
    ),
    hover_data=["metric", "delta_title", "top_n", "vocab_proportion", "value"],
    height=1000,
    title="Value ranges for all profiles",
)
utils.realign_subplot_axes(fig)
fig
```

```{code-cell} ipython3
best_performing_top_n = (
    three_metrics.reset_index("top_n")
    .groupby(["metric", "delta_title"])
    .apply(
        lambda df: df.loc[
            df.groupby("profiles").value.idxmax(),
            ["profiles", "top_n", "vocab_proportion", "value"],
        ]
    )
)
best_performing_top_n.loc[best_performing_top_n.groupby(level=[0]).value.idxmax()]
```

```{code-cell} ipython3
best_performing_top_n.groupby(level=[0]).apply(
    lambda df: df.groupby(level=1).max().describe()
).unstack()
```

```{raw-cell}
make_bar_plot(
    best_performing_top_n.reset_index(),
    x_col="delta_title",
    y_col="vocab_proportion",
    facet_row="metric",
    color="profiles",
    color_discrete_map=color_map,
    #facet_col="delta_title",
    category_orders=dict(profiles=best_performing_top_n.groupby("profiles").vocab_proportion.median().sort_values().index.to_list()),
    hover_data=["metric", "delta_title"],
    height=1000,
    barmode="group"
    #title="For each delta, the best performing vocab proportions for ARI + Simple Score"
)
```

```{raw-cell}
make_box_plot(
    best_performing_top_n.reset_index(),
    x_col="profiles",
    y_col="vocab_proportion",
    color="profiles",
    color_discrete_map=color_map,
    #facet_col="delta_title",
    facet_row="metric",
    category_orders=dict(profiles=best_performing_top_n.query("metric.str.startswith('Adjust')").groupby("profiles").value.mean().sort_values(ascending=False).index.to_list()),
    hover_data=["metric", "delta_title", "top_n"],
    height=1000,
    title="For each metric, the best performing vocab proportions for all deltas"
)
```

From this plot comes the idea to show vocabulary-size boxes of the best performing values.

```{code-cell} ipython3
def get_upper_quantile_values(metrics, quantile=0.05):
    result = (
        metrics.reset_index("top_n")
        .groupby(["profiles", "metric", "delta_title"])
        .apply(
            lambda df: df.loc[
                df.value >= df.value.quantile(1.0 - quantile),
                ["value", "top_n", "vocab_proportion"],
            ],
            include_groups=False,
        )
    )
    metric_wise_ranking = result.groupby(["metric"]).apply(
        lambda df: df.groupby(["profiles", "delta_title"])
        .value.mean()
        .rank(ascending=False)
    )
    result = result.join(
        metric_wise_ranking.T.stack().rename("overall_rank").astype(int)
    )
    return result


best_ari_rankings = {}
for upper_quantile in (0.05, 0.03, 0.01):
    best_values = get_upper_quantile_values(three_metrics, upper_quantile)
    best_ari_rankings[f"top {upper_quantile:.0%}"] = (
        best_values.query("metric.str.startswith('Adjust')")
        .groupby("profiles")
        .value.mean()
        .rank(ascending=False)
        .astype(int)
        .rename("rank")
    )
best_ari_rankings = pd.concat(best_ari_rankings, names=["quantile"]).to_frame()
make_line_plot(
    best_ari_rankings.reset_index(),
    x_col="quantile",
    y_col="rank",
    color="profiles",
    color_discrete_map=color_map,
    category_orders=dict(
        profiles=best_ari_rankings.groupby("profiles")["rank"]
        .mean()
        .sort_values()
        .index.to_list()
    ),
    markers=True,
    width=400,
    height=800,
    y_axis=dict(autorange="reversed"),
)
```

```{code-cell} ipython3
y_axis = "value"  # choose between "vocab_proportion", "top_n", and "value"
upper_quantile = 0.005
top_n_for_upper_quantile = get_upper_quantile_values(three_metrics, upper_quantile)


def format_percent(perc: float, precision=1) -> str:
    """Converts a decimal fraction to a percentage, rounding to the indicated precision,
    with trailing zeros removed. E.g., whatever precision is indicated, format_percent(0.5)
    always returns "50%".
    """
    return re.sub(r"\.?0+%$", "%", f"{perc:.{precision}%}")


fig = make_box_plot(
    top_n_for_upper_quantile.reset_index(),
    x_col="profiles",
    y_col=y_axis,
    color="profiles",
    color_discrete_map=color_map,
    # facet_col="delta_title",
    facet_row="metric",
    category_orders=dict(
        delta_title=delta_order,
        profiles=top_n_for_upper_quantile.query("metric.str.startswith('Adjust')")
        .groupby("profiles")
        .value.mean()
        .sort_values(ascending=False)
        .index.to_list(),
    ),
    hover_data=["metric", "delta_title", "top_n", "vocab_proportion", "value"],
    height=1000,
    title=f"Value ranges summarized for the top-performing {format_percent(upper_quantile)} scores",
)
utils.realign_subplot_axes(fig)
fig
```

```{code-cell} ipython3
fig = make_box_plot(
    top_n_for_upper_quantile.reset_index(),
    x_col="profiles",
    y_col="value",
    color="profiles",
    color_discrete_map=color_map,
    facet_col="delta_title",
    facet_row="metric",
    category_orders=dict(
        delta_title=delta_order,
        profiles=top_n_for_upper_quantile.query("metric.str.startswith('Adjust')")
        .groupby("profiles")
        .value.mean()
        .sort_values(ascending=False)
        .index.to_list(),
    ),
    hover_data=["metric", "delta_title", "top_n", "vocab_proportion", "value"],
    height=1000,
    title=f"Evaluation metrics for the peak {format_percent(upper_quantile)} values",
)
utils.realign_subplot_axes(fig)
fig
```

```{code-cell} ipython3
fig = make_box_plot(
    top_n_for_upper_quantile.reset_index(),
    x_col="overall_rank",
    y_col="value",
    color="profiles",
    color_discrete_map=color_map,
    # facet_col="delta_title",
    facet_row="metric",
    category_orders=dict(
        delta_title=delta_order,
        profiles=top_n_for_upper_quantile.query("metric.str.startswith('Adjust')")
        .groupby("profiles")
        .value.mean()
        .sort_values(ascending=False)
        .index.to_list(),
    ),
    hover_data=["metric", "delta_title", "top_n", "vocab_proportion", "value"],
    height=1000,
    layout=dict(hovermode="x unified"),
    title=f"Peak {format_percent(upper_quantile)} values ordered by their mean",
)
utils.realign_subplot_axes(fig)
fig
```

```{code-cell} ipython3
top_n_ward = (
    top_n_for_upper_quantile.loc(axis=0)[:, "Adjusted Rand Index (Ward)"]
    .query("overall_rank in [1,2,3,4,5]")
    .sort_values(["overall_rank", "value"])
)
top_n_ward
```

```{code-cell} ipython3
top_n_for_upper_quantile.loc(axis=0)[:, "Adjusted Rand Index (PAM)"].query(
    "overall_rank in [1,2,3,4,5]"
).sort_values(["overall_rank", "value"])
```

```{code-cell} ipython3
test = (
    get_upper_quantile_values(three_metrics, 0.005)
    .loc(axis=0)[:, "Adjusted Rand Index (Ward)"]
    .query("overall_rank in [1]")
    .groupby(["overall_rank", "profiles", "metric", "delta_title"])
    .describe(percentiles=[0, 0.5, 1])
)
test
```

```{code-cell} ipython3
height = 1479
width = 1000


def make_top_ranks_scatter_data(
    measurements,
    upper_quantile: float = 0.01,
    top_k: int = 10,
    percentiles: Tuple[float, float, float] = (0.25, 0.5, 0.75),
):
    assert (
        len(percentiles) == 3
    ), f"Percentiles must be three decimals that designate (lower_error, value, upper_error). Got: {percentiles}"
    lower, middle, upper = (format_percent(perc) for perc in percentiles)
    top_n_for_upper_quantile = get_upper_quantile_values(measurements, upper_quantile)
    show_ranks = list(range(1, top_k + 1))  # noqa: F841
    scatter_data = (
        top_n_for_upper_quantile.query("overall_rank in @show_ranks")
        .groupby(["overall_rank", "profiles", "metric", "delta_title"])
        .describe(percentiles=percentiles)
    )
    scatter_data.columns = utils.merge_index_levels(scatter_data.columns)
    scatter_data.rename(
        columns=lambda col: re.sub(f", {middle}", "", col), inplace=True
    )
    # turn percentiles into error bar values via subtraction
    error_bar_columns = []
    for col_name in ("value", "top_n", "vocab_proportion"):
        medians = scatter_data[col_name]
        error_bar_columns.extend(
            (
                (scatter_data[col_name + f", {upper}"] - medians).rename(
                    col_name + "_err"
                ),
                (medians - scatter_data[col_name + f", {lower}"]).rename(
                    col_name + "_err_minus"
                ),
            )
        )
    scatter_data = pd.concat([scatter_data] + error_bar_columns, axis=1)
    return scatter_data


def subselect_scatter(scatter_data, col_start):
    result = scatter_data.filter(regex=f"^(value|{col_start})")
    result = result.rename(
        columns=lambda col: "x" + col[len(col_start) :]
        if col.startswith(col_start)
        else col
    )
    return result


scatter_data = make_top_ranks_scatter_data(
    three_metrics, upper_quantile=0.005, top_k=15, percentiles=(0, 0.5, 1)
)
top_n_scatter_data = subselect_scatter(scatter_data, "top_n")
vocab_proportion_scatter_data = subselect_scatter(scatter_data, "vocab_proportion")
double_scatter_data = pd.concat(
    [top_n_scatter_data, vocab_proportion_scatter_data],
    keys=["Absolute vocabulary size", "Relative Vocabulary Size"],
    names=["vocabulary size"],
).reset_index()
plot_args = dict(
    df=double_scatter_data,
    x_col="x",
    error_x="x_err",
    error_x_minus="x_err_minus",
    y_col="value",
    error_y="value_err",
    error_y_minus="value_err_minus",
    color="profiles",
    color_discrete_map=color_map,
    symbol="delta_title",
    symbol_sequence=[
        # 'hexagram',
        "star",
        "asterisk",
        "y-up",
        "y-down",
        "x-thin",
        "hash",
        # 'circle',
        "y-left",
        "y-right",
        # 'x',
        # 'triangle-up',
        # 'octagon',
        # 'circle-x',
        # 'hourglass',
        # 'square',
        # 'diamond',
        "line-ne",
        "line-nw",
        "cross",
        "triangle-down",
        "triangle-left",
        "triangle-right",
        "triangle-ne",
        "triangle-se",
        "triangle-sw",
        "triangle-nw",
        "pentagon",
        "hexagon",
        "hexagon2",
        "star-triangle-up",
        "star-triangle-down",
        "star-square",
        "star-diamond",
        "diamond-tall",
        "diamond-wide",
        "bowtie",
        "circle-cross",
        "square-cross",
        "square-x",
        "diamond-cross",
        "diamond-x",
        "cross-thin",
        "line-ew",
        "line-ns",
        "arrow-up",
        "arrow-down",
        "arrow-left",
        "arrow-right",
        "arrow-bar-up",
        "arrow-bar-down",
        "arrow-bar-left",
        "arrow-bar-right",
    ],
    traces_settings=dict(marker_size=12, marker_line_width=2),
    facet_col="vocabulary size",
    facet_row="metric",
    category_orders=dict(
        metric=sorted(double_scatter_data.metric.unique()),
        delta_title=double_scatter_data.groupby("delta_title")
        .size()
        .sort_values(ascending=False)  # sort by occurrence under the top n
        .index.unique()
        .to_list(),
        profiles=double_scatter_data.groupby("profiles")
        .size()
        .sort_values(ascending=False)  # sort by occurrence under the top n
        .index.unique()
        .to_list(),
    ),
    hover_data=[
        "overall_rank",
        "delta_title",
        "metric",
        "value, max",
        "value, min",
    ],
    labels=dict(x=""),
    layout=dict(
        legend=dict(
            x=-0.05,
            y=1.13,
            orientation="h",
            itemsizing="constant",
            title_text=None,
            font_size=13,
        )
    ),
    height=height,
    width=width,
)
fig = make_scatter_plot(**plot_args)
for trace in fig.data:
    trace.marker.line.color = trace.marker.color
utils.realign_subplot_axes(fig, True, True)
save_figure_as(fig, "clustering_quality_best_profiles.pdf", height=height, width=width)
fig
```

```{raw-cell}
def show_single_top_ranks_scatter(
    measurements,
    metric="Adjusted Rand Index (Ward)",
    upper_quantile: float = 0.01,
    top_k: int = 10,
    x_axis: Literal["top_n", "vocab_proportion"] = "vocab_proportion",
    percentiles: Tuple[float, float, float] = (0.25, 0.5, 0.75),
    delta_symbols=True,
    **kwargs,
):
    scatter_data = make_top_ranks_scatter_data(
        measurements, upper_quantile, top_k, percentiles
    )
    scatter_data = scatter_data.loc(axis=0)[:, :, metric]
    plot_args = dict(
        df=scatter_data.reset_index(),
        x_col=x_axis,
        error_x=f"{x_axis}_err",
        error_x_minus=f"{x_axis}_err_minus",
        y_col="value",
        error_y="value_err",
        error_y_minus="value_err_minus",
        color="profiles",
        color_discrete_map=color_map,
        category_orders=dict(
            profiles=scatter_data.sort_values("value", ascending=False)
            .index.get_level_values("profiles")
            .unique()
            .to_list()
        ),
        hover_data=[
            "overall_rank",
            "delta_title",
            "metric",
            "top_n",
            "vocab_proportion",
            "value, max",
            "value, min",
        ],
        title=(f"{metric} values vs. vocabulary sizes<br><sup>of the peak {format_percent(upper_quantile)} "
               f"measurements ")
        f"for each of the {top_k} top-performing configurations</sup>",
    )
    if delta_symbols:
        plot_args.update(
            symbol="delta_title",
            traces_settings=dict(marker_size=12),
        )
    plot_args.update(kwargs)
    return make_scatter_plot(**plot_args)


fig = show_single_top_ranks_scatter(
    three_metrics,
    top_k=20,
    upper_quantile=0.005,
    percentiles=(0, 0.5, 1),
    height=800,
    width=1000,
    # x_axis="top_n"
)
fig
```

```{code-cell} ipython3
get_upper_quantile_values(three_metrics, 0.005).query(
    "profiles.str.contains('root') & overall_rank < 16"
)
```

```{code-cell} ipython3
double_scatter_data.query("profiles == '<i>P<sup>G, ct, ||root||</sup></i>'")
```

```{raw-cell}
show_single_top_ranks_scatter(
    three_metrics,
    top_k=20,
    upper_quantile=0.005,
    percentiles=(0, 0.5, 1),
    height=800,
    width=1000,
    x_axis="top_n",
)
```

```{raw-cell}
show_single_top_ranks_scatter(
    three_metrics,
    metric="Adjusted Rand Index (PAM)",
    top_k=20,
    upper_quantile=0.005,
    percentiles=(0, 0.5, 1),
    height=800,
    width=1000,
    # x_axis="top_n"
)
```

```{raw-cell}
show_single_top_ranks_scatter(
    three_metrics,
    metric="Simple Score",
    top_k=20,
    upper_quantile=0.005,
    percentiles=(0, 0.5, 1),
    height=800,
    width=1000,
    # x_axis="top_n"
)
```

```{code-cell} ipython3
winners = get_n_best(metrics_complete, n=1)
winners
```

```{code-cell} ipython3
len(winners)
```

```{code-cell} ipython3
winners.groupby(["feature_name", "norm", "top_n"]).size()
```

```{code-cell} ipython3
metrics_complete.head(1)
```

```{code-cell} ipython3
fishers_only = metrics_complete.query('metric == "Fisher\'s LD"')
fishers_only = pd.concat(
    [fishers_only, fishers_only.value.rank(ascending=False).rename("rank")], axis=1
)
```

```{code-cell} ipython3
def make_outliers_mask(
    mask_support,
    iqr_coefficient: float = 1.5,
    reverse: bool = False,
    verbose: bool = False,
):
    is_dataframe = isinstance(mask_support, pd.DataFrame)
    if is_dataframe and len(mask_support.columns) == 1:
        mask_support = mask_support.iloc(axis=1)[0]
        is_dataframe = False
    Q1 = mask_support.quantile(0.25)
    Q3 = mask_support.quantile(0.75)
    distance = iqr_coefficient * (Q3 - Q1)
    lower_fence, upper_fence = Q1 - distance, Q3 + distance
    if verbose:
        print(f"Q1={Q1}")
        print(f"l={lower_fence}")
        print(f"Q3={Q3}")
        print(f"u={upper_fence}")
    within_mask = mask_support.ge(lower_fence) & mask_support.le(upper_fence)
    if is_dataframe:
        within_mask = within_mask.all(axis=1)
    if not reverse:
        within_mask = ~within_mask
    return within_mask


def remove_outliers(
    df_or_s: pd.DataFrame | pd.Series,
    columns: Optional[str | Iterable[str]] = "value",
    groups: Optional[str | Iterable[str]] = None,
    iqr_coefficient: float = 1.5,
    reverse: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | pd.Series:
    is_dataframe = isinstance(df_or_s, pd.DataFrame)
    on_subset = bool(columns)
    if on_subset:
        assert (
            is_dataframe
        ), f"Input is not a dataframe so I don't accept the columns argument {columns!r}."
        if isinstance(columns, str):
            columns = [columns]
        else:
            columns = list(columns)
    if groups:
        if on_subset:
            within_mask = df_or_s.groupby(groups, group_keys=False)[columns].apply(
                make_outliers_mask,
                iqr_coefficient=iqr_coefficient,
                reverse=not reverse,
                verbose=verbose,
            )
        else:
            within_mask = df_or_s.groupby(groups, group_keys=False).apply(
                make_outliers_mask,
                iqr_coefficient=iqr_coefficient,
                reverse=not reverse,
                verbose=verbose,
            )
        within_mask = within_mask.reindex(df_or_s.index)
    else:
        mask_support = df_or_s[columns] if on_subset else df_or_s
        within_mask = make_outliers_mask(
            mask_support, iqr_coefficient, not reverse, verbose
        )
    return df_or_s[within_mask]
```

```{code-cell} ipython3
def compare_single_metric(metrics_complete, metric, iqr_coefficient=10):
    """Displays faceted plot with all values for a single metric, removing extreme outliers from each subplot.
    All y-axes range up to the global maximum value for better comparability.
    """
    filtered = metrics_complete.query(f'metric == "{metric}"')
    # filtered = pd.concat(
    #     [filtered, filtered.value.rank(ascending=False).rename("rank")], axis=1
    # )
    n_before = len(filtered)
    filtered = remove_outliers(
        filtered, iqr_coefficient=iqr_coefficient, groups=["norm", "delta_title"]
    )
    n_after = len(filtered)
    print(f"Dropped {n_before - n_after} outliers.")
    fig = make_scatter_plot(
        filtered,
        x_col="top_n",
        y_col="value",
        symbol="norm",
        color="features",
        color_discrete_map=utils.CHORD_PROFILE_COLORS,
        facet_col="norm",
        facet_row="delta_title",
        y_axis=dict(matches=None),
        traces_settings=dict(marker_size=3, opacity=0.5),
        layout=dict(legend=dict(y=-0.05, orientation="h")),
        title=f"{metric} without extreme outliers ({iqr_coefficient}*IQR)",
        height=1500,
        log_y=False,
    )
    # fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
    fig.update_yaxes(matches="y")
    # for row_idx, row_figs in enumerate(fig._grid_ref):
    #     for col_idx, col_fig in enumerate(row_figs):
    #         fig.update_yaxes(
    #             row=row_idx + 1,
    #             col=col_idx + 1,
    #             matches="y" + str(len(row_figs) * row_idx + 1),
    #         )
    # save_figure_as(fig, "chord_tone_profiles_evaluation", height=4000, width=1300)
    return fig


if not ONLY_PAPER_METRICS:
    compare_single_metric(metrics_complete, "Fisher's LD")
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
def show_gridsearch(df, save_as: Optional[str], height=height, width=1300, **kwargs):
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
    return fig
```

## Comapring piecenorm vs. rootnorm for single type of profile

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
root_per_globalkey = load_feature_metrics("root_per_globalkey")
if ONLY_PAPER_METRICS:
    root_per_globalkey = keep_only_paper_metrics(root_per_globalkey)
show_gridsearch(root_per_globalkey, "root_per_globalkey_gridsearch")
```

+++ {"jupyter": {"outputs_hidden": false}}

### Localkey

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
root_per_local = load_feature_metrics("root_per_localkey")
if ONLY_PAPER_METRICS:
    root_per_local = keep_only_paper_metrics(root_per_local)
show_gridsearch(root_per_globalkey, "root_per_localkey_gridsearch")
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
root_per_tonicization = load_feature_metrics("root_per_tonicization")
if ONLY_PAPER_METRICS:
    root_per_tonicization = keep_only_paper_metrics(root_per_tonicization)
show_gridsearch(root_per_tonicization, "root_per_tonicization_gridsearch")
```

```{code-cell} ipython3
global_root_ct = load_feature_metrics("global_root_ct")
if ONLY_PAPER_METRICS:
    global_root_ct = keep_only_paper_metrics(global_root_ct)
show_gridsearch(global_root_ct, "global_root_ct_gridsearch")
```

```{code-cell} ipython3
local_root_ct = load_feature_metrics("local_root_ct")
if ONLY_PAPER_METRICS:
    local_root_ct = keep_only_paper_metrics(local_root_ct)
show_gridsearch(local_root_ct, "local_root_ct_gridsearch")
```

```{code-cell} ipython3
tonicization_root_ct = load_feature_metrics("tonicization_root_ct")
if ONLY_PAPER_METRICS:
    tonicization_root_ct = keep_only_paper_metrics(tonicization_root_ct)
show_gridsearch(tonicization_root_ct, "tonicization_root_ct_gridsearch")
```

Best-performing: Burrow's, Cosine Delta, Eder's Delta, Eder's Simple, Manhattan
