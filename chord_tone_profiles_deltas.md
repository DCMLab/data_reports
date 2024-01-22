---
jupytext:
  formats: md:myst,ipynb,py:percent
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: pydelta
  language: python
  name: pydelta
---

# Chord Profiles

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide imports
  code_prompt_show: Show imports
tags: [hide-cell]
---
import math

%load_ext autoreload
%autoreload 2
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, NamedTuple, Optional
from zipfile import ZipFile

import delta
import dimcat as dc
import ms3
import pandas as pd
from dimcat import resources
from dimcat.plotting import make_scatter_plot, write_image
from git import Repo
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

import utils

plt.rcParams["figure.dpi"] = 300

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
```

```{code-cell}
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
D
```

```{code-cell}
chord_slices = utils.get_sliced_notes(D)
chord_slices.head(5)
```

## pydelta Corpus objects

```{code-cell}
describer = delta.TsvDocumentDescriber(D.get_metadata().reset_index())


def make_pydelta_corpus(
    matrix: pd.DataFrame,
    info: str = "chords",
    absolute: bool = True,
) -> delta.Corpus:
    metadata = delta.Metadata(
        features=info,
        ordered=False,
        words=None,
        corpus="Distant Listening Corpus",
        complete=True,
        frequencies=not absolute,
    )
    corpus = delta.Corpus(matrix, document_describer=describer, metadata=metadata)
    corpus.index = utils.merge_index_levels(corpus.index)
    return corpus
```

```{code-cell}
features = {

    "chord_reduced": (
        ["chord_reduced_and_mode", "fifths_over_local_tonic"],
        "reduced chords",
    ),
    "numerals": (
        ["effective_localkey_is_minor", "numeral", "fifths_over_local_tonic"],
        "effective numerals",
    ),
    "roots_local": (
        ["root", "fifths_over_local_tonic"],
        "root intervals over local tonic",
    ),
    "roots_global": (
        ["root_fifths_over_global_tonic", "fifths_over_local_tonic"],
        "root intervals over global tonic",
    ),
}

data = {}


class DataTuple(NamedTuple):
    corpus: delta.Corpus
    groupwise: delta.Corpus = None
    prevalence_matrix: Optional[resources.PrevalenceMatrix] = None


analyzer_config = dc.DimcatConfig(
    "PrevalenceAnalyzer",
    index=["corpus", "piece"],
)

for feature_name, (feature_columns, info) in features.items():
    print(f"Computing prevalence matrix and pydelta corpus for {info}")
    analyzer_config.update(columns=feature_columns)
    prevalence_matrix = chord_slices.apply_step(analyzer_config)
    corpus = make_pydelta_corpus(prevalence_matrix.relative, info=info, absolute=False)
    groupwise_prevalence = prevalence_matrix.get_groupwise_prevalence(
        column_levels=feature_columns[:-1]
    )
    groupwise = make_pydelta_corpus(
        groupwise_prevalence.relative, info=f"groupwise {info}", absolute=False
    )
    data[feature_name] = DataTuple(corpus, groupwise, prevalence_matrix)
```

## Compute deltas

```{code-cell}
delta.functions.deltas.keys()
```

```{code-cell}
selected_deltas = {
    name: func
    for name, func in delta.functions.deltas.items()
    if name in ["cosine", "cosine-z_score", "manhattan-z_score", "manhattan-z_score-eder_std"]
}
```

```{code-cell}

# n_vocab_sizes = 5
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
    vocab_sizes: int | Iterable[int] = 5,
) -> Dict[str, Dict[int, Dict[str, DataTuple]]]:
    distance_matrices = defaultdict(dict)
    for feature_name, (corpus, groupwise, prevalence_matrix) in data.items():
        if isinstance(vocab_sizes, int):
            vocabulary_size = prevalence_matrix.n_types
            stride = math.ceil(vocabulary_size / vocab_sizes)
            vocab_sizes = [i * stride for i in range(1, vocab_sizes + 1)]
        for vocab_size in vocab_sizes:
            print(f"Computing deltas for {feature_name} (vocab size {vocab_size})")
            corpus_top_n = corpus.top_n(vocab_size)
            groupwise_top_n = groupwise.top_n(vocab_size)
            # small = delta.Corpus(
            #             corpus=corpus_top_n.iloc[:10],
            #             document_describer=corpus_top_n.document_describer,
            #             metadata=corpus_top_n.metadata,
            #             complete=False,
            #             frequencies=True)
            # small_groupwise = delta.Corpus(
            #             corpus=groupwise_top_n.iloc[:10],
            #             document_describer=groupwise_top_n.document_describer,
            #             metadata=groupwise_top_n.metadata,
            #             complete=False,
            #             frequencies=True)
            n_types = corpus_top_n.shape[
                1
            ]  # may be difference from vocab_size for the last iteration
            corpus_top_n.metadata.ntypes = n_types
            groupwise_top_n.metadata.ntypes = n_types
            distance_matrices[feature_name][vocab_size] = apply_deltas(
                corpus_top_n, groupwise_top_n
            )
    return distance_matrices


def store_distance_matrices(distance_matrices, filepath):
    for feature_name, feature_data in distance_matrices.items():
        for vocab_size, delta_data in feature_data.items():
            for delta_name, delta_results in delta_data.items():
                print(".", end="")
                name = f"{feature_name}_{vocab_size}_{delta_name}"
                delta_results.corpus.save_to_zip(name + ".tsv", filepath)
                delta_results.groupwise.save_to_zip(name + "_groupwise.tsv", filepath)


def load_distance_matrices(filepath):
    distance_matrices = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: [None, None]))
    )
    print(f"Loading {filepath}", end="")
    with ZipFile(filepath, "r") as zip_handler:
        for file in zip_handler.namelist():
            print(".", end="")
            match = re.match(r"(.+)_(\d+)_(.+?)(_groupwise)?\.tsv", file)
            if not match:
                continue
            with zip_handler.open(file) as f:
                matrix = pd.read_csv(f, sep="\t", index_col=0)
            metadata = delta.Metadata.from_zip_file(file, zip_handler)
            dm = delta.DistanceMatrix(
                matrix, metadata=metadata, document_describer=describer
            )
            feature_name, vocab_size, delta_name, groupwise = match.groups()
            position = int(bool(groupwise))
            distance_matrices[feature_name][int(vocab_size)][delta_name][position] = dm
    result = {}
    for feature_name, feature_data in distance_matrices.items():
        feature_results = {}
        for vocab_size, delta_data in feature_data.items():
            feature_results[vocab_size] = {
                delta_name: DataTuple(*delta_data)
                for delta_name, delta_data in delta_data.items()
            }
        result[feature_name] = feature_results
    return result


def get_distance_matrices(
    data: Dict[str, DataTuple],
    vocab_sizes: int | Iterable[int] = 5,
    name: Optional[str] = "chord_tone_profile_deltas",
    basepath: Optional[str] = None,
):
    filepath = None
    if name:
        if basepath is None:
            basepath = dc.get_setting("default_basepath")
        basepath = utils.resolve_dir(basepath)
        name = "chord_tone_profile_deltas"
        if isinstance(vocab_sizes, int):
            zip_file = f"{name}_{vocab_sizes}.zip"
        else:
            zip_file = f"{name}_{'-'.join(map(str, vocab_sizes))}.zip"
        filepath = os.path.join(basepath, zip_file)
        if os.path.isfile(filepath):
            return load_distance_matrices(filepath)
    distance_matrices = compute_deltas(data, vocab_sizes)
    if filepath:
        print(f"Storing distance matrices to {filepath}", end="")
        store_distance_matrices(distance_matrices, filepath)
    return distance_matrices


distance_matrices = get_distance_matrices(data, vocab_sizes=(10, 100, 1000))
```

```{code-cell}
distance_matrices["roots_local"][679]["cosine"].corpus.metadata
```

```{code-cell}
def get_distance_evaluations(distance_matrices):
    distance_metrics_rows = []
    try:
        for feature_name, feature_data in distance_matrices.items():
            for vocab_size, delta_data in feature_data.items():
                for delta_name, delta_results in delta_data.items():
                    for norm, distance_matrix in delta_results._asdict().items():
                        if distance_matrix is None:
                            continue
                        print(".", end="")
                        row = {
                            "feature_name": feature_name,
                            "vocab_size": vocab_size,
                            "delta": delta_name,
                            "norm": norm,
                        }
                        # row.update(distance_matrix.metadata)
                        for metric, value in distance_matrix.evaluate().items():
                            distance_metrics_rows.append(dict(
                                row,
                                metric=metric,
                                value=value
                            ))
                        clustering = delta.Clustering(distance_matrix)
                        for metric, value in clustering.evaluate().items():
                            distance_metrics_rows.append(dict(
                                row,
                                metric=metric,
                                value=value
                            ))
    except KeyboardInterrupt:
        return pd.DataFrame(distance_metrics_rows)
    distance_evaluations = pd.DataFrame(distance_metrics_rows)
    return distance_evaluations

distance_evaluations = get_distance_evaluations(distance_matrices)
distance_evaluations
```

```{code-cell}
# vocab_size = distance_evaluations.vocab_size.vocab_size(distance_evaluations.vocab_size <= 100, 1000)
# distance_evaluations.feature_name += "<br>" + vocab_size.astype(str)
# distance_evaluations
```

```{code-cell}
fig = make_scatter_plot(
    distance_evaluations,
    x_col="vocab_size",
    y_col="value",
    symbol="norm",
    color="feature_name",
    facet_col="delta",
    facet_row="metric",
    y_axis=dict(matches=None),
    traces_settings=dict(marker_size=10),
    height=2000,
)
fig.for_each_xaxis(lambda xaxis: xaxis.update(showticklabels=True))
save_figure_as(fig, "chord_tone_profiles_evaluation", height=2300, width=1200)
fig
```
