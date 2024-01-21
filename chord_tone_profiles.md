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
%load_ext autoreload
%autoreload 0
import os
from importlib import reload
from typing import Iterable, List, Literal, Optional, Tuple, Union

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
    str2pd_interval,
    transpose_notes_to_c,
)
from dimcat.plotting import make_bar_plot, make_scatter_plot, write_image
from dimcat.steps.analyzers import prevalence
from git import Repo
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram  # , linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import Normalizer, RobustScaler, StandardScaler

import utils
from dendrograms import Dendrogram, TableDocumentDescriber

plt.rcParams["figure.dpi"] = 300

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
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


def plot_document_frequency(
    chord_tones: resources.PrevalenceMatrix, info: str = "chord tones", **kwargs
):
    df = chord_tones.document_frequencies()
    vocabulary = merge_columns_into_one(df.index.to_frame(index=False), join_str=True)
    doc_freq_data = pd.DataFrame(
        dict(
            chord_tones=vocabulary,
            document_frequency=df.values,
            rank=range(1, len(vocabulary) + 1),
        )
    )
    D, V = chord_tones.shape
    settings = dict(
        x_col="rank",
        y_col="document_frequency",
        hover_data="chord_tones",
        log_x=True,
        log_y=True,
        title=f"Document frequency of {info} (D = {D}, V = {V})",
    )
    if kwargs:
        settings.update(kwargs)
    fig = make_scatter_plot(
        doc_freq_data,
        **settings,
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


def replace_boolean_column_level_with_mode(
    matrix: resources.PrevalenceMatrix,
    level: int = 0,
    name: str = "mode",
):
    """Replaces True with 'minor' and False with 'major' in the given column index level and renames it.
    Function operates inplace.
    """
    old_columns = matrix._df.columns
    bool_values = old_columns.levels[level]
    mode_values = bool_values.map(
        {True: "minor", False: "major", "True": "minor", "False": "major"}
    )
    new_columns = old_columns.set_levels(mode_values, level=0)
    new_columns.set_names(name, level=level, inplace=True)
    matrix._df.columns = new_columns


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


def compare_corpus_frequencies(
    chord_slices: resources.DimcatResource, features: str | Iterable[str]
):
    if isinstance(features, str):
        features = [features]
    doc_freqs = []
    for feature in features:
        analyzer = prevalence.PrevalenceAnalyzer(index="corpus", columns=feature)
        matrix = analyzer.process(chord_slices)
        doc_freqs.append(
            matrix.document_frequencies(name="corpus_frequency")
            .sort_values(ascending=False)
            .astype("Int64")
        )
    if len(doc_freqs) == 1:
        return doc_freqs[0]
    return pd.concat([series.reset_index() for series in doc_freqs], axis=1)


def make_cosine_distances(
    tf,
    standardize=False,
    norm: Optional[Literal["l1", "l2", "max"]] = None,
    flat_index: bool = False,  # useful for plotting
):
    if standardize:
        scaler = StandardScaler()
        scaler.set_output(transform="pandas")
        tf = scaler.fit_transform(tf)
    if norm:
        scaler = Normalizer(norm=norm)
        scaler.set_output(transform="pandas")
        tf = scaler.fit_transform(tf)
    if flat_index:
        index = utils.merge_index_levels(tf.index)
    else:
        index = tf.index
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
def get_sliced_notes(
    D: dc.Dataset,
    basepath: Optional[str] = None,
    cache_name: Optional[str] = "chord_slices",
):
    if basepath is None:
        basepath = utils.resolve_dir(dc.get_setting("default_basepath"))
    if cache_name:
        cache_path = os.path.join(basepath, f"{cache_name}.resource.json")
        if os.path.isfile(cache_path):
            chord_slices = dc.deserialize_json_file(cache_path)
            chord_slices.load()
            # work around for correct IntervalIndex deserialization
            converted = list(map(str2pd_interval, chord_slices.index.levels[2]))
            chord_slices._df.index = chord_slices._df.index.set_levels(
                converted, level=2
            )
            return chord_slices
    label_slicer = slicers.HarmonyLabelSlicer()
    sliced_D = label_slicer.process(D)
    sliced_notes = sliced_D.get_feature(resources.Notes)
    slice_info = label_slicer.slice_metadata.droplevel(-1)
    slice_info["root_fifths_over_global_tonic"] = (
        ms3.transform(
            slice_info[["effective_localkey_resolved", "globalkey_is_minor"]],
            ms3.roman_numeral2fifths,
        ).rename("root_fifths_over_global_tonic")
        + slice_info.root
    )
    merge_columns = [
        col for col in slice_info.columns if col not in sliced_notes.columns
    ]
    slice_info = join_df_on_index(
        slice_info[merge_columns], sliced_notes.index, how="right"
    )
    chord_slices = pd.concat([sliced_notes, slice_info], axis=1)
    chord_slices = pd.concat([chord_slices, transpose_notes_to_c(chord_slices)], axis=1)
    chord_slices = resources.DimcatResource.from_dataframe(chord_slices, "chord_slices")
    chord_slices.store_resource(basepath=basepath, name="chord_slices")
    return chord_slices


chord_slices = get_sliced_notes(D)
chord_slices.head(5)
```

## Document frequencies of chord features

```{code-cell} ipython3
compare_corpus_frequencies(
    chord_slices,
    [
        "chord_reduced_and_mode",
        ["effective_localkey_is_minor", "numeral"],
        "root",
        "root_fifths_over_global_tonic",
    ],
)
```

## Create chord-tone profiles for multiple chord features

Tokens are `(feature, ..., chord_tone)` tuples.

```{code-cell} ipython3
chord_reduced: resources.PrevalenceMatrix = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["chord_reduced_and_mode", "fifths_over_local_tonic"],
        index=["corpus", "piece"],
    )
)
print(f"Shape: {chord_reduced.shape}")
```

```{code-cell} ipython3
numerals: resources.PrevalenceMatrix = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["effective_localkey_is_minor", "numeral", "fifths_over_local_tonic"],
        index=["corpus", "piece"],
    )
)
print(f"Shape: {numerals.shape}")
replace_boolean_column_level_with_mode(numerals)
```

```{code-cell} ipython3
roots: resources.PrevalenceMatrix = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["root", "fifths_over_local_tonic"],
        index=["corpus", "piece"],
    )
)
print(f"Shape: {roots.shape}")
```

```{code-cell} ipython3
root_fifths_over_global_tonic = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["root_fifths_over_global_tonic", "fifths_over_local_tonic"],
        index=["corpus", "piece"],
    )
)
print(f"Shape: {root_fifths_over_global_tonic.shape}")
```

### Document frequencies of the tokens

```{code-cell} ipython3
fig = plot_document_frequency(chord_reduced)
save_figure_as(fig, "document_frequency_of_chord_tones")
fig
```

```{code-cell} ipython3
plot_document_frequency(numerals, info="numerals")
```

```{code-cell} ipython3
plot_document_frequency(roots, info="roots")
```

```{code-cell} ipython3
plot_document_frequency(
    root_fifths_over_global_tonic, info="root relative to global tonic"
)
```

## Principal Component Analyses

```{code-cell} ipython3
# chord_reduced.query("piece in ['op03n12a', 'op03n12b']").dropna(axis=1, how='all')
```

```{code-cell} ipython3
metadata = D.get_metadata()
CORPUS_YEARS = utils.corpus_mean_composition_years(metadata)
PIECE_YEARS = metadata.get_composition_years().rename("mean_composition_year")
utils.plot_pca(
    chord_reduced.relative,
    info="chord-tone profiles of reduced chords",
    color=PIECE_YEARS,
)
```

```{code-cell} ipython3
utils.plot_pca(
    chord_reduced.combine_results("corpus").relative,
    info="chord-tone profiles of reduced chords",
    color=CORPUS_YEARS,
    size=5,
)
```

```{code-cell} ipython3
utils.plot_pca(
    numerals.relative, info="numeral profiles of numerals", color=PIECE_YEARS
)
```

```{code-cell} ipython3
utils.plot_pca(
    numerals.combine_results("corpus").relative,
    info="chord-tone profiles of numerals",
    color=CORPUS_YEARS,
    size=5,
)
```

```{code-cell} ipython3
utils.plot_pca(
    roots.relative, info="root profiles of chord roots (local)", color=PIECE_YEARS
)
```

```{code-cell} ipython3
utils.plot_pca(
    roots.combine_results("corpus").relative,
    info="chord-tone profiles of chord roots (local)",
    color=CORPUS_YEARS,
    size=5,
)
```

```{code-cell} ipython3
utils.plot_pca(
    root_fifths_over_global_tonic.relative,
    info="root profiles of chord roots (global)",
    color=PIECE_YEARS,
)
```

```{code-cell} ipython3
utils.plot_pca(
    root_fifths_over_global_tonic.combine_results("corpus").relative,
    info="chord-tone profiles of chord roots (global)",
    color=CORPUS_YEARS,
    size=5,
)
```

### Scaled PCA

minuscule augmentation of explained variance.

```{code-cell} ipython3
scaler = RobustScaler()
scaler.set_output(transform="pandas")
scaled = scaler.fit_transform(chord_reduced.relative)
utils.plot_pca(scaled, info="chord-tone profiles", color=PIECE_YEARS)
```

# Classification

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    LeaveOneOut,
    cross_val_score,
    cross_validate,
    train_test_split,
)

```{code-cell} ipython3
from sklearn.svm import LinearSVC


def make_split(
    matrix: resources.PrevalenceMatrix,
):
    X = matrix.relative
    # first, drop corpora containing only one piece
    pieces_per_corpus = X.groupby(level="corpus").size()
    more_than_one = pieces_per_corpus[pieces_per_corpus > 1].index
    X = X.loc[more_than_one]
    # get the labels from the index level, then drop the level
    y = X.index.get_level_values("corpus")
    X = X.reset_index(level="corpus", drop=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=np.random.RandomState(42)
    )
    return X_train, X_test, y_train, y_test


class Classification:
    def __init__(
        self,
        matrix: resources.PrevalenceMatrix,
        clf,
        cv,
    ):
        self.matrix = matrix
        self.clf = clf
        self.cv = cv
        self.X_train, self.X_test, self.y_train, self.y_test = make_split(self.matrix)
        self.score = None

    def fit(
        self,
    ):
        self.clf.fit(self.X_train, self.y_train)
        self.y_pred = self.clf.predict(self.X_test)
        self.score = self.clf.score(self.X_test, self.y_test)
        self.classification_report = classification_report(
            self.y_test, self.y_pred, output_dict=True
        )
        self.confusion_matrix = confusion_matrix(
            self.y_test, self.y_pred, labels=self.clf.classes_
        )
        return self.score

    def show_confusion_matrix(self):
        return ConfusionMatrixDisplay(
            confusion_matrix=self.confusion_matrix, display_labels=self.clf.classes_
        )


class CrossValidated(Classification):
    def __init__(
        self,
        matrix: resources.PrevalenceMatrix,
        clf,
        cv,
    ):
        super().__init__(matrix, clf, cv)
        self.cv_results = None
        self.estimators = None
        self.scores = None
        self.best_estimator = None
        self.best_score = None
        self.best_params = None
        self.best_index = None
        self.best_estimator = None

    def cross_validate(
        self,
    ):
        self.cv_results = cross_validate(
            self.clf,
            self.X_train,
            self.y_train,
            cv=self.cv,
            n_jobs=-1,
            return_estimator=True,
        )
        self.estimators = self.cv_results["estimator"]
        self.scores = pd.DataFrame(
            {
                "RandomForestClassifier": self.cv_results["test_score"],
            }
        )
        self.best_index = self.scores.idxmax()
        self.best_estimator = self.estimators[self.best_index]
        self.best_score = self.scores.max()
        self.best_params = self.best_estimator.get_params()
        return self.cv_results


# clf = RandomForestClassifier()
clf = LinearSVC()
cv = LeaveOneOut()
RFC = Classification(matrix=chord_reduced, clf=clf, cv=cv)
RFC.fit()
```

```{code-cell} ipython3
RFC.show_confusion_matrix().plot()
```

```{code-cell} ipython3
import seaborn as sns

clf_report = RFC.classification_report
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, cmap="RdBu")
```

```{code-cell} ipython3
scores = pd.DataFrame(
    {
        "RandomForestClassifier": cv_results["test_score"],
    }
)
ax = scores.plot.kde(legend=True)
ax.set_xlabel("Accuracy score")
# ax.set_xlim([0, 0.7])
_ = ax.set_title(
    "Density of the accuracy scores for the different multiclass strategies"
)
```

```{code-cell} ipython3
best_index = scores.idxmax()
best_estimator = cv_results["estimator"][best_index]
best_score = scores.max()
best_params = best_estimator.get_params()
print(f"Best score: {best_score}")
print(f"Best params: {best_params}")
```

```{code-cell} ipython3
scores
```

```{code-cell} ipython3
best_index
```

```{code-cell} ipython3
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
```

```{code-cell} ipython3
cos_dist_chord_tones = make_cosine_distances(
    chord_reduced.relative, standardize=False, flat_index=False
)
# np.fill_diagonal(cos_dist_chord_tones.values, np.nan)
cos_dist_chord_tones.iloc[:10, :10]
```

```{code-cell} ipython3
ABC = cos_dist_chord_tones.loc(axis=1)[["ABC"]]
ABC.shape
```

```{code-cell} ipython3
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
```

```{code-cell} ipython3
numerals.iloc[:10, :10]
```

```{code-cell} ipython3
corpus_numerals: resources.PrevalenceMatrix = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["effective_localkey_is_minor", "numeral"],
        index="corpus",
    )
)
replace_boolean_column_level_with_mode(corpus_numerals)
corpus_numerals.document_frequencies()
```

```{code-cell} ipython3
# selected_numerals = corpus_frequencies_old.loc[
#     corpus_frequencies_old.iloc[:, -1] == 39, "numeral"
# ]
# absolute_numeral_data = chord_slices[chord_slices.numeral.isin(selected_numerals)]
# absolute_numeral_data.head(5)
```

```{code-cell} ipython3
reload(utils)
```

```{code-cell} ipython3
culled_chord_tones = chord_reduced.get_culled_matrix(1 / 3)
culled_chord_tones.shape
```

```{code-cell} ipython3
plot_cosine_distances(culled_chord_tones.relative, standardize=True)
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


cos_distance_matrix = make_cosine_distances(culled_chord_tones.relative)
cos_distance_matrix
```

```{code-cell} ipython3
labels = cos_distance_matrix.index.to_list()
ac = AgglomerativeClustering(
    metric="precomputed", linkage="complete", distance_threshold=0, n_clusters=None
)
ac.fit_predict(cos_distance_matrix)
plot_dendrogram(ac, truncate_mode="level", p=0)
plt.title("Hierarchical Clustering using maximum cosine distances")
# plt.savefig('aggl_mvt_max_cos.png', bbox_inches='tight')
```

```{raw-cell}
sliced_notes.store_resource(
    basepath="~/dimcat_data",
    name="sliced_notes"
)
```

```{raw-cell}
restored = dc.deserialize_json_file("/home/laser/dimcat_data/sliced_notes.resource.json")
restored.df
```

```{code-cell} ipython3
ac.fit_predict(cos_distance_matrix)
lm = linkage_matrix(ac)  # probably want to use this to have better control
# lm = linkage(cos_distance_matrix)
describer = TableDocumentDescriber(metadata.reset_index())
plt.figure(figsize=(10, 60))
ddg = Dendrogram(lm, describer, labels)
```

```{code-cell} ipython3
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
```

```{code-cell} ipython3
chord_tone_profiles = make_chord_tone_profile(chord_slices)
```

```{code-cell} ipython3
plot_chord_profiles(chord_tone_profiles, "i, minor", log_y=True)
```

```{code-cell} ipython3
plot_chord_profiles(chord_tone_profiles, "V7, minor")
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
sliced_D = analyzed_D.apply_step(slicers.KeySlicer)
sliced_D
```

```{code-cell} ipython3
key_slices = sliced_D.get_feature("HarmonyLabels")
```

```{code-cell} ipython3
sliced_notes = sliced_D.get_feature(resources.Notes)
sliced_notes
```

```{code-cell} ipython3
slicer = sliced_D.get_last_step("Slicer")
type(slicer)
```

```{code-cell} ipython3
normal_notes = analyzed_D.get_feature("Notes")
normal_notes
```

```{code-cell} ipython3
normal_notes
```

```{code-cell} ipython3
sn = slicer.process_resource(normal_notes)
sn
```

```{code-cell} ipython3
normal_notes = analyzed_D.get_feature(resources.Notes)
normal_notes
```

```{code-cell} ipython3
harmony_labels.loc[
    harmony_labels["scale_degrees_major"] == ("1", "3", "5", "b7"), "chord"
].value_counts()
```

```{code-cell} ipython3
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
px.scatter(
    sd_major_occurrences,
    x="rank",
    y="frequency",
    log_x=True,
    log_y=True,
)
```

```{code-cell} ipython3


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
sd_major_occurrences.iloc[130:150]
```

```{code-cell} ipython3
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
chord_proportions.make_ranking_table(["mode"], drop_cols="chord_and_mode")
```

```{code-cell} ipython3
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
piece_profiles.sort_index(
    key=lambda _: pd.Index(piece_profiles.sum(axis=1)), ascending=False
).iloc[:50, :50]
```

```{code-cell} ipython3
corpus_proportions = chord_proportions.combine_results().droplevel("chord_and_mode")
corpus_proportions
```

```{code-cell} ipython3
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
mask_with_sum = pd.concat(
    [chord_occurrence_mask, chord_occurrence_mask.sum(axis=1).rename("sum")], axis=1
)
mask_with_sum.to_csv(make_output_path("chord_occurrence_mask", "tsv"), sep="\t")
```

```{code-cell} ipython3
analyzed_D
```
