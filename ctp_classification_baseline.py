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

# %%
import contextlib
import os
import re
from collections import defaultdict
from typing import Dict, Literal, Optional, Tuple
from zipfile import ZipFile

import delta
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from dimcat import resources
from dimcat.plotting import make_scatter_plot, write_image
from matplotlib import pyplot as plt
from sklearn.covariance import OAS
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from tqdm.auto import tqdm

import utils

# %% [markdown]
# # Chord-Tone Profiles Classification Baselines

# %% mystnb={"code_prompt_hide": "Hide imports", "code_prompt_show": "Show imports"} tags=["hide-cell"]
# %load_ext autoreload
# %autoreload 2


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


# %% jupyter={"outputs_hidden": false}
def make_info(corpus, name) -> str:
    info = f"{corpus.metadata.features}, {corpus.metadata.norm}"
    if name:
        info = f"{name} of the {info}"
    return info


def iter_zipped_matrices(
    filepath: str = "data.zip",
    document_describer: delta.util.DocumentDescriber = None,
):
    with ZipFile(filepath, "r") as zip_handler:
        for i, file in enumerate(zip_handler.namelist()):
            match = re.match(r"^(.+)_(rootnorm|piecenorm)\.tsv$", file)
            if not match:
                continue
            feature_name, norm = match.groups()
            print(f"Extracting {file}")
            with zip_handler.open(file) as f:
                matrix = pd.read_csv(f, sep="\t", index_col=0)
            try:
                metadata = delta.Metadata.from_zip_file(file, zip_handler)
            except KeyError:
                metadata = None
            corpus = delta.Corpus(
                matrix, metadata=metadata, document_describer=document_describer
            )
            corpus.index.rename("corpus, piece", inplace=True)
            yield feature_name, norm, corpus


def load_original_data(
    directory: str,
    zip_name: str = "data.zip",
    metadata_name: str = "metadata.tsv",
    drop_with_less_than: Optional[int] = 3,
) -> Tuple[Dict[Tuple[str, str], delta.Corpus], resources.Metadata]:
    directory = utils.resolve_dir(directory)
    zip_path = os.path.join(directory, zip_name)
    metadata_path = os.path.join(directory, metadata_name)
    dd = delta.TsvDocumentDescriber(metadata_path)
    metadata = resources.Metadata.from_resource_path(metadata_path)
    data = {
        (feature_name, norm): corpus
        for feature_name, norm, corpus in iter_zipped_matrices(zip_path, dd)
    }
    if drop_with_less_than:
        print("DROPPING...")
        data = {
            k: utils.drop_groups_with_less_than(v, drop_with_less_than)
            for k, v in data.items()
        }
        metadata = utils.drop_groups_with_less_than(metadata, drop_with_less_than)
    return data, metadata


data, metadata = load_original_data("~/git/chord_profile_search/")
PIECE_YEARS = metadata.get_composition_years().rename("mean_composition_year")
PIECE_MODE = metadata.annotated_key.str.islower().map({True: "minor", False: "major"})

# %% jupyter={"outputs_hidden": false}
corpus = data[("local_root_ct", "rootnorm")]
corpus.shape

# %% jupyter={"outputs_hidden": false}
X_train, X_test, y_train, y_test = utils.make_split(corpus)

# %% jupyter={"outputs_hidden": false}


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument.
    Amazing out-of-the-box solution by featuredpeow via https://stackoverflow.com/a/58936697
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# %% jupyter={"outputs_hidden": false}


def get_scores(name, clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return {
        (name, "accuracy"): report["accuracy"],
        (name, "weighted_avg_f1"): report["weighted avg"]["f1-score"],
    }


oracle = OAS(store_precision=False, assume_centered=False)

classifiers = dict(
    lda=LinearDiscriminantAnalysis(solver="lsqr", shrinkage=None),
    lda_shrink=LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    lda_oas=LinearDiscriminantAnalysis(solver="lsqr", covariance_estimator=oracle),
)
classifiers.update(
    dict(
        lda_scaled=make_pipeline(StandardScaler(), classifiers["lda"]),
        lda_scaled_shrink=make_pipeline(StandardScaler(), classifiers["lda_shrink"]),
        lda_scaled_oas=make_pipeline(StandardScaler(), classifiers["lda_oas"]),
    )
)

n_classifiers = len(classifiers)
search_range = range(1, X_train.shape[1] + 1)
total_rows = len(search_range) * n_classifiers
with tqdm_joblib(tqdm(desc="Computing scores", total=total_rows)) as _:
    with joblib.Parallel(n_jobs=-1, return_as="generator") as parallel:
        rows = defaultdict(dict)
        try:
            reduced_data_iterator = (
                (X_train.iloc[:, :n_features], X_test.iloc[:, :n_features])
                for n_features in search_range
            )
            for i, scores in enumerate(
                parallel(
                    joblib.delayed(get_scores)(
                        name, clf, x_train, y_train, x_test, y_test
                    )
                    for x_train, x_test in reduced_data_iterator
                    for name, clf in classifiers.items()
                )
            ):
                rows[i // n_classifiers + 1].update(scores)
        finally:
            results = pd.DataFrame.from_dict(rows, orient="index")
            results.index.rename("n_features", inplace=True)
            results.columns.set_names(["classifier", "score"], inplace=True)
            results = results.stack(-1).reset_index().melt(["n_features", "score"])
results.head()

# %%
results = results.stack(-1).reset_index().melt(["n_features", "score"])
results.head()

# %%

make_scatter_plot(
    results,
    x_col="n_features",
    y_col="value",
    facet_col="score",
    color="classifier",
    opacity=0.4,
    title="LDA classification scores for local_root_ct, rootnorm",
    color_discrete_sequence=px.colors.qualitative.G10,
)

# %%
df = pd.DataFrame(
    {
        ("AA", "A"): {0: "a", 1: "b", 2: "c"},
        ("AA", "B"): {0: "d", 1: "e", 2: "f"},
        ("AA", "C"): {0: "g", 1: "h", 2: "i"},
        ("BB", "D"): {0: 1, 1: 2, 2: 3},
        ("BB", "E"): {0: 4, 1: 5, 2: 6},
    }
)
df


# %% jupyter={"outputs_hidden": false}
def show_pca(corpus, **kwargs):
    info = f"{corpus.metadata.features}, {corpus.metadata.norm}"
    return utils.plot_pca(corpus, info=info, **kwargs)


show_pca(corpus, color=corpus.group_index_level, n_components=3)

# %% jupyter={"outputs_hidden": false}
random_state = np.random.RandomState(42)
pca = PCA(n_components=2, random_state=random_state)
scaled_pca = make_pipeline(StandardScaler(), pca)
lda = LinearDiscriminantAnalysis(n_components=2, shrinkage="auto", solver="eigen")
scaled_lda = make_pipeline(StandardScaler(), lda)
nca = NeighborhoodComponentsAnalysis(n_components=2, random_state=random_state)
scaled_nca = make_pipeline(
    StandardScaler(),
    nca,
)

# %% jupyter={"outputs_hidden": false}
transformations = {
    "PCA": pca,
    "PCA (standardized)": scaled_pca,
    "LDA": lda,
    "LDA (standardized)": scaled_lda,
    "NCA": nca,
    "NCA (standardized)": scaled_nca,
}
for name, transformation in transformations.items():
    info = make_info(corpus, name)
    coordinates = transformation.fit_transform(corpus, corpus.group_index_level)
    x, y = coordinates.columns
    fig = utils.plot_component_analysis(
        coordinates, color=corpus.group_index_level, info=info
    )
    fig.show()


# %% jupyter={"outputs_hidden": false}
def show_lda(
    corpus,
    standardize=False,
    shrinkage: Literal["auto"] | float = "auto",
    solver: Literal["svd", "lsqr", "eigen"] = "eigen",
    **kwargs,
):
    name = "LDA (standardized)" if standardize else "LDA"
    info = make_info(corpus, name)
    y = corpus.group_index_level
    data = StandardScaler().fit_transform(corpus) if standardize else corpus
    return utils.plot_lda(
        data, y=y, shrinkage=shrinkage, solver=solver, info=info, **kwargs
    )


def show_nca(corpus, standardize=False, **kwargs):
    name = "NCA (standardized)" if standardize else "NCA"
    info = make_info(corpus, name)
    y = corpus.group_index_level
    data = StandardScaler().fit_transform(corpus) if standardize else corpus
    return utils.plot_nca(data, y=y, info=info, **kwargs)


show_lda(corpus, color=corpus.group_index_level)

# %% jupyter={"outputs_hidden": false}
show_lda(corpus, standardize=True, color=corpus.group_index_level)

# %% jupyter={"outputs_hidden": false}
show_lda(corpus.top_n(280), color=corpus.group_index_level)

# %%
# class CrossValidated(Classification):
#     def __init__(
#         self,
#         matrix: resources.PrevalenceMatrix,
#         clf,
#         cv,
#     ):
#         super().__init__(matrix, clf, cv)
#         self.cv_results = None
#         self.estimators = None
#         self.scores = None
#         self.best_estimator = None
#         self.best_score = None
#         self.best_params = None
#         self.best_index = None
#         self.best_estimator = None
#
#     def cross_validate(
#         self,
#     ):
#         self.cv_results = cross_validate(
#             self.clf,
#             self.X_train,
#             self.y_train,
#             cv=self.cv,
#             n_jobs=-1,
#             return_estimator=True,
#         )
#         self.estimators = self.cv_results["estimator"]
#         self.scores = pd.DataFrame(
#             {
#                 "RandomForestClassifier": self.cv_results["test_score"],
#             }
#         )
#         self.best_index = self.scores.idxmax()
#         self.best_estimator = self.estimators[self.best_index]
#         self.best_score = self.scores.max()
#         self.best_params = self.best_estimator.get_params()
#         return self.cv_results

# %% jupyter={"outputs_hidden": false}
svc = utils.Classification(
    matrix=data["local_root_ct"],
    clf=LinearSVC(dual="auto"),
)

# %% jupyter={"outputs_hidden": false}
svc.show_confusion_matrix(fontsize=10)


# %% jupyter={"outputs_hidden": false}
def compare_feature_performance(
    data: Dict[str, resources.PrevalenceMatrix],
    classifier=RandomForestClassifier(),
):
    results = []
    for feature, matrix in data.items():
        clf = utils.Classification(matrix=matrix, clf=classifier)
        report = clf.classification_report
        result = dict(
            feature=feature,
            weighted_avg_f1=report["weighted avg"]["f1-score"],
            macro_avg_f1=report["macro avg"]["f1-score"],
            accuracy=report["accuracy"],
            weighted_avg_precision=report["weighted avg"]["precision"],
            weighted_avg_recall=report["weighted avg"]["recall"],
            macro_avg_precision=report["macro avg"]["precision"],
            macro_avg_recall=report["macro avg"]["recall"],
        )
        results.append(result)
    return pd.DataFrame(results).sort_values("weighted_avg_f1", ascending=False)


def doubly_compare_feature_performance(
    data: Dict[str, resources.PrevalenceMatrix],
    groupwise_data: Dict[str, resources.PrevalenceMatrix],
    classifier=RandomForestClassifier(),
):
    report1 = compare_feature_performance(data, classifier=classifier)
    report2 = compare_feature_performance(groupwise_data, classifier=classifier)
    return pd.concat([report1, report2], keys=["corpuswise", "groupwise"]).sort_values(
        "weighted_avg_f1", ascending=False
    )


groupwise_data = data

doubly_compare_feature_performance(
    data, groupwise_data, classifier=RandomForestClassifier()
)

# %% jupyter={"outputs_hidden": false}
# performs worse
# doubly_compare_feature_performance(data_pca, groupwise_data_pca, classifier=RandomForestClassifier())

# %% jupyter={"outputs_hidden": false}
doubly_compare_feature_performance(
    data, groupwise_data, classifier=LinearSVC(dual="auto")
)

# %% jupyter={"outputs_hidden": false}
# performs the same
# doubly_compare_feature_performance(data_pca, groupwise_data_pca, classifier=LinearSVC(dual="auto"))

# %% jupyter={"outputs_hidden": false}
doubly_compare_feature_performance(
    data, groupwise_data, classifier=SVC(gamma=2, C=1, random_state=42)
)

# %% jupyter={"outputs_hidden": false}
doubly_compare_feature_performance(
    data,
    groupwise_data,
    classifier=GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
)

# %% jupyter={"outputs_hidden": false}
doubly_compare_feature_performance(
    data, groupwise_data, classifier=KNeighborsClassifier(39)
)

# %% jupyter={"outputs_hidden": false}
doubly_compare_feature_performance(
    data, groupwise_data, classifier=QuadraticDiscriminantAnalysis()
)


# # %%
# scores = pd.DataFrame(
#     {
#         "RandomForestClassifier": cv_results["test_score"],
#     }
# )
# ax = scores.plot.kde(legend=True)
# ax.set_xlabel("Accuracy score")
# # ax.set_xlim([0, 0.7])
# _ = ax.set_title(
#     "Density of the accuracy scores for the different multiclass strategies"
# )
#
# # %%
# best_index = scores.idxmax()
# best_estimator = cv_results["estimator"][best_index]
# best_score = scores.max()
# best_params = best_estimator.get_params()
# print(f"Best score: {best_score}")
# print(f"Best params: {best_params}")
#
# # %%
# scores
#
# # %%
# best_index
