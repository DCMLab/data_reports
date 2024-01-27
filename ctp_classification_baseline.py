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
import os
import re
from typing import Dict, Tuple
from zipfile import ZipFile

import delta
import numpy as np
import pandas as pd
from dimcat import resources
from dimcat.plotting import write_image
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

import utils

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


# %%
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
    return data, metadata


data, metadata = load_original_data("~/git/chord_profile_search/")

# %%
corpus = data[("local_root_ct", "rootnorm")]
corpus.shape

# %%
PIECE_YEARS = metadata.get_composition_years().rename("mean_composition_year")
PIECE_MODE = metadata.annotated_key.str.islower().map({True: "minor", False: "major"})


def show_pca(corpus, **kwargs):
    info = f"{corpus.metadata.features}, {corpus.metadata.norm}"
    return utils.plot_pca(corpus, info=info, **kwargs)


show_pca(corpus, color=corpus.group_index_level, n_components=3)

# %%
random_state = np.random.RandomState(42)
pca = PCA(n_components=2, random_state=random_state)
scaled_pca = make_pipeline(StandardScaler(), pca)
lda = LinearDiscriminantAnalysis(n_components=2)
scaled_lda = make_pipeline(StandardScaler(), lda)
nca = NeighborhoodComponentsAnalysis(n_components=2, random_state=random_state)
scaled_nca = make_pipeline(
    StandardScaler(),
    nca,
)


# %%
def make_info(corpus, name) -> str:
    info = f"{corpus.metadata.features}, {corpus.metadata.norm}"
    if name:
        info = f"{name} of the {info}"
    return info


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


# %%
def show_lda(corpus, standardize=False, **kwargs):
    name = "LDA (standardized)" if standardize else "LDA"
    info = make_info(corpus, name)
    y = corpus.group_index_level
    data = StandardScaler().fit_transform(corpus) if standardize else corpus
    return utils.plot_lda(data, y=y, info=info, **kwargs)


def show_nca(corpus, standardize=False, **kwargs):
    name = "NCA (standardized)" if standardize else "NCA"
    info = make_info(corpus, name)
    y = corpus.group_index_level
    data = StandardScaler().fit_transform(corpus) if standardize else corpus
    return utils.plot_nca(data, y=y, info=info, **kwargs)


show_nca(corpus, color=corpus.group_index_level)

# %%
show_lda(corpus.top_n(280), color=corpus.group_index_level)


# %%
def make_split(matrix: resources.PrevalenceMatrix | pd.DataFrame, test_size=0.2):
    if isinstance(matrix, pd.DataFrame):
        X = matrix
    else:
        X = matrix.relative
    # first, drop corpora containing only one piece
    pieces_per_corpus = X.groupby(level="corpus").size()
    more_than_one = pieces_per_corpus[pieces_per_corpus > 1].index
    X = X.loc[more_than_one]
    # get the labels from the index level, then drop the level
    y = X.index.get_level_values("corpus")
    X = X.reset_index(level="corpus", drop=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=np.random.RandomState(42)
    )
    return X_train, X_test, y_train, y_test


class Classification:
    def __init__(self, matrix: resources.PrevalenceMatrix, clf, cv=None, test_size=0.2):
        self.matrix = matrix
        self.clf = clf
        self.cv = cv
        self.X_train, self.X_test, self.y_train, self.y_test = make_split(
            self.matrix, test_size=test_size
        )
        self.score = None
        self.confusion_matrix = None
        self.classification_report = None
        self.fit()

    def fit(
        self,
    ):
        self.clf.fit(self.X_train, self.y_train)
        self.y_pred = self.clf.predict(self.X_test)
        self.score = self.clf.score(self.X_test, self.y_test)
        self.classification_report = classification_report(
            self.y_test, self.y_pred, output_dict=True, zero_division=0.0
        )
        self.confusion_matrix = confusion_matrix(
            self.y_test, self.y_pred, labels=self.clf.classes_
        )
        return self.score

    def show_confusion_matrix(self, fontsize=22):
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 12

        plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
        plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
        # sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, cmap="RdBu")
        disp = ConfusionMatrixDisplay(
            confusion_matrix=self.confusion_matrix, display_labels=self.clf.classes_
        )
        disp.plot()
        plt.xticks(rotation=90)
        plt.show()


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

# %%
svc = Classification(
    matrix=data["local_root_ct"],
    clf=LinearSVC(dual="auto"),
)

# %%
svc.show_confusion_matrix(fontsize=10)


# %%
def compare_feature_performance(
    data: Dict[str, resources.PrevalenceMatrix],
    classifier=RandomForestClassifier(),
):
    results = []
    for feature, matrix in data.items():
        clf = Classification(matrix=matrix, clf=classifier)
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

# %%
# performs worse
# doubly_compare_feature_performance(data_pca, groupwise_data_pca, classifier=RandomForestClassifier())

# %%
doubly_compare_feature_performance(
    data, groupwise_data, classifier=LinearSVC(dual="auto")
)

# %%
# performs the same
# doubly_compare_feature_performance(data_pca, groupwise_data_pca, classifier=LinearSVC(dual="auto"))

# %%
doubly_compare_feature_performance(
    data, groupwise_data, classifier=SVC(gamma=2, C=1, random_state=42)
)

# %%
doubly_compare_feature_performance(
    data,
    groupwise_data,
    classifier=GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
)

# %%
doubly_compare_feature_performance(
    data, groupwise_data, classifier=KNeighborsClassifier(39)
)

# %%
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
