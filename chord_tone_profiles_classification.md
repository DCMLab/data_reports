---
jupytext:
  formats: md:myst,ipynb,py:percent
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: revamp
  language: python
  name: revamp
---

# Chord Profiles


```{code-cell}
---
mystnb:
  code_prompt_hide: Hide imports
  code_prompt_show: Show imports
tags: [hide-cell]
---
%load_ext autoreload
%autoreload 2

import os
from typing import Dict, List, Tuple

import dimcat as dc
import ms3
import numpy as np
import pandas as pd
from dimcat import resources
from dimcat.plotting import write_image
from git import Repo
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

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

```{code-cell}

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


def make_data(
    chord_slices: resources.DimcatResource,
    features: Dict[str, Tuple[str | List[str], str]],
) -> Dict[str, resources.PrevalenceMatrix]:
    data = {}
    for feature_name, (feature_columns, info) in features.items():
        print(f"Computing prevalence matrix for {info}")
        analyzer_config.update(columns=feature_columns)
        prevalence_matrix = chord_slices.apply_step(analyzer_config)
        data[feature_name] = prevalence_matrix
    return data


data = make_data(chord_slices, features)
groupwise_data = {
    feature: matrix.get_groupwise_prevalence()
    for feature, matrix in data.items()
    if matrix.columns.nlevels > 1
}
```

```{code-cell}
pca = PCA()
pca.set_output(transform="pandas")
data_pca = {
    feature: pca.fit_transform(matrix.relative) for feature, matrix in data.items()
}
groupwise_data_pca = {
    feature: pca.fit_transform(matrix.relative)
    for feature, matrix in groupwise_data.items()
}
```

```{code-cell}
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
```

```{code-cell}
svc = Classification(
    matrix=data["local_root_ct"],
    clf=LinearSVC(dual="auto"),
)
```

```{code-cell}
svc.show_confusion_matrix(fontsize=10)
```

```{code-cell}
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


doubly_compare_feature_performance(
    data, groupwise_data, classifier=RandomForestClassifier()
)
```

```{code-cell}
# performs worse
# doubly_compare_feature_performance(data_pca, groupwise_data_pca, classifier=RandomForestClassifier())
```

```{code-cell}
doubly_compare_feature_performance(
    data, groupwise_data, classifier=LinearSVC(dual="auto")
)
```

```{code-cell}
# performs the same
# doubly_compare_feature_performance(data_pca, groupwise_data_pca, classifier=LinearSVC(dual="auto"))
```

```{code-cell}
doubly_compare_feature_performance(
    data, groupwise_data, classifier=SVC(gamma=2, C=1, random_state=42)
)
```

```{code-cell}
doubly_compare_feature_performance(
    data,
    groupwise_data,
    classifier=GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
)
```

```{code-cell}
doubly_compare_feature_performance(
    data, groupwise_data, classifier=KNeighborsClassifier(39)
)
```

```{code-cell}
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
```
