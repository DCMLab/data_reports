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

```{code-cell} ipython3
import contextlib
import os
from collections import defaultdict
from typing import Dict, Literal, Optional

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
```

# Chord-Tone Profiles Classification Baselines

```{code-cell} ipython3
---
mystnb:
  code_prompt_hide: Hide imports
  code_prompt_show: Show imports
tags: [hide-cell]
---
%load_ext autoreload
%autoreload 2


plt.rcParams["figure.dpi"] = 300

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
```

```{code-cell} ipython3
RESULTS_PATH = os.path.expanduser("~/git/diss/31_profiles/")
os.makedirs(RESULTS_PATH, exist_ok=True)


def make_output_path(filename, extension=None):
    return utils.make_output_path(
        filename, extension, RESULTS_PATH, use_subfolders=True
    )


def save_figure_as(fig, filename, directory=RESULTS_PATH, **kwargs):
    if not any(key in kwargs for key in ("height", "width")):
        kwargs["width"] = 1280
        kwargs["height"] = 720
    write_image(fig, filename, directory, **kwargs)
```

```{code-cell} ipython3
def make_info(corpus, name) -> str:
    info = f"{corpus.metadata.features}, {corpus.metadata.norm}"
    if name:
        info = f"{name} of the {info}"
    return info


data, metadata = utils.load_profiles()
PIECE_YEARS = metadata.get_composition_years().rename("mean_composition_year")
PIECE_MODE = metadata.annotated_key.str.islower().map({True: "minor", False: "major"})
```

```{code-cell} ipython3
corpus = data[("local_root_ct", "rootnorm")]
corpus.shape
```

```{code-cell} ipython3


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
```

```{code-cell} ipython3


def get_scores(name, clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return {
        (name, "accuracy"): report["accuracy"],
        (name, "weighted_avg_f1"): report["weighted avg"]["f1-score"],
    }


def compare_lda_classifiers(corpus):
    X_train, X_test, y_train, y_test = utils.make_split(corpus)
    oracle = OAS(store_precision=False, assume_centered=False)

    classifiers = dict(
        lda=LinearDiscriminantAnalysis(solver="lsqr", shrinkage=None),
        lda_shrink=LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        lda_oas=LinearDiscriminantAnalysis(solver="lsqr", covariance_estimator=oracle),
    )
    classifiers.update(
        dict(
            lda_scaled=make_pipeline(StandardScaler(), classifiers["lda"]),
            lda_scaled_shrink=make_pipeline(
                StandardScaler(), classifiers["lda_shrink"]
            ),
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
                return results
```

```{code-cell} ipython3
def get_lda_comparison(corpus, feature_acronym: Optional[str] = None) -> pd.DataFrame:
    if feature_acronym:
        output_path = make_output_path(f"lda_comparison_{feature_acronym}", "tsv")
    if os.path.isfile(output_path):
        print(f"Loading {output_path}")
        return pd.read_csv(output_path, sep="\t")
    results = compare_lda_classifiers(corpus)
    results.to_csv(output_path, sep="\t", index=False)
    print(f"Saved {output_path}")
    return results


lctr_results = get_lda_comparison(data[("local_root_ct", "rootnorm")], "lctr")
lctr_results.head()
```

```{code-cell} ipython3
make_scatter_plot(
    lctr_results,
    x_col="n_features",
    y_col="value",
    facet_col="score",
    color="classifier",
    opacity=0.4,
    title="LDA classification scores for local_root_ct, rootnorm",
    color_discrete_sequence=px.colors.qualitative.G10,
)
```

```{code-cell} ipython3
rpgr_results = compare_lda_classifiers(data[("root_per_globalkey", "rootnorm")])
rpgr_results.head()
```

```{code-cell} ipython3

make_scatter_plot(
    rpgr_results,
    x_col="n_features",
    y_col="value",
    facet_col="score",
    color="classifier",
    opacity=0.4,
    title="LDA classification scores for local_root_ct, rootnorm",
    color_discrete_sequence=px.colors.qualitative.G10,
)
```

```{code-cell} ipython3
rpgr_results.groupby(["classifier", "score"]).value.mean().sort_values(ascending=False)
```

```{code-cell} ipython3
def show_pca(corpus, **kwargs):
    info = f"{corpus.metadata.features}, {corpus.metadata.norm}"
    return utils.plot_pca(corpus, info=info, **kwargs)


show_pca(corpus, color=corpus.group_index_level, n_components=3)
```

```{code-cell} ipython3
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
```

```{code-cell} ipython3
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
```

```{code-cell} ipython3
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
```

```{code-cell} ipython3
show_lda(corpus, standardize=True, color=corpus.group_index_level)
```

```{code-cell} ipython3
show_lda(corpus.top_n(280), color=corpus.group_index_level)
```

```{code-cell} ipython3
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

```{code-cell} ipython3
svc = utils.Classification(
    matrix=data["local_root_ct"],
    clf=LinearSVC(dual="auto"),
)
```

```{code-cell} ipython3
svc.show_confusion_matrix(fontsize=10)
```

```{code-cell} ipython3
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
```

```{code-cell} ipython3
# performs worse
# doubly_compare_feature_performance(data_pca, groupwise_data_pca, classifier=RandomForestClassifier())
```

```{code-cell} ipython3
doubly_compare_feature_performance(
    data, groupwise_data, classifier=LinearSVC(dual="auto")
)
```

```{code-cell} ipython3
# performs the same
# doubly_compare_feature_performance(data_pca, groupwise_data_pca, classifier=LinearSVC(dual="auto"))
```

```{code-cell} ipython3
doubly_compare_feature_performance(
    data, groupwise_data, classifier=SVC(gamma=2, C=1, random_state=42)
)
```

```{code-cell} ipython3
doubly_compare_feature_performance(
    data,
    groupwise_data,
    classifier=GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
)
```

```{code-cell} ipython3
doubly_compare_feature_performance(
    data, groupwise_data, classifier=KNeighborsClassifier(39)
)
```

```{code-cell} ipython3
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
