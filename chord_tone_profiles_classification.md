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

import dimcat as dc
import ms3
import numpy as np
import pandas as pd
import seaborn as sns
from dimcat import resources
from dimcat.plotting import write_image
from git import Repo
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import LeaveOneOut, cross_validate, train_test_split
from sklearn.svm import LinearSVC

import utils

plt.rcParams["figure.dpi"] = 300

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
```

```{code-cell}
RESULTS_PATH = os.path.expanduser("~/git/diss/32_profiles/figs")
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

## Document frequencies of chord features

```{code-cell}
utils.compare_corpus_frequencies(
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

```{code-cell}
chord_reduced: resources.PrevalenceMatrix = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["chord_reduced_and_mode", "fifths_over_local_tonic"],
        index=["corpus", "piece"],
    )
)
print(f"Shape: {chord_reduced.shape}")
```

```{code-cell}
numerals: resources.PrevalenceMatrix = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["effective_localkey_is_minor", "numeral", "fifths_over_local_tonic"],
        index=["corpus", "piece"],
    )
)
print(f"Shape: {numerals.shape}")
utils.replace_boolean_column_level_with_mode(numerals)
```

```{code-cell}
roots: resources.PrevalenceMatrix = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["root", "fifths_over_local_tonic"],
        index=["corpus", "piece"],
    )
)
print(f"Shape: {roots.shape}")
```

```{code-cell}
root_fifths_over_global_tonic = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["root_fifths_over_global_tonic", "fifths_over_local_tonic"],
        index=["corpus", "piece"],
    )
)
print(f"Shape: {root_fifths_over_global_tonic.shape}")
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

```{code-cell}
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
CV = CrossValidated(matrix=chord_reduced, clf=clf, cv=cv)
cv_results = CV.cross_validate()
```

```{code-cell}
RFC.show_confusion_matrix().plot()
```

```{code-cell}
clf_report = RFC.classification_report
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, cmap="RdBu")
```

```{code-cell}
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

```{code-cell}
best_index = scores.idxmax()
best_estimator = cv_results["estimator"][best_index]
best_score = scores.max()
best_params = best_estimator.get_params()
print(f"Best score: {best_score}")
print(f"Best params: {best_params}")
```

```{code-cell}
scores
```

```{code-cell}
best_index
```
