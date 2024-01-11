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

# Chord profiles for phrases in the DLC

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
from math import ceil
from typing import List

import dimcat as dc
import ms3
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px
from dimcat import resources
from dimcat.plotting import write_image
from git import Repo
from sklearn.decomposition import PCA

import utils

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
```

```{code-cell}
RESULTS_PATH = os.path.abspath(os.path.join(utils.OUTPUT_FOLDER, "phrases"))
os.makedirs(RESULTS_PATH, exist_ok=True)


def make_output_path(
    filename: str,
    extension=None,
    path=RESULTS_PATH,
) -> str:
    return utils.make_output_path(
        filename,
        extension=extension,
        path=path,
    )


def save_figure_as(fig, filename, directory=RESULTS_PATH, **kwargs):
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
chronological_corpus_names = D.get_metadata().get_corpus_names(func=None)
D
```

```{code-cell}
phrase_annotations: resources.PhraseAnnotations = D.get_feature("PhraseAnnotations")
phrase_annotations
```

```{code-cell}
def make_phrase_data(phrase_annotations, columns, components="phrase", **kwargs):
    phrase_data = phrase_annotations.get_phrase_data(
        columns, components=components, **kwargs
    )
    return phrase_data


def make_phrase_bigram_table(
    phrase_annotations: resources.PhraseAnnotations, columns: str | List[str]
) -> resources.NgramTable:
    phrase_data = make_phrase_data(
        phrase_annotations,
        columns,
        components="phrase",
        drop_levels="phrase_component",  # otherwise, no bigrams spanning body and codetta
    )
    phrase_bgt = phrase_data.apply_step("BigramAnalyzer")
    return phrase_bgt


phrase_data = make_phrase_data(phrase_annotations, ["chord_and_mode", "duration_qb"])
phrase_data_df = phrase_data.df.dropna(subset="chord_and_mode")
```

```{code-cell}
overall_chord_distribution = (
    phrase_data_df.groupby("chord_and_mode")
    .duration_qb.sum()
    .sort_values(ascending=False)
)
overall_chord_distribution
```

```{code-cell}
CF_abs = phrase_data_df.pivot_table(
    index=["corpus", "piece", "phrase_id"],
    columns="chord_and_mode",
    values="duration_qb",
    aggfunc="sum",
)
CF_abs.shape
```

```{code-cell}
PF = CF_abs.notna().sum().sort_values(ascending=False)
PF
```

```{code-cell}
smooth = 1e-20
CF = CF_abs.fillna(0.0).add(smooth).div(CF_abs.sum(axis=1), axis=0)
CF.iloc[:10, :10]
```

```{code-cell}
phrase_pca = PCA(3)
decomposed_phrases = pd.DataFrame(phrase_pca.fit_transform(CF), index=CF.index)
print(f"Explained variance: {phrase_pca.explained_variance_}")
fig = px.scatter_3d(
    decomposed_phrases.reset_index(),
    x=0,
    y=1,
    z=2,
    color="corpus",
    hover_name="piece",
    title="3 principal components of the chord frequency matrix",
)
fig
```

```{code-cell}
def do_pca_grid_search(
    data: pd.DataFrame,
    features: npt.ArrayLike,
    max_components: int = 10,
):
    max_features = len(features)
    n_columns = max_components if max_components > 0 else ceil(max_features / 2)
    grid_search = np.zeros((max_features, n_columns))
    for n_features in range(1, max_features + 1):
        print(n_features, end=" ")
        selected_features = features[:n_features]
        selected_data = data.loc(axis=1)[selected_features]
        if max_components > 0:
            up_to = min(max_components, n_features)
        else:
            up_to = ceil(n_features / 2)
        for n_components in range(1, up_to + 1):
            pca = PCA(n_components)
            _ = pca.fit_transform(selected_data)
            variance = pca.explained_variance_.sum()
            grid_search[n_features - 1, n_components - 1] = variance
            print(f"{variance:.1%}", end=" ")
        print()
    result = pd.DataFrame(
        grid_search,
        index=pd.RangeIndex(1, max_features + 1, name="features"),
        columns=pd.RangeIndex(1, n_columns + 1, name="components"),
    )
    return result


grid_search_by_occurrence = do_pca_grid_search(CF, PF.index[:100])
```

```{code-cell}
grid_search_by_duration = do_pca_grid_search(CF, overall_chord_distribution.index[:100])
```

```{code-cell}
grid_search_by_duration - grid_search_by_occurrence
```

```{code-cell}
phrase_labels = phrase_annotations.extract_feature("PhraseLabels")
phrase_labels
```

```{code-cell}
def add_bass_progressions(
    phrase_bodies: resources.PhraseData,
    reverse=False,
) -> resources.NgramTable:
    bgt: resources.NgramTable = phrase_bodies.apply_step("BigramAnalyzer")
    if reverse:
        bgt.loc(axis=1)["b", "bass_note"] = (
            bgt.loc(axis=1)["a", "bass_note"] - bgt.loc(axis=1)["b", "bass_note"]
        )
    else:
        bgt.loc(axis=1)["b", "bass_note"] -= bgt.loc(axis=1)["a", "bass_note"]
    new_index = pd.MultiIndex.from_tuples(
        [
            ("b", "bass_progression") if t == ("b", "bass_note") else t
            for t in bgt.columns
        ]
    )
    bgt.df.columns = new_index
    return bgt


ending_on_tonic_data = phrase_labels.get_phrase_data(
    ["bass_note", "intervals_over_bass"],
    drop_levels="phrase_component",
    reverse=True,
    query="end_chord == ['I', 'i']",
)
ending_on_tonic = add_bass_progressions(ending_on_tonic_data, reverse=True)
ending_on_tonic
```

```{code-cell}
sonority_progression_tuples = ending_on_tonic.make_bigram_tuples(
    "intervals_over_bass", None, terminal_symbols="DROP"
)
sonority_progression_tuples
```

```{code-cell}
sonority_progression_tuples.query("i == 0").n_gram.value_counts()
```

```{code-cell}
ending_on_tonic = phrase_labels.get_phrase_data(
    ["chord"],
    drop_levels="phrase_component",
    reverse=True,
    query="end_chord == ['I', 'i']",
)
ending_on_tonic.query("i == 1").chord.value_counts()
```

```{code-cell}
bodies_reversed = phrase_labels.get_phrase_data(
    ["chord", "numeral"],
    drop_levels="phrase_component",
    reverse=True,
)
bgt = bodies_reversed.apply_step("BigramAnalyzer")
bgt
```

```{code-cell}
bgt.query("@bgt.a.numeral == 'V'").b.chord.value_counts()
```

```{code-cell}
phrase_bodies = phrase_annotations.get_phrase_data(
    ["bass_note", "intervals_over_bass"], drop_levels="phrase_component"
)
bgt = add_bass_progressions(phrase_bodies)
chord_type_pairs = bgt.make_bigram_tuples(
    "intervals_over_bass", None, terminal_symbols="DROP"
)
chord_type_pairs.make_ranking_table()
```

```{code-cell}
chord_type_transitions = bgt.get_transitions(
    "intervals_over_bass", None, terminal_symbols="DROP"
)
chord_type_transitions.head(50)
```

```{code-cell}
new_idx = chord_type_transitions.index.copy()
antecedent_counts = (
    chord_type_transitions.groupby("antecedent")["count"].sum().to_dict()
)
level_0_values = pd.Series(
    new_idx.get_level_values(0).map(antecedent_counts.get),
    index=new_idx,
    name="antecedent_count",
)
pd.concat([level_0_values, chord_type_transitions], axis=1).sort_values(
    ["antecedent_count", "count"], ascending=False
)
```

```{code-cell}
chord_type_transitions.sort_values("count", ascending=False).iloc[:100]
```

```{code-cell}
bgt.make_bigram_tuples(terminal_symbols="DROP").make_ranking_table()
```
