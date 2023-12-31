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

# Detecting diatonic bands

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide imports
  code_prompt_show: Show imports
tags: [hide-cell]
---
# %load_ext autoreload
# %autoreload 2

import os
from typing import Dict, Hashable, List, Optional, Tuple

import dimcat as dc
import ms3
import pandas as pd
from dimcat import resources
from dimcat.data.resources.results import _entropy
from dimcat.data.resources.utils import merge_columns_into_one
from dimcat.plotting import make_bar_plot, make_box_plot, write_image
from git import Repo

from utils import (
    DEFAULT_OUTPUT_FORMAT,
    OUTPUT_FOLDER,
    get_repo_name,
    make_stage_data,
    print_heading,
    resolve_dir,
)

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
```

```{code-cell}
RESULTS_PATH = os.path.abspath(os.path.join(OUTPUT_FOLDER, "phrases"))
os.makedirs(RESULTS_PATH, exist_ok=True)


def make_output_path(filename, extension=None):
    if extension:
        extension = "." + extension.lstrip(".")
    else:
        extension = DEFAULT_OUTPUT_FORMAT
    return os.path.join(RESULTS_PATH, f"{filename}{extension}")


def save_figure_as(fig, filename, directory=RESULTS_PATH, **kwargs):
    write_image(fig, filename, directory, **kwargs)
```

```{code-cell}
def make_criterion(
    phrase_feature,
    criterion_name: Optional[str] = None,
    columns="chord",
    components="body",
    drop_levels=3,
    reverse=True,
    level_name="stage",
    query=None,
    join_str: Optional[str | bool] = None,
    fillna: Optional[Hashable] = None,
) -> pd.Series:
    """Function sets the defaults for the stage TSVs produced in the following."""
    phrase_data = phrase_feature.get_phrase_data(
        columns=columns,
        components=components,
        drop_levels=drop_levels,
        reverse=reverse,
        level_name=level_name,
        wide_format=False,
        query=query,
    )
    if not isinstance(columns, str) and len(columns) > 1:
        phrase_data = merge_columns_into_one(
            phrase_data, join_str=join_str, fillna=fillna
        )
        if criterion_name is None:
            criterion_name = "_and_".join(columns)
    else:
        phrase_data = phrase_data.iloc(axis=1)[0]
        if criterion_name is None:
            if isinstance(columns, str):
                criterion_name = columns
            else:
                criterion_name = columns[0]
    result = phrase_data.rename(criterion_name)
    return result


def make_criterion_stages(
    phrase_annotations: resources.PhraseAnnotations,
    criteria_dict: Dict[str, str | List[str]],
    join_str=True,
):
    """Takes a {name -> [columns]} dict."""
    uncompressed = make_stage_data(
        phrase_annotations,
        columns=["chord_and_mode", "duration_qb"],
        wide_format=False,
    )
    name2phrase_data = {"uncompressed": uncompressed}
    for name, columns in criteria_dict.items():
        criterion = make_criterion(
            phrase_annotations,
            columns=columns,
            criterion_name=name,
            join_str=join_str,
        )
        name2phrase_data[name] = uncompressed.regroup_phrases(criterion)
    return name2phrase_data


def get_stage_durations(phrase_data: resources.PhraseData):
    return phrase_data.groupby(
        ["corpus", "piece", "phrase_id", "stage"]
    ).duration_qb.sum()


def get_criterion_phrase_lengths(phrase_data: resources.PhraseData):
    """In terms of number of stages after merging."""
    stage_index = phrase_data.index.to_frame(index=False)
    phrase_id_col = stage_index.columns.get_loc("phrase_id")
    groupby = stage_index.columns.to_list()[: phrase_id_col + 1]
    stage_lengths = stage_index.groupby(groupby).stage.max() + 1
    return stage_lengths.rename("phrase_length")


def get_criterion_phrase_entropies(
    phrase_data: resources.PhraseData, criterion_name: Optional[str] = None
):
    if not criterion_name:
        criterion_name = phrase_data.columns.to_list()[0]
    criterion_distributions = phrase_data.groupby(
        ["corpus", criterion_name]
    ).duration_qb.sum()
    return criterion_distributions.groupby("corpus").agg(_entropy).rename("entropy")


def get_metrics_means(name2phrase_data: Dict[str, resources.PhraseData]):
    criterion_metric2value = {}
    for name, stages in name2phrase_data.items():
        stage_durations = get_stage_durations(stages)
        criterion_metric2value[
            (name, "mean stage duration", "mean")
        ] = stage_durations.mean()
        criterion_metric2value[
            (name, "mean stage duration", "sem")
        ] = stage_durations.sem()
        phrase_lengths = get_criterion_phrase_lengths(stages)
        criterion_metric2value[
            (name, "mean phrase length", "mean")
        ] = phrase_lengths.mean()
        criterion_metric2value[
            (name, "mean phrase length", "sem")
        ] = phrase_lengths.sem()
        phrase_entropies = get_criterion_phrase_entropies(stages)
        criterion_metric2value[
            (name, "mean phrase entropy", "mean")
        ] = phrase_entropies.mean()
        criterion_metric2value[
            (name, "mean phrase entropy", "sem")
        ] = phrase_entropies.sem()
    metrics = pd.Series(criterion_metric2value, name="value").unstack(sort=False)
    metrics.index.names = ["criterion", "metric"]
    return metrics


def compare_criteria_metrics(
    name2phrase_data: Dict[str, resources.PhraseData], **kwargs
):
    metrics = get_metrics_means(name2phrase_data).reset_index()
    return make_bar_plot(
        metrics,
        facet_row="metric",
        color="criterion",
        x_col="mean",
        y_col="criterion",
        x_axis=dict(matches=None, showticklabels=True),
        layout=dict(showlegend=False),
        error_x="sem",
        orientation="h",
        labels=dict(entropy="entropy of stage distributions in bits", corpus=""),
        **kwargs,
    )


def plot_corpuswise_criteria_means(
    criterion2values: Dict[str, pd.Series],
    category_title="stage_type",
    y_axis_label="mean duration of stages in ♩",
    **kwargs,
):
    """Takes a {trace_name -> values} dict where each entry will be turned into a bar plot trace for comparison."""
    aggregated = {
        name: durations.groupby("corpus").agg(["mean", "sem"])
        for name, durations in criterion2values.items()
    }
    df = pd.concat(aggregated, names=[category_title])
    corpora = df.index.get_level_values("corpus").unique()
    corpus_order = [
        corpus for corpus in chronological_corpus_names if corpus in corpora
    ]
    return make_bar_plot(
        df,
        x_col="corpus",
        y_col="mean",
        error_y="sem",
        color=category_title,
        category_orders=dict(corpus=corpus_order),
        labels=dict(mean=y_axis_label, corpus=""),
        **kwargs,
    )


def plot_corpuswise_criteria(
    criterion2values,
    category_title="stage_type",
    y_axis_label="entropy of stage distributions in bits",
    **kwargs,
):
    """Takes a {trace_name -> PhraseData} dict where each entry will be turned into a bar plot trace for comparison."""
    df = pd.concat(criterion2values, names=[category_title])
    corpora = df.index.get_level_values("corpus").unique()
    corpus_order = [
        corpus for corpus in chronological_corpus_names if corpus in corpora
    ]
    return make_bar_plot(
        df,
        x_col="corpus",
        y_col="entropy",
        color=category_title,
        category_orders=dict(corpus=corpus_order),
        labels=dict(entropy=y_axis_label, corpus=""),
        **kwargs,
    )


def _compare_criteria_stage_durations(
    name2phrase_data: Dict[str, resources.PhraseData]
):
    durations_dict = {
        name: get_stage_durations(stages) for name, stages in name2phrase_data.items()
    }
    return plot_corpuswise_criteria_means(durations_dict, height=800)


def compare_criteria_stage_durations(
    phrase_annotations: resources.PhraseAnnotations,
    criteria_dict: Dict[str, str | List[str]],
    join_str=True,
):
    name2phrase_data = make_criterion_stages(
        phrase_annotations, criteria_dict, join_str=join_str
    )
    return _compare_criteria_stage_durations(name2phrase_data)


def _compare_criteria_phrase_lengths(name2phrase_data: Dict[str, resources.PhraseData]):
    lengths_dict = {
        name: get_criterion_phrase_lengths(durations)
        for name, durations in name2phrase_data.items()
    }
    return plot_corpuswise_criteria_means(
        lengths_dict, y_axis_label="mean number of stages per phrase", height=800
    )


def compare_criteria_phrase_lengths(
    phrase_annotations: resources.PhraseAnnotations,
    criteria_dict: Dict[str, str | List[str]],
    join_str=True,
):
    name2phrase_data = make_criterion_stages(
        phrase_annotations, criteria_dict, join_str=join_str
    )
    return _compare_criteria_phrase_lengths(name2phrase_data)


def _compare_criteria_entropies(name2phrase_data: Dict[str, resources.PhraseData]):
    entropies = {
        name: get_criterion_phrase_entropies(durations)
        for name, durations in name2phrase_data.items()
    }
    return plot_corpuswise_criteria(entropies, height=800)


def compare_criteria_entropies(phrase_annotations, criteria_dict, join_str=True):
    name2phrase_data = make_criterion_stages(
        phrase_annotations, criteria_dict, join_str=join_str
    )
    return _compare_criteria_entropies(name2phrase_data)
```

```{code-cell}
:tags: [hide-input]

package_path = resolve_dir(
    "~/distant_listening_corpus/distant_listening_corpus.datapackage.json"
)
repo = Repo(os.path.dirname(package_path))
print_heading("Data and software versions")
print(f"Data repo '{get_repo_name(repo)}' @ {repo.commit().hexsha[:7]}")
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
CRITERIA = dict(
    chord_reduced_and_localkey=["chord_reduced", "localkey"],
    chord_reduced_and_mode=["chord_reduced_and_mode"],
    bass_degree=["bass_note"],
    root_roman=["root_roman", "localkey_mode"],
    root_degree=["root"],
    numeral_or_applied_to_numeral=["numeral_or_applied_to_numeral", "localkey_mode"],
)
criterion2stages = make_criterion_stages(phrase_annotations, CRITERIA)
```

```{code-cell}
chord_tones = phrase_annotations.get_phrase_data(
    reverse=True,
    columns=[
        "chord",
        "localkey",
        "globalkey",
        "globalkey_is_minor",
        "chord_tones",
    ],
    drop_levels="phrase_component",
)
chord_tones.df.chord_tones.where(chord_tones.chord_tones != (), inplace=True)
chord_tones.df.chord_tones.ffill(inplace=True)
chord_tones = ms3.transpose_chord_tones_by_localkey(chord_tones.df, by_global=True)
chord_tones["lowest_tpc"] = chord_tones.chord_tones.apply(min)
chord_tones["highest_tpc"] = chord_tones.chord_tones.apply(max)
chord_tones["tpc_width"] = chord_tones.highest_tpc - chord_tones.lowest_tpc
```

```{code-cell}
chord_tones
```

```{code-cell}
chord_tones.tpc_width.value_counts()
```

```{code-cell}
chord_tones.query("phrase_id == 1").iloc[:, -3:]
```

```{code-cell}
input = chord_tones.query("phrase_id == 1").iloc[:, -3:]
input
```

```{code-cell}
def get_max_range(widths) -> Tuple[int, int, int]:
    """Index range capturing the first until last occurrence of the maximum value."""
    maximum, first_ix, last_ix = 0, 0, 0
    for i, width in enumerate(widths):
        if width > maximum:
            maximum = width
            first_ix = i
            last_ix = i
        elif width == maximum:
            last_ix = i
    return first_ix, last_ix + 1, maximum


def compute_smallest_fifth_ranges(width, smallest=7):
    first_max_ix, last_max_ix, max_val = get_max_range(width)
    if max_val < smallest:
        return [max_val] * len(width)
    left = width[:first_max_ix]
    middle = width[first_max_ix:last_max_ix]
    right = width[last_max_ix:]
    return (
        compute_smallest_fifth_ranges(left)
        + [max_val] * len(middle)
        + compute_smallest_fifth_ranges(right)
    )


compute_smallest_fifth_ranges(input.tpc_width.to_list())
```

```{code-cell}
def merge_up_to_max_width(input, largest):
    lowest, highest = None, None
    merge_n = 0
    result = []

    def do_merge():
        nonlocal lowest, highest, merge_n
        if merge_n:
            result.extend([(lowest, highest - lowest)] * merge_n)
        lowest, highest = None, None
        merge_n = 0

    for i, (low, high, width) in enumerate(input, start=1):
        if width > largest:
            do_merge()
            result.append((low, width))
            continue
        if lowest is None:
            lowest = low
            highest = high
            continue
        merge_low_point = min((low, lowest))
        merge_high_point = max((high, highest))
        merge_width = merge_high_point - merge_low_point
        if merge_width <= largest:
            # merge
            lowest = merge_low_point
            highest = merge_high_point
            merge_n += 1
        else:
            do_merge()
            lowest = low
            highest = high
    do_merge()
    return pd.DataFrame(result, columns=["lowest_tpc", "tpc_width"])


merge_up_to_max_width(list(input.itertuples(index=False, name=None)), largest=7)
```

```{code-cell}
def compute_smallest_fifth_ranges(input, smallest=7, largest=10):
    if len(input) == 1:
        return input.iloc[:, [0, 2]]
    first_max_ix, last_max_ix, max_val = get_max_range(input.tpc_width)
    if max_val == smallest:
        return input.iloc[:, [0, 2]]
    if max_val < smallest:
        return merge_up_to_max_width(input.values, largest=smallest)
    left = input.iloc[:first_max_ix]
    middle = input.iloc[first_max_ix:last_max_ix]
    right = input.iloc[last_max_ix:]
    if max_val < largest:
        middle = merge_up_to_max_width(middle.values, largest=largest)
    return pd.concat(
        [
            compute_smallest_fifth_ranges(left),
            middle,
            compute_smallest_fifth_ranges(right),
        ]
    )


compute_smallest_fifth_ranges(input, largest=9)
```

```{code-cell}
numeral_type_effective_key = phrase_annotations.get_phrase_data(
    reverse=True,
    columns=[
        "numeral",
        "chord_type",
        "effective_localkey",
        "effective_localkey_is_minor",
    ],
    drop_levels="phrase_component",
)
is_dominant = numeral_type_effective_key.numeral.eq(
    "V"
) & numeral_type_effective_key.chord_type.isin({"Mm7", "M"})
leading_tone_is_root = (
    numeral_type_effective_key.numeral.eq("#vii")
    & numeral_type_effective_key.effective_localkey_is_minor
) | (
    numeral_type_effective_key.numeral.eq("vii")
    & ~numeral_type_effective_key.effective_localkey_is_minor
)
is_rootless_dominant = (
    leading_tone_is_root & numeral_type_effective_key.chord_type.isin({"o", "o7", "%7"})
)
dominants_and_resolutions = ms3.transform(
    numeral_type_effective_key,
    ms3.rel2abs_key,
    ["numeral", "effective_localkey", "effective_localkey_is_minor"],
).rename("effective_numeral_or_its_dominant")
dominants_and_resolutions.where(
    ~(is_dominant | is_rootless_dominant),
    numeral_type_effective_key.effective_localkey,
    inplace=True,
)
effective_numeral_or_its_dominant = criterion2stages["uncompressed"].regroup_phrases(
    dominants_and_resolutions
)
criterion2stages[
    "effective_numeral_or_its_dominant"
] = effective_numeral_or_its_dominant
effective_numeral_or_its_dominant.head(100)
```

```{code-cell}
compare_criteria_metrics(criterion2stages, height=1000)
```

```{code-cell}
_compare_criteria_stage_durations(criterion2stages)
```

```{code-cell}
_compare_criteria_phrase_lengths(criterion2stages)
```

```{code-cell}
_compare_criteria_entropies(criterion2stages)
```

```{code-cell}
uncompressed_lengths = get_criterion_phrase_lengths(criterion2stages["uncompressed"])
uncompressed_lengths.groupby("corpus").describe()
```

```{code-cell}
make_box_plot(
    uncompressed_lengths,
    x_col="corpus",
    y_col="phrase_length",
    height=800,
    category_orders=dict(corpus=chronological_corpus_names),
)
```
