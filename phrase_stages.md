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

ToDo

* n01op18-1_01, phrase_id 4, viio/vi => #viio/
* 07-1, phrase_id 2415, vi/V in D would be f# but this is clearly in a. It is a minor key, so bVI should be VI

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
from random import choice
from typing import Hashable, Optional

import dimcat as dc
import ms3
import numpy as np
import pandas as pd
from dimcat import resources
from dimcat.data.resources.utils import (
    make_group_start_mask,
    merge_columns_into_one,
    subselect_multiindex_from_df,
)
from dimcat.plotting import make_box_plot, write_image
from git import Repo

import utils
from utils import (
    DEFAULT_OUTPUT_FORMAT,
    OUTPUT_FOLDER,
    get_repo_name,
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
    effective_localkey=["effective_localkey"],
)
criterion2stages = utils.make_criterion_stages(phrase_annotations, CRITERIA)
```

```{code-cell}
uncompressed_lengths = utils.get_criterion_phrase_lengths(
    criterion2stages["uncompressed"]
)
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

```{code-cell}
def group_operation(group_df):
    return utils._compute_smallest_fifth_ranges(
        group_df.lowest_tpc.values, group_df.tpc_width.values
    )


def _make_diatonics_criterion(
    chord_tones,
) -> pd.DataFrame:
    lowest, width = zip(
        *chord_tones.groupby("phrase_id", sort=False, group_keys=False).apply(
            group_operation
        )
    )
    lowest = np.concatenate(lowest)
    width = np.concatenate(width)
    result = pd.DataFrame(
        {"lowest_tpc": lowest, "tpc_width": width}, index=chord_tones.index
    )
    return result


def make_diatonics_criterion(
    chord_tones,
    join_str: Optional[str | bool] = None,
    fillna: Optional[Hashable] = None,
) -> pd.Series:
    result = _make_diatonics_criterion(chord_tones)
    result = merge_columns_into_one(result, join_str=join_str, fillna=fillna)
    return result.rename("diatonics")
```

```{code-cell}
def make_root_roman_or_dominant_criterion(phrase_annotations):
    numeral_type_effective_key = utils.get_phrase_chord_tones(
        phrase_annotations,
        additional_columns=[
            "effective_localkey_resolved",
            "effective_localkey_is_minor",
            "numeral",
            "root_roman",
            "chord_type",
        ],
    )
    dominant_selector = utils.make_dominant_selector(numeral_type_effective_key)
    expected_root = ms3.transform(
        numeral_type_effective_key,
        ms3.roman_numeral2fifths,
        ["effective_localkey", "globalkey_is_minor"],
    ).rename("expected_root")
    expected_root = expected_root.where(dominant_selector).astype("Int64")
    effective_numeral_df = pd.concat(
        [
            ms3.transform(
                numeral_type_effective_key,
                ms3.rel2abs_key,
                ["numeral", "effective_localkey", "effective_localkey_is_minor"],
            ).rename("effective_numeral"),
            numeral_type_effective_key.globalkey_is_minor,
        ],
        axis=1,
    )
    subsequent_root = (
        ms3.transform(
            effective_numeral_df,
            ms3.roman_numeral2fifths,
        )
        .shift()
        .astype("Int64")
        .rename("subsequent_root")
    )
    all_but_ultima_selector = ~make_group_start_mask(subsequent_root, "phrase_id")
    subsequent_root.where(all_but_ultima_selector, inplace=True)
    subsequent_root_roman = numeral_type_effective_key.root_roman.shift().rename(
        "subsequent_root_roman"
    )
    subsequent_root_roman.where(all_but_ultima_selector, inplace=True)
    merge_with_previous = (expected_root == subsequent_root).fillna(False)
    copy_decision_from_previous = (expected_root.eq(expected_root.shift())).fillna(
        False
    )
    fill_preparation_chain = (
        copy_decision_from_previous & merge_with_previous.shift().fillna(False)
    ) & all_but_ultima_selector
    keep_root = ~(merge_with_previous | fill_preparation_chain)
    root_roman_criterion = numeral_type_effective_key.root_roman.where(
        keep_root, subsequent_root_roman
    )
    root_roman_criterion = (
        root_roman_criterion.where(~fill_preparation_chain)
        .ffill()
        .rename("root_roman_or_its_dominant")
    )
    numeral_type_effective_key = numeral_type_effective_key.from_resource_and_dataframe(
        numeral_type_effective_key,
        pd.concat(
            [
                numeral_type_effective_key,
                expected_root,
                subsequent_root,
                subsequent_root_roman,
            ],
            axis=1,
        ),
    )
    return numeral_type_effective_key.regroup_phrases(root_roman_criterion)


root_roman_or_its_dominant = make_root_roman_or_dominant_criterion(phrase_annotations)
criterion2stages["root_roman_or_its_dominant"] = root_roman_or_its_dominant
root_roman_or_its_dominant.head(100)
```

```{code-cell}
utils.compare_criteria_metrics(criterion2stages, height=1000)
```

```{code-cell}
utils._compare_criteria_stage_durations(
    criterion2stages, chronological_corpus_names=chronological_corpus_names
)
```

```{code-cell}
utils._compare_criteria_phrase_lengths(
    criterion2stages, chronological_corpus_names=chronological_corpus_names
)
```

```{code-cell}
utils._compare_criteria_entropies(
    criterion2stages, chronological_corpus_names=chronological_corpus_names
)
```

```{code-cell}
def make_simple_resource_column(timeline_data):
    is_dominant = timeline_data.expected_root.notna()
    group_levels = is_dominant.index.names[:-1]
    stage_has_dominant = is_dominant.groupby(group_levels).any()
    is_tonic_resolution = ~is_dominant & stage_has_dominant.reindex(timeline_data.index)
    resource_column = pd.Series("other", index=timeline_data.index, name="Resource")
    resource_column.where(~is_dominant, "dominant", inplace=True)
    resource_column.where(~is_tonic_resolution, "tonic resolution", inplace=True)
    return resource_column


def make_timeline_data(root_roman_or_its_dominant):
    timeline_data = pd.concat(
        [
            root_roman_or_its_dominant,
            root_roman_or_its_dominant.groupby(
                "phrase_id", group_keys=False, sort=False
            ).duration_qb.apply(utils.make_start_finish),
            ms3.transform(
                root_roman_or_its_dominant,
                ms3.roman_numeral2fifths,
                ["effective_localkey_resolved", "globalkey_is_minor"],
            ).rename("effective_local_tonic_tpc"),
        ],
        axis=1,
    )
    exploded_chord_tones = root_roman_or_its_dominant.chord_tone_tpcs.explode()
    exploded_chord_tones = pd.DataFrame(
        dict(
            chord_tone_tpc=exploded_chord_tones,
            Task=ms3.transform(exploded_chord_tones, ms3.fifths2name),
        ),
        index=exploded_chord_tones.index,
    )
    timeline_data = pd.merge(
        timeline_data, exploded_chord_tones, left_index=True, right_index=True
    )
    timeline_data = pd.concat(
        [timeline_data, make_simple_resource_column(timeline_data)], axis=1
    ).rename(columns=dict(chord="Description"))
    return timeline_data
```

```{code-cell}
timeline_data = make_timeline_data(root_roman_or_its_dominant)
timeline_data.head(50)
```

```{code-cell}
n_phrases = max(timeline_data.index.levels[2])
color_shade = 500
colorscale = {
    resource: utils.TailwindColorsHex.get_color(color_name, color_shade)
    for resource, color_name in zip(
        ("dominant", "tonic resolution", "other"), ("red", "blue", "gray")
    )
}
```

```{code-cell}
phrase_timeline_data = timeline_data.query(f"phrase_id == {choice(range(n_phrases))}")
```

```{code-cell}
fig = utils.plot_phrase(
    phrase_timeline_data,
    colorscale=colorscale,
    # shapes=make_diatonics_rectangles(phrase_timeline_data)
)
fig
```

```{code-cell}
phrase_timeline_data
```

```{code-cell}
def subselect_dominant_stages(timeline_data):
    """Returns a copy where all remaining stages contain at least one dominant."""
    dominant_stage_mask = (
        timeline_data.expected_root.notna().groupby(level=[0, 1, 2, 3]).any()
    )
    dominant_stage_index = dominant_stage_mask[dominant_stage_mask].index
    all_dominant_stages = subselect_multiindex_from_df(
        timeline_data, dominant_stage_index
    )
    return all_dominant_stages


all_dominant_stages = subselect_dominant_stages(timeline_data)
all_dominant_stages
```

```{code-cell}
gpb = all_dominant_stages.groupby(level=[0, 1, 2, 3])
expected_roots = gpb.expected_root.nunique()
expected_roots[expected_roots.gt(1)]
```

```{code-cell}
unique_resource_vals = gpb.Resource.unique()
unique_resource_vals.head()
```

```{code-cell}
n_root_roman = gpb.root_roman_or_its_dominant.nunique()
n_root_roman[n_root_roman.gt(1)]
```
