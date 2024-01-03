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

* n01op18-1_01, phrase_id 4, viio/vi => #viio/vi

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
from typing import Hashable, Optional

import dimcat as dc
import ms3
import numpy as np
import pandas as pd
from dimcat import resources
from dimcat.data.resources.utils import merge_columns_into_one
from dimcat.plotting import write_image
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
def get_phrase_chord_tones(phrase_annotations) -> resources.PhraseData:
    chord_tones = phrase_annotations.get_phrase_data(
        reverse=True,
        columns=[
            "chord",
            "duration_qb",
            "localkey",
            "globalkey",
            "globalkey_is_minor",
            "chord_tones",
        ],
        drop_levels="phrase_component",
    )
    df = chord_tones.df
    df.chord_tones.where(df.chord_tones != (), inplace=True)
    df.chord_tones.ffill(inplace=True)
    df = ms3.transpose_chord_tones_by_localkey(df, by_global=True)
    df["lowest_tpc"] = df.chord_tones.map(min)
    highest_tpc = df.chord_tones.map(max)
    df["tpc_width"] = highest_tpc - df.lowest_tpc
    df["highest_tpc"] = highest_tpc
    return chord_tones.from_resource_and_dataframe(chord_tones, df)


def group_operation(group_df):
    return utils._compute_smallest_fifth_ranges(
        group_df.lowest_tpc.values, group_df.tpc_width.values
    )


def make_diatonics_criterion(
    chord_tones,
    join_str: Optional[str | bool] = None,
    fillna: Optional[Hashable] = None,
):
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
    result = merge_columns_into_one(result, join_str=join_str, fillna=fillna)
    return result.rename("diatonics")
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
chord_tones = get_phrase_chord_tones(phrase_annotations)
chord_tones.head()
```

```{code-cell}
chord_tones.tpc_width.value_counts()
```

```{code-cell}
diatonics_criterion = make_diatonics_criterion(chord_tones)
diatonics_stages = chord_tones.regroup_phrases(diatonics_criterion)
criterion2stages["diatonics"] = diatonics_stages
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
diatonics_stages
```
