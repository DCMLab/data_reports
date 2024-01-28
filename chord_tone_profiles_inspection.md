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

# Inspection of Chord-Tone Profiles

PCA, LDA etc.

```{code-cell} ipython3
%load_ext autoreload
%autoreload 2

import os

import pandas as pd
from dimcat.plotting import write_image
from matplotlib import pyplot as plt

import utils

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

## Principal Component Analyses
### Chord profiles

```{code-cell} ipython3
data.keys()
```

```{code-cell} ipython3
def show_pca(feature, norm="piecenorm", **kwargs):
    global data
    corpus = data[(feature, norm)]
    info = make_info(corpus, "PCA")
    return utils.plot_pca(corpus, info=info, **kwargs)


show_pca("root_per_globalkey", color=PIECE_YEARS, symbol=PIECE_MODE)
```

```{code-cell} ipython3
show_pca("root_per_localkey", color=PIECE_YEARS, symbol=PIECE_MODE)
```

```{code-cell} ipython3
show_pca("root_per_tonicization", color=PIECE_YEARS, symbol=PIECE_MODE)
```

```{raw-cell}
## Pitch-class profiles
```

```{raw-cell}
show_pca("globalkey_profiles", color=PIECE_YEARS, symbol=PIECE_MODE)
```

```{raw-cell}
show_pca("localkey_profiles", color=PIECE_YEARS, symbol=PIECE_MODE)
```

```{raw-cell}
show_pca("tonicization_profiles", color=PIECE_YEARS, symbol=PIECE_MODE)
```

### Chord-tone profiles

```{code-cell} ipython3
show_pca("global_root_ct", color=PIECE_YEARS, symbol=PIECE_MODE)
```

```{code-cell} ipython3
show_pca("local_root_ct", color=PIECE_YEARS, symbol=PIECE_MODE)
```

```{code-cell} ipython3
show_pca("tonicization_root_ct", color=PIECE_YEARS, symbol=PIECE_MODE)
```

```{raw-cell}
## Create chord-tone profiles for multiple chord features

Tokens are `(feature, ..., chord_tone)` tuples.
```

```{raw-cell}
chord_reduced: resources.PrevalenceMatrix = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["chord_reduced_and_mode", "fifths_over_local_tonic"],
        index=["corpus", "piece"],
    )
)
print(f"Shape: {chord_reduced.shape}")
utils.replace_boolean_column_level_with_mode(chord_reduced)
```

```{raw-cell}
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

```{raw-cell}
roots: resources.PrevalenceMatrix = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["root", "fifths_over_local_tonic"],
        index=["corpus", "piece"],
    )
)
print(f"Shape: {roots.shape}")
```

```{raw-cell}
root_per_globalkey = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["root_per_globalkey", "fifths_over_local_tonic"],
        index=["corpus", "piece"],
    )
)
print(f"Shape: {root_per_globalkey.shape}")
```

```{raw-cell}
fig = utils.plot_document_frequency(chord_reduced)
save_figure_as(fig, "document_frequency_of_chord_tones")
fig
```

```{raw-cell}
utils.plot_document_frequency(numerals, info="numerals")
```

```{raw-cell}
utils.plot_document_frequency(roots, info="roots")
```

```{raw-cell}
utils.plot_document_frequency(
    root_per_globalkey, info="root relative to global tonic"
)
```

```{raw-cell}
## Principal Component Analyses
```

```{raw-cell}
# chord_reduced.query("piece in ['op03n12a', 'op03n12b']").dropna(axis=1, how='all')
```

```{raw-cell}
metadata = D.get_metadata()
CORPUS_YEARS = utils.corpus_mean_composition_years(metadata)
PIECE_YEARS = metadata.get_composition_years().rename("mean_composition_year")
utils.plot_pca(
    chord_reduced.relative,
    info="chord-tone profiles of reduced chords",
    color=PIECE_YEARS,
)
```

```{raw-cell}
utils.plot_pca(
    chord_reduced.combine_results("corpus").relative,
    info="chord-tone profiles of reduced chords",
    color=CORPUS_YEARS,
    size=5,
)
```

```{raw-cell}
utils.plot_pca(
    numerals.relative, info="numeral profiles of numerals", color=PIECE_YEARS
)
```

```{raw-cell}
utils.plot_pca(
    numerals.combine_results("corpus").relative,
    info="chord-tone profiles of numerals",
    color=CORPUS_YEARS,
    size=5,
)
```

```{raw-cell}
utils.plot_pca(
    roots.relative, info="root profiles of chord roots (local)", color=PIECE_YEARS
)
```

```{raw-cell}
utils.plot_pca(
    roots.combine_results("corpus").relative,
    info="chord-tone profiles of chord roots (local)",
    color=CORPUS_YEARS,
    size=5,
)
```

```{raw-cell}
utils.plot_pca(
    root_per_globalkey.relative,
    info="root profiles of chord roots (global)",
    color=PIECE_YEARS,
)
```

```{raw-cell}
utils.plot_pca(
    root_per_globalkey.combine_results("corpus").relative,
    info="chord-tone profiles of chord roots (global)",
    color=CORPUS_YEARS,
    size=5,
)
```
