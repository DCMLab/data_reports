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

# Chopin Profiles

Motivation: Chopin's dominant is often attributed a special characteristic due to the characteristic 13

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
import pandas as pd
from dimcat import resources
from dimcat.plotting import make_bar_plot, write_image
from git import Repo
from matplotlib import pyplot as plt

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

```{raw-cell}
harmony_labels = D.get_feature("harmonylabels")
harmony_labels.head()
raw_labels = harmony_labels.numeral.str.upper() + harmony_labels.figbass.fillna('')
ll = raw_labels.to_list()
from suffix_tree import Tree
sfx_tree = Tree({"dlc": ll})
# query = ["I", "I6", "VII6", "II6", "I", "V7"]
query = ["VII6", "II6", "I"]
sfx_tree.find_all(query)
```

```{code-cell}
chord_slices = utils.get_sliced_notes(D)
chord_slices.head(5)
```

**First intuition: Compare `V7` chord profiles**

```{code-cell}
chord_tone_profiles = utils.make_chord_tone_profile(chord_slices)
chord_tone_profiles.head()
```

```{code-cell}
utils.plot_chord_profiles(chord_tone_profiles, "V7, major")
```

**It turns out that the scale degree in question (3) is more frequent in `V7` chords in `bach_solo` and
`peri_euridice` than in Chopin's Mazurkas. We might suspect that the Chopin chord is not included because it is
highlighted as a different label, 7e.g. `V7(13)`.**

```{code-cell}
utils.plot_chord_profiles(chord_tone_profiles, "V7(13), major")
```

**From here, it is interesting to ask, either, if these special labels show up more frequently in Chopin's corpus
than in others, and if 3 shows up prominently in Chopin's dominants if we combine all dominant chord profiles with
each other.**

```{code-cell}
harmony_labels = D.get_feature("harmonylabels")
all_V7 = harmony_labels.query("numeral == 'V' & figbass == '7'")
all_V7.head()
```

```{code-cell}
all_V7["tonicization_chord"] = all_V7.chord.str.split("/").str[0]
```

```{code-cell}
all_V7_absolute = all_V7.groupby(["corpus", "tonicization_chord"]).duration_qb.agg(
    ["sum", "size"]
)
all_V7_absolute.columns = ["duration_qb", "count"]
all_V7_absolute
```

```{code-cell}
all_V7_relative = all_V7_absolute / all_V7_absolute.groupby("corpus").sum()
make_bar_plot(
    all_V7_relative.reset_index(),
    x_col="tonicization_chord",
    y_col="duration_qb",
    color="corpus",
    log_y=True,
)
```

```{code-cell}
all_V7_relative.loc["chopin_mazurkas"].sort_values("count", ascending=False) * 100
```

**This is not a good way of comparing dominant chords. We could now start summing up all the different chords we
consider to be part of the "Chopin chord" category. Chord-tone-profiles are probably the better way to see.**

+++

## Create chord-tone profiles for multiple chord features

Tokens are `(feature, ..., chord_tone)` tuples.

```{code-cell}
tonicization_profiles: resources.PrevalenceMatrix = chord_slices.apply_step(
    dc.DimcatConfig(
        "PrevalenceAnalyzer",
        columns=["root_per_tonicization", "fifths_over_tonicization"],
        index="corpus",
    )
)
tonicization_profiles._df.columns = ms3.map2elements(
    tonicization_profiles.columns, int
).set_names(["root_per_tonicization", "fifths_over_tonicization"])
```

```{code-cell}
tonicization_profiles.head()
dominant_ct = tonicization_profiles.loc(axis=1)[[1]].stack()
dominant_ct.columns = ["duration_qb"]
dominant_ct["proportion"] = dominant_ct["duration_qb"] / dominant_ct.groupby(
    "corpus"
).duration_qb.agg("sum")
dominant_ct.head()
```

```{code-cell}
fig = make_bar_plot(
    dominant_ct.reset_index(),
    x_col="fifths_over_tonicization",
    y_col="proportion",
    facet_row="corpus",
    facet_row_spacing=0.001,
    height=10000,
    y_axis=dict(matches=None),
)
fig
```

**Chopin does not show a specifically high bar for 4 (the major third of the scale), Mozart's is higher, for example.
This could have many reasons, e.g. that the pieces are mostly in minor, or that the importance of scale degree 3 as
lower neighbor to the dominant seventh is statistically more important, or that the "characteristic 13" is not
actually important duration-wise.**

```{code-cell}
tonic_thirds_in_dominants = (
    dominant_ct.loc[(slice(None), [4, -3]), "proportion"].groupby("corpus").sum()
)
```

```{code-cell}
make_bar_plot(
    tonic_thirds_in_dominants,
    x_col="corpus",
    y_col="proportion",
    category_orders=dict(corpus=D.get_metadata().get_corpus_names(func=None)),
    title="Proportion of scale degree 3 in dominant chords, chronological order",
)
```

**No chronological trend visible.**
