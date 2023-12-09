---
jupytext:
  formats: md:myst,ipynb
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

# Annotations

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
from git import Repo
import dimcat as dc
import ms3
import pandas as pd
from dimcat import slicers

from utils import CORPUS_COLOR_SCALE, color_background, corpus_mean_composition_years, \
  get_corpus_display_name, get_repo_name, print_heading, resolve_dir, make_sunburst, rectangular_sunburst, STD_LAYOUT

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
```

```{code-cell}
from utils import OUTPUT_FOLDER
from dimcat.plotting import write_image
RESULTS_PATH = os.path.abspath(os.path.join(OUTPUT_FOLDER, "scale_degrees"))
os.makedirs(RESULTS_PATH, exist_ok=True)
SUNBURST_WIDTH = 1620
TERMINAL_SYMBOL = "∎"
def save_figure_as(fig, filename, directory=RESULTS_PATH, **kwargs):
    if not "width" in kwargs:
        kwargs['width'] = SUNBURST_WIDTH
    write_image(fig, filename, directory, **kwargs)
```

```{code-cell}
:tags: [hide-input]

package_path = resolve_dir("~/distant_listening_corpus/distant_listening_corpus.datapackage.json")
repo = Repo(os.path.dirname(package_path))
print_heading("Data and software versions")
print(f"Data repo '{get_repo_name(repo)}' @ {repo.commit().hexsha[:7]}")
print(f"dimcat version {dc.__version__}")
print(f"ms3 version {ms3.__version__}")
D = dc.Dataset.from_package(package_path)
D
```

```{code-cell}
all_metadata = D.get_metadata()
assert len(all_metadata) > 0, "No pieces selected for analysis."
all_notes = D.get_feature('notes').df
all_measures = D.get_feature('measures').df
mean_composition_years = corpus_mean_composition_years(all_metadata)
chronological_order = mean_composition_years.index.to_list()
corpus_colors = dict(zip(chronological_order, CORPUS_COLOR_SCALE))
corpus_names = {corp: get_corpus_display_name(corp) for corp in chronological_order}
chronological_corpus_names = list(corpus_names.values())
corpus_name_colors = {corpus_names[corp]: color for corp, color in corpus_colors.items()}
```

## Key areas

```{code-cell}
keys_segmented = slicers.KeySlicer().process(D)
keys_segmented
```

```{code-cell}
notes = keys_segmented.get_feature('notes')
notes
```

```{code-cell}
result = notes.get_default_analysis()
result
```

```{code-cell}
result.plot_grouped()
```

```{code-cell}
from ms3 import roman_numeral2fifths, transform

keys_segmented = dc.LocalKeySlicer().process_data(D)
keys = keys_segmented.get_slice_info()
print(f"Overall number of key segments is {len(keys.index)}")
keys["localkey_fifths"] = transform(keys, roman_numeral2fifths, ['localkey', 'globalkey_is_minor'])
keys.head(5).style.apply(color_background, subset="localkey")
```

```{code-cell}
mode_slices = dc.ModeGrouper().process_data(keys_segmented)
```

### Whole dataset

```{code-cell}
mode_slices.get_slice_info()
```

```{code-cell}
chords_by_localkey = mode_slices.get_facet('expanded')
chords_by_localkey
```

```{code-cell}
for is_minor, df in chords_by_localkey.groupby(level=0, group_keys=False):
    df = df.droplevel(0)
    df = df[df.bass_note.notna()]
    sd = ms3.fifths2sd(df.bass_note, minor=is_minor).rename('sd')
    sd.index = df.index
    sd_progression = df.groupby(level=[0,1,2], group_keys=False).bass_note.apply(lambda S: S.shift(-1) - S).rename('sd_progression')
    if is_minor:
        chords_by_localkey_minor = pd.concat([df, sd, sd_progression], axis=1)
    else:
        chords_by_localkey_major = pd.concat([df, sd, sd_progression], axis=1)
```

## Scale degrees

```{code-cell}
chords_by_localkey_minor
```

```{code-cell}
fig = make_sunburst(chords_by_localkey_major, parent='major', terminal_symbol=TERMINAL_SYMBOL)
fig.update_layout(**STD_LAYOUT)
save_figure_as(fig, "bass_degree_major_sunburst")
fig.show()
```

```{code-cell}
fig = make_sunburst(chords_by_localkey_minor, parent='minor', terminal_symbol=TERMINAL_SYMBOL)
fig.update_layout(**STD_LAYOUT)
save_figure_as(fig, "bass_degree_minor_sunburst")
fig.show()
```

```{code-cell}
fig = rectangular_sunburst(chords_by_localkey_major, path=['sd', 'figbass', 'interval'], title="MAJOR", terminal_symbol=TERMINAL_SYMBOL)
fig.update_layout(**STD_LAYOUT)
save_figure_as(fig, "bass_degree-figbass-progression_major_sunburst")
fig.show()
```

```{code-cell}
fig = rectangular_sunburst(chords_by_localkey_major, path=['sd', 'interval', 'figbass'], title="MAJOR", terminal_symbol=TERMINAL_SYMBOL)
fig.update_layout(**STD_LAYOUT)
save_figure_as(fig, "bass_degree-progression-figbass_major_sunburst")
fig.show()
```

```{code-cell}
def make_table(sd_major, sd_progression_major, sd_minor=None, sd_progression_minor=None):
    selected_chords = chords_by_localkey_major[(
        (chords_by_localkey_major.sd == sd_major) &
        (chords_by_localkey_major.sd_progression == sd_progression_major)
    )]
    result = selected_chords.figbass.fillna('3').value_counts().rename(f"{sd_major} in major")
    if sd_minor is not None:
        selected_chords = chords_by_localkey_minor[(
            (chords_by_localkey_minor.sd == sd_minor) &
            (chords_by_localkey_minor.sd_progression == sd_progression_minor)
        )]
        minor_result = selected_chords.figbass.fillna('3').value_counts().rename(f"{sd_minor} in minor")
        result = pd.concat([result, minor_result], axis=1).fillna(0).astype(int)
    sum_row = pd.DataFrame(result.sum(), columns=["sum"]).T
    result = pd.concat([result, sum_row], names=["figbass"])
    return result

comparison_table = make_table("4", 5, "4", -2)
comparison_table #.to_clipboard()
```

```{code-cell}
selector = (
            (chords_by_localkey_minor.sd == "4") &
            (chords_by_localkey_minor.sd_progression == -2) &
            (chords_by_localkey_minor.figbass == "65")
        )
selector |= selector.shift()
selected_chords = chords_by_localkey_minor[selector]
selected_chords[["mn", "chord"]].droplevel([0, 2, 3]).to_clipboard()
```

```{code-cell}
fig = rectangular_sunburst(chords_by_localkey_minor, path=['sd', 'figbass', 'interval'], title="MINOR", terminal_symbol=TERMINAL_SYMBOL)
fig.update_layout(**STD_LAYOUT)
save_figure_as(fig, "bass_degree-figbass-progression_minor_sunburst")
fig.show()
```

```{code-cell}
fig = rectangular_sunburst(chords_by_localkey_minor, path=['sd', 'interval', 'figbass'], title="MINOR", terminal_symbol=TERMINAL_SYMBOL)
fig.update_layout(**STD_LAYOUT)
save_figure_as(fig, "bass_degree-progression-figbass_minor_sunburst")
fig.show()
```

```{code-cell}
fig = rectangular_sunburst(chords_by_localkey_major, path=['sd', 'interval', 'figbass', 'following_figbass'], title="MAJOR", terminal_symbol=TERMINAL_SYMBOL)
fig.update_layout(**STD_LAYOUT)
save_figure_as(fig, "bass_degree-progression-figbass-subsequent_figbass_major_sunburst")
fig.show()
```

```{code-cell}
fig = rectangular_sunburst(chords_by_localkey_minor, path=['sd', 'interval', 'figbass', 'following_figbass'], title="MINOR", terminal_symbol=TERMINAL_SYMBOL)
fig.update_layout(**STD_LAYOUT)
save_figure_as(fig, "bass_degree-progression-figbass-subsequent_figbass_minor_sunburst")
fig.show()
```
