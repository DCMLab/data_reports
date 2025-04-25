---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: revamp
  language: python
  name: revamp
---

# Sunburst

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide imports
  code_prompt_show: Show imports
tags: [hide-cell]
---
from dimcat.steps import slicers, groupers
%load_ext autoreload
%autoreload 2
import os
import ms3
import pandas as pd
from dimcat import plotting, Pipeline

import utils
import plotly.io as pio
# workaround to remove the "loading mathjax" box from the PDF figure
# see https://github.com/plotly/plotly.py/issues/3469#issuecomment-994907721
pio.kaleido.scope.mathjax = None
# if mathjax was needed to render math, one could try
# pio.full_figure_for_development(fig, warn=False)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
```

```{code-cell}
SUNBURST_WIDTH = 1620
TERMINAL_SYMBOL = "∎"
RESULTS_PATH = os.path.abspath(os.path.join(utils.OUTPUT_FOLDER, "scale_degrees"))
os.makedirs(RESULTS_PATH, exist_ok=True)


def make_output_path(
    filename: str,
    extension=None,
    path=RESULTS_PATH,
) -> str:
    return utils.make_output_path(filename=filename, extension=extension, path=path)


def save_figure_as(
    fig, filename, formats=("png", "pdf"), directory=RESULTS_PATH, **kwargs
):
    if formats is not None:
        for fmt in formats:
            plotting.write_image(fig, filename, directory, format=fmt, **kwargs)
    else:
        plotting.write_image(fig, filename, directory, **kwargs)
```

**Loading data**

```{code-cell}
D = utils.get_dataset("couperin_concerts", corpus_release="v2.2")
D
```

## Key areas

```{code-cell}
sliced_D = slicers.KeySlicer().process(D)
grouped_D = groupers.ModeGrouper().process(sliced_D)
grouped_D
```

```{code-cell}
notes = sliced_D.get_feature('notes')
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
notes.head()
```

```{code-cell}
keys = grouped_D.get_feature("KeyAnnotations")
print(f"Overall number of key segments is {len(keys.index)}")
keys.head()
```

```{code-cell}
# this bit is copied from the annotations notebook
keys_data = keys[[col for col in keys.columns if col not in notes.columns]].droplevel(-1)
notes_joined_with_keys = notes.join(keys_data, how="left",)
notes_by_keys_transposed = ms3.transpose_notes_to_localkey(notes_joined_with_keys)
tpc_distribution = notes_by_keys_transposed.reset_index(drop=True).groupby(['localkey_is_minor', 'tpc']).duration_qb.sum()
mode_tpcs = tpc_distribution.reset_index(-1).sort_values('tpc').reset_index()
mode_tpcs['sd'] = ms3.fifths2sd(mode_tpcs.tpc)
mode_tpcs['duration_pct'] = mode_tpcs.groupby('localkey_is_minor', group_keys=False).duration_qb.apply(lambda S: S / S.sum())
mode_tpcs['mode'] = mode_tpcs.localkey_is_minor.map({False: 'major', True: 'minor'})
corpuswise_tpc_distribution = notes_by_keys_transposed.groupby(["corpus", "localkey_is_minor", "tpc"]).duration_qb.sum().reset_index()
corpuswise_tpc_distribution['duration_pct'] = corpuswise_tpc_distribution.groupby(["corpus", "localkey_is_minor"], group_keys=False).duration_qb.apply(lambda S: S / S.sum())
std_err_mean = corpuswise_tpc_distribution.groupby(["localkey_is_minor", "tpc"]).duration_pct.sem().rename("std_err")
additional_columns = dict(
    sd = ms3.fifths2sd(mode_tpcs.tpc),
    duration_pct = mode_tpcs.groupby('localkey_is_minor', group_keys=False).duration_qb.apply(lambda S: S / S.sum()),
    mode = mode_tpcs.localkey_is_minor.map({False: 'major', True: 'minor'}),
    std_err = std_err_mean
)
mode_tpcs = mode_tpcs.join(std_err_mean, on=["localkey_is_minor", "tpc"])
mode_tpcs
```

```{code-cell}
sd_order = ['b1', '1', '#1', 'b2', '2', '#2', 'b3', '3', '4', '#4', 'b5', '5', '#5', 'b6','6', '#6', 'b7', '7']
selector = (mode_tpcs.tpc > -8) & (mode_tpcs.tpc < 11)
legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99
)

fig = plotting.make_bar_plot(
    mode_tpcs[selector], #.query("duration_pct > 0.001")
    x_col='sd',
    y_col='duration_pct',
    #title="Scale degree distribution over major and minor segments",
    color='mode',
    barmode='group',
    color_discrete_map=utils.MAJOR_MINOR_COLORS,
    labels=dict(
        duration_pct='normalized duration',
        duration_qb="duration in ♩",
        sd="Notes transposed to the local key, as major-scale degrees",
        ),
    error_y="std_err",
    layout=dict(
        margin=dict(
            t=0
        )
    ),
    x_axis=dict(
        showgrid=False,
    ),
    #log_y=True,
    category_orders=dict(sd=sd_order)
    )
save_figure_as(fig, 'scale_degree_distributions_maj_min_normalized_bars', height=350, width=1200)
fig.show()
other_sum = mode_tpcs[~selector].sum()
print(f"{(~selector).sum()} scale degrees with a total duration of {other_sum.duration_qb} ♩ "
      f"({other_sum.duration_pct:.2%}) are not in the range -7 to 10 and have been omitted.")
```

### Whole dataset

```{code-cell}
chords_by_localkey = grouped_D.get_feature('HarmonyLabels')
chords_by_localkey
```

```{code-cell}
for is_minor, df in chords_by_localkey.groupby(level=0, group_keys=False):
    df = df.droplevel(0)
    df = df[df.bass_note.notna()]
    sd = ms3.fifths2sd(df.bass_note, minor=is_minor).rename('sd')
    sd.index = df.index
    sd_progression = df.groupby(level=[0,1,2], group_keys=False).bass_note.apply(lambda S: S.shift(-1) - S).rename('sd_progression')
    if is_minor == "minor":
        chords_by_localkey_minor = pd.concat([df, sd, sd_progression], axis=1)
    else:
        chords_by_localkey_major = pd.concat([df, sd, sd_progression], axis=1)
```

## Scale degrees

```{code-cell}
chords_by_localkey_minor
```

```{code-cell}
fig = utils.make_sunburst(chords_by_localkey_major, parent='major', terminal_symbol=TERMINAL_SYMBOL)
fig.update_layout(**utils.STD_LAYOUT)
save_figure_as(fig, "bass_degree_major_sunburst")
fig.show()
```

```{code-cell}
fig = utils.make_sunburst(chords_by_localkey_minor, parent='minor', terminal_symbol=TERMINAL_SYMBOL)
fig.update_layout(**utils.STD_LAYOUT)
save_figure_as(fig, "bass_degree_minor_sunburst")
fig.show()
```

```{code-cell}
fig = utils.rectangular_sunburst(chords_by_localkey_major, path=['sd', 'figbass', 'interval'], title="MAJOR", terminal_symbol=TERMINAL_SYMBOL)
fig.update_layout(**utils.STD_LAYOUT)
save_figure_as(fig, "bass_degree-figbass-progression_major_sunburst")
fig.show()
```

```{code-cell}
fig = utils.rectangular_sunburst(chords_by_localkey_major, path=['sd', 'interval', 'figbass'], title="MAJOR", terminal_symbol=TERMINAL_SYMBOL)
fig.update_layout(**utils.STD_LAYOUT)
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
fig = utils.rectangular_sunburst(chords_by_localkey_minor, path=['sd', 'figbass', 'interval'], title="MINOR", terminal_symbol=TERMINAL_SYMBOL)
fig.update_layout(**utils.STD_LAYOUT)
save_figure_as(fig, "bass_degree-figbass-progression_minor_sunburst")
fig.show()
```

```{code-cell}
fig = utils.rectangular_sunburst(chords_by_localkey_minor, path=['sd', 'interval', 'figbass'], title="MINOR", terminal_symbol=TERMINAL_SYMBOL)
fig.update_layout(**utils.STD_LAYOUT)
save_figure_as(fig, "bass_degree-progression-figbass_minor_sunburst")
fig.show()
```

```{code-cell}
---
jupyter:
  is_executing: true
---
fig = utils.rectangular_sunburst(chords_by_localkey_major, path=['sd', 'interval', 'figbass', 'following_figbass'], title="MAJOR", terminal_symbol=TERMINAL_SYMBOL)
fig.update_layout(**utils.STD_LAYOUT)
save_figure_as(fig, "bass_degree-progression-figbass-subsequent_figbass_major_sunburst")
fig.show()
```

```{code-cell}
fig = utils.rectangular_sunburst(chords_by_localkey_minor, path=['sd', 'interval', 'figbass', 'following_figbass'], title="MINOR", terminal_symbol=TERMINAL_SYMBOL)
fig.update_layout(**utils.STD_LAYOUT)
save_figure_as(fig, "bass_degree-progression-figbass-subsequent_figbass_minor_sunburst")
fig.show()
```
