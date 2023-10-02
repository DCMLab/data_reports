---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.7
kernelspec:
  display_name: ms3
  language: python
  name: ms3
---

# Keys

```{code-cell} ipython3
import os
from collections import defaultdict, Counter

from git import Repo
import dimcat as dc
import ms3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import STD_LAYOUT, CADENCE_COLORS, chronological_corpus_order, color_background, get_corpus_display_name, get_repo_name, resolve_dir, value_count_df, get_repo_name, resolve_dir
```

```{code-cell} ipython3
CORPUS_PATH = os.environ.get('CORPUS_PATH', "~/dcml_corpora")
print(f"CORPUS_PATH: '{CORPUS_PATH}'")
CORPUS_PATH = resolve_dir(CORPUS_PATH)
```

```{code-cell} ipython3
repo = Repo(CORPUS_PATH)
notebook_repo = Repo('.', search_parent_directories=True)
print(f"Notebook repository '{get_repo_name(notebook_repo)}' @ {notebook_repo.commit().hexsha[:7]}")
print(f"Data repo '{get_repo_name(CORPUS_PATH)}' @ {repo.commit().hexsha[:7]}")
print(f"dimcat version {dc.__version__}")
print(f"ms3 version {ms3.__version__}")
```

## Data loading

### Detected files

```{code-cell} ipython3
dataset = dc.Dataset()
dataset.load(directory=CORPUS_PATH)
dataset.data
```

```{code-cell} ipython3
print(f"N = {dataset.data.count_pieces()} annotated pieces, {dataset.data.count_parsed_tsvs()} parsed dataframes.")
```

## Metadata

```{code-cell} ipython3
all_metadata = dataset.data.metadata()
print(f"Concatenated 'metadata.tsv' files cover {len(all_metadata)} of the {dataset.data.count_pieces()} scores.")
all_metadata.reset_index(level=1).groupby(level=0).nth(0).iloc[:,:20]
```

## All annotation labels from the selected pieces

```{code-cell} ipython3
all_labels = dataset.data.get_facet('expanded')

print(f"{len(all_labels.index)} hand-annotated harmony labels:")
all_labels.iloc[:20].style.apply(color_background, subset="chord")
```

## Computing extent of key segments from annotations

**In the following, major and minor keys are distinguished as boolean `localkey_is_minor=(False|True)`**

```{code-cell} ipython3
segmented_by_keys = dc.Pipeline([
                         dc.LocalKeySlicer(),
                         dc.ModeGrouper()])\
                        .process_data(dataset)
key_segments = segmented_by_keys.get_slice_info()
```

```{code-cell} ipython3
print(key_segments.duration_qb.dtype)
key_segments.duration_qb = pd.to_numeric(key_segments.duration_qb)
```

```{code-cell} ipython3
key_segments.iloc[:15, 11:].fillna('').style.apply(color_background, subset="localkey")
```

## Ratio between major and minor key segments by aggregated durations
### Overall

```{code-cell} ipython3
maj_min_ratio = key_segments.groupby(level="localkey_is_minor").duration_qb.sum().to_frame()
maj_min_ratio['fraction'] = (100.0 * maj_min_ratio.duration_qb / maj_min_ratio.duration_qb.sum()).round(1)
maj_min_ratio
```

### By dataset

```{code-cell} ipython3
segment_duration_per_dataset = key_segments.groupby(level=["corpus", "localkey_is_minor"]).duration_qb.sum().round(2)
norm_segment_duration_per_dataset = 100 * segment_duration_per_dataset / segment_duration_per_dataset.groupby(level="corpus").sum()
maj_min_ratio_per_dataset = pd.concat([segment_duration_per_dataset,
                                      norm_segment_duration_per_dataset.rename('fraction').round(1).astype(str)+" %"],
                                     axis=1)
```

```{code-cell} ipython3
segment_duration_per_dataset = key_segments.groupby(level=["corpus", "localkey_is_minor"]).duration_qb.sum().reset_index()
```

```{code-cell} ipython3
maj_min_ratio_per_dataset
```

```{code-cell} ipython3
chronological_order = chronological_corpus_order(all_metadata)
corpus_names = {corp: get_corpus_display_name(corp) for corp in chronological_order}
chronological_corpus_names = list(corpus_names.values())
#corpus_name_colors = {corpus_names[corp]: color for corp, color in corpus_colors.items()}
maj_min_ratio_per_dataset['corpus_name'] = maj_min_ratio_per_dataset.index.get_level_values('corpus').map(corpus_names)
maj_min_ratio_per_dataset['mode'] = maj_min_ratio_per_dataset.index.get_level_values('localkey_is_minor').map({False: 'major', True: 'minor'})
```

```{code-cell} ipython3
maj_min_ratio_per_dataset
```

```{code-cell} ipython3
os.chdir("/home/hentsche/ismir2023_dimcat/notebooks/")
```

```{code-cell} ipython3
fig = px.bar(maj_min_ratio_per_dataset.reset_index(),
       x="corpus_name",
       y="duration_qb",
       color="mode",
       text='fraction',
       labels=dict(dataset='', duration_qb="duration in 𝅘𝅥", corpus_name='Key segments grouped by corpus'),
       category_orders=dict(dataset=chronological_order)
    )
fig.update_layout(**STD_LAYOUT, height=270, width=1200)
fig.write_image(os.path.join(os.path.abspath("../latex/figs/mode.pdf")))
fig.show()
```

```{raw-cell}
D = dc.Dataset("~/my_dataset")
grpd_slcs = dc.Pipeline(
    [dc.KeySlicer(),
     dc.ModeGrouper()]
).process(D)
F = dc.PitchesConfig(as=SCALE_DEGREES)
grpd_slcs.get_feature(F).plot_groups()
corpus_groups = dc.CorpusGrouper().process(grpd_slcs)
corpus_groups.get_slice_info().plot_groups()
```

## Annotation table sliced by key segments

```{code-cell} ipython3
notes_by_keys = segmented_by_keys.get_facet("notes")
notes_by_keys
```

```{code-cell} ipython3
slice_info = segmented_by_keys.get_slice_info()
slice_info = slice_info[[col for col in slice_info.columns if col not in notes_by_keys]]
notes_joined_with_keys = notes_by_keys.join(slice_info, on=slice_info.index.names)
```

```{code-cell} ipython3
notes_by_keys_transposed = ms3.transpose_notes_to_localkey(notes_joined_with_keys)
```

```{code-cell} ipython3
mode_tpcs = notes_by_keys_transposed.reset_index(drop=True).groupby(['localkey_is_minor', 'tpc']).duration_qb.sum().reset_index(-1).sort_values('tpc').reset_index()
mode_tpcs
```

```{code-cell} ipython3
mode_tpcs['sd'] = ms3.fifths2sd(mode_tpcs.tpc)
mode_tpcs['duration_pct'] = mode_tpcs.groupby('localkey_is_minor', group_keys=False).duration_qb.apply(lambda S: S / S.sum())
mode_tpcs['mode'] = mode_tpcs.localkey_is_minor.map({False: 'major', True: 'minor'})
mode_tpcs
```

```{code-cell} ipython3
#mode_tpcs = mode_tpcs[mode_tpcs['duration_pct'] > 0.001]
#sd_order = ['b1', '1', '#1', 'b2', '2', '#2', 'b3', '3', 'b4', '4', '#4', '##4', 'b5', '5', '#5', 'b6','6', '#6', 'b7', '7']
xaxis = dict(
        tickmode = 'array',
        tickvals = mode_tpcs.tpc,
        ticktext = mode_tpcs.sd
    )
legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99
)
fig = px.bar(mode_tpcs,
       x='tpc',
       y='duration_pct',
       color='mode',
       barmode='group',
       labels=dict(duration_pct='normalized duration',
                   tpc="Notes transposed to the local key, as major-scale degrees",
                  ),
       #log_y=True,
       #category_orders=dict(sd=sd_order)
      )
fig.update_layout(**STD_LAYOUT, xaxis=xaxis, legend=legend, height=210, width=1200)
fig.write_image(os.path.join(os.path.abspath("../latex/figs/mode_sds.pdf")))
fig.show()
```

```{code-cell} ipython3
fig = px.bar(maj_min_ratio_per_dataset.reset_index(),
       x="corpus_name",
       y="duration_qb",
       color="mode",
       text='fraction',
       labels=dict(dataset='', duration_qb="duration in 𝅘𝅥", corpus_name='Key segments grouped by corpus'),
       category_orders=dict(dataset=chronological_order)
    )
fig.update_layout(**STD_LAYOUT, height=270, width=1200)
fig.write_image(os.path.join(os.path.abspath("../latex/figs/mode.pdf")))
fig.show()
```

```{code-cell} ipython3

```
