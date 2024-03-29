---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
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

from utils import STD_LAYOUT, CADENCE_COLORS, color_background, get_repo_name, resolve_dir, value_count_df, get_repo_name, resolve_dir
```

```{code-cell} ipython3
CORPUS_PATH = os.environ.get('CORPUS_PATH', "~/dcml_corpora")
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

```{code-cell} ipython3
dataset = dc.Dataset()
dataset.load(directory=CORPUS_PATH)
dataset.data
```

### Filtering out pieces without cadence annotations

```{code-cell} ipython3
hascadence = dc.HasCadenceAnnotationsFilter().process_data(dataset)
print(f"Before: {len(dataset.indices[()])} pieces; after removing those without cadence labels: {len(hascadence.indices[()])}")
```

### Show corpora containing pieces with cadence annotations

```{code-cell} ipython3
grouped_by_dataset = dc.CorpusGrouper().process_data(hascadence)
corpora = {group[0]: f"{len(ixs)} pieces" for group, ixs in  grouped_by_dataset.indices.items()}
print(f"{len(corpora)} corpora with {sum(map(len, grouped_by_dataset.indices.values()))} pieces containing cadence annotations:")
corpora
```

### All annotation labels from the selected pieces

```{code-cell} ipython3
all_labels = hascadence.get_facet('expanded')

print(f"{len(all_labels.index)} hand-annotated harmony labels:")
all_labels.iloc[:10, 14:].style.apply(color_background, subset="chord")
```

### Metadata

```{code-cell} ipython3
dataset_metadata = hascadence.data.metadata()
hascadence_metadata = dataset_metadata.loc[hascadence.indices[()]]
hascadence_metadata.index.rename('dataset', level=0, inplace=True)
hascadence_metadata.head()
```

```{code-cell} ipython3
mean_composition_years = hascadence_metadata.groupby(level=0).composed_end.mean().astype(int).sort_values()
chronological_order = mean_composition_years.index.to_list()
bar_data = pd.concat([mean_composition_years.rename('year'), 
                      hascadence_metadata.groupby(level='dataset').size().rename('pieces')],
                     axis=1
                    ).reset_index()
fig = px.bar(bar_data, x='year', y='pieces', color='dataset', title='Pieces contained in the dataset')
fig.update_traces(width=5)
```

## Computing extent of key segments from annotations

**In the following, major and minor keys are distinguished as boolean `localkey_is_minor=(False|True)`**

```{code-cell} ipython3
segmented_by_keys = dc.Pipeline([
                         dc.LocalKeySlicer(), 
                         dc.ModeGrouper()])\
                        .process_data(hascadence)
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
maj_min_ratio_per_dataset.reset_index()
```

```{code-cell} ipython3
chronological_order
```

```{code-cell} ipython3
fig = px.bar(maj_min_ratio_per_dataset.reset_index(), 
       x="corpus", 
       y="duration_qb", 
       color="localkey_is_minor", 
       text='fraction',
       labels=dict(dataset='', duration_qb="aggregated duration in quarter notes"),
       category_orders=dict(dataset=chronological_order)
    )
fig.update_layout(**STD_LAYOUT)
```

## Annotation table sliced by key segments

```{code-cell} ipython3
annotations_by_keys = segmented_by_keys.get_facet("expanded")
annotations_by_keys
```
