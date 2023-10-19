---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: corpus_docs
  language: python
  name: corpus_docs
---

# Keys

```{code-cell}
import os

from git import Repo
import dimcat as dc
import ms3
import pandas as pd
import plotly.express as px

from utils import STD_LAYOUT, chronological_corpus_order, color_background, get_corpus_display_name, get_repo_name, resolve_dir, print_heading
```

```{code-cell}
from utils import OUTPUT_FOLDER, write_image
RESULTS_PATH = os.path.abspath(os.path.join(OUTPUT_FOLDER, "keys"))
os.makedirs(RESULTS_PATH, exist_ok=True)
def save_figure_as(fig, filename, directory=RESULTS_PATH, **kwargs):
    write_image(fig, filename, directory, **kwargs)
```

```{code-cell}
# CORPUS_PATH = os.path.abspath(os.path.join('..', '..')) # for running the notebook in the homepage deployment workflow
CORPUS_PATH = "~/distant_listening_corpus/couperin_concerts"                # for running the notebook locally
print_heading("Notebook settings")
print(f"CORPUS_PATH: {CORPUS_PATH!r}")
CORPUS_PATH = resolve_dir(CORPUS_PATH)
```

```{code-cell}
repo = Repo(CORPUS_PATH)
print_heading("Data and software versions")
print(f"Data repo '{get_repo_name(repo)}' @ {repo.commit().hexsha[:7]}")
print(f"dimcat version {dc.__version__}")
print(f"ms3 version {ms3.__version__}")
```

## Data loading

### Detected files

```{code-cell}
dataset = dc.Dataset()
dataset.load(directory=CORPUS_PATH, parse_tsv=False)
```

```{code-cell}
annotated_view = dataset.data.get_view('annotated')
annotated_view.include('facets', 'measures', 'notes$', 'expanded')
annotated_view.pieces_with_incomplete_facets = False
dataset.data.set_view(annotated_view)
dataset.data.parse_tsv(choose='auto')
dataset.get_indices()
dataset.data
```

```{code-cell}
print(f"N = {dataset.data.count_pieces()} annotated pieces, {dataset.data.count_parsed_tsvs()} parsed dataframes.")
```

## Metadata

```{code-cell}
all_metadata = dataset.data.metadata()
print(f"Concatenated 'metadata.tsv' files cover {len(all_metadata)} of the {dataset.data.count_pieces()} scores.")
all_metadata.reset_index(level=1).groupby(level=0).nth(0).iloc[:,:20]
```

## All annotation labels from the selected pieces

```{code-cell}
all_labels = dataset.data.get_facet('expanded')

print(f"{len(all_labels.index)} hand-annotated harmony labels:")
all_labels.iloc[:20].style.apply(color_background, subset="chord")
```

## Computing extent of key segments from annotations

**In the following, major and minor keys are distinguished as boolean `localkey_is_minor=(False|True)`**

```{code-cell}
segmented_by_keys = dc.Pipeline([
                         dc.LocalKeySlicer(),
                         dc.ModeGrouper()])\
                        .process_data(dataset)
key_segments = segmented_by_keys.get_slice_info()
```

```{code-cell}
print(key_segments.duration_qb.dtype)
key_segments.duration_qb = pd.to_numeric(key_segments.duration_qb)
```

```{code-cell}
key_segments.iloc[:15, 11:].fillna('').style.apply(color_background, subset="localkey")
```

## Ratio between major and minor key segments by aggregated durations
### Overall

```{code-cell}
maj_min_ratio = key_segments.groupby(level="localkey_is_minor").duration_qb.sum().to_frame()
maj_min_ratio['fraction'] = (100.0 * maj_min_ratio.duration_qb / maj_min_ratio.duration_qb.sum()).round(1)
maj_min_ratio
```

### By dataset

```{code-cell}
segment_duration_per_corpus = key_segments.groupby(level=["corpus", "localkey_is_minor"]).duration_qb.sum().round(2)
norm_segment_duration_per_corpus = 100 * segment_duration_per_corpus / segment_duration_per_corpus.groupby(level="corpus").sum()
maj_min_ratio_per_corpus = pd.concat([segment_duration_per_corpus,
                                      norm_segment_duration_per_corpus.rename('fraction').round(1).astype(str)+" %"],
                                     axis=1)
```

```{code-cell}
segment_duration_per_corpus = key_segments.groupby(level=["corpus", "localkey_is_minor"]).duration_qb.sum().reset_index()
```

```{code-cell}
maj_min_ratio_per_corpus
```

```{code-cell}
chronological_order = chronological_corpus_order(all_metadata)
corpus_names = {corp: get_corpus_display_name(corp) for corp in chronological_order}
chronological_corpus_names = list(corpus_names.values())
#corpus_name_colors = {corpus_names[corp]: color for corp, color in corpus_colors.items()}
maj_min_ratio_per_corpus['corpus_name'] = maj_min_ratio_per_corpus.index.get_level_values('corpus').map(corpus_names)
maj_min_ratio_per_corpus['mode'] = maj_min_ratio_per_corpus.index.get_level_values('localkey_is_minor').map({False: 'major', True: 'minor'})
```

```{code-cell}
maj_min_ratio_per_corpus
```

```{code-cell}
fig = px.bar(
    maj_min_ratio_per_corpus.reset_index(),
    x="corpus_name",
    y="duration_qb",
    title="Fractions of summed corpus duration that are in major vs. minor",
    color="mode",
    text='fraction',
    labels=dict(duration_qb="duration in 𝅘𝅥", corpus_name='Key segments grouped by corpus'),
    category_orders=dict(corpus_name=chronological_corpus_names)
    )
fig.update_layout(**STD_LAYOUT)
fig.update_xaxes(tickangle=45)
save_figure_as(fig, 'major_minor_key_segments_corpuswise_absolute_stacked_bars', height=900)
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

```{code-cell}
notes_by_keys = segmented_by_keys.get_facet("notes")
notes_by_keys
```

```{code-cell}
slice_info = segmented_by_keys.get_slice_info()
slice_info = slice_info[[col for col in slice_info.columns if col not in notes_by_keys]]
notes_joined_with_keys = notes_by_keys.join(slice_info, on=slice_info.index.names)
```

```{code-cell}
notes_by_keys_transposed = ms3.transpose_notes_to_localkey(notes_joined_with_keys)
```

```{code-cell}
mode_tpcs = notes_by_keys_transposed.reset_index(drop=True).groupby(['localkey_is_minor', 'tpc']).duration_qb.sum().reset_index(-1).sort_values('tpc').reset_index()
mode_tpcs
```

```{code-cell}
mode_tpcs['sd'] = ms3.fifths2sd(mode_tpcs.tpc)
mode_tpcs['duration_pct'] = mode_tpcs.groupby('localkey_is_minor', group_keys=False).duration_qb.apply(lambda S: S / S.sum())
mode_tpcs['mode'] = mode_tpcs.localkey_is_minor.map({False: 'major', True: 'minor'})
mode_tpcs
```
