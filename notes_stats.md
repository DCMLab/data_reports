---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
---

# Notes

```{code-cell} ipython3
import os
from collections import defaultdict, Counter

from git import Repo
import dimcat as dc
import ms3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import STD_LAYOUT, CADENCE_COLORS, CORPUS_COLOR_SCALE, color_background, get_repo_name, resolve_dir, value_count_df, get_repo_name, resolve_dir
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
dataset.load(directory=CORPUS_PATH, parse_tsv=False)
dataset.data
```

### Filtering

```{code-cell} ipython3
annotated_view = dataset.data.get_view('annotated')
annotated_view.include('facets', 'measures', 'notes$', 'expanded')
annotated_view.fnames_with_incomplete_facets = False
dataset.data.set_view(annotated_view)
dataset.data
```

```{code-cell} ipython3
print(f"N = {dataset.data.count_pieces()} annotated pieces.")
```

## Metadata

```{code-cell} ipython3
all_metadata = dataset.data.metadata()
print(f"Concatenated 'metadata.tsv' files cover {len(all_metadata)} of the {dataset.data.count_pieces()} scores.")
all_metadata.reset_index(level=1).groupby(level=0).nth(0).iloc[:,:20]
```

**Compute chronological order**

```{code-cell} ipython3
mean_composition_years = all_metadata.groupby(level=0).composed_end.mean().astype(int).sort_values()
chronological_order = mean_composition_years.index.to_list()
dataset_colors = dict(zip(chronological_order, CORPUS_COLOR_SCALE))
chronological_order
```

```{code-cell} ipython3
all_notes = dataset.data.get_all_parsed('notes', force=True, flat=True)
print(f"{len(all_notes.index)} notes over {len(all_notes.groupby(level=[0,1]))} files.")
all_notes.head()
```

```{code-cell} ipython3
def weight_notes(nl, group_col='midi', precise=True):
    summed_durations = nl.groupby(group_col).duration_qb.sum()
    shortest_duration = summed_durations[summed_durations > 0].min()
    summed_durations /= shortest_duration # normalize such that the shortest duration results in 1 occurrence
    if not precise:
        # This simple trick reduces compute time but also precision:
        # The rationale is to have the smallest value be slightly larger than 0.5 because
        # if it was exactly 0.5 it would be rounded down by repeat_notes_according_to_weights()
        summed_durations /= 1.9999999
    return repeat_notes_according_to_weights(summed_durations)
    
def repeat_notes_according_to_weights(weights):
    counts = weights.round().astype(int)
    counts_reflecting_weights = []
    for pitch, count in counts.items():
        counts_reflecting_weights.extend([pitch]*count)
    return pd.Series(counts_reflecting_weights)
```

## Ambitus

```{code-cell} ipython3
corpus_names = dict(
    corelli='Corelli Trio Sonatas',
    mozart_piano_sonatas='Mozart Piano Sonatas',
    ABC='Beethoven String Quartets',
    beethoven_piano_sonatas='Beethoven Sonatas',
    chopin_mazurkas='Chopin Mazurkas',
    debussy_suite_bergamasque='Debussy Suite',
    dvorak_silhouettes="Dvořák Silhouettes",
    grieg_lyric_pieces="Grieg Lyric Pieces",
    liszt_pelerinage="Liszt Années",
    medtner_tales="Medtner Tales",
    schumann_kinderszenen="Schumann Kinderszenen",
    tchaikovsky_seasons="Tchaikovsky Seasons"
)
dataset_name_colors = {corpus_names[corp]: color for corp, color in dataset_colors.items()}
chronological_corpus_names = [corpus_names[corp] for corp in chronological_order]
all_notes['dataset_name'] = all_notes.index.get_level_values(0).map(corpus_names)
```

```{code-cell} ipython3
grouped_notes = all_notes.groupby('dataset_name')
weighted_midi = pd.concat([weight_notes(nl, 'midi', precise=False) for _, nl in grouped_notes], keys=grouped_notes.groups.keys()).reset_index(level=0)
weighted_midi.columns = ['dataset', 'midi']
weighted_midi
```

```{code-cell} ipython3
yaxis=dict(tickmode= 'array',
           tickvals= [12, 24, 36, 48, 60, 72, 84, 96],
           ticktext = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"],
           gridcolor='lightgrey',
           )
fig = px.violin(weighted_midi, 
                x='dataset', 
                y='midi', 
                color='dataset', 
                box=True,
                labels=dict(
                    dataset='',
                    midi='distribution of pitches by duration'
                ),
                category_orders=dict(dataset=chronological_corpus_names),
                color_discrete_map=dataset_name_colors,
                width=1000, height=600,
               )
fig.update_traces(spanmode='hard') # do not extend beyond outliers
fig.update_layout(yaxis=yaxis, 
                  **STD_LAYOUT,
                 showlegend=False)
fig.show()
```

## Tonal Pitch Classes (TPC)

```{code-cell} ipython3
weighted_tpc = pd.concat([weight_notes(nl, 'tpc') for _, nl in grouped_notes], keys=grouped_notes.groups.keys()).reset_index(level=0)
weighted_tpc.columns = ['dataset', 'tpc']
weighted_tpc
```

### As violin plot

```{code-cell} ipython3
yaxis=dict(
    tickmode= 'array',
    tickvals= [-12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18],
    ticktext = ["Dbb", "Bbb", "Gb", "Eb", "C", "A", "F#", "D#", "B#", "G##", "E##"],
    gridcolor='lightgrey',
    zerolinecolor='lightgrey',
    zeroline=True
           )
fig = px.violin(weighted_tpc, 
                x='dataset', 
                y='tpc', 
                color='dataset', 
                box=True,
                labels=dict(
                    dataset='',
                    tpc='distribution of tonal pitch classes by duration'
                ),
                category_orders=dict(dataset=chronological_corpus_names),
                color_discrete_map=dataset_name_colors,
                width=1000, 
                height=600,
               )
fig.update_traces(spanmode='hard') # do not extend beyond outliers
fig.update_layout(yaxis=yaxis, 
                  **STD_LAYOUT,
                 showlegend=False)
fig.show()
```

### As bar plots

```{code-cell} ipython3
bar_data = all_notes.groupby('tpc').duration_qb.sum().reset_index()
x_values = list(range(bar_data.tpc.min(), bar_data.tpc.max()+1))
x_names = ms3.fifths2name(x_values)
fig = px.bar(bar_data, x='tpc', y='duration_qb',
             labels=dict(tpc='Named pitch class',
                             duration_qb='Duration in quarter notes'
                            ),
             color_discrete_sequence=CORPUS_COLOR_SCALE,
             width=1000, height=300,
             )
fig.update_layout(**STD_LAYOUT)
fig.update_yaxes(gridcolor='lightgrey')
fig.update_xaxes(gridcolor='lightgrey', zerolinecolor='grey', tickmode='array', 
                 tickvals=x_values, ticktext = x_names, dtick=1, ticks='outside', tickcolor='black', 
                 minor=dict(dtick=6, gridcolor='grey', showgrid=True),
                )
fig.show()
```

```{code-cell} ipython3
scatter_data = all_notes.groupby(['dataset_name', 'tpc']).duration_qb.sum().reset_index()
px.bar(scatter_data, x='tpc', y='duration_qb', color='dataset_name', 
                 labels=dict(
                     duration_qb='duration',
                     tpc='named pitch class',
                 ),
                 category_orders=dict(dataset=chronological_corpus_names),
                 color_discrete_map=dataset_name_colors,
                 width=1000, height=500,
                )
```

### As scatter plots

```{code-cell} ipython3
fig = px.scatter(scatter_data, x='tpc', y='duration_qb', color='dataset_name', 
                 labels=dict(
                     duration_qb='duration',
                     tpc='named pitch class',
                 ),
                 category_orders=dict(dataset=chronological_corpus_names),
                 color_discrete_map=dataset_name_colors,
                 facet_col='dataset_name', facet_col_wrap=3, facet_col_spacing=0.03,
                 width=1000, height=500,
                )
fig.update_traces(mode='lines+markers')
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_layout(**STD_LAYOUT, showlegend=False)
fig.update_xaxes(gridcolor='lightgrey', zerolinecolor='lightgrey', tickmode='array', tickvals= [-12, -6, 0, 6, 12, 18],
    ticktext = ["Dbb", "Gb", "C", "F#", "B#", "E##"], visible=True, )
fig.update_yaxes(gridcolor='lightgrey', zeroline=False, matches=None, showticklabels=True)
fig.show()
```

```{code-cell} ipython3
no_accidental = bar_data[bar_data.tpc.between(-1,5)].duration_qb.sum()
with_accidental = bar_data[~bar_data.tpc.between(-1,5)].duration_qb.sum()
```

```{code-cell} ipython3
entire = no_accidental + with_accidental
f"Fraction of note duration without accidental of the entire durations: {no_accidental} / {entire} = {no_accidental / entire}"
```

### Notes and staves

```{code-cell} ipython3
print("Distribution of notes over staves:")
value_count_df(all_notes.staff)
```
