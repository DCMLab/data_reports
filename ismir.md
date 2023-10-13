---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: revamp
  language: python
  name: revamp
---

# Plots for ISMIR 2023

Notebook created by copying and adapting `annotations.ipynb`.

```{code-cell} ipython3
---
mystnb:
  code_prompt_hide: Hide imports
  code_prompt_show: Show imports
tags: [hide-cell]
---

from matplotlib import pyplot as plt
%load_ext autoreload
%autoreload 2
import os

import dimcat as dc
from dimcat.steps import groupers
import ms3
import pandas as pd
import plotly.express as px
from git import Repo

from utils import STD_LAYOUT, CORPUS_COLOR_SCALE, TYPE_COLORS, color_background, corpus_mean_composition_years, \
  get_corpus_display_name, get_repo_name, print_heading, resolve_dir, plot_cum, value_count_df, \
  plot_pitch_class_distribution, remove_none_labels, remove_non_chord_labels, plot_transition_heatmaps
```

```{code-cell} ipython3
from utils import OUTPUT_FOLDER, write_image
RESULTS_PATH = os.path.abspath(os.path.join(OUTPUT_FOLDER, "ismir"))
os.makedirs(RESULTS_PATH, exist_ok=True)
def save_figure_as(fig, filename, directory=RESULTS_PATH, **kwargs):
    write_image(fig, filename, directory, format=".pdf", **kwargs)
```

```{code-cell} ipython3
:tags: [remove-output]

package_path = resolve_dir("~/dcml_corpora/dcml_corpora.datapackage.json")
repo = Repo(os.path.dirname(package_path))
print_heading("Data and software versions")
print(f"Data repo '{get_repo_name(repo)}' @ {repo.commit().hexsha[:7]}")
print(f"dimcat version {dc.__version__}")
print(f"ms3 version {ms3.__version__}")
D = dc.Dataset.from_package(package_path)
D
```

```{code-cell} ipython3
all_metadata = D.get_metadata()
assert len(all_metadata) > 0, "No pieces selected for analysis."
mean_composition_years = corpus_mean_composition_years(all_metadata)
chronological_order = mean_composition_years.index.to_list()
corpus_colors = dict(zip(chronological_order, CORPUS_COLOR_SCALE))
corpus_names = {corp: get_corpus_display_name(corp) for corp in chronological_order}
chronological_corpus_names = list(corpus_names.values())
corpus_name_colors = {corpus_names[corp]: color for corp, color in corpus_colors.items()}
```

## FeatureExtractor

```{code-cell} ipython3
notes = D.get_feature('notes')
fig = plot_pitch_class_distribution(
  notes.df,
  title=None,
  modin=False)
save_figure_as(fig, "complete_pitch_class_distribution_absolute_bars", height=800)
fig.show()
```

## Groupers

```{code-cell} ipython3
grouping = dc.Pipeline([
    groupers.CorpusGrouper(),
    groupers.ModeGrouper(),
])
GD = grouping.process(D)
grouped_keys = GD.get_feature('keyannotations')
grouped_keys_df = grouped_keys.df
grouped_keys_df
```

```{code-cell} ipython3
segment_duration_per_corpus = grouped_keys.groupby(["corpus", "mode"]).duration_qb.sum().round(2)
norm_segment_duration_per_corpus = 100 * segment_duration_per_corpus / segment_duration_per_corpus.groupby("corpus").sum()
maj_min_ratio_per_corpus = pd.concat([segment_duration_per_corpus,
                                      norm_segment_duration_per_corpus.rename('fraction').round(1).astype(str)+" %"],
                                     axis=1)
maj_min_ratio_per_corpus['corpus_name'] = maj_min_ratio_per_corpus.index.get_level_values('corpus').map(corpus_names)
fig = px.bar(
    maj_min_ratio_per_corpus.reset_index(),
    x="corpus_name",
    y="duration_qb",
    title=None, #f"Fractions of summed corpus duration that are in major vs. minor",
    color="mode",
    text='fraction',
    labels=dict(duration_qb="duration in 𝅘𝅥", corpus_name='Key segments grouped by corpus'),
    category_orders=dict(corpus_name=chronological_corpus_names)
    )
fig.update_layout(**STD_LAYOUT)
fig.update_xaxes(tickangle=45)
save_figure_as(fig, 'major_minor_key_segments_corpuswise_absolute_stacked_bars', height=800)
fig.show()
```

```{raw-cell}
to_be_filled = grouped_keys_df.quarterbeats_all_endings == ''
grouped_keys_df.quarterbeats_all_endings = grouped_keys_df.quarterbeats_all_endings.where(~to_be_filled, grouped_keys_df.quarterbeats)
ms3.make_interval_index_from_durations(grouped_keys_df, position_col="quarterbeats_all_endings")
```

## Slicer

```{code-cell} ipython3
:tags: [hide-input]

try:
    labels = D.get_feature('harmonylabels')
    all_annotations = labels.df
except Exception:
    all_annotations = pd.DataFrame()
n_annotations = len(all_annotations)
includes_annotations = n_annotations > 0
if includes_annotations:
    all_chords = remove_none_labels(all_annotations)
    all_chords = remove_non_chord_labels(all_chords)
    display(all_chords.head())
    print(f"Concatenated annotation tables contain {n_annotations} rows.")
    no_chord = all_annotations.root.isna()
    print(f"Dataset contains {len(all_chords)} tokens and {len(all_chords.chord.unique())} types over {len(all_chords.groupby(level=[0,1]))} documents.")
    all_annotations['corpus_name'] = all_annotations.index.get_level_values(0).map(corpus_names)
    all_chords['corpus_name'] = all_chords.index.get_level_values(0).map(corpus_names)
else:
    print(f"Dataset contains no annotations.")
```

```{code-cell} ipython3
group_keys, group_dict = dc.data.resources.utils.make_adjacency_groups(
            all_chords.localkey, groupby=["corpus", "piece"]
        )
segment2bass_note_series = {seg: bn for seg, bn in all_chords.groupby(group_keys).bass_note}
full_grams = {i: S[( S!=S.shift() ).fillna(True)].to_list() for i, S in segment2bass_note_series.items()}
full_grams_major, full_grams_minor = [], []
for i, bass_notes in segment2bass_note_series.items():
    #progression = bass_notes[(bass_notes != bass_notes.shift()).fillna(True)].to_list()
    is_minor = group_dict[i].islower()
    progression = ms3.fifths2sd(bass_notes.to_list(), is_minor) + ['∅']
    if is_minor:
        full_grams_minor.append(progression)
    else:
        full_grams_major.append(progression)
```

```{code-cell} ipython3
plot_transition_heatmaps(full_grams_major, full_grams_minor, top=20)
save_pdf_path = os.path.join(RESULTS_PATH, 'bass_degree_bigrams.pdf')
plt.savefig(save_pdf_path, dpi=400)
plt.show()
```

```{code-cell} ipython3
#font_dict = {'font': {'size': 20}}2
fig = plot_cum(
  all_chords.chord,
  font_size=35,
  markersize=10,
  **STD_LAYOUT)
save_figure_as(fig, 'chord_type_distribution_cumulative')
fig.show()
```

```{code-cell} ipython3
grouped_chords = groupers.ModeGrouper().process(labels)
grouped_chords.get_default_groupby()
```

```{code-cell} ipython3
value_count_df(grouped_chords.chord)
```

```{code-cell} ipython3
ugs_dict = {mode: value_count_df(chords).reset_index() for mode, chords in
            grouped_chords.groupby("mode").chord}
ugs_df = pd.concat(ugs_dict, axis=1)
ugs_df.columns = ['_'.join(map(str, col)) for col in ugs_df.columns]
ugs_df.index = (ugs_df.index + 1).rename('k')
ugs_df
```

## Key areas

```{code-cell} ipython3
from ms3 import roman_numeral2fifths, transform, resolve_all_relative_numerals, replace_boolean_mode_by_strings

keys_segmented = dc.LocalKeySlicer().process_data(D)
keys = keys_segmented.get_slice_info()
print(f"Overall number of key segments is {len(keys.index)}")
keys["localkey_fifths"] = transform(keys, roman_numeral2fifths, ['localkey', 'globalkey_is_minor'])
keys.head(5).style.apply(color_background, subset="localkey")
```

### Durational distribution of local keys

All durations given in quarter notes

```{code-cell} ipython3
key_durations = keys.groupby(['globalkey_is_minor', 'localkey']).duration_qb.sum().sort_values(ascending=False)
print(f"{len(key_durations)} keys overall including hierarchical such as 'III/v'.")
```

```{code-cell} ipython3
keys_resolved = resolve_all_relative_numerals(keys)
key_resolved_durations = keys_resolved.groupby(['globalkey_is_minor', 'localkey']).duration_qb.sum().sort_values(ascending=False)
print(f"{len(key_resolved_durations)} keys overall after resolving hierarchical ones.")
key_resolved_durations
```

#### Distribution of local keys for piece in major and in minor

`globalkey_mode=minor` => Piece is in Minor

```{code-cell} ipython3
pie_data = replace_boolean_mode_by_strings(key_resolved_durations.reset_index())
fig = px.pie(
  pie_data,
  title="Distribution of local keys for major vs. minor pieces",
  names='localkey',
  values='duration_qb',
  facet_col='globalkey_mode',
  labels=dict(globalkey_mode="Mode of global key")
)
fig.update_layout(**STD_LAYOUT)
fig.update_traces(
  textposition='inside',
  textinfo='percent+label',
)
fig.update_legends(
  orientation='h',
)
save_figure_as(fig, 'localkey_distributions_major_minor_pies', height=700, width=900)
fig.show()
```

#### Distribution of intervals between localkey tonic and global tonic

```{code-cell} ipython3
localkey_fifths_durations = keys.groupby(['localkey_fifths', 'localkey_is_minor']).duration_qb.sum()
bar_data = replace_boolean_mode_by_strings(localkey_fifths_durations.reset_index())
bar_data.localkey_fifths = bar_data.localkey_fifths.map(ms3.fifths2iv)
fig = px.bar(bar_data, x='localkey_fifths', y='duration_qb', color='localkey_mode', log_y=True, barmode='group',
             labels=dict(localkey_fifths='Roots of local keys as intervallic distance from the global tonic',
                   duration_qb='total duration in quarter notes',
                   localkey_mode='mode'
                  ),
             color_discrete_sequence=CORPUS_COLOR_SCALE,
             )
fig.update_layout(**STD_LAYOUT)
save_figure_as(fig, 'scale_degree_distributions_maj_min_absolute_bars')
fig.show()
```

### Ratio between major and minor key segments by aggregated durations
#### Overall

```{code-cell} ipython3
keys.duration_qb = pd.to_numeric(keys.duration_qb)
maj_min_ratio = keys.groupby("localkey_is_minor").duration_qb.sum().to_frame()
maj_min_ratio['fraction'] = (100.0 * maj_min_ratio.duration_qb / maj_min_ratio.duration_qb.sum()).round(1)
maj_min_ratio
```

#### By dataset

```{code-cell} ipython3
segment_duration_per_corpus = keys.groupby(["corpus", "localkey_is_minor"]).duration_qb.sum().round(2)
norm_segment_duration_per_corpus = 100 * segment_duration_per_corpus / segment_duration_per_corpus.groupby(level="corpus").sum()
maj_min_ratio_per_corpus = pd.concat([segment_duration_per_corpus,
                                      norm_segment_duration_per_corpus.rename('fraction').round(1).astype(str)+" %"],
                                     axis=1)
maj_min_ratio_per_corpus['corpus_name'] = maj_min_ratio_per_corpus.index.get_level_values('corpus').map(corpus_names)
maj_min_ratio_per_corpus['mode'] = maj_min_ratio_per_corpus.index.get_level_values('localkey_is_minor').map({False: 'major', True: 'minor'})
```

```{code-cell} ipython3
fig = px.bar(maj_min_ratio_per_corpus.reset_index(),
       x="corpus_name",
       y="duration_qb",
       color="mode",
       text='fraction',
       labels=dict(dataset='', duration_qb="duration in 𝅘𝅥", corpus_name='Key segments grouped by corpus'),
       category_orders=dict(dataset=chronological_order)
    )
fig.update_layout(**STD_LAYOUT)
save_figure_as(fig, 'major_minor_key_segments_corpuswise_bars')
fig.show()
```

### Tone profiles for all major and minor local keys

```{code-cell} ipython3
notes_by_keys = keys_segmented.get_facet("notes")
notes_by_keys
```

```{code-cell} ipython3
keys = keys[[col for col in keys.columns if col not in notes_by_keys]]
notes_joined_with_keys = notes_by_keys.join(keys, on=keys.index.names)
notes_by_keys_transposed = ms3.transpose_notes_to_localkey(notes_joined_with_keys)
mode_tpcs = notes_by_keys_transposed.reset_index(drop=True).groupby(['localkey_is_minor', 'tpc']).duration_qb.sum().reset_index(-1).sort_values('tpc').reset_index()
mode_tpcs['sd'] = ms3.fifths2sd(mode_tpcs.tpc)
mode_tpcs['duration_pct'] = mode_tpcs.groupby('localkey_is_minor', group_keys=False).duration_qb.apply(lambda S: S / S.sum())
mode_tpcs['mode'] = mode_tpcs.localkey_is_minor.map({False: 'major', True: 'minor'})
```

```{code-cell} ipython3
#mode_tpcs = mode_tpcs[mode_tpcs['duration_pct'] > 0.001]
#sd_order = ['b1', '1', '#1', 'b2', '2', '#2', 'b3', '3', 'b4', '4', '#4', '##4', 'b5', '5', '#5', 'b6','6', '#6', 'b7', '7']
legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99
)
fig = px.bar(mode_tpcs,
    x='tpc',
    y='duration_pct',
    title="Scale degree distribution over major and minor segments",
    color='mode',
    barmode='group',
    labels=dict(
        duration_pct='normalized duration',
        tpc="Notes transposed to the local key, as major-scale degrees",
        ),
    #log_y=True,
    #category_orders=dict(sd=sd_order)
    )
fig.update_layout(**STD_LAYOUT, legend=legend)
fig.update_xaxes(
    tickmode='array',
    tickvals=mode_tpcs.tpc,
    ticktext=mode_tpcs.sd
)
save_figure_as(fig, 'scale_degree_distributions_maj_min_normalized_bars', height=600)
fig.show()
```

## Harmony labels
### Unigrams
For computing unigram statistics, the tokens need to be grouped by their occurrence within a major or a minor key because this changes their meaning. To that aim, the annotated corpus needs to be sliced into contiguous localkey segments which are then grouped into a major (`is_minor=False`) and a minor group.

```{code-cell} ipython3
root_durations = all_chords[all_chords.root.between(-5,6)].groupby(['root', 'chord_type']).duration_qb.sum()
# sort by stacked bar length:
#root_durations = root_durations.sort_values(key=lambda S: S.index.get_level_values(0).map(S.groupby(level=0).sum()), ascending=False)
bar_data = root_durations.reset_index()
bar_data.root = bar_data.root.map(ms3.fifths2iv)
fig = px.bar(
  bar_data,
  x='root',
  y='duration_qb',
  color='chord_type',
  title="Distribution of chord types over chord roots",
  labels=dict(root="Chord root expressed as interval above the local (or secondary) tonic",
              duration_qb="duration in quarter notes",
              chord_type="chord type",
             ),

)
fig.update_layout(**STD_LAYOUT)
save_figure_as(fig, 'chord_type_distribution_over_scale_degrees_absolute_stacked_bars')
fig.show()
```

```{code-cell} ipython3
relative_roots = all_chords[['numeral', 'duration_qb', 'relativeroot', 'localkey_is_minor', 'chord_type']].copy()
relative_roots['relativeroot_resolved'] = transform(relative_roots, ms3.resolve_relative_keys, ['relativeroot', 'localkey_is_minor'])
has_rel = relative_roots.relativeroot_resolved.notna()
relative_roots.loc[has_rel, 'localkey_is_minor'] = relative_roots.loc[has_rel, 'relativeroot_resolved'].str.is_minor()
relative_roots['root'] = transform(relative_roots, roman_numeral2fifths, ['numeral', 'localkey_is_minor'])
chord_type_frequency = all_chords.chord_type.value_counts()
replace_rare = ms3.map_dict({t: 'other' for t in chord_type_frequency[chord_type_frequency < 500].index})
relative_roots['type_reduced'] = relative_roots.chord_type.map(replace_rare)
#is_special = relative_roots.chord_type.isin(('It', 'Ger', 'Fr'))
#relative_roots.loc[is_special, 'root'] = -4
```

```{code-cell} ipython3
root_durations = relative_roots.groupby(['root', 'type_reduced']).duration_qb.sum().sort_values(ascending=False)
bar_data = root_durations.reset_index()
bar_data.root = bar_data.root.map(ms3.fifths2iv)
root_order = bar_data.groupby('root').duration_qb.sum().sort_values(ascending=False).index.to_list()
fig = px.bar(bar_data, x='root', y='duration_qb', color='type_reduced', barmode='group', log_y=True,
             color_discrete_map=TYPE_COLORS,
             category_orders=dict(root=root_order,
                                  type_reduced=relative_roots.type_reduced.value_counts().index.to_list(),
                                 ),
            labels=dict(root="intervallic difference between chord root to the local or secondary tonic",
                        duration_qb="duration in quarter notes",
                        type_reduced="chord type",
                       ),
             width=1000,
             height=400,
            )
fig.update_layout(**STD_LAYOUT,
                  legend=dict(
                      orientation='h',
                      xanchor="right",
                      x=1,
                      y=1,
                  )
                 )
save_figure_as(fig, 'chord_type_distribution_over_scale_degrees_absolute_grouped_bars')
fig.show()
```

```{code-cell} ipython3
print(f"Reduced to {len(set(bar_data.iloc[:,:2].itertuples(index=False, name=None)))} types. Paper cites the sum of types in major and types in minor (see below), treating them as distinct.")
```

```{code-cell} ipython3
dim_or_aug = bar_data[bar_data.root.str.startswith("a") | bar_data.root.str.startswith("d")].duration_qb.sum()
complete = bar_data.duration_qb.sum()
print(f"On diminished or augmented scale degrees: {dim_or_aug} / {complete} = {dim_or_aug / complete}")
```

```{code-cell} ipython3
mode_slices = dc.ModeGrouper().process_data(keys_segmented)
```

### Whole dataset

```{code-cell} ipython3
mode_slices.get_slice_info()
```

```{code-cell} ipython3
unigrams = dc.ChordSymbolUnigrams(once_per_group=True).process_data(mode_slices)
```

```{code-cell} ipython3
unigrams.group2pandas = "group_of_series2series"
```

```{code-cell} ipython3
unigrams.get(as_pandas=True)
```

```{code-cell} ipython3
k = 20
modes = {True: 'MINOR', False: 'MAJOR'}
for (is_minor,), ugs in unigrams.iter():
    print(f"TOP {k} {modes[is_minor]} UNIGRAMS\n{ugs.shape[0]} types, {ugs.sum()} tokens")
    print(ugs.head(k).to_string())
```

```{code-cell} ipython3
ugs_dict = {modes[is_minor].lower(): (ugs/ugs.sum() * 100).round(2).rename('%').reset_index() for (is_minor,), ugs in unigrams.iter()}
ugs_df = pd.concat(ugs_dict, axis=1)
ugs_df.columns = ['_'.join(map(str, col)) for col in ugs_df.columns]
ugs_df.index = (ugs_df.index + 1).rename('k')
print(ugs_df.iloc[:50].to_markdown())
```

### Per corpus

```{code-cell} ipython3
corpus_wise_unigrams = dc.Pipeline([dc.CorpusGrouper(), dc.ChordSymbolUnigrams(once_per_group=True)]).process_data(mode_slices)
```

```{code-cell} ipython3
corpus_wise_unigrams.get()
```

```{code-cell} ipython3
for (is_minor, corpus_name), ugs in corpus_wise_unigrams.iter():
    print(f"{corpus_name} {modes[is_minor]} unigrams ({ugs.shape[0]} types, {ugs.sum()} tokens)")
    print(ugs.head(5).to_string())
```

```{code-cell} ipython3
types_shared_between_corpora = {}
for (is_minor, corpus_name), ugs in corpus_wise_unigrams.iter():
    if is_minor in types_shared_between_corpora:
        types_shared_between_corpora[is_minor] = types_shared_between_corpora[is_minor].intersection(ugs.index)
    else:
        types_shared_between_corpora[is_minor] = set(ugs.index)
types_shared_between_corpora = {k: sorted(v, key=lambda x: unigrams.get()[(k, x)], reverse=True) for k, v in types_shared_between_corpora.items()}
n_types = {k: len(v) for k, v in types_shared_between_corpora.items()}
print(f"Chords which occur in all corpora, sorted by descending global frequency:\n{types_shared_between_corpora}\nCounts: {n_types}")
```

### Per piece

```{code-cell} ipython3
piece_wise_unigrams = dc.Pipeline([dc.PieceGrouper(), dc.ChordSymbolUnigrams(once_per_group=True)]).process_data(mode_slices)
```

```{code-cell} ipython3
piece_wise_unigrams.get()
```

```{code-cell} ipython3
types_shared_between_pieces = {}
for (is_minor, corpus_name), ugs in piece_wise_unigrams.iter():
    if is_minor in types_shared_between_pieces:
        types_shared_between_pieces[is_minor] = types_shared_between_pieces[is_minor].intersection(ugs.index)
    else:
        types_shared_between_pieces[is_minor] = set(ugs.index)
print(types_shared_between_pieces)
```

## Bigrams

+++

### Whole dataset

```{code-cell} ipython3
bigrams = dc.ChordSymbolBigrams(once_per_group=True).process_data(mode_slices)
```

```{code-cell} ipython3
bigrams.get()
```

```{code-cell} ipython3
modes = {True: 'MINOR', False: 'MAJOR'}
for (is_minor,), ugs in bigrams.iter():
    print(f"{modes[is_minor]} BIGRAMS\n{ugs.shape[0]} transition types, {ugs.sum()} tokens")
    print(ugs.head(20).to_string())
```

### Per corpus

```{code-cell} ipython3
corpus_wise_bigrams = dc.Pipeline([dc.CorpusGrouper(), dc.ChordSymbolBigrams(once_per_group=True)]).process_data(mode_slices)
```

```{code-cell} ipython3
corpus_wise_bigrams.get()
```

```{code-cell} ipython3
for (is_minor, corpus_name), ugs in corpus_wise_bigrams.iter():
    print(f"{corpus_name} {modes[is_minor]} bigrams ({ugs.shape[0]} transition types, {ugs.sum()} tokens)")
    print(ugs.head(5).to_string())
```

```{code-cell} ipython3
normalized_corpus_unigrams = {group: (100 * ugs / ugs.sum()).round(1).rename("frequency") for group, ugs in corpus_wise_unigrams.iter()}
```

```{code-cell} ipython3
transitions_from_shared_types = {
    False: {},
    True: {}
}
for (is_minor, corpus_name), bgs in corpus_wise_bigrams.iter():
    transitions_normalized_per_from = bgs.groupby(level="from", group_keys=False).apply(lambda S: (100 * S / S.sum()).round(1))
    most_frequent_transition_per_from = transitions_normalized_per_from.rename('fraction').reset_index(level=1).groupby(level=0).nth(0)
    most_frequent_transition_per_shared = most_frequent_transition_per_from.loc[types_shared_between_corpora[is_minor]]
    unigram_frequency_of_shared = normalized_corpus_unigrams[(is_minor, corpus_name)].loc[types_shared_between_corpora[is_minor]]
    combined = pd.concat([unigram_frequency_of_shared, most_frequent_transition_per_shared], axis=1)
    transitions_from_shared_types[is_minor][corpus_name] = combined
```

```{code-cell} ipython3
pd.concat(transitions_from_shared_types[False].values(), keys=transitions_from_shared_types[False].keys(), axis=1)
```

```{code-cell} ipython3
pd.concat(transitions_from_shared_types[True].values(), keys=transitions_from_shared_types[False].keys(), axis=1)
```

### Per piece

```{code-cell} ipython3
piece_wise_bigrams = dc.Pipeline([dc.PieceGrouper(), dc.ChordSymbolBigrams(once_per_group=True)]).process_data(mode_slices)
```

```{code-cell} ipython3
piece_wise_bigrams.get()
```
