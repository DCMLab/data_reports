---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
---

+++ {"tags": []}

# General overview

```{code-cell} ipython3
import os
from collections import defaultdict, Counter
from fractions import Fraction

from git import Repo
import dimcat as dc
import ms3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import CADENCE_COLORS, CORPUS_COLOR_SCALE, STD_LAYOUT, TYPE_COLORS, color_background, value_count_df, get_repo_name, resolve_dir
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
dataset.data.parse_tsv(choose='auto')
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

```{code-cell} ipython3
print(f"Composition dates range from {all_metadata.composed_start.min()} {all_metadata.composed_start.idxmin()} "
      f"to {all_metadata.composed_end.max()} {all_metadata.composed_end.idxmax()}.")
```

```{code-cell} ipython3
:tags: []

annotated = dc.IsAnnotatedFilter().process_data(dataset)
print(f"Before: {len(dataset.indices[()])} IDs, after filtering: {len(annotated.indices[()])}")
```

**Choose here if you want to see stats for all or only for annotated scores.**

```{code-cell} ipython3
#selected = dataset
selected = annotated
```

## Measures

```{code-cell} ipython3
all_measures = selected.get_facet('measures')
print(f"{len(all_measures.index)} measures over {len(all_measures.groupby(level=[0,1]))} files.")
all_measures.head()
```

```{code-cell} ipython3
print("Distribution of time signatures per XML measure (MC):")
all_measures.timesig.value_counts(dropna=False)
```

## Notes

```{code-cell} ipython3
all_notes = selected.get_facet('notes')
print(f"{len(all_notes.index)} notes over {len(all_notes.groupby(level=[0,1]))} files.")
all_notes.head()
```

### Notes and staves

```{code-cell} ipython3
print("Distribution of notes over staves:")
all_notes.staff.value_counts()
```

```{code-cell} ipython3
print("Distribution of notes over staves for all pieces with more than two staves\n")
for group, df in all_notes.groupby(level=[0,1]):
    if (df.staff > 2).any():
        print(group)
        print(df.staff.value_counts().to_dict())
```

```{code-cell} ipython3
all_notes[all_notes.staff > 2].groupby(level=[0,1]).staff.value_counts()
```

## Harmony labels

All symbols, independent of the local key (the mode of which changes their semantics).

```{code-cell} ipython3
all_annotations = annotated.get_facet('expanded')
all_annotations.head()
```

```{code-cell} ipython3
no_chord = all_annotations.root.isna()
print(f"Concatenated annotation tables contains {all_annotations.shape[0]} rows. {no_chord.sum()} of them are not chords. Their values are:")
all_annotations.label[no_chord].value_counts(dropna=False).to_dict()
```

```{code-cell} ipython3
all_chords = all_annotations[~no_chord].copy()
print(f"Corpus contains {all_chords.shape[0]} tokens and {len(all_chords.chord.unique())} types over {len(all_chords.groupby(level=[0,1]))} documents.")
```

```{code-cell} ipython3
#from ms3 import write_tsv
#write_tsv(all_annotations[all_annotations.pedalend.notna()], './issues/pedalpoints.tsv', pre_process=False)
```

## Corpus summary

```{code-cell} ipython3
summary = all_metadata
if selected == annotated:
    summary = summary[summary.label_count > 0].copy()
summary.length_qb = all_measures.groupby(level=[0,1]).act_dur.sum() * 4.0
summary = pd.concat([summary,
                     all_notes.groupby(level=[0,1]).size().rename('notes'),
                    ], axis=1)
summary.groupby(level=0).describe().dropna(axis=1, how='all')
```

```{code-cell} ipython3
mean_composition_years = summary.groupby(level=0).composed_end.mean().astype(int).sort_values()
chronological_order = mean_composition_years.index.to_list()
corpus_colors = dict(zip(chronological_order, CORPUS_COLOR_SCALE))
bar_data = pd.concat([mean_composition_years.rename('year'), 
                      summary.groupby(level='corpus').size().rename('pieces')],
                     axis=1
                    ).reset_index()
fig = px.bar(bar_data, x='year', y='pieces', color='corpus',
             color_discrete_map=corpus_colors,
            height=350, width=800,
            )
fig.update_traces(width=5)
fig.update_layout(**STD_LAYOUT)
fig.update_yaxes(gridcolor='lightgrey')
fig.update_traces(width=5)
```

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
chronological_corpus_names = [corpus_names[corp] for corp in chronological_order]
corpus_name_colors = {corpus_names[corp]: color for corp, color in corpus_colors.items()}
all_annotations['corpus_name'] = all_annotations.index.get_level_values(0).map(corpus_names)
all_chords['corpus_name'] = all_chords.index.get_level_values(0).map(corpus_names)
```

```{code-cell} ipython3
bar_data = summary.reset_index().groupby(['composed_end', 'corpus']).size().rename('counts').reset_index()
px.bar(bar_data, x='composed_end', y='counts', color='corpus', color_discrete_map=corpus_colors)
```

```{code-cell} ipython3
hist_data = summary.reset_index()
hist_data.corpus = hist_data.corpus.map(corpus_names)
hist_data.head()
```

```{code-cell} ipython3
fig = px.histogram(hist_data, x='composed_end', color='corpus',
                   labels=dict(composed_end='decade',
                               count='pieces',
                              ),
                   color_discrete_map=corpus_name_colors,
                   width=1000, height=400,
                  )
fig.update_traces(xbins=dict(
    size=10
))
fig.update_layout(**STD_LAYOUT)
fig.update_yaxes(gridcolor='lightgrey')
fig.show()
```

```{code-cell} ipython3
summary.columns
```

```{code-cell} ipython3
corpus_metadata = summary.groupby(level=0)
n_pieces = corpus_metadata.size().rename('pieces')
absolute_numbers = dict(
    measures = corpus_metadata.last_mn.sum(),
    length = corpus_metadata.length_qb.sum(),
    notes = corpus_metadata.notes.sum(),
    labels = corpus_metadata.label_count.sum(),
)
absolute = pd.DataFrame.from_dict(absolute_numbers)
relative = absolute.div(n_pieces, axis=0)
complete_summary = pd.concat([pd.concat([n_pieces, absolute], axis=1), relative, absolute.iloc[:,2:].div(absolute.measures, axis=0)], axis=1, keys=['absolute', 'per piece', 'per measure'])
complete_summary = complete_summary.apply(pd.to_numeric).round(2)
complete_summary.index = complete_summary.index.map(corpus_names)
complete_summary
```

```{code-cell} ipython3
sum_row = pd.DataFrame(complete_summary.sum(), columns=['sum']).T
sum_row.iloc[:,5:] = ''
summary_with_sum = pd.concat([complete_summary, sum_row])
summary_with_sum.loc[:, [('absolute', 'notes'), ('absolute', 'labels')]] = summary_with_sum[[('absolute', 'notes'), ('absolute', 'labels')]].astype(int)
summary_with_sum
```

```{code-cell} ipython3
summary[summary.ambitus.isna()]
```

```{code-cell} ipython3
summary.ambitus.str.extract(r"^(\d+)-(\d+)")
```

```{code-cell} ipython3
ambitus = summary.ambitus.str.extract(r"^(\d+)-(\d+)").astype(int)
ambitus.columns = ['low', 'high']
ambitus['range'] = ambitus.high - ambitus.low
ambitus.head()
```

```{code-cell} ipython3
ambitus.groupby(level=0).high.max()
```

```{code-cell} ipython3
ambitus.groupby(level=0).low.min()
```

```{code-cell} ipython3
ambitus.groupby(level=0).range.max()
```

```{code-cell} ipython3
ambitus.groupby(level=0).high.max() - ambitus.groupby(level=0).low.min()
```

## Phrases

```{code-cell} ipython3
:tags: []

phrase_segmented = dc.PhraseSlicer().process_data(selected)
phrases = phrase_segmented.get_slice_info()
print(f"Overall number of phrases is {len(phrases.index)}")
phrases.head(20)
```

```{code-cell} ipython3
phrase_segments = phrase_segmented.get_facet('expanded')
phrase_segments
```

```{code-cell} ipython3
phrases[phrases.duration_qb > 50]
```

```{code-cell} ipython3
phrase2timesigs = phrase_segments.groupby(level=[0,1,2]).timesig.unique()
n_timesignatures_per_phrase = phrase2timesigs.map(len)
uniform_timesigs = phrase2timesigs[n_timesignatures_per_phrase == 1].map(lambda l: l[0])
more_than_one = n_timesignatures_per_phrase > 1
print(f"Filtered out the {more_than_one.sum()} phrases incorporating more than one time signature.")
n_timesigs = n_timesignatures_per_phrase.value_counts()
display(n_timesigs.reset_index().rename(columns=dict(index='#time signatures', timesig='#phrases')))
uniform_timesig_phrases = phrases.loc[uniform_timesigs.index]
timesig_in_quarterbeats = uniform_timesigs.map(Fraction) * 4
exact_measure_lengths = uniform_timesig_phrases.duration_qb / timesig_in_quarterbeats
uniform_timesigs = pd.concat([exact_measure_lengths.rename('duration_measures'), uniform_timesig_phrases], axis=1)
fig = px.histogram(uniform_timesigs, x='duration_measures', log_y=True,
                   labels=dict(duration_measures='phrase length bin in number of measures'),
                   color_discrete_sequence=CORPUS_COLOR_SCALE,
                   height=400,
                   width = 1000,
                  )
fig.update_traces(xbins=dict( # bins used for histogram
        #start=0.0,
        #end=100.0,
        size=1
    ))
fig.update_layout(**STD_LAYOUT)
fig.update_xaxes(dtick=4, gridcolor='lightgrey')
fig.update_yaxes(gridcolor='lightgrey')
fig.show()
```

```{code-cell} ipython3
uniform_timesigs[uniform_timesigs.duration_measures > 80]
```

## Keys

```{code-cell} ipython3
from ms3 import roman_numeral2fifths, transform, resolve_all_relative_numerals, replace_boolean_mode_by_strings
keys_segmented = dc.LocalKeySlicer().process_data(selected)
keys = keys_segmented.get_slice_info()
print(f"Overall number of key segments is {len(keys.index)}")
keys["localkey_fifths"] = transform(keys, roman_numeral2fifths, ['localkey', 'globalkey_is_minor'])
keys.head(20)
```

```{code-cell} ipython3
keys.duration_qb.sum()
```

```{code-cell} ipython3
phrases.duration_qb.sum()
```

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

```{code-cell} ipython3
pie_data = replace_boolean_mode_by_strings(key_resolved_durations.reset_index())
px.pie(pie_data, names='localkey', values='duration_qb', facet_col='globalkey_mode', height=700)
```

```{code-cell} ipython3
localkey_fifths_durations = keys.groupby(['localkey_fifths', 'localkey_is_minor']).duration_qb.sum()
# sort by stacked bar length:
localkey_fifths_durations = localkey_fifths_durations.sort_values(key=lambda S: S.index.get_level_values(0).map(S.groupby(level=0).sum()), ascending=False)
bar_data = replace_boolean_mode_by_strings(localkey_fifths_durations.reset_index())
bar_data.localkey_fifths = bar_data.localkey_fifths.map(ms3.fifths2iv)
fig = px.bar(bar_data, x='localkey_fifths', y='duration_qb', color='localkey_mode', log_y=True, barmode='group',
             labels=dict(localkey_fifths='Roots of local keys as intervallic distance from the global tonic', 
                   duration_qb='total duration in quarter notes',
                   localkey_mode='mode'
                  ),
             color_discrete_sequence=CORPUS_COLOR_SCALE,
             width=1000)
fig.update_layout(**STD_LAYOUT)
fig.update_yaxes(gridcolor='lightgrey')
fig.show()
```

```{code-cell} ipython3
localkey_fifths_durations = keys.groupby(['localkey_fifths', 'localkey_is_minor']).duration_qb.sum()
# sort by stacked bar length:
bar_data = replace_boolean_mode_by_strings(localkey_fifths_durations.reset_index())
bar_data.localkey_fifths = bar_data.localkey_fifths.map(ms3.fifths2iv)
fig = px.bar(bar_data, x='localkey_fifths', y='duration_qb', color='localkey_mode', log_y=True, barmode='group',
             labels=dict(localkey_fifths='Roots of local keys as intervallic distance from the global tonic', 
                   duration_qb='total duration in quarter notes',
                   localkey_mode='mode'
                  ),
             color_discrete_sequence=CORPUS_COLOR_SCALE,
             width=1000)
fig.update_layout(**STD_LAYOUT)
fig.update_yaxes(gridcolor='lightgrey')
fig.show()
```

```{code-cell} ipython3
localkey_fifths_durations_stacked = localkey_fifths_durations.groupby(level=0).sum().sort_values()
pd.concat([localkey_fifths_durations_stacked, localkey_fifths_durations_stacked.rename('fraction') / localkey_fifths_durations_stacked.sum()], axis=1)
```

```{code-cell} ipython3
keys[keys.localkey_fifths == -9]
```

```{code-cell} ipython3
keys[keys.localkey_fifths == 10]
```

## Cadences

```{code-cell} ipython3
all_annotations.cadence.value_counts()
```

```{code-cell} ipython3
all_annotations.groupby("corpus_name").cadence.value_counts()
```

```{code-cell} ipython3
cadence_count_per_corpus = all_annotations.groupby("corpus_name").cadence.value_counts().sort_values(ascending=False)
cadence_count_per_corpus.groupby(level=0).sum()
```

```{code-cell} ipython3
cadence_fraction_per_corpus = cadence_count_per_corpus / cadence_count_per_corpus.groupby(level=0).sum()
fig = px.bar(cadence_fraction_per_corpus.rename('count').reset_index(), x='corpus_name', y='count', color='cadence',
             labels=dict(count='fraction', corpus=''), 
             height=400, width=900,
       category_orders=dict(corpus_name=chronological_corpus_names))
      #color_discrete_map=cadence_colors, 

fig.update_layout(**STD_LAYOUT)
fig.update_yaxes(gridcolor='lightgrey')
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
px.bar(bar_data, x='root', y='duration_qb', color='chord_type')
```

```{code-cell} ipython3
relative_roots = all_chords[['numeral', 'duration_qb', 'relativeroot', 'localkey_is_minor', 'chord_type']].copy()
relative_roots['relativeroot_resolved'] = transform(relative_roots, ms3.resolve_relative_keys, ['relativeroot', 'localkey_is_minor'])
has_rel = relative_roots.relativeroot_resolved.notna()
relative_roots.loc[has_rel, 'localkey_is_minor'] = relative_roots.loc[has_rel, 'relativeroot_resolved'].str.islower()
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
fig.update_yaxes(gridcolor='lightgrey')
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
modes = {True: 'MINOR', False: 'MAJOR'}
for (is_minor,), ugs in unigrams.iter():
    print(f"{modes[is_minor]} UNIGRAMS\n{ugs.shape[0]} types, {ugs.sum()} tokens")
    print(ugs.head(20).to_string())
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
    transitions_normalized_per_from = bgs.groupby(level="from").apply(lambda S: (100 * S / S.sum()).round(1))
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
