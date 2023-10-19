---
jupytext:
  formats: md:myst,ipynb
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

# Cadences

```{code-cell}
---
mystnb:
  code_prompt_hide: Hide imports
  code_prompt_show: Show imports
tags: [hide-cell]
---
import os
from collections import defaultdict, Counter

from git import Repo
import dimcat as dc
import ms3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import STD_LAYOUT, CADENCE_COLORS, color_background, value_count_df, get_repo_name, print_heading, \
  resolve_dir, get_corpus_display_name
```

```{code-cell}
from utils import OUTPUT_FOLDER, write_image
RESULTS_PATH = os.path.abspath(os.path.join(OUTPUT_FOLDER, "cadences"))
os.makedirs(RESULTS_PATH, exist_ok=True)
def save_figure_as(fig, filename, directory=RESULTS_PATH, **kwargs):
    write_image(fig, filename, directory, **kwargs)
```

```{code-cell}
:tags: [hide-input]

# CORPUS_PATH = os.path.abspath(os.path.join('..', '..')) # for running the notebook in the homepage deployment workflow
CORPUS_PATH = "~/distant_listening_corpus/couperin_concerts"                # for running the notebook locally
print_heading("Notebook settings")
print(f"CORPUS_PATH: {CORPUS_PATH!r}")
CORPUS_PATH = resolve_dir(CORPUS_PATH)
```

```{code-cell}
:tags: [hide-input]

repo = Repo(CORPUS_PATH)
print_heading("Data and software versions")
print(f"Data repo '{get_repo_name(repo)}' @ {repo.commit().hexsha[:7]}")
print(f"dimcat version {dc.__version__}")
print(f"ms3 version {ms3.__version__}")
```

```{code-cell}
:tags: [remove-output]

dataset = dc.Dataset()
dataset.load(directory=CORPUS_PATH, parse_tsv=False)
```

```{code-cell}
:tags: [remove-input]

annotated_view = dataset.data.get_view('annotated')
annotated_view.include('facets', 'expanded')
annotated_view.pieces_with_incomplete_facets = False
dataset.data.set_view(annotated_view)
dataset.data.parse_tsv(choose='auto')
dataset.get_indices()
dataset.data
```

```{code-cell}
:tags: [remove-input]

print(f"N = {dataset.data.count_pieces()} annotated pieces, {dataset.data.count_parsed_tsvs()} parsed dataframes.")
```

## Metadata

```{code-cell}
all_metadata = dataset.data.metadata()
assert len(all_metadata) > 0, "No pieces selected for analysis."
print(f"Concatenated 'metadata.tsv' files cover {len(all_metadata)} of the {dataset.data.count_pieces()} scores.")
all_metadata.reset_index(level=1).groupby(level=0).nth(0).iloc[:,:20]
```

## All annotation labels from the selected pieces

```{code-cell}
all_labels = dataset.data.get_facet('expanded')

print(f"{len(all_labels.index)} hand-annotated harmony labels:")
all_labels.iloc[:20].style.apply(color_background, subset="chord")
```

### Filtering out pieces without cadence annotations

```{code-cell}
hascadence = dc.HasCadenceAnnotationsFilter().process_data(dataset)
assert () in hascadence.indices and len(hascadence.indices[()]) > 0, "No cadences found."
print(f"Before: {len(dataset.indices[()])} pieces; after removing those without cadence labels: {len(hascadence.indices[()])}")
```

### Show corpora containing pieces with cadence annotations

```{code-cell}
grouped_by_corpus = dc.CorpusGrouper().process_data(hascadence)
corpora = {group[0]: f"{len(ixs)} pieces" for group, ixs in  grouped_by_corpus.indices.items()}
print(f"{len(corpora)} corpora with {sum(map(len, grouped_by_corpus.indices.values()))} pieces containing cadence annotations:")
corpora
```

### All annotation labels from the selected pieces

```{code-cell}
all_labels = hascadence.get_facet('expanded')

print(f"{len(all_labels)} hand-annotated harmony labels:")
all_labels.iloc[:10, 13:].style.apply(color_background, subset="chord")
```

### Metadata

```{code-cell}
dataset_metadata = hascadence.data.metadata()
hascadence_metadata = dataset_metadata.loc[hascadence.indices[()]]
hascadence_metadata.index.rename('dataset', level=0, inplace=True)
hascadence_metadata.head()
```

```{code-cell}
mean_composition_years = hascadence_metadata.groupby(level=0).composed_end.mean().astype(int).sort_values()
chronological_order = mean_composition_years.index.to_list()
corpus_names = {corp: get_corpus_display_name(corp) for corp in chronological_order}
chronological_corpus_names = list(corpus_names.values())
bar_data = pd.concat([mean_composition_years.rename('year'),
                      hascadence_metadata.groupby(level='dataset').size().rename('pieces')],
                     axis=1
                    ).reset_index()
fig = px.bar(bar_data, x='year', y='pieces', color='dataset', title='Pieces contained in the dataset')
fig.update_traces(width=5)
```

## Overall

* **PAC**: Perfect Authentic Cadence
* **IAC**: Imperfect Authentic Cadence
* **HC**: Half Cadence
* **DC**: Deceptive Cadence
* **EC**: Evaded Cadence
* **PC**: Plagal Cadence

```{code-cell}
print(f"{all_labels.cadence.notna().sum()} cadence labels.")
value_count_df(all_labels.cadence)
```

```{code-cell}
fig = px.pie(
    all_labels[all_labels.cadence.notna()],
    title="Distribution of cadence types over the DLC",
    names="cadence",
    color="cadence",
    color_discrete_map=CADENCE_COLORS
)
fig.update_layout(**STD_LAYOUT,)
fig.update_traces(
  textposition='auto',
  textinfo='percent+label',
textfont_size=30
)
save_figure_as(fig, 'all_cadences_pie', width=1000, height=1000)
fig.show()
```

## Per dataset

```{code-cell}
cadence_count_per_dataset = all_labels.groupby("corpus").cadence.value_counts()
cadence_fraction_per_dataset = cadence_count_per_dataset / cadence_count_per_dataset.groupby(level=0).sum()
cadence_fraction_per_dataset = cadence_fraction_per_dataset.rename('fraction').reset_index()
cadence_fraction_per_dataset["corpus_name"] = cadence_fraction_per_dataset.corpus.map(corpus_names)
fig = px.bar(
    cadence_fraction_per_dataset,
    x='corpus_name',
    y='fraction',
    title="Distribution of cadence types per corpus",
    color='cadence',
    color_discrete_map=CADENCE_COLORS,
    labels=dict(corpus_name="", fraction="Fraction of all cadences"),
    category_orders=dict(corpus_name=chronological_corpus_names)
)
fig.update_layout(**STD_LAYOUT)
save_figure_as(fig, 'all_cadences_corpuswise_stacked_bars', height=1000)
fig.show()
```

```{code-cell}
fig = px.pie(cadence_count_per_dataset.rename('count').reset_index(), names='cadence', color='cadence', values='count',
       facet_col='corpus', facet_col_wrap=4, height=2000, color_discrete_map=CADENCE_COLORS)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_layout(**STD_LAYOUT)
save_figure_as(fig, 'all_cadences_corpuswise_pies')
fig.show
```

## Per phrase
### Number of cadences per phrase

```{code-cell}
segmented = dc.PhraseSlicer().process_data(grouped_by_corpus)
phrases = segmented.get_slice_info()
phrase_segments = segmented.get_facet("expanded")
phrase_gpb = phrase_segments.groupby(level=[0,1,2])
local_keys_per_phrase = phrase_gpb.localkey.unique().map(tuple)
n_local_keys_per_phrase = local_keys_per_phrase.map(len)
phrases_with_keys = pd.concat([n_local_keys_per_phrase.rename('n_local_keys'),
                               local_keys_per_phrase.rename('local_keys'),
                               phrases], axis=1)
phrases_with_cadences = pd.concat([
    phrase_gpb.cadence.nunique().rename('n_cadences'),
    phrase_gpb.cadence.unique().rename('cadences').map(lambda l: tuple(e for e in l if not pd.isnull(e))),
    phrases_with_keys
], axis=1)
value_count_df(phrases_with_cadences.n_cadences, counts_column="#phrases")
```

```{code-cell}
n_cad = phrases_with_cadences.groupby(level='corpus').n_cadences.value_counts().rename('counts').reset_index().sort_values('n_cadences')
n_cad.n_cadences = n_cad.n_cadences.astype(str)
fig = px.bar(n_cad, x='corpus', y='counts', color='n_cadences', height=800, barmode='group',
             labels=dict(n_cadences="#cadences in a phrase"),
             category_orders=dict(dataset=chronological_order)
      )
save_figure_as(fig, 'n_cadences_per_phrase_corpuswise_absolute_grouped_bars')
fig.show()
```

### Combinations of cadence types for phrases with more than one cadence

```{code-cell}
value_count_df(phrases_with_cadences[phrases_with_cadences.n_cadences > 1].cadences)
```

### Positioning of cadences within phrases

```{code-cell}
df_rows = []
y_position = 0
for ix in phrases_with_cadences[phrases_with_cadences.n_cadences > 0].sort_values('duration_qb').index:
    df = phrase_segments.loc[ix]
    description = str(ix)
    if df.cadence.notna().any():
        interval = ix[2]
        df_rows.append((y_position, interval.length, "end of phrase", description))
        start_pos = interval.left
        cadences = df.loc[df.cadence.notna(), ['quarterbeats', 'cadence']]
        cadences.quarterbeats -= start_pos
        for cadence_x, cadence_type in cadences.itertuples(index=False, name=None):
            df_rows.append((y_position, cadence_x, cadence_type, description))
        y_position += 1
    #else:
    #    df_rows.append((y_position, pd.NA, pd.NA, description))

data = pd.DataFrame(df_rows, columns=["phrase_ix", "x", "marker", "description"])
```

```{code-cell}
fig = px.scatter(data[data.x.notna()], x='x', y="phrase_ix", color="marker", hover_name="description", height=3000,
                labels=dict(marker='legend'), color_discrete_map=CADENCE_COLORS)
fig.update_traces(marker_size=5)
fig.update_yaxes(autorange="reversed")
save_figure_as(fig, 'cadence_positions_within_all_phrases')
fig.show()
```

## Cadence ultima

```{code-cell}
phrase_segments = segmented.get_facet("expanded")
cadence_selector = phrase_segments.cadence.notna()
missing_chord_selector = phrase_segments.chord.isna()
cadence_with_missing_chord_selector = cadence_selector & missing_chord_selector
missing = phrase_segments[cadence_with_missing_chord_selector]
expanded = ms3.expand_dcml.expand_labels(phrase_segments[cadence_with_missing_chord_selector], propagate=False, chord_tones=True, skip_checks=True)
phrase_segments.loc[cadence_with_missing_chord_selector] = expanded
print(f"Ultima harmony missing for {(phrase_segments.cadence.notna() & phrase_segments.bass_note.isna()).sum()} cadence labels.")
```

### Ultimae as Roman numeral

```{code-cell}
def highlight(row, color="#ffffb3"):
    if row.counts < 10:
        return [None, None, None, None]
    else:
        return ["background-color: {color};"] * 4

cadence_counts = all_labels.cadence.value_counts()
ultima_root = phrase_segments.groupby(['localkey_is_minor', 'cadence']).numeral.value_counts().rename('counts').to_frame().reset_index()
ultima_root.localkey_is_minor = ultima_root.localkey_is_minor.map({False: 'in major', True: 'in minor'})
#ultima_root.style.apply(highlight, axis=1)
```

```{code-cell}
fig = px.pie(ultima_root, names='numeral', values='counts',
             facet_row='cadence', facet_col='localkey_is_minor',
             height=1500,
             category_orders={'cadence': cadence_counts.index},
            )
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(**STD_LAYOUT)
save_figure_as(fig, 'ultima_root_distributions_over_cadence_types_maj_min_pies')
fig.show()
```

```{code-cell}
#phrase_segments.groupby(level=[0,1,2], group_keys=False).apply(lambda df: df if ((df.cadence == 'PAC') & (df.numeral == 'V')).any() else None)
```

### Ultimae bass note as scale degree

```{code-cell}
ultima_bass = phrase_segments.groupby(['localkey_is_minor','cadence']).bass_note.value_counts().rename('counts').reset_index()
ultima_bass.bass_note = ms3.transform(ultima_bass, ms3.fifths2sd, dict(fifths='bass_note', minor='localkey_is_minor'))
ultima_bass.localkey_is_minor = ultima_bass.localkey_is_minor.map({False: 'in major', True: 'in minor'})
#ultima_bass.style.apply(highlight, axis=1)
```

```{code-cell}
fig = px.pie(ultima_bass, names='bass_note', values='counts',
             facet_row='cadence', facet_col='localkey_is_minor',
             height=1500,
             category_orders={'cadence': cadence_counts.index},
            )
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(**STD_LAYOUT)
save_figure_as(fig, 'ultima_degree_distributions_over_cadence_types_maj_min_pies')
fig.show()
```

## Chord progressions

+++

### PACs with ultima I/i

```{code-cell}
def remove_immediate_duplicates(l):
    return tuple(a for a, b in zip(l, (None, ) + l) if a != b)

def get_progressions(selected='PAC', last_row={}, feature='chord', dataset=None, as_series=True, remove_duplicates=False):
    """Uses the nonlocal variable phrase_segments."""
    last_row = {k: v if isinstance(v, tuple) else (v,) for k, v in last_row.items()}
    progressions = []

    for (corp, fname, *_), df in phrase_segments[phrase_segments[feature].notna()].groupby(level=[0,1,2]):
        if dataset is not None and dataset not in corp:
            continue
        if (df.cadence == selected).fillna(False).any():
            # remove chords after the last cadence label
            df = df[df.cadence.bfill().notna()]
            # group segments leading up to a cadence label
            cadence_groups = df.cadence.notna().shift().fillna(False).cumsum()
            for i, cadence in df.groupby(cadence_groups):
                last_r = cadence.iloc[-1]
                typ = last_r.cadence
                if typ != selected:
                    continue
                if any(last_r[feat] not in values for feat, values in last_row.items()):
                    continue
                if remove_duplicates:
                    progressions.append(remove_immediate_duplicates(cadence[feature].to_list()))
                else:
                    progressions.append(tuple(cadence[feature]))
    if as_series:
        return pd.Series(progressions, dtype='object')
    return progressions
```

```{code-cell}
chord_progressions = get_progressions('PAC', dict(numeral=('I', 'i')), 'chord')
print(f"Progressions for {len(chord_progressions)} cadences:")
value_count_df(chord_progressions, "chord progressions")
```

```{code-cell}
numeral_progressions = get_progressions('PAC', dict(numeral=('I', 'i')), 'numeral')
value_count_df(numeral_progressions, "numeral progressions")
```

```{code-cell}
numeral_prog_no_dups = numeral_progressions.map(remove_immediate_duplicates)
value_count_df(numeral_prog_no_dups)
```

### PACs ending on scale degree 1

**Scale degrees expressed w.r.t. major scale, regardless of actual key.**

```{code-cell}
bass_progressions = get_progressions('PAC', dict(bass_note=0), 'bass_note')
bass_prog = bass_progressions.map(ms3.fifths2sd)
print(f"Progressions for {len(bass_progressions)} cadences:")
value_count_df(bass_prog, "bass progressions")
```

```{code-cell}
bass_prog_no_dups = bass_prog.map(remove_immediate_duplicates)
value_count_df(bass_prog_no_dups)
```

```{code-cell}
def make_sankey(
        data,
        labels,
        node_pos=None,
        margin={'l': 10, 'r': 10, 'b': 10, 't': 10},
        pad=20,
        color='auto',
        **kwargs):
    if color=='auto':
        unique_labels = set(labels)
        color_step = 100 / len(unique_labels)
        unique_colors = {label: f'hsv({round(i*color_step)}%,100%,100%)' for i, label in enumerate(unique_labels)}
        color = list(map(lambda l: unique_colors[l], labels))
    fig = go.Figure(go.Sankey(
        arrangement = 'snap',
        node = dict(
          pad = pad,
          #thickness = 20,
          #line = dict(color = "black", width = 0.5),
          label = labels,
          x = [node_pos[i][0] if i in node_pos else 0 for i in range(len(labels))] if node_pos is not None else None,
          y = [node_pos[i][1] if i in node_pos else 0 for i in range(len(labels))] if node_pos is not None else None,
          color = color,
          ),
        link = dict(
          source = data.source,
          target = data.target,
          value = data.value
          ),
        ),
     )

    fig.update_layout(margin=margin, **kwargs)
    return fig

def progressions2graph_data(progressions, cut_at_stage=None):
    stage_nodes = defaultdict(dict)
    edge_weights = Counter()
    node_counter = 0
    for progression in progressions:
        previous_node = None
        for stage, current in enumerate(reversed(progression)):
            if cut_at_stage and stage > cut_at_stage:
                break
            if current in stage_nodes[stage]:
                current_node = stage_nodes[stage][current]
            else:
                stage_nodes[stage][current] = node_counter
                current_node = node_counter
                node_counter += 1
            if previous_node is not None:
                edge_weights.update([(current_node, previous_node)])
            previous_node = current_node
    return stage_nodes, edge_weights

def graph_data2sankey(stage_nodes, edge_weights, **kwargs):
    data = pd.DataFrame([(u, v, w) for (u, v), w in edge_weights.items()], columns = ['source', 'target', 'value'])
    node2label = {node: label for stage, nodes in stage_nodes.items() for label, node in nodes.items()}
    labels = [node2label[i] for i in range(len(node2label))]
    return make_sankey(data, labels, **kwargs)

def plot_progressions(progressions, cut_at_stage=None, **kwargs):
    stage_nodes, edge_weights = progressions2graph_data(progressions, cut_at_stage=cut_at_stage)
    return graph_data2sankey(stage_nodes, edge_weights, **kwargs)
```

#### Chordal roots for the 3 last stages

```{code-cell}
fig = plot_progressions(
  numeral_prog_no_dups,
  cut_at_stage=3,
  font=dict(size=30),
)
save_figure_as(fig, 'last_3_roots_before_pacs_ending_on_1_sankey', height=800)
fig.show()
```

#### Complete chords for the last four stages in major

```{code-cell}
pac_major = get_progressions('PAC', dict(numeral='I', localkey_is_minor=False), 'chord')
fig = plot_progressions(pac_major, cut_at_stage=4)
save_figure_as(fig, 'last_4_stages_before_pacs_in_major_sankey')
fig.show()
```

#### Bass degrees for the last 6 stages.

```{code-cell}
fig = plot_progressions(bass_prog_no_dups, cut_at_stage=7)
save_figure_as(fig, 'last_7_degrees_before_pacs_ending_on_1_sankey')
fig.show()
```

#### Bass degrees without accidentals

```{code-cell}
def remove_sd_accidentals(t):
    return tuple(map(lambda sd: sd[-1], t))

bass_prog_no_acc_no_dup = bass_prog.map(remove_sd_accidentals).map(remove_immediate_duplicates)
fig = plot_progressions(bass_prog_no_acc_no_dup, cut_at_stage=7)
save_figure_as(fig, 'last_7_degrees_before_pacs_ending_on_1_without_accdentals_sankey')
fig.show()
```

### HCs ending on V

```{code-cell}
half = get_progressions('HC', dict(numeral='V'), 'bass_note').map(ms3.fifths2sd)
print(f"Progressions for {len(half)} cadences:")
fig = plot_progressions(half.map(remove_immediate_duplicates), cut_at_stage=5)
save_figure_as(fig, 'last_7_degrees_before_hcs_ending_on_V_sankey')
fig.show()
```
