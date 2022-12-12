---
jupyter:
  jupytext:
    formats: md,py:percent
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: dimcat
    language: python
    name: dimcat
---

`pip install dimcat Jinja2 colorlover GitPython plotly`

```python
%load_ext autoreload
%autoreload 2
import os
from collections import defaultdict, Counter
from fractions import Fraction
from IPython.display import HTML
import ms3
import dimcat as dc
from git import Repo
import plotly.express as px
import colorlover
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 500)
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
plt.set_loglevel('error')

def value_count_df(S, thing=None, counts='counts'):
    thing = S.name if thing is None else thing
    df = S.value_counts().rename(counts).to_frame()
    df.index.rename(thing, inplace=True)
    return df

def color_background(x, color="#ffffb3"):
    return np.where(x.notna().to_numpy(), f"background-color: {color};", None)


```

This javascript allows to add a "Toggle Code" button to every cell as per http://www.eointravers.com/post/jupyter-toggle/

<!-- #raw -->
function toggler(){
    if(window.already_toggling){
        // Don't add multiple buttons.
        return 0
    }
    let btn = $('.input').append('<button>Toggle Code</button>')
        .children('button');
    btn.on('click', function(e){
        let tgt = e.currentTarget;
        $(tgt).parent().children('.inner_cell').toggle()
    })
    window.already_toggling = true;
}
// Since javascript cells are executed as soon as we load
// the notebook (if it's trusted), and this cell might be at the
// top of the notebook (so is executed first), we need to
// allow time for all of the other code cells to load before
// running. Let's give it 5 seconds.

setTimeout(toggler, 5000);
<!-- #endraw -->

```python
HTML('''<script>
  function code_toggle() {
    if (code_shown){
      $('div.input').hide('500');
      $('#toggleButton').val('Show Code')
    } else {
      $('div.input').show('500');
      $('#toggleButton').val('Hide Code')
    }
    code_shown = !code_shown
  }
  $( document ).ready(function(){
    code_shown=false;
    $('div.input').hide()
  });
</script>
<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>''')
```

# Preliminaries
## Software versions and configurations

```python
dataset_path = "~/all_subcorpora"

repo = Repo(dataset_path)
print(f"{os.path.basename(dataset_path)} repository @ commit {repo.commit().hexsha[:7]}")
print(f"dimcat version {dc.__version__}")
print(f"ms3 version {ms3.__version__}")
```

```python
STD_LAYOUT = {
 'paper_bgcolor': '#FFFFFF',
 'plot_bgcolor': '#FFFFFF',
 'margin': {'l': 40, 'r': 0, 'b': 0, 't': 40, 'pad': 0},
 'font': {'size': 15}
}
import colorlover
#for name, scales in colorlover.scales['6']['qual'].items():
#    print(name)
#    display(HTML(colorlover.to_html(scales)))
cadence_colors = dict(zip(('HC', 'PAC', 'PC', 'IAC', 'DC', 'EC'), colorlover.scales['6']['qual']['Set1']))
```

## Data loading

```python
dataset = dc.Dataset()
for folder in ['bach_solo', 'beethoven_piano_sonatas', 'c_schumann_lieder', 'chopin_mazurkas', 'corelli', 'debussy_suite_bergamasque', 'dvorak_silhouettes', 'grieg_lyrical_pieces', 'handel_keyboard', 'jc_bach_sonatas', 'liszt_pelerinage', 'mahler_kindertotenlieder', 'medtner_tales', 'pleyel_quartets', 'scarlatti_sonatas', 'schubert_dances', 'schumann_kinderszenen', 'tchaikovsky_seasons', 'wf_bach_sonatas']:
    print("Loading", folder)
    path = os.path.join(dataset_path, folder)
    dataset.load(directory=path)
```

```python
dataset.data
```

```python
#dataset = Corpus(directory=dataset_path)
#dataset.data
```

### Filtering out pieces without cadence annotations

```python
hascadence = dc.HasCadenceAnnotationsFilter().process_data(dataset)
display(HTML(f"<h4>Before: {len(dataset.indices[()])} pieces; "
             f"after removing those without cadence labels: {len(hascadence.indices[()])}</h4>"))
```

### Show corpora containing pieces with cadence annotations

```python
grouped_by_dataset = dc.CorpusGrouper().process_data(hascadence)
corpora = {group[0]: f"{len(ixs)} pieces" for group, ixs in  grouped_by_dataset.indices.items()}
print(f"{len(corpora)} corpora with {sum(map(len, grouped_by_dataset.indices.values()))} pieces containing cadence annotations:")
corpora
```

### All annotation labels from the selected pieces

```python
all_labels = hascadence.get_facet('expanded')

print(f"{len(all_labels.index)} hand-annotated harmony labels:")
all_labels.iloc[:10, 14:].style.apply(color_background, subset="chord")
```

```python
HTML('''<script>
  function code_toggle() {
    if (code_shown){
      $('div.input').hide('500');
      $('#toggleButton').val('Show Code')
    } else {
      $('div.input').show('500');
      $('#toggleButton').val('Hide Code')
    }
    code_shown = !code_shown
  }
  $( document ).ready(function(){
    code_shown=false;
    $('div.input').hide()
  });
</script>
<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>''')
```

### Metadata

```python
dataset_metadata = hascadence.data.metadata()
hascadence_metadata = dataset_metadata.loc[hascadence.indices[()]]
hascadence_metadata.index.rename('dataset', level=0, inplace=True)
hascadence_metadata.head()
```

```python
mean_composition_years = hascadence_metadata.groupby(level=0).composed_end.mean().astype(int).sort_values()
chronological_order = mean_composition_years.index.to_list()
bar_data = pd.concat([mean_composition_years.rename('year'), 
                      hascadence_metadata.groupby(level='dataset').size().rename('pieces')],
                     axis=1
                    ).reset_index()
fig = px.bar(bar_data, x='year', y='pieces', color='dataset', title='Pieces contained in the dataset')
fig.update_traces(width=5)
```

# Keys

## Computing extent of key segments from annotations

**In the following, major and minor keys are distinguished as boolean `localkey_is_minor=(False|True)`**

```python
segmented_by_keys = dc.Pipeline([
                         dc.LocalKeySlicer(), 
                         dc.ModeGrouper()])\
                        .process_data(hascadence)
key_segments = segmented_by_keys.get_slice_info()
```

```python
print(key_segments.duration_qb.dtype)
key_segments.duration_qb = pd.to_numeric(key_segments.duration_qb)
```

```python
key_segments.iloc[:15, 11:].fillna('').style.apply(color_background, subset="localkey")
```

## Ratio between major and minor key segments by aggregated durations
### Overall

```python
maj_min_ratio = key_segments.groupby(level="localkey_is_minor").duration_qb.sum().to_frame()
maj_min_ratio['fraction'] = (100.0 * maj_min_ratio.duration_qb / maj_min_ratio.duration_qb.sum()).round(1)
maj_min_ratio
```

### By dataset

```python
segment_duration_per_dataset = key_segments.groupby(level=["corpus", "localkey_is_minor"]).duration_qb.sum().round(2)
norm_segment_duration_per_dataset = 100 * segment_duration_per_dataset / segment_duration_per_dataset.groupby(level="corpus").sum()
maj_min_ratio_per_dataset = pd.concat([segment_duration_per_dataset, 
                                      norm_segment_duration_per_dataset.rename('fraction').round(1).astype(str)+" %"], 
                                     axis=1)
```

```python
segment_duration_per_dataset = key_segments.groupby(level=["corpus", "localkey_is_minor"]).duration_qb.sum().reset_index()
```


```python
maj_min_ratio_per_dataset.reset_index()
```

```python
chronological_order
```

```python
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

```python
annotations_by_keys = segmented_by_keys.get_facet("expanded")
annotations_by_keys
```

```python
HTML('''<script>
  function code_toggle() {
    if (code_shown){
      $('div.input').hide('500');
      $('#toggleButton').val('Show Code')
    } else {
      $('div.input').show('500');
      $('#toggleButton').val('Hide Code')
    }
    code_shown = !code_shown
  }
  $( document ).ready(function(){
    code_shown=false;
    $('div.input').hide()
  });
</script>
<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>''')
```

# Phrases
## Overview
### Presence of phrase annotation symbols per dataset:

```python
all_labels.groupby(["corpus"]).phraseend.value_counts()
```

### Presence of legacy phrase endings

```python
all_labels[all_labels.phraseend == r'\\'].style.apply(color_background, subset="label")
```

### A table with the extents of all annotated phrases
**Relevant columns:**
* `quarterbeats`: start position for each phrase
* `duration_qb`: duration of each phrase, measured in quarter notes 
* `phrase_slice`: time interval of each annotated phrases (for segmenting chord progressions and notes)

```python
# segmented = PhraseSlicer().process_data(hascadence)
segmented = dc.PhraseSlicer().process_data(grouped_by_dataset)
phrases = segmented.get_slice_info()
print(f"Overall number of phrases is {len(phrases.index)}")
phrases.head(10).style.apply(color_background, subset=["quarterbeats", "duration_qb"])
```

```python
print(phrases.duration_qb.dtype)
phrases.duration_qb = pd.to_numeric(phrases.duration_qb)
```

### Annotation table sliced by phrase annotations

ToDo: Example for overlap / phrase beginning without new chord

```python
phrase_segments = segmented.get_facet("expanded")
phrase_segments.head(10)
```

```python
print(phrase_segments.duration_qb.dtype)
phrase_segments.duration_qb = pd.to_numeric(phrase_segments.duration_qb)
```

## Distribution of phrase lengths
### Histogram summarizing the lengths of all phrases measured in quarter notes

```python
phrase_durations = phrases.duration_qb.value_counts() 
histogram = px.histogram(x=phrase_durations.index, y=phrase_durations, labels=dict(x='phrase lengths binned to a quarter note', y='#phrases within length bin'))
histogram.update_traces(xbins=dict( # bins used for histogram
        #start=0.0,
        end=100.0,
        size=1
    ))
histogram.update_xaxes(dtick=4)
histogram.show()
```

### Bar plot showing approximative phrase length in measures

**Simply by subtracting for the span of every phrase the first measure measure number from the last.**

```python
phrase_gpb = phrase_segments.groupby(level=[0,1,2])
phrase_length_in_measures = phrase_gpb.mn.max() - phrase_gpb.mn.min()
measure_length_counts = phrase_length_in_measures.value_counts()
fig = px.bar(x=measure_length_counts.index, y=measure_length_counts, labels=dict(x="approximative size of all phrases (difference between end and start measure number)",
                                                                           y="#phrases"))
fig.update_xaxes(dtick=4)
```

### Histogram summarizing phrase lengths by precise length expressed in measures

**In order to divide the phrase length by the length of a measure, the phrases containing more than one time signature are filtered out.**


**Durations computed by dividing the duration by the measure length**

```python
phrase2timesigs = phrase_gpb.timesig.unique()
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
fig = px.histogram(uniform_timesigs, x='duration_measures', 
                   labels=dict(duration_measures='phrase length in measures, factoring in time signatures'))
fig.update_traces(xbins=dict( # bins used for histogram
        #start=0.0,
        #end=100.0,
        size=1
    ))
fig.update_xaxes(dtick=4)
```

```python
uniform_timesigs.head(10).style.apply(color_background, subset='duration_measures')
```

### Inspecting long phrases

```python
timsig_counts = uniform_timesigs.timesig.value_counts()
fig = px.bar(timsig_counts, labels=dict(index="time signature", value="#phrases"))
fig.update_layout(**STD_LAYOUT)
fig.update_yaxes(gridcolor='lightgrey')
```

```python
filter_counts_smaller_than = 5
filtered_timesigs = timsig_counts[timsig_counts < filter_counts_smaller_than].index.to_list()
```

```python
fig = px.histogram(uniform_timesigs[~uniform_timesigs.timesig.isin(filtered_timesigs)], 
                   x='duration_measures', facet_col='timesig', facet_col_wrap=2, height=1500)
fig.update_xaxes(matches=None, showticklabels=True, visible=True, dtick=4)
fig.update_yaxes(matches=None, showticklabels=True, visible=True)
fig.update_traces(xbins=dict( # bins used for histogram
        #start=0.0,
        end=50.0,
        size=1
    ))
```

```python
see_greater_equal = 33
longest_measure_length = uniform_timesigs.loc[uniform_timesigs.duration_measures >= see_greater_equal, ["duration_measures", "timesig"]]
for timesig, long_phrases in longest_measure_length.groupby('timesig'):
    L = len(long_phrases)
    plural = 's' if L > 1 else ''
    display(HTML(f"<h3>{L} long phrase{plural} in {timesig} meter:</h3>"))
    display(long_phrases.sort_values('duration_measures'))
```

## Local keys

```python
local_keys_per_phrase = phrase_gpb.localkey.unique().map(tuple)
n_local_keys_per_phrase = local_keys_per_phrase.map(len)
phrases_with_keys = pd.concat([n_local_keys_per_phrase.rename('n_local_keys'),
                               local_keys_per_phrase.rename('local_keys'),
                               phrases], axis=1)
phrases_with_keys.head(10).style.apply(color_background, subset=['n_local_keys', 'local_keys'])
```

### Number of unique local keys per phrase

```python
count_n_keys = phrases_with_keys.n_local_keys.value_counts().rename("#phrases").to_frame()
count_n_keys.index.rename("unique keys", inplace=True)
count_n_keys
```

### The most frequent keys for non-modulating phrases

```python
unique_key_selector = phrases_with_keys.n_local_keys == 1
phrases_with_unique_key = phrases_with_keys[unique_key_selector].copy()
phrases_with_unique_key.local_keys = phrases_with_unique_key.local_keys.map(lambda t: t[0])
value_count_df(phrases_with_unique_key.local_keys, counts="#phrases")
```

### Most frequent modulations within one phrase

```python
two_keys_selector = phrases_with_keys.n_local_keys > 1
phrases_with_unique_key = phrases_with_keys[two_keys_selector].copy()
value_count_df(phrases_with_unique_key.local_keys, "modulations")
```

# Cadences
## Overall

* **PAC**: Perfect Authentic Cadence
* **IAC**: Imperfect Authentic Cadence
* **HC**: Half Cadence
* **DC**: Deceptive Cadence
* **EC**: Evaded Cadence
* **PC**: Plagal Cadence

```python
HTML('''<script>
  function code_toggle() {
    if (code_shown){
      $('div.input').hide('500');
      $('#toggleButton').val('Show Code')
    } else {
      $('div.input').show('500');
      $('#toggleButton').val('Hide Code')
    }
    code_shown = !code_shown
  }
  $( document ).ready(function(){
    code_shown=false;
    $('div.input').hide()
  });
</script>
<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>''')
```

```python
print(f"{all_labels.cadence.notna().sum()} cadence labels.")
value_count_df(all_labels.cadence)
```

```python
px.pie(all_labels[all_labels.cadence.notna()], names="cadence", color="cadence", color_discrete_map=cadence_colors)
```

## Per dataset

```python
cadence_count_per_dataset = all_labels.groupby("corpus").cadence.value_counts()
cadence_fraction_per_dataset = cadence_count_per_dataset / cadence_count_per_dataset.groupby(level=0).sum()
px.bar(cadence_fraction_per_dataset.rename('count').reset_index(), x='corpus', y='count', color='cadence',
      color_discrete_map=cadence_colors, category_orders=dict(dataset=chronological_order))
```

```python
fig = px.pie(cadence_count_per_dataset.rename('count').reset_index(), names='cadence', color='cadence', values='count', 
       facet_col='corpus', facet_col_wrap=4, height=2000, color_discrete_map=cadence_colors)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_layout(**STD_LAYOUT)
```

## Per phrase
### Number of cadences per phrase

```python
phrases_with_cadences = pd.concat([
    phrase_gpb.cadence.nunique().rename('n_cadences'),
    phrase_gpb.cadence.unique().rename('cadences').map(lambda l: tuple(e for e in l if not pd.isnull(e))),
    phrases_with_keys
], axis=1)
value_count_df(phrases_with_cadences.n_cadences, counts="#phrases")
```

```python
n_cad = phrases_with_cadences.groupby(level='corpus').n_cadences.value_counts().rename('counts').reset_index().sort_values('n_cadences')
n_cad.n_cadences = n_cad.n_cadences.astype(str)
fig = px.bar(n_cad, x='corpus', y='counts', color='n_cadences', height=800, barmode='group',
             labels=dict(n_cadences="#cadences in a phrase"),
             category_orders=dict(dataset=chronological_order)
      )
fig.show()
```

### Combinations of cadence types for phrases with more than one cadence

```python
value_count_df(phrases_with_cadences[phrases_with_cadences.n_cadences > 1].cadences)
```

### Positioning of cadences within phrases

```python
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

```python
fig = px.scatter(data[data.x.notna()], x='x', y="phrase_ix", color="marker", hover_name="description", height=3000,
                labels=dict(marker='legend'), color_discrete_map=cadence_colors)
fig.update_traces(marker_size=5)
fig.update_yaxes(autorange="reversed")
fig.show()
```

## Cadence ultima

```python
HTML('''<script>
  function code_toggle() {
    if (code_shown){
      $('div.input').hide('500');
      $('#toggleButton').val('Show Code')
    } else {
      $('div.input').show('500');
      $('#toggleButton').val('Hide Code')
    }
    code_shown = !code_shown
  }
  $( document ).ready(function(){
    code_shown=false;
    $('div.input').hide()
  });
</script>
<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>''')
```

```python
phrase_segments = segmented.get_facet("expanded")
cadence_selector = phrase_segments.cadence.notna()
missing_chord_selector = phrase_segments.chord.isna()
cadence_with_missing_chord_selector = cadence_selector & missing_chord_selector
#print(f"Ultima missing for {cadence_with_missing_chord_selector.sum()} cadences.")
missing = phrase_segments[cadence_with_missing_chord_selector]
expanded = ms3.expand_dcml.expand_labels(phrase_segments[cadence_with_missing_chord_selector], propagate=False, chord_tones=True, skip_checks=True)
phrase_segments.loc[cadence_with_missing_chord_selector] = expanded
print(f"Ultima harmony missing for {(phrase_segments.cadence.notna() & phrase_segments.bass_note.isna()).sum()} cadence labels.")
```

### Ultimae as Roman numeral

```python
def highlight(row, color="#ffffb3"):
    if row.counts < 10:
        return [None, None, None, None]
    else:
        return ["background-color: #ffffb3;"] * 4

cadence_counts = all_labels.cadence.value_counts()
ultima_root = phrase_segments.groupby(['localkey_is_minor', 'cadence']).numeral.value_counts().rename('counts').to_frame().reset_index()
ultima_root.localkey_is_minor = ultima_root.localkey_is_minor.map({False: 'in major', True: 'in minor'})
#ultima_root.style.apply(highlight, axis=1)
```

```python
fig = px.pie(ultima_root, names='numeral', values='counts', 
             facet_row='cadence', facet_col='localkey_is_minor', 
             height=1500,
             category_orders={'cadence': cadence_counts.index},
            )
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(**STD_LAYOUT)
fig.show()
```

```python
#phrase_segments.groupby(level=[0,1,2], group_keys=False).apply(lambda df: df if ((df.cadence == 'PAC') & (df.numeral == 'V')).any() else None)
```

### Ultimae bass note as scale degree

```python
ultima_bass = phrase_segments.groupby(['localkey_is_minor','cadence']).bass_note.value_counts().rename('counts').reset_index()
ultima_bass.bass_note = ms3.transform(ultima_bass, ms3.fifths2sd, dict(fifths='bass_note', minor='localkey_is_minor'))
ultima_bass.localkey_is_minor = ultima_bass.localkey_is_minor.map({False: 'in major', True: 'in minor'})
#ultima_bass.style.apply(highlight, axis=1)
```

```python
fig = px.pie(ultima_bass, names='bass_note', values='counts', 
             facet_row='cadence', facet_col='localkey_is_minor', 
             height=1500, 
             category_orders={'cadence': cadence_counts.index},
            )
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(**STD_LAYOUT)
fig.show()
```

## Chord progressions


### PACs with ultima I/i

```python
#pac_on_i = phrase_segments.groupby(level=[0,1,2], group_keys=False).apply(lambda df: df if ((df.cadence == 'PAC') & (df.numeral.isin(('I', 'i')))).any() else None)
#pac_on_i.cadence.value_counts()
#pac_on_i.droplevel(-1).index.nunique()
```

```python
def get_progressions(selected='PAC', last_row={}, feature='chord', dataset=None, as_series=True):
    """Uses the nonlocal variable phrase_segments."""
    last_row = {k: v if isinstance(v, tuple) else (v,) for k, v in last_row.items()}
    progressions = []

    for (corp, fname, *_), df in phrase_segments[phrase_segments[feature].notna()].groupby(level=[0,1,2]):
        if dataset is not None and dataset not in corp:
            continue
        if (df.cadence == selected).fillna(False).any():
            # remove chords after the last cadence label
            df = df[df.cadence.fillna(method='bfill').notna()]
            # group segments leading up to a cadence label
            cadence_groups = df.cadence.notna().shift().fillna(False).cumsum()
            for i, cadence in df.groupby(cadence_groups):
                last_r = cadence.iloc[-1]
                typ = last_r.cadence
                if typ != selected:
                    continue
                if any(last_r[feat] not in values for feat, values in last_row.items()):
                    continue
                progressions.append(tuple(cadence[feature]))
    if as_series:
        return pd.Series(progressions)
    return progressions


```

```python
chord_progressions = get_progressions('PAC', dict(numeral=('I', 'i')), 'chord')
print(f"Progressions for {len(chord_progressions)} cadences:")
value_count_df(chord_progressions, "chord progressions")
```

```python
numeral_progressions = get_progressions('PAC', dict(numeral=('I', 'i')), 'numeral')
value_count_df(numeral_progressions, "numeral progressions")
```

```python
def remove_immediate_duplicates(l):
    return tuple(a for a, b in zip(l, (None, ) + l) if a != b)

numeral_prog_no_dups = numeral_progressions.map(remove_immediate_duplicates)
value_count_df(numeral_prog_no_dups)
```

### PACs ending on scale degree 1

**Scale degrees expressed w.r.t. major scale, regardless of actual key.**

```python
bass_progressions = get_progressions('PAC', dict(bass_note=0), 'bass_note')
bass_prog = bass_progressions.map(ms3.fifths2sd)
print(f"Progressions for {len(bass_progressions)} cadences:")
value_count_df(bass_prog, "bass progressions")
```

```python
bass_prog_no_dups = bass_prog.map(remove_immediate_duplicates)
value_count_df(bass_prog_no_dups)
```

```python
def make_sankey(data, labels, node_pos=None, margin={'l': 10, 'r': 10, 'b': 10, 't': 10}, pad=20, color='auto', **kwargs):
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

def graph_data2sankey(stage_nodes, edge_weights):
    data = pd.DataFrame([(u, v, w) for (u, v), w in edge_weights.items()], columns = ['source', 'target', 'value'])
    node2label = {node: label for stage, nodes in stage_nodes.items() for label, node in nodes.items()}
    labels = [node2label[i] for i in range(len(node2label))]
    return make_sankey(data, labels)

def plot_progressions(progressions, cut_at_stage=None):
    stage_nodes, edge_weights = progressions2graph_data(progressions, cut_at_stage=cut_at_stage)
    return graph_data2sankey(stage_nodes, edge_weights)

plot_progressions(numeral_prog_no_dups, cut_at_stage=3)
```

```python
chord_progressions_minor = get_progressions('PAC', dict(numeral='i', localkey_is_minor=True), 'root')
chord_progressions_minor
```

```python
pac_major = get_progressions('PAC', dict(numeral='I', localkey_is_minor=False), 'chord')
plot_progressions(pac_major, cut_at_stage=4)
```

```python
deceptive = get_progressions('DC', dict(localkey_is_minor=False), 'chord')
deceptive.value_counts()
```

```python
plot_progressions(deceptive, cut_at_stage=4)
```

```python
plot_progressions(bass_prog_no_dups, cut_at_stage=7)
```

```python
def remove_sd_accidentals(t):
    return tuple(map(lambda sd: sd[-1], t))
                  
bass_prog_no_acc_no_dup = bass_prog.map(remove_sd_accidentals).map(remove_immediate_duplicates)
plot_progressions(bass_prog_no_acc_no_dup, cut_at_stage=7)
```

### HCs ending on V

```python
half = get_progressions('HC', dict(numeral='V'), 'bass_note').map(ms3.fifths2sd)
print(f"Progressions for {len(half)} cadences:")
plot_progressions(half.map(remove_immediate_duplicates), cut_at_stage=5)
```

```python
HTML('''<script>
  function code_toggle() {
    if (code_shown){
      $('div.input').hide('500');
      $('#toggleButton').val('Show Code')
    } else {
      $('div.input').show('500');
      $('#toggleButton').val('Hide Code')
    }
    code_shown = !code_shown
  }
  $( document ).ready(function(){
    code_shown=false;
    $('div.input').hide()
  });
</script>
<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>''')
```

# Profiles

```python

```

# Gantt approach

```python
HTML('''<script>
  function code_toggle() {
    if (code_shown){
      $('div.input').hide('500');
      $('#toggleButton').val('Show Code')
    } else {
      $('div.input').show('500');
      $('#toggleButton').val('Hide Code')
    }
    code_shown = !code_shown
  }
  $( document ).ready(function(){
    code_shown=false;
    $('div.input').hide()
  });
</script>
<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>''')
```

```python
chord_progressions = dict(
    PAC = [],
    IAC = [],
    HC = [],
    EC = [],
    DC = [],
    PC = [],
)
```

```python
import plotly.graph_objects as go

import networkx as nx

G = nx.random_geometric_graph(200, 0.125)

edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='YlGnBu',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))

node_adjacencies = []
node_text = []
for node, adjacencies in enumerate(G.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    node_text.append('# of connections: '+str(len(adjacencies[1])))

node_trace.marker.color = node_adjacencies
node_trace.text = node_text

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
fig.show()
```

```python
from ms3 import make_gantt_data
from gantt import create_gantt, create_modulation_plan, get_phraseends
```

```python
df = all_labels.loc[('beethoven_piano_sonatas', '01-3')]
data = make_gantt_data(df)
data
```

```python
create_gantt(data, task_column='fifths')
```

```python
df[df.phraseend.notna()]
```

```python
create_modulation_plan(data, title="Beethoven 01-3", globalkey='f', task_column='semitones', phraseends=get_phraseends(df))
```

```python
for ix, df in phrase_segments.groupby(level=["corpus", "fname", "phrase_slice"]):
    display(df)
    break
```

```python
make_gantt_data(df)
```
