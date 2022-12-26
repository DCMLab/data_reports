# ---
# jupyter:
#   jupytext:
#     formats: py:percent,md
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: dimcat
#     language: python
#     name: dimcat
# ---

# %% pycharm={"name": "#%%\n"}
# %load_ext autoreload
# %autoreload 2
import os
from fractions import Fraction
from IPython.display import HTML
import ms3
import dimcat as dc
from git import Repo
import plotly.express as px
import colorlover
import pandas as pd
pd.set_option("display.max_columns", 100)

# %%
corpus_path = "~/romantic_piano_corpus"
corpus_path = os.path.expanduser(corpus_path)
repo = Repo(corpus_path)
print(f"{os.path.basename(corpus_path)} @ {repo.commit().hexsha[:7]}")
print(f"dimcat version {dc.__version__}")
print(f"ms3 version {ms3.__version__}")

# %%
STD_LAYOUT = {
 'paper_bgcolor': '#FFFFFF',
 'plot_bgcolor': '#FFFFFF',
 'margin': {'l': 40, 'r': 0, 'b': 0, 't': 40, 'pad': 0},
 'font': {'size': 15}
}
#OUTPUT_DIR = "/home/hentsche/Documents/phd/romantic_piano_corpus_report/figures/"
OUTPUT_DIR = os.path.join(corpus_path, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)
#HTML(colorlover.to_html(colorlover.scales))
HTML(colorlover.to_html(colorlover.scales['9']['qual']['Paired']))

# %%
fig = px.colors.qualitative.swatches()
fig.show()

# %%
corpus_color_scale = px.colors.qualitative.D3

# %% [markdown]
# # Overview

# %% pycharm={"name": "#%%\n"}
dataset = dc.Dataset(directory=corpus_path)
dataset.data

# %%
all_metadata = dataset.data.metadata()
print(f"Concatenated 'metadata.tsv' files cover {len(all_metadata)} of the {dataset.data.n_pieces} scores.")
all_metadata.groupby(level=0).nth(0)

# %%
annotated = dc.IsAnnotatedFilter().process_data(dataset)
print(f"Before: {len(dataset.indices[()])} IDs, after filtering: {len(annotated.indices[()])}")

# %% [markdown]
# **Choose here if you want to see stats for all or only for annotated scores.**

# %%
#selected = dataset
selected = annotated

# %% [markdown]
# **Compute chronological order**

# %%
summary = all_metadata[all_metadata.label_count > 0]
print(f"Selected metadata rows cover {len(summary)} of the {len(sum((ixs for _, ixs in selected.iter_groups()), start=[]))} scores.")
mean_composition_years = summary.groupby(level=0).composed_end.mean().astype(int).sort_values()
chronological_order = mean_composition_years.index.to_list()
dataset_colors = dict(zip(chronological_order, corpus_color_scale))
chronological_order

# %% [markdown]
# ## Notes

# %%
all_notes = selected.get_facet('notes')
print(f"{len(all_notes.index)} notes over {len(all_notes.groupby(level=[0,1]))} files.")
all_notes.head()


# %%
def weight_notes(nl, group_col='midi', precise=True):
    summed_durations = nl.groupby(group_col).duration_qb.sum()
    summed_durations /= summed_durations.min() # normalize such that the shortest duration results in 1 occurrence
    if not precise:
        # This simple trick reduces compute time but also precision:
        # The rationale is to have the smallest value be slightly larger than 0.5 because
        # if it was exactly 0.5 it would be rounded down by repeat_notes_according_to_weights()
        summed_durations /= 1.9999999
    return repeat_notes_according_to_weights(summed_durations)
    
def repeat_notes_according_to_weights(weights):
    counts = weights.round().astype(int)
    counts_reflecting_weights = []
    for pitch, count in counts.iteritems():
        counts_reflecting_weights.extend([pitch]*count)
    return pd.Series(counts_reflecting_weights)



# %% [raw]
# grouped_notes = all_notes.groupby(level=0)
# weighted_midi = pd.concat([weight_notes(nl, 'midi', precise=False) for _, nl in grouped_notes], axis=1, keys=grouped_notes.groups.keys())
#
# yaxis=dict(tickmode= 'array',
#            tickvals= [12, 24, 36, 48, 60, 72, 84, 96],
#            ticktext = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"],
#            gridcolor='lightgrey',
#            )
# fig = px.violin(weighted_midi, labels=dict(variable='', value='pitch'), box=True, height=500,
#                category_orders=chronological_order,
#                ) #, title="Distribution of pitches per dataset"
# fig.update_layout(yaxis=yaxis, **STD_LAYOUT)
# fig.write_image(os.path.join(OUTPUT_DIR, "ambitus_per_dataset.png"), scale=2)
# fig.show()

# %%
dataset_names = dict(
    beethoven_piano_sonatas='Beethoven Sonatas',
    chopin_mazurkas='Chopin Mazurkas',
    debussy_suite_bergamasque='Debussy Suite',
    dvorak_silhouettes="Dvořák Silhouettes",
    grieg_lyrical_pieces="Grieg Lyrical Pieces",
    liszt_pelerinage="Liszt Années",
    medtner_tales="Medtner Tales",
    schumann_kinderszenen="Schumann Kinderszenen",
    tchaikovsky_seasons="Tchaikovsky Seasons"
)
dataset_name_colors = {dataset_names[corp]: color for corp, color in dataset_colors.items()}
chronological_dataset_names = [dataset_names[corp] for corp in chronological_order]
all_notes['dataset_name'] = all_notes.index.get_level_values(0).map(dataset_names)

# %%
grouped_notes = all_notes.groupby('dataset_name')
weighted_midi = pd.concat([weight_notes(nl, 'midi', precise=False) for _, nl in grouped_notes], keys=grouped_notes.groups.keys()).reset_index(level=0)
weighted_midi.columns = ['dataset', 'midi']
weighted_midi

# %%
yaxis=dict(tickmode= 'array',
           tickvals= [12, 24, 36, 48, 60, 72, 84, 96],
           ticktext = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"],
           gridcolor='lightgrey',
           )
fig = px.violin(weighted_midi, x='dataset', y='midi', color='dataset', box=True,
                labels=dict(
                    dataset='',
                    midi='distribution of pitches by duration'
                ),
                category_orders=dict(dataset=chronological_dataset_names),
                color_discrete_map=dataset_name_colors,
                width=1000, height=600,
               )
fig.update_traces(spanmode='hard') # do not extend beyond outliers
fig.update_layout(yaxis=yaxis, **STD_LAYOUT,
                 showlegend=False)
fig.write_image(os.path.join(OUTPUT_DIR, "ambitus_per_dataset_colored.png"), scale=2)
fig.show()

# %% [raw]
# weighted_tpc = pd.concat([weight_notes(nl, 'tpc') for _, nl in grouped_notes], axis=1, keys=grouped_notes.groups.keys())
# weighted_tpc

# %% [raw]
# yaxis=dict(
#     tickmode= 'array',
#     tickvals= [-12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18],
#     ticktext = ["Dbb", "Bbb", "Gb", "Eb", "C", "A", "F#", "D#", "B#", "G##", "E##"],
#     gridcolor='lightgrey',
#     zerolinecolor='lightgrey',
#     zeroline=True
#            )
# fig = px.violin(weighted_tpc, labels=dict(variable='', value='pitch class'), box=True, height=500)
# fig.update_layout(yaxis=yaxis, **STD_LAYOUT)
# fig.write_image(os.path.join(OUTPUT_DIR, "tpc_per_dataset.png"), scale=2)
# fig.show()

# %% [raw]
# # adapted from https://plotly.com/python/violin/#ridgeline-plot
# fig = go.Figure()
# for corp, data_line in weighted_tpc.iteritems():
#     fig.add_trace(go.Violin(x=data_line, name=corp))
#
# fig.update_traces(side='positive', orientation='h', width=2, points=False)
# fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=True, height=600)
# fig.show()

# %%
bar_data = all_notes.groupby('tpc').duration_qb.sum().reset_index()
x_values = list(range(bar_data.tpc.min(), bar_data.tpc.max()+1))
x_names = ms3.fifths2name(x_values)
fig = px.bar(bar_data, x='tpc', y='duration_qb',
                 labels=dict(tpc='Named pitch class',
                             duration_qb='Duration in quarter notes'
                            ),
             color_discrete_sequence=corpus_color_scale,
             width=1000, height=300,
            )
fig.update_layout(**STD_LAYOUT)
fig.update_yaxes(gridcolor='lightgrey')
fig.update_xaxes(gridcolor='lightgrey', zerolinecolor='grey', tickmode='array', 
                 tickvals=x_values, ticktext = x_names, dtick=1, ticks='outside', tickcolor='black', 
                 minor=dict(dtick=6, gridcolor='grey', showgrid=True),
                )
fig.write_image(os.path.join(OUTPUT_DIR, "tpc_distribution_overall.png"), scale=2)
fig.show()

# %%
scatter_data = all_notes.groupby(['dataset_name', 'tpc']).duration_qb.sum().reset_index()
fig = px.scatter(scatter_data, x='tpc', y='duration_qb', color='dataset_name', 
                 labels=dict(
                     duration_qb='duration',
                     tpc='named pitch class',
                 ),
                 category_orders=dict(dataset=chronological_dataset_names),
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
fig.write_image(os.path.join(OUTPUT_DIR, "tpc_line_per_dataset_compact.png"), scale=2)
fig.show()

# %%
px.bar(scatter_data, x='tpc', y='duration_qb', color='dataset_name', 
                 labels=dict(
                     duration_qb='duration',
                     tpc='named pitch class',
                 ),
                 category_orders=dict(dataset=chronological_dataset_names),
                 color_discrete_map=dataset_name_colors,
                 width=1000, height=500,
                )

# %%
no_accidental = bar_data[bar_data.tpc.between(-1,5)].duration_qb.sum()
with_accidental = bar_data[~bar_data.tpc.between(-1,5)].duration_qb.sum()

# %%
entire = no_accidental + with_accidental
f"Fraction of note duration without accidental of the entire durations: {no_accidental} / {entire} = {no_accidental / entire}"

# %% [raw]
# fig = make_subplots(rows=len(grouped_notes), cols=1, subplot_titles=list(grouped_notes.groups.keys()), shared_xaxes=True)
# for i, (corp, notes) in enumerate(grouped_notes, 1):
#     tpc_durations = notes.groupby('tpc').duration_qb.sum()
#     tpc_durations /= tpc_durations.sum()
#     fig.add_trace(go.Scatter(x=tpc_durations.index, y=tpc_durations, name=corp, mode='lines+markers'), row=i, col=1)
#
# #fig.update_traces(side='positive', orientation='h', width=2, points=False)
# fig.update_layout(**STD_LAYOUT, showlegend=False, height=800, width=300)
# fig.update_xaxes(gridcolor='lightgrey', zerolinecolor='lightgrey', tickmode='array', tickvals= [-12, -6, 0, 6, 12, 18],
#     ticktext = ["Dbb", "Gb", "C", "F#", "B#", "E##"],)
# fig.update_yaxes(showgrid=False, zeroline=False)
# fig.write_image(os.path.join(OUTPUT_DIR, "tpc_line_per_dataset.png"), scale=2)
# fig.show()

# %% [markdown]
# ### Notes and staves

# %%
print("Distribution of notes over staves:")
all_notes.staff.value_counts()

# %%
print("Distribution of notes over staves for all pieces with more than two staves\n")
for group, df in all_notes.groupby(level=[0,1]):
    if (df.staff > 2).any():
        print(group)
        print(df.staff.value_counts().to_dict())

# %%
all_notes[all_notes.staff > 2].groupby(level=[0,1]).staff.value_counts()
