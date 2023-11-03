# ---
# jupyter:
#   jupytext:
#     formats: md:myst,ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: revamp
#     language: python
#     name: revamp
# ---

# %% [markdown]
# # Plots for ISMIR 2023
#
# Notebook created by copying and adapting `annotations.ipynb`.

# %% mystnb={"code_prompt_hide": "Hide imports", "code_prompt_show": "Show imports"} tags=["hide-cell"]
# %load_ext autoreload
# %autoreload 2
import os
from matplotlib import pyplot as plt

import dimcat as dc
from dimcat import groupers
import ms3
import pandas as pd
import plotly.express as px
from git import Repo

from utils import STD_LAYOUT, CORPUS_COLOR_SCALE, corpus_mean_composition_years, \
  get_corpus_display_name, get_repo_name, print_heading, resolve_dir, plot_cum, value_count_df, \
    remove_none_labels, remove_non_chord_labels, plot_transition_heatmaps

# %%
from docs.notebooks.utils import DEFAULT_OUTPUT_FORMAT
from utils import OUTPUT_FOLDER
from dimcat.plotting import write_image
RESULTS_PATH = os.path.abspath(os.path.join(OUTPUT_FOLDER, "ismir"))
os.makedirs(RESULTS_PATH, exist_ok=True)
def make_output_path(filename):
    return os.path.join(RESULTS_PATH, f"{filename}{DEFAULT_OUTPUT_FORMAT}")
def save_figure_as(fig, filename, directory=RESULTS_PATH, **kwargs):
    write_image(fig, filename, directory, **kwargs)


# %% tags=["remove-output"]
package_path = resolve_dir("dcml_corpora.datapackage.json")
repo = Repo(os.path.dirname(package_path))
print_heading("Data and software versions")
print(f"Data repo '{get_repo_name(repo)}' @ {repo.commit().hexsha[:7]}")
print(f"dimcat version {dc.__version__}")
print(f"ms3 version {ms3.__version__}")
D = dc.Dataset.from_package(package_path)
D

# %%
all_metadata = D.get_metadata()
assert len(all_metadata) > 0, "No pieces selected for analysis."
mean_composition_years = corpus_mean_composition_years(all_metadata)
chronological_order = mean_composition_years.index.to_list()
corpus_colors = dict(zip(chronological_order, CORPUS_COLOR_SCALE))
corpus_names = {corp: get_corpus_display_name(corp) for corp in chronological_order}
chronological_corpus_names = list(corpus_names.values())
corpus_name_colors = {corpus_names[corp]: color for corp, color in corpus_colors.items()}

# %% [markdown]
# ## FeatureExtractor

# %%
notes = D.get_feature('notes')
notes.plot_grouped(output=make_output_path("complete_pitch_class_distribution_absolute_bars"), height=800)

# %%
labels = D.get_feature('harmonylabels')
labels.plot(output=make_output_path("fun_fun_fun"))


# %% [markdown]
# ## Slicer

# %%
def globalminor_localkey_expressed_in_globalmajor(key):
    return ms3.abs2rel_key(key, 'I', True)

def map_GLEIG_to_list(globalminor_localkeys):
    return list(map(globalminor_localkey_expressed_in_globalmajor, globalminor_localkeys))

def piecewise_localkeys_expressed_in_globalmajor(localkeys):
    gpb = localkeys.groupby(["corpus", "piece"])
    piecewise_localkeys = gpb.localkey.apply(list)
    is_major_mask = ~gpb.globalkey_is_minor.first()
    expressed_in_globalmajor = piecewise_localkeys.apply(map_GLEIG_to_list)
    return piecewise_localkeys.where(is_major_mask, expressed_in_globalmajor)

keys = D.get_feature('keyannotations')
keys.load()
piecewise_localkey_transitions = piecewise_localkeys_expressed_in_globalmajor(keys)

# %%
plot_transition_heatmaps(
  piecewise_localkey_transitions.to_list(),
  top=10,
  bottom_margin=0.05,
  left_margin=0.14)
save_pdf_path = os.path.join(RESULTS_PATH, 'localkey_transition_matrix.pdf')
plt.savefig(save_pdf_path, dpi=400)
plt.show()

# %% [markdown]
# ## Groupers

# %%
grouping = dc.Pipeline([
    groupers.CorpusGrouper(),
    groupers.ModeGrouper(),
])
GD = grouping.process(D)
grouped_keys = GD.get_feature('keyannotations')
grouped_keys_df = grouped_keys.df
grouped_keys_df

# %%
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

# %% [raw]
# to_be_filled = grouped_keys_df.quarterbeats_all_endings == ''
# grouped_keys_df.quarterbeats_all_endings = grouped_keys_df.quarterbeats_all_endings.where(~to_be_filled, grouped_keys_df.quarterbeats)
# ms3.make_interval_index_from_durations(grouped_keys_df, position_col="quarterbeats_all_endings")

# %% [markdown]
# ## Slicer

# %% tags=["hide-input"]
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

# %%
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

# %%
plot_transition_heatmaps(full_grams_major, full_grams_minor, top=20)
plt.savefig(make_output_path("bass_degree_bigrams"), dpi=400)
plt.show()

# %%
#font_dict = {'font': {'size': 20}}2
fig = plot_cum(
  all_chords.chord,
  font_size=35,
  markersize=10,
  **STD_LAYOUT)
save_figure_as(fig, 'chord_type_distribution_cumulative', )
fig.show()

# %%
grouped_chords = groupers.ModeGrouper().process(labels)
grouped_chords.get_default_groupby()

# %%
value_count_df(grouped_chords.chord)

# %%
ugs_dict = {mode: value_count_df(chords).reset_index() for mode, chords in
            grouped_chords.groupby("mode").chord}
ugs_df = pd.concat(ugs_dict, axis=1)
ugs_df.columns = ['_'.join(map(str, col)) for col in ugs_df.columns]
ugs_df.index = (ugs_df.index + 1).rename('k')
ugs_df
