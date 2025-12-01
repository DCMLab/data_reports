# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
#     formats: ipynb,md:myst,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: revamp
#     language: python
#     name: revamp
# ---

# %% [markdown]
# # Annotations

# %% mystnb={"code_prompt_hide": "Hide imports", "code_prompt_show": "Show imports"} tags=["hide-input"]
# %load_ext autoreload
# %autoreload 2

import os

import dimcat as dc
import ms3
import plotly.express as px
from dimcat import groupers, plotting, slicers

import utils

# %% editable=true slideshow={"slide_type": ""} tags=["hide-input"]
RESULTS_PATH = os.path.abspath(os.path.join(utils.OUTPUT_FOLDER, "annotations"))
os.makedirs(RESULTS_PATH, exist_ok=True)


def make_output_path(
    filename: str,
    extension=None,
    path=RESULTS_PATH,
) -> str:
    return utils.make_output_path(filename=filename, extension=extension, path=path)


def save_figure_as(
    fig, filename, formats=("png", "pdf"), directory=RESULTS_PATH, **kwargs
):
    if formats is not None:
        for fmt in formats:
            plotting.write_image(fig, filename, directory, format=fmt, **kwargs)
    else:
        plotting.write_image(fig, filename, directory, **kwargs)


# %% editable=true slideshow={"slide_type": ""} tags=["hide-input"]
D = dc.get_dataset("couperin_concerts")

# %% editable=true slideshow={"slide_type": ""} tags=["hide-input"]
package = D.inputs.get_package()
package_info = package._package.custom
git_tag = package_info.get("git_tag")
utils.print_heading("Data and software versions")
print(f"datapackage version: {package.package_name} {git_tag}")
print(f"dimcat version {dc.__version__}")

# %% editable=true slideshow={"slide_type": ""}
filtered_D = D.apply_step("HasHarmonyLabelsFilter")
all_metadata = filtered_D.get_metadata()

# %%
assert len(all_metadata) > 0, "No pieces selected for analysis."
chronological_corpus_names = all_metadata.get_corpus_names()

# %% [markdown]
# ## DCML harmony labels

# %% tags=["hide-input"]
all_annotations = filtered_D.get_feature("DcmlAnnotations")
is_annotated_mask = all_metadata.label_count > 0
is_annotated_index = all_metadata.index[is_annotated_mask]
annotated_notes = filtered_D.get_feature("notes").subselect(is_annotated_index)
print(f"The annotated pieces have {len(annotated_notes)} notes.")

# %%
all_chords = filtered_D.get_feature("harmonylabels")
all_chords.subselect([("couperin_concerts", "c03n06_musette_1")])

# %%
print(
    f"{len(all_annotations)} annotations, of which {len(all_chords)} are harmony labels."
)

# %% [markdown]
# ## Harmony labels
# ### Unigrams
# For computing unigram statistics, the tokens need to be grouped by their occurrence within a major or a minor key
# because this changes their meaning. To that aim, the annotated corpus needs to be sliced into contiguous localkey
# segments which are then grouped into a major (`is_minor=False`) and a minor group.

# %%
root_durations = (
    all_chords[all_chords.root.between(-5, 6)]
    .groupby(["root", "chord_type"])
    .duration_qb.sum()
)
# sort by stacked bar length:
# root_durations = root_durations.sort_values(key=lambda S: S.index.get_level_values(0).map(S.groupby(level=0).sum()),
# ascending=False)
bar_data = root_durations.reset_index()
bar_data.root = bar_data.root.map(ms3.fifths2iv)
fig = px.bar(
    bar_data,
    x="root",
    y="duration_qb",
    color="chord_type",
    title="Distribution of chord types over chord roots",
    labels=dict(
        root="Chord root expressed as interval above the local (or secondary) tonic",
        duration_qb="duration in quarter notes",
        chord_type="chord type",
    ),
)
fig.update_layout(**utils.STD_LAYOUT)
save_figure_as(fig, "chord_type_distribution_over_scale_degrees_absolute_stacked_bars")
fig.show()

# %%
relative_roots = all_chords[
    ["numeral", "duration_qb", "relativeroot", "localkey_is_minor", "chord_type"]
].copy()
relative_roots["relativeroot_resolved"] = ms3.transform(
    relative_roots, ms3.resolve_relative_keys, ["relativeroot", "localkey_is_minor"]
)
has_rel = relative_roots.relativeroot_resolved.notna()
relative_roots.loc[has_rel, "localkey_is_minor"] = relative_roots.loc[
    has_rel, "relativeroot_resolved"
].str.islower()
relative_roots["root"] = ms3.transform(
    relative_roots, ms3.roman_numeral2fifths, ["numeral", "localkey_is_minor"]
)
chord_type_frequency = all_chords.chord_type.value_counts()
replace_rare = ms3.map_dict(
    {t: "other" for t in chord_type_frequency[chord_type_frequency < 500].index}
)
relative_roots["type_reduced"] = relative_roots.chord_type.map(replace_rare)
# is_special = relative_roots.chord_type.isin(('It', 'Ger', 'Fr'))
# relative_roots.loc[is_special, 'root'] = -4

# %%
root_durations = (
    relative_roots.groupby(["root", "type_reduced"])
    .duration_qb.sum()
    .sort_values(ascending=False)
)
bar_data = root_durations.reset_index()
bar_data.root = bar_data.root.map(ms3.fifths2iv)
root_order = (
    bar_data.groupby("root")
    .duration_qb.sum()
    .sort_values(ascending=False)
    .index.to_list()
)
fig = px.bar(
    bar_data,
    x="root",
    y="duration_qb",
    color="type_reduced",
    barmode="group",
    log_y=True,
    color_discrete_map=utils.TYPE_COLORS,
    category_orders=dict(
        root=root_order,
        type_reduced=relative_roots.type_reduced.value_counts().index.to_list(),
    ),
    labels=dict(
        root="intervallic difference between chord root to the local or secondary tonic",
        duration_qb="duration in quarter notes",
        type_reduced="chord type",
    ),
    width=1000,
    height=400,
)
fig.update_layout(
    **utils.STD_LAYOUT,
    legend=dict(
        orientation="h",
        xanchor="right",
        x=1,
        y=1,
    ),
)
save_figure_as(fig, "chord_type_distribution_over_scale_degrees_absolute_grouped_bars")
fig.show()

# %%
print(
    f"Reduced to {len(set(bar_data.iloc[:,:2].itertuples(index=False, name=None)))} types. "
    f"Paper cites the sum of types in major and types in minor (see below), treating them as distinct."
)

# %%
dim_or_aug = bar_data[
    bar_data.root.str.startswith("a") | bar_data.root.str.startswith("d")
].duration_qb.sum()
complete = bar_data.duration_qb.sum()
print(
    f"On diminished or augmented scale degrees: {dim_or_aug} / {complete} = {dim_or_aug / complete}"
)

# %%
chords_by_mode = groupers.ModeGrouper().process(all_chords)
chords_by_mode.format = "scale_degree"

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# #### Whole dataset

# %%
unigram_proportions = chords_by_mode.get_default_analysis()
unigram_proportions.make_ranking_table()

# %%
chords_by_mode.apply_step("Counter")

# %%
chords_by_mode.format = "scale_degree"
chords_by_mode.get_default_analysis().make_ranking_table()

# %%
unigram_proportions.plot_grouped()

# %% [raw]
# k = 20
# modes = {True: 'MINOR', False: 'MAJOR'}
# for (is_minor,), ugs in unigram_proportions.iter():
#     print(f"TOP {k} {modes[is_minor]} UNIGRAMS\n{ugs.shape[0]} types, {ugs.sum()} tokens")
#     print(ugs.head(k).to_string())

# %% [raw]
# ugs_dict = {modes[is_minor].lower(): (ugs/ugs.sum() * 100).round(2).rename('%').reset_index() for (is_minor,),
# ugs in unigram_proportions.iter()}
# ugs_df = pd.concat(ugs_dict, axis=1)
# ugs_df.columns = ['_'.join(map(str, col)) for col in ugs_df.columns]
# ugs_df.index = (ugs_df.index + 1).rename('k')
# print(ugs_df.iloc[:50].to_markdown())

# %% [raw]
# #### Per corpus

# %% [raw]
# corpus_wise_unigrams = dc.Pipeline([
# dc.CorpusGrouper(),
# dc.ChordSymbolUnigrams(once_per_group=True)]).process(mode_slices)

# %% [raw]
# corpus_wise_unigrams.get()

# %% [raw]
# for (is_minor, corpus_name), ugs in corpus_wise_unigrams.iter():
#     print(f"{corpus_name} {modes[is_minor]} unigrams ({ugs.shape[0]} types, {ugs.sum()} tokens)")
#     print(ugs.head(5).to_string())

# %% [raw]
# types_shared_between_corpora = {}
# for (is_minor, corpus_name), ugs in corpus_wise_unigrams.iter():
#     if is_minor in types_shared_between_corpora:
#         types_shared_between_corpora[is_minor] = types_shared_between_corpora[is_minor].intersection(ugs.index)
#     else:
#         types_shared_between_corpora[is_minor] = set(ugs.index)
# types_shared_between_corpora = {k: sorted(v, key=lambda x: unigram_proportions.get()[(k, x)], reverse=True)
# for k, v in types_shared_between_corpora.items()}
# n_types = {k: len(v) for k, v in types_shared_between_corpora.items()}
# print(f"Chords which occur in all corpora, sorted by descending global frequency:\n{types_shared_between_corpora}\n
# Counts: {n_types}")

# %% [raw]
# #### Per piece

# %% [raw]
# piece_wise_unigrams = dc.Pipeline([
# dc.PieceGrouper(),
# dc.ChordSymbolUnigrams(once_per_group=True)]).process(mode_slices)

# %% [raw]
# piece_wise_unigrams.get()

# %% [raw]
# types_shared_between_pieces = {}
# for (is_minor, corpus_name), ugs in piece_wise_unigrams.iter():
#     if is_minor in types_shared_between_pieces:
#         types_shared_between_pieces[is_minor] = types_shared_between_pieces[is_minor].intersection(ugs.index)
#     else:
#         types_shared_between_pieces[is_minor] = set(ugs.index)
# print(types_shared_between_pieces)

# %% [markdown]
# ### Bigrams

# %% [markdown]
# #### Tone profiles for all major and minor local keys

# %% editable=true slideshow={"slide_type": ""}
key_slicer = slicers.KeySlicer()
keys_segmented = key_slicer.process(D)
notes = keys_segmented.get_feature("Notes")
notes

# %% editable=true slideshow={"slide_type": ""}
keys = key_slicer.slice_metadata

# %% editable=true slideshow={"slide_type": ""}
keys = keys[keys.columns.difference(notes.columns)]
notes_joined_with_keys = notes.join(keys, on=keys.index.names)
notes_by_keys_transposed = ms3.transpose_notes_to_localkey(notes_joined_with_keys)
mode_tpcs = (
    notes_by_keys_transposed.reset_index(drop=True)
    .groupby(["localkey_is_minor", "tpc"])
    .duration_qb.sum()
    .reset_index(-1)
    .sort_values("tpc")
    .reset_index()
)
mode_tpcs["sd"] = ms3.fifths2sd(mode_tpcs.tpc)
mode_tpcs["duration_pct"] = mode_tpcs.groupby(
    "localkey_is_minor", group_keys=False
).duration_qb.apply(lambda S: S / S.sum())
mode_tpcs["mode"] = mode_tpcs.localkey_is_minor.map({False: "major", True: "minor"})

# %% editable=true slideshow={"slide_type": ""}
# mode_tpcs = mode_tpcs[mode_tpcs['duration_pct'] > 0.001]
# sd_order = ['b1', '1', '#1', 'b2', '2', '#2', 'b3', '3', 'b4', '4', '#4', '##4', 'b5', '5', '#5', 'b6','6', '#6',
# 'b7', '7']
legend = dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
fig = px.bar(
    mode_tpcs,
    x="tpc",
    y="duration_pct",
    title="Scale degree distribution over major and minor segments",
    color="mode",
    barmode="group",
    labels=dict(
        duration_pct="normalized duration",
        tpc="Notes transposed to the local key, as major-scale degrees",
    ),
    # log_y=True,
    # category_orders=dict(sd=sd_order)
)
# fig.update_layout(**utils.STD_LAYOUT, legend=legend)
# fig.update_xaxes(tickmode="array", tickvals=mode_tpcs.tpc, ticktext=mode_tpcs.sd)
# save_figure_as(fig, "scale_degree_distributions_maj_min_normalized_bars", height=600)
fig.show()
