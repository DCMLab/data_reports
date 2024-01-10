# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md:myst,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: revamp
#     language: python
#     name: revamp
# ---

# %% [markdown]
# # Overview
#
# This notebook gives a general overview of the features included in the dataset.

# %% mystnb={"code_prompt_hide": "Hide imports", "code_prompt_show": "Show imports"} tags=["hide-cell"]
# %load_ext autoreload
# %autoreload 2
import os
from collections import defaultdict
from typing import Dict, Iterator, List, Optional

import dimcat as dc
import ms3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dimcat import filters
from dimcat.plotting import update_figure_layout, write_image
from dimcat.utils import get_middle_composition_year
from git import Repo
from IPython.display import display
from plotly.subplots import make_subplots

import utils

# %%


RESULTS_PATH = os.path.abspath("/home/laser/git/DLC/img")
os.makedirs(RESULTS_PATH, exist_ok=True)


def save_figure_as(fig, filename, format=None, directory=RESULTS_PATH, **kwargs):
    write_image(fig, filename, format=format, directory=directory, **kwargs)


# %% [markdown]
# **Loading data**

# %%
package_path = utils.resolve_dir(
    "~/distant_listening_corpus/distant_listening_corpus.datapackage.json"
)
repo = Repo(os.path.dirname(package_path))
utils.print_heading("Data and software versions")
print(f"Data repo '{utils.get_repo_name(repo)}' @ {repo.commit().hexsha[:7]}")
print(f"dimcat version {dc.__version__}")
print(f"ms3 version {ms3.__version__}")
D = dc.Dataset.from_package(package_path)
D

# %%
filtered_D = filters.HasHarmonyLabelsFilter(keep_values=[True]).process(D)
all_metadata = filtered_D.get_metadata()
assert len(all_metadata) > 0, "No pieces selected for analysis."
all_metadata

# %%
mean_composition_years = utils.corpus_mean_composition_years(all_metadata)
chronological_order = mean_composition_years.index.to_list()
corpus_colors = dict(zip(chronological_order, utils.CORPUS_COLOR_SCALE))
corpus_names = {
    corp: utils.get_corpus_display_name(corp) for corp in chronological_order
}
chronological_corpus_names = list(corpus_names.values())
corpus_name_colors = {
    corpus_names[corp]: color for corp, color in corpus_colors.items()
}

# %%
mean_composition_years

# %% [markdown]
# ## Composition dates
#
# This section relies on the dataset's metadata.

# %%
valid_composed_start = pd.to_numeric(all_metadata.composed_start, errors="coerce")
valid_composed_end = pd.to_numeric(all_metadata.composed_end, errors="coerce")
print(
    f"Composition dates range from {int(valid_composed_start.min())} {valid_composed_start.idxmin()} "
    f"to {int(valid_composed_end.max())} {valid_composed_end.idxmax()}."
)


# %% [markdown]
# ### Mean composition years per corpus


# %%
def make_summary(metadata_df):
    piece_is_annotated = metadata_df.label_count > 0
    return metadata_df[piece_is_annotated].copy()


# %%
summary = make_summary(all_metadata)
N = len(summary)
summary

# %% [markdown]
# ### Composition years histogram

# %% tags=["hide-input"]
hist_data = summary.reset_index()
hist_data.corpus = hist_data.corpus.map(corpus_names)
fig = px.histogram(
    hist_data,
    x="composed_end",
    color="corpus",
    labels=dict(
        composed_end="decade",
        count="pieces",
    ),
    color_discrete_map=corpus_name_colors,
    title=f"Temporal coverage of the {N} annotated pieces in the Distant Listening Corpus",
)
update_figure_layout(
    fig, traces_settings=dict(xbins=dict(size=10)), legend=dict(font=dict(size=16))
)
# save_figure_as(fig, "pieces_timeline_histogram", height=1250)
fig.show()


# %%
all_subcorpora = pd.read_csv("/home/laser/git/workflow_deployment/all_subcorpora.csv")
is_public = all_subcorpora.dcml_corpora.notna()
# is_dlc = all_subcorpora.distant_listening_corpus.notna()
# is_dlc_and_unpublished = all_subcorpora[~is_public & is_dlc]
# new_corpora = set(is_dlc_and_unpublished.repo_name)
# summary["unpublished"] = summary.index.get_level_values("corpus").isin(new_corpora)
# summary["novelty"] = summary.unpublished.map({True: "new addition", False: "published elsewhere"})
published_corpora = all_subcorpora.loc[is_public, "repo_name"]
repo2published_mapper = defaultdict(lambda: "Romantic Piano Corpus")
repo2published_mapper.update(
    dict(
        ABC="Annotated Beethoven Corpus",
        corelli="Corelli Trio Sonatas",
        mozart_piano_sonatas="Mozart Piano Sonatas",
    )
)
repo2published_mapper = dict(
    zip(published_corpora, published_corpora.map(repo2published_mapper))
)
repo2publication_mapper = defaultdict(lambda: "DLC")
repo2publication_mapper.update(repo2published_mapper)
summary["first published"] = summary.index.get_level_values("corpus").map(
    repo2publication_mapper
)
hist_data = summary.reset_index().sort_values(
    "first published", key=lambda S: S.eq("DLC")
)  # DLC always last bar
hist_data.corpus = hist_data.corpus.map(corpus_names)

# %%
hist_data.head()

# %%
fig = px.histogram(
    hist_data,
    x="composed_end",
    color="first published",
    labels=dict(
        composed_end="composition year",
        count="pieces",
    ),
    color_discrete_map=corpus_name_colors,
    title=f"Temporal coverage of the {N} annotated pieces in the Distant Listening Corpus",
)
update_figure_layout(
    fig, traces_settings=dict(xbins=dict(size=10)), legend=dict(font=dict(size=16))
)

# %%
COLOR_MAPPING = dict(
    zip(
        (
            "DLC",
            "Annotated Beethoven Corpus",
            "Romantic Piano Corpus",
            "Corelli Trio Sonatas",
            "Mozart Piano Sonatas",
        ),
        px.colors.qualitative.Set1,
    )
)


def make_histogram_traces(
    hist_data,
    x: str = "composed_end",
    y: Optional[str] = None,
    color="first published",
    discrete_colors: Optional[List | Dict] = COLOR_MAPPING,
    **kwargs,
) -> Iterator[go.Histogram]:
    color_palette_is_dict = None
    if discrete_colors is not None:
        color_palette_is_dict = isinstance(discrete_colors, dict)

    def get_group_color(group, i):
        if color_palette_is_dict is None:
            return
        if color_palette_is_dict:
            return discrete_colors[group]
        return discrete_colors[i]

    for i, (group, group_df) in enumerate(hist_data.groupby(color, sort=False)):
        y_data = None if not y else group_df[y]
        trace = go.Histogram(
            x=group_df[x],
            y=y_data,
            name=group,
            marker_color=get_group_color(group, i),
            **kwargs,
        )
        yield trace


def make_single_go_histogram(
    hist_data,
    x="composed_end",
    y=None,
    color="first published",
    histfunc=None,
    discrete_colors: Optional[List | Dict] = None,
    barmode="stack",
    **kwargs,
) -> go.Figure:
    fig = go.Figure()
    for trace in make_histogram_traces(
        hist_data,
        x=x,
        y=y,
        color=color,
        histfunc=histfunc,
        discrete_colors=discrete_colors,
    ):
        fig.add_trace(trace)
    if barmode:
        fig.update_layout(barmode=barmode)
    return fig


fig = make_single_go_histogram(
    hist_data,
    xbins_start=1600,
)
update_figure_layout(
    fig,
    x_axis=dict(title_text="composition year"),
    y_axis=dict(title_text="# pieces"),
)
fig


# %%
def make_stacked_go_histograms(
    hist_data,
    y: List[Optional[str]],
    histfunc: List[Optional[str]],
    x="composed_end",
    color="first published",
    xaxis_labels: Optional[str | List[str]] = None,
    yaxis_labels: Optional[str | List[str]] = None,
    xbins=20,
    barmode="stack",
) -> go.Figure:
    n_histograms = len(y)
    assert len(histfunc) == n_histograms
    fig = make_subplots(rows=n_histograms)
    for histogram_i, (y, histfunc) in enumerate(zip(y, histfunc), start=1):
        showlegend = histogram_i == n_histograms  # show legend for last histogram
        for trace_group, trace in enumerate(
            make_histogram_traces(
                hist_data,
                x=x,
                y=y,
                histfunc=histfunc,
                color=color,
                showlegend=showlegend,
            )
        ):
            trace["legendgroup"] = f"group{trace_group}"
            fig.append_trace(trace, histogram_i, 1)
    if barmode:
        fig.update_layout(barmode=barmode)
    if xbins:
        fig.update_traces(xbins=dict(size=xbins))
    if xaxis_labels:
        if isinstance(xaxis_labels, str):
            fig.update_xaxes(title_text=xaxis_labels, row=n_histograms)
        else:
            assert len(xaxis_labels) == n_histograms
            for row, label in enumerate(xaxis_labels, start=1):
                fig.update_xaxes(title_text=label, row=row)
    if yaxis_labels:
        if isinstance(yaxis_labels, str):
            fig.update_yaxes(title_text=yaxis_labels, row=n_histograms)
        else:
            assert len(yaxis_labels) == n_histograms
            for row, label in enumerate(yaxis_labels, start=1):
                fig.update_yaxes(title_text=label, row=row)
    return fig


fig = make_stacked_go_histograms(
    hist_data,
    y=[None, "n_onsets", "label_count"],
    histfunc=[None, "sum", "sum"],
    xaxis_labels="composition year",
    yaxis_labels=["# pieces", "# notes", "# annotation labels"],
)
update_figure_layout(
    fig,
    title_text="Size of the Distant Listening Corpus",
    traces_settings=dict(xbins=dict(size=25, start=1575)),
    legend=dict(
        title_text="Data report",
        traceorder="reversed",
        itemsizing="constant",
        xanchor="right",
        bgcolor="lightgray",
        bordercolor="gray",
        borderwidth=2,
        x=1.1,
        y=1.1,
        # orientation="h",  # show entries horizontally
        # xanchor="center",  # use center of legend as anchor
        # x=0.5,
        # y=-0.1,
    ),
    font_size=25,
)
save_figure_as(fig, "corpus_size", height=1200, width=1440, format="pdf")
fig

# %% [markdown]
# ## Dimensions
#
# ### Overview


# %%
def piece_order(_):
    return get_middle_composition_year(all_metadata)


all_metadata.sort_values("composed_start", key=piece_order)


# %%
def make_overview_table(groupby, group_name="pieces"):
    n_groups = groupby.size().rename(group_name)
    absolute_numbers = dict(
        measures=groupby.last_mn.sum(),
        length=groupby.length_qb.sum(),
        notes=groupby.n_onsets.sum(),
        labels=groupby.label_count.sum(),
    )
    absolute = pd.DataFrame.from_dict(absolute_numbers)
    absolute = pd.concat([n_groups, absolute], axis=1)
    sum_row = pd.DataFrame(absolute.sum(), columns=["sum"]).T
    absolute = pd.concat([absolute, sum_row])
    return absolute


absolute = make_overview_table(summary.groupby("workTitle"))
# print(absolute.astype(int).to_markdown())
absolute.astype(int)

# %%
public = dc.Dataset.from_package(
    "/home/laser/git/meta_repositories/dcml_corpora/dcml_corpora.datapackage.json"
)
public


# %%
def summarize_dataset(D):
    all_metadata = D.get_metadata()
    summary = make_summary(all_metadata)
    return make_overview_table(summary.groupby(level=0))


dcml_corpora = summarize_dataset(public)
print(dcml_corpora.astype(int).to_markdown())

# %%
distant_listening = summarize_dataset(D)
print(distant_listening.astype(int).to_markdown())

# %% [markdown]
# ### Measures

# %%
all_measures = D.get_feature("measures").df
print(
    f"{len(all_measures.index)} measures over {len(all_measures.groupby(level=[0,1]))} files."
)
all_measures.head()

# %%
print("Distribution of time signatures per XML measure (MC):")
all_measures.timesig.value_counts(dropna=False)

# %% [markdown]
# ### Harmony labels
#
# All symbols, independent of the local key (the mode of which changes their semantics).

# %%
try:
    all_annotations = D.get_feature("harmonylabels").df
except Exception:
    all_annotations = pd.DataFrame()
n_annotations = len(all_annotations.index)
includes_annotations = n_annotations > 0
if includes_annotations:
    display(all_annotations.head())
    print(f"Concatenated annotation tables contains {all_annotations.shape[0]} rows.")
    no_chord = all_annotations.root.isna()
    if no_chord.sum() > 0:
        print(
            f"{no_chord.sum()} of them are not chords. Their values are:"
            f" {all_annotations.label[no_chord].value_counts(dropna=False).to_dict()}"
        )
    all_chords = all_annotations[~no_chord].copy()
    print(
        f"Dataset contains {all_chords.shape[0]} tokens and {len(all_chords.chord.unique())} types over "
        f"{len(all_chords.groupby(level=[0,1]))} documents."
    )
    all_annotations["corpus_name"] = all_annotations.index.get_level_values(0).map(
        utils.get_corpus_display_name
    )
    all_chords["corpus_name"] = all_chords.index.get_level_values(0).map(
        utils.get_corpus_display_name
    )
else:
    print("Dataset contains no annotations.")
