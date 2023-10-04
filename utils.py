import os
import re
from collections import Counter, defaultdict
from functools import cache
from typing import List, Optional

import colorlover
import frictionless as fl
import ms3
import numpy as np

# import modin.pandas as pd
import pandas as pd
import plotly.express as px
import seaborn as sns
from git import Repo
from IPython.display import display
from kaleido.scopes.plotly import PlotlyScope
from matplotlib import gridspec as gridspec
from matplotlib import pyplot as plt
from plotly import graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from scipy.stats import entropy

OUTPUT_FOLDER = os.path.abspath("outputs")
DEFAULT_OUTPUT_FORMAT = ".png"
AVAILABLE_FIGURE_FORMATS = PlotlyScope._all_formats
CORPUS_COLOR_SCALE = px.colors.qualitative.D3
COLOR_SCALE_SETTINGS = dict(
    color_continuous_scale="RdBu_r", color_continuous_midpoint=2
)
TPC_DISCRETE_COLOR_MAP = dict(zip(range(-15, 20), sample_colorscale("RdBu_r", 35)))
STD_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin={"l": 40, "r": 0, "b": 0, "t": 80, "pad": 0},
    font={"size": 25},
    xaxis=dict(gridcolor="lightgrey"),
    yaxis=dict(gridcolor="lightgrey"),
)

CADENCE_COLORS = dict(
    zip(("HC", "PAC", "PC", "IAC", "DC", "EC"), colorlover.scales["6"]["qual"]["Set1"])
)
CORPUS_COLOR_SCALE = px.colors.qualitative.D3

CORPUS_NAMES = {
    "ABC": "Beethoven String Quartets",
    "bach_en_fr_suites": "Bach Suites",
    "bach_solo": "Bach Solo",
    "bartok_bagatelles": "Bartok Bagatelles",
    "beethoven_piano_sonatas": "Beethoven Sonatas",
    "c_schumann_lieder": "C Schumann Lieder",
    "chopin_mazurkas": "Chopin Mazurkas",
    "corelli": "Corelli Trio Sonatas",
    "couperin_clavecin": "Couperin Clavecin",
    "couperin_concerts": "Couperin Concerts Royaux",
    "cpe_bach_keyboard": "CPE Bach Keyboard",
    "debussy_suite_bergamasque": "Debussy Suite Bergamasque",
    "dvorak_silhouettes": "Dvořák Silhouettes",
    "frescobaldi_fiori_musicali": "Frescobaldi Fiori Musicali",
    "gastoldi_baletti": "Gastoldi Baletti",
    "grieg_lyric_pieces": "Grieg Lyric Pieces",
    "handel_keyboard": "Handel Keyboard",
    "jc_bach_sonatas": "JC Bach Sonatas",
    "kleine_geistliche_konzerte": "Schütz Kleine Geistliche Konzerte",
    "kozeluh_sonatas": "Kozeluh Sonatas",
    "liszt_pelerinage": "Liszt Années",
    "mahler_kindertotenlieder": "Mahler Kindertotenlieder",
    "medtner_tales": "Medtner Tales",
    "mendelssohn_quartets": "Mendelssohn Quartets",
    "monteverdi_madrigals": "Monteverdi Madrigals",
    "mozart_piano_sonatas": "Mozart Piano Sonatas",
    "pergolesi_stabat_mater": "Pergolesi Stabat Mater",
    "peri_euridice": "Peri Euridice",
    "pleyel_quartets": "Pleyel Quartets",
    "poulenc_mouvements_perpetuels": "Poulenc Mouvements Perpetuels",
    "rachmaninoff_piano": "Rachmaninoff Piano",
    "ravel_piano": "Ravel Piano",
    "scarlatti_sonatas": "Scarlatti Sonatas",
    "schubert_dances": "Schubert Dances",
    "schubert_winterreise": "Schubert Winterreise",
    "schulhoff_suite_dansante_en_jazz": "Schulhoff Suite Dansante En Jazz",
    "schumann_kinderszenen": "R Schumann Kinderszenen",
    "schumann_liederkreis": "R Schumann Liederkreis",
    "sweelinck_keyboard": "Sweelinck Keyboard",
    "tchaikovsky_seasons": "Tchaikovsky Seasons",
    "wagner_overtures": "Wagner Overtures",
    "wf_bach_sonatas": "WF Bach Sonatas",
}

TRACES_SETTINGS = dict(marker_line_color="black")
TYPE_COLORS = dict(
    zip(
        ("Mm7", "M", "o7", "o", "mm7", "m", "%7", "MM7", "other"),
        colorlover.scales["9"]["qual"]["Paired"],
    )
)
X_AXIS = dict(gridcolor="lightgrey", zerolinecolor="grey")
Y_AXIS = dict()

COLUMN2SUNBURST_TITLE = dict(
    sd="bass degree",
    figbass="figured bass",
    interval="bass progression",
    following_figbass="subsequent figured bass",
)


def cnt(
    S: pd.Series,
    interval: int | List[int],
    k_min: int = 1,
    include_zero: bool = True,
    df: bool = True,
):
    """Count subsequent occurrences of one or several numbers in a sequence.

    Parameters
    ----------
    S : pd.Series
    interval: int or list
    k_min : int
        Minimal sequence length to take into account, defaults to 1
    include_zero : bool
        By default, zero is always accepted as part of the sequence. Zeros never increase
        sequence length.
    df : bool
        Defaults to True, so the function returns a DataFrame with index segments and sequence lengths.
        Pass False to return a list of index segments only.
    """
    try:
        interval_list = [int(interval)]
        if include_zero:
            interval_list.append(0)
    except Exception:
        interval_list = interval
        if include_zero and 0 not in interval_list:
            interval_list.append(0)

    ix_chunks = pd.DataFrame(columns=["ixs", "n"]) if df else []
    current = []
    n = 0
    s = pd.concat([S, pd.Series([pd.NA])])  # so that else is executed in the end
    for i, iv in s.items():
        if not pd.isnull(iv) and iv in interval_list:
            current.append(i)
            if iv != 0:
                n += 1
        else:
            if n >= k_min:
                if df:
                    ix_chunks.loc[len(ix_chunks)] = (current, n)
                else:
                    ix_chunks.append((current, n))
            current = [i]
            n = 0
    return ix_chunks


def color_background(x, color="#ffffb3"):
    """Format DataFrame cells with given background color."""
    return np.where(x.notna().to_numpy(), f"background-color: {color};", None)


def corpus_mean_composition_years(
    df: pd.DataFrame, year_column: str = "composed_end"
) -> pd.Series:
    """Expects a dataframe containing ``year_column`` and computes its means by grouping on the first index level
    ('corpus' by default).
    Returns the result as a series where the index contains corpus names and the values are mean composition years.
    """
    years = pd.to_numeric(df[year_column], errors="coerce")
    return years.groupby(level=0).mean().sort_values()


def chronological_corpus_order(
    df: pd.DataFrame, year_column: str = "composed_end"
) -> List[str]:
    """Expects a dataframe containing ``year_column`` and corpus names in the first index level.
    Returns the corpus names in chronological order
    """
    mean_composition_years = corpus_mean_composition_years(
        df=df, year_column=year_column
    )
    return mean_composition_years.index.to_list()


def cumulative_fraction(S, start_from_zero=False):
    """Accumulate the value counts of a Series so they can be plotted."""
    values_df = S.value_counts().to_frame("x").reset_index()
    total = values_df.x.sum()
    values_df["y"] = values_df.x.cumsum() / total
    if start_from_zero:
        return pd.concat(
            [pd.DataFrame({"chord": pd.NA, "x": 0, "y": 0.0}, index=[0]), values_df],
            ignore_index=True,
        )
    return values_df


def fifths_bar_plot(
    bar_data,
    x_col="tpc",
    y_col="duration_qb",
    labels=None,
    title="Pitch-class distribution",
    fifth_transform=ms3.fifths2name,
    shift_color_midpoint=2,
    showlegend=False,
    width=1500,
    height=400,
    output=None,
    **kwargs,
):
    """bar_data with x_col ('tpc'), y_col ('duration_qb')"""

    color_values = list(bar_data[x_col])
    if labels is None:
        labels = {str(x_col): "Tonal pitch class", str(y_col): "Duration in ♩"}
    fig = px.bar(
        bar_data,
        x=x_col,
        y=y_col,
        title=title,
        labels=labels,
        color=color_values,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=shift_color_midpoint,
        width=width,
        height=height,
        **kwargs,
    )
    x_values = list(set(color_values))
    x_names = list(map(fifth_transform, x_values))
    fig.update_coloraxes(showscale=False)
    fig.update_layout(**STD_LAYOUT, showlegend=showlegend)
    fig.update_yaxes(gridcolor="lightgrey")
    fig.update_xaxes(
        gridcolor="lightgrey",
        zerolinecolor="grey",
        tickmode="array",
        tickvals=x_values,
        ticktext=x_names,
        dtick=1,
        ticks="outside",
        tickcolor="black",
        minor=dict(dtick=6, gridcolor="grey", showgrid=True),
    )
    if output is not None:
        fig.write_image(output)
    return fig


def frictionless_field2modin_dtype(name, _) -> Optional[str]:
    category_fields = [  # often recurring string values
        "act_dur",
        "corpus",
        "duration",
        "gracenote",
        "mc_offset",
        "mc_onset",
        "mn_onset",
        "name",
        "nominal_duration",
        "piece",
        "scalar",
        "timesig",
        "tremolo",
    ]
    string_fields = [  # mostly distinct string values
        "next",
        "quarterbeats",  # (mostly) unique fractions
    ]
    int_fields = [  # sparse integer columns (many NA values)
        "numbering_offset",
        "tied",
        "tpc",
        "volta",
    ]
    boolean_fields = [
        "dont_count",
        "globalkey_is_minor",
        "localkey_is_minor",
    ]
    # left for inference, no NA values:
    # chord_id: int
    # duration_qb: float
    # i: int
    # keysig: int
    # mc: int
    # midi: int
    # mn: int
    # octave: int
    # staff: int
    # tpc: int
    # voice: int
    if name in category_fields:
        return "category"
    if name in string_fields:
        return "string"
    if name in int_fields:
        return "Int64"
    if name in boolean_fields:
        return "boolean"
    # print(f"{name} ({dtype}): infer")


def frictionless_types2modin_types(schema):
    return {
        name: outcome
        for name, dtype in schema.items()
        if (outcome := frictionless_field2modin_dtype(name, dtype))
    }


@cache
def get_corpus_display_name(repo_name: str) -> str:
    """Looks up a repository name in the CORPUS_NAMES constant. If not present,
    the repo name is returned as title case.
    """
    name = CORPUS_NAMES.get(repo_name, "")
    if name == "":
        name = " ".join(s.title() for s in repo_name.split("_"))
    return name


def get_middle_composition_year(
    metadata: pd.DataFrame,
    composed_start_column: str = "composed_start",
    composed_end_column: str = "composed_end",
) -> pd.Series:
    """Returns the middle of the composition year range."""
    composed_start = pd.to_numeric(metadata[composed_start_column], errors="coerce")
    composed_end = pd.to_numeric(metadata[composed_end_column], errors="coerce")
    composed_start.fillna(composed_end, inplace=True)
    composed_end.fillna(composed_start, inplace=True)
    return (composed_start + composed_end) / 2


def get_modin_dtypes(path):
    descriptor_path = os.path.join(path, "all_subcorpora.datapackage.json")
    fl_package = fl.Package(descriptor_path)
    facet_schemas = {}
    for fl_resource in fl_package.resources:
        _, facet_name = fl_resource.name.split(".")
        facet_schemas[facet_name] = fl_resource.schema
    facet_schema_types = {
        facet_name: {field.name: field.type for field in fl_schema.fields}
        for facet_name, fl_schema in facet_schemas.items()
    }
    modin_dtypes = {
        facet_name: frictionless_types2modin_types(fl_types)
        for facet_name, fl_types in facet_schema_types.items()
    }
    return modin_dtypes


def get_pitch_class_distribution(
    df,
    pitch_column="tpc",
    duration_column="duration_qb",
):
    return (
        df.groupby(pitch_column)[duration_column].sum().to_frame(name=duration_column)
    )


def get_repo_name(repo: Repo) -> str:
    """Gets the repo name from the origin's URL, or from the local path if there is None."""
    if isinstance(repo, str):
        repo = Repo(repo)
    if len(repo.remotes) == 0:
        return repo.git.rev_parse("--show-toplevel")
    remote = repo.remotes[0]
    return remote.url.split(".git")[0].split("/")[-1]


def grams(lists_of_symbols, n=2):
    """Returns a list of n-gram tuples for given list. List can be nested.

    Use nesting to exclude transitions between pieces or other units.

    """
    if nest_level(lists_of_symbols) > 1:
        ngrams = []
        no_sublists = []
        for item in lists_of_symbols:
            if isinstance(item, list):
                ngrams.extend(grams(item, n))
            else:
                no_sublists.append(item)
        if len(no_sublists) > 0:
            ngrams.extend(grams(no_sublists, n))
        return ngrams
    else:
        # if len(l) < n:
        #    print(f"{l} is too small for a {n}-gram.")
        # ngrams = [l[i:(i+n)] for i in range(len(l)-n+1)]
        ngrams = list(zip(*(lists_of_symbols[i:] for i in range(n))))
        # convert to tuple of strings
        return [tuple(str(g) for g in gram) for gram in ngrams]


def load_facets(
    path,
    suffix="",
):
    modin_types = get_modin_dtypes(path)
    facets = {}
    for file in os.listdir(path):
        facet_regex = "^all_subcorpora" + suffix + r"\.(.+)\.tsv$"
        facet_match = re.match(facet_regex, file)
        if not facet_match:
            continue
        facet_name = facet_match.group(1)
        facet_path = os.path.join(path, file)
        # if facet_name == "metadata":
        #     index_col = [0, 1]
        # else:
        #     index_col = [0, 1, 2]
        dtypes = modin_types[facet_name]
        facet_df = pd.read_csv(
            facet_path,
            sep="\t",
            # index_col=index_col,
            dtype=dtypes,
        )
        facets[facet_name] = facet_df
    return facets


def make_sunburst(chords, mode, inspect=False):
    in_scale = []
    sd2prog = defaultdict(Counter)
    for sd, sd_prog in chords[["sd", "sd_progression"]].itertuples(index=False):
        if len(sd) == 1:
            in_scale.append(sd)
            sd2prog[sd].update(["∎"] if pd.isnull(sd_prog) else [str(sd_prog)])
    label_counts = Counter(in_scale)
    labels, values = list(label_counts.keys()), list(label_counts.values())
    # labels, values = zip(*list((sd, label_counts[sd]) for sd in sorted(label_counts)))
    parents = [mode] * len(labels)
    labels = [mode] + labels
    parents = [""] + parents
    values = [len(chords)] + values
    # print(sd2prog)
    if inspect:
        print(len(labels), len(parents), len(values))
        for scad, prog_counts in sd2prog.items():
            for prog, cnt in prog_counts.most_common():
                labels.append(prog)
                parents.append(scad)
                values.append(cnt)
                if cnt < 3000:
                    break
                print(f"added {prog}, {scad}, {cnt}")
            break

    fig = go.Figure(
        go.Sunburst(labels=labels, parents=parents, values=values, branchvalues="total")
    )
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    return fig


def nest_level(obj, include_tuples=False):
    """Recursively calculate the depth of a nested list."""
    if obj.__class__ != list:
        if include_tuples:
            if obj.__class__ != tuple:
                return 0
        else:
            return 0
    max_level = 0
    for item in obj:
        max_level = max(max_level, nest_level(item, include_tuples=include_tuples))
    return max_level + 1


def plot_bigram_tables(
    major_unigrams,
    minor_unigrams,
    major_bigrams,
    minor_bigrams,
    top,
    two_col_width=1,
    frequencies=False,
):
    if isinstance(major_unigrams, pd.core.frame.DataFrame):
        major_unigrams = major_unigrams.iloc[:, 0]
    if isinstance(minor_unigrams, pd.core.frame.DataFrame):
        minor_unigrams = minor_unigrams.iloc[:, 0]
    # set custom context for this plot
    with plt.rc_context(
        {
            # disable spines for entropy bars
            "axes.spines.top": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.spines.right": False,
            "font.family": "sans-serif",
        }
    ):
        # settings for margins etc.
        barsize = [0.0, 0.7]
        gridspec_ratio = [0.25, 2.0]
        top_margin = 0.99
        bottom_margin = 0.12
        hspace = None
        wspace = 0.0
        right_margin = 0.005
        left_margin = 0.085

        fig = plt.figure(figsize=(two_col_width, two_col_width * 0.5))

        # ## MAJOR BIGRAMS

        gs1 = gridspec.GridSpec(1, 2, width_ratios=gridspec_ratio)
        gs1.update(
            left=left_margin,
            right=0.5 - right_margin,
            wspace=wspace,
            hspace=hspace,
            bottom=bottom_margin,
            top=top_margin,
        )

        ax1 = plt.subplot(gs1[0, 0])

        # vmin = 0
        # vmax = 5

        s_maj = pd.Series(
            (
                major_bigrams.apply(lambda x: entropy(x, base=2), axis=1)
                / np.log2(major_bigrams.shape[0])
            )[:top].values,
            index=[
                i + f" ({str(round(fr*100, 1))})" if frequencies else i
                for i, fr in zip(major_bigrams.index[:top], major_unigrams.values[:top])
            ],
        )
        ax = s_maj.plot(kind="barh", ax=ax1, color="k")

        # create a list to collect the plt.patches data
        totals_maj = []
        # find the values and append to list
        for i in ax.patches:
            totals_maj.append(round(i.get_width(), 2))

        for i, p in enumerate(ax.patches):
            # entropy values
            ax1.text(
                totals_maj[i] - 0.01,
                p.get_y() + 0.3,
                f"${totals_maj[i]}$",
                color="w",
                fontsize=4,
                verticalalignment="center",
                horizontalalignment="left",
            )
        ax1.set_xlim(barsize)
        ax1.invert_yaxis()
        ax1.invert_xaxis()
        ax1.set_xticklabels([])
        ax1.tick_params(
            axis="both",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False,
            bottom=False,
            labelleft=True,
        )

        ax2 = plt.subplot(gs1[0, 1])

        sns.heatmap(
            major_bigrams[major_bigrams > 0].iloc[
                :top, :top
            ],  # only display non-zero values
            annot=True,
            fmt=".1f",
            cmap="Blues",
            ax=ax2,
            # vmin=vmin,
            # vmax=vmax,
            annot_kws={"fontsize": 6.5, "rotation": 60},
            cbar=False,
        )
        ax2.set_yticks([])
        ax2.tick_params(bottom=False)

        # ## MINOR BIGRAMS

        gs2 = gridspec.GridSpec(1, 2, width_ratios=gridspec_ratio)
        gs2.update(
            left=0.5 + left_margin,
            right=1.0 - right_margin,
            wspace=wspace,
            hspace=hspace,
            bottom=bottom_margin,
            top=top_margin,
        )

        ax3 = plt.subplot(gs2[0, 0])

        s_min = pd.Series(
            (
                minor_bigrams.apply(lambda x: entropy(x, base=2), axis=1)
                / np.log2(minor_bigrams.shape[0])
            )[:top].values,
            index=[
                i + f" ({str(round(fr*100, 1))})" if frequencies else i
                for i, fr in zip(minor_bigrams.index, minor_unigrams[:top].values)
            ],
        )
        ax = s_min.plot(kind="barh", ax=ax3, color="k")

        # create a list to collect the plt.patches data
        totals_min = []

        # find the values and append to list
        for i in ax.patches:
            totals_min.append(round(i.get_width(), 2))

        for i, p in enumerate(ax.patches):
            ax3.text(
                totals_min[i] - 0.01,
                p.get_y() + 0.3,
                f"${totals_min[i]}$",
                color="w",
                fontsize=4,
                verticalalignment="center",
                horizontalalignment="left",
            )
        ax3.set_xlim(barsize)

        ax3.invert_yaxis()
        ax3.invert_xaxis()
        ax3.set_xticklabels([])
        ax3.tick_params(
            axis="both",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False,
            bottom=False,
            labelleft=True,
        )

        ax4 = plt.subplot(gs2[0, 1])

        sns.heatmap(
            minor_bigrams[minor_bigrams > 0].iloc[
                :top, :top
            ],  # only display non-zero values
            annot=True,
            fmt=".1f",
            cmap="Reds",
            ax=ax4,
            # vmin=vmin,
            # vmax=vmax,
            annot_kws={"fontsize": 6.5, "rotation": 60},
            cbar=False,
        )

        ax4.set_yticks([])
        ax4.tick_params(bottom=False)

        fig.align_labels()
    return fig


def plot_cum(
    S=None,
    cum=None,
    x_log=False,
    markersize=2,
    left_range=(-0.1, 4.40),
    right_range=(-0.023, 1.099),
    **kwargs,
):
    """Pass either a Series or cumulative_fraction(S).reset_index()"""
    if cum is None:
        cum = cumulative_fraction(S).reset_index()
        cum.index = cum.index + 1
    fig = make_subplots(
        specs=[
            [
                {
                    "secondary_y": True,
                }
            ]
        ]
    )
    ix = cum.index
    fig.add_trace(
        go.Scatter(
            x=ix,
            y=cum.x,
            text=cum["index"],
            name="Absolute count",
            mode="markers",
            marker=dict(size=markersize),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=ix,
            y=cum.y,
            text=cum["index"],
            name="Cumulative fraction",
            mode="markers",
            marker=dict(size=markersize),
        ),
        secondary_y=True,
    )
    fig.update_xaxes(
        title_text="Rank of chord label", zeroline=False, gridcolor="lightgrey"
    )
    if x_log:
        ranks = np.log(len(ix)) / np.log(10)
        fig.update_xaxes(type="log", range=(-0.01 * ranks, 1.01 * ranks))
    else:
        ranks = len(ix)
        fig.update_xaxes(range=(-0.02 * ranks, 1.02 * ranks))
    fig.update_yaxes(
        title_text="Absolute label count",
        secondary_y=False,
        type="log",
        gridcolor="grey",
        zeroline=True,
        dtick=1,
        range=left_range,
    )
    fig.update_yaxes(
        title_text="Cumulative fraction",
        secondary_y=True,
        gridcolor="lightgrey",
        zeroline=False,
        dtick=0.1,
        range=right_range,
    )
    fig.update_layout(**kwargs)
    return fig


def plot_pitch_class_distribution(
    df: pd.DataFrame,
    pitch_column="tpc",
    duration_column="duration_qb",
    title="Pitch class distribution",
    fifths_transform=ms3.fifths2name,
    width=1500,
    height=500,
    labels=None,
    modin=True,
    output=None,
):
    bar_data = get_pitch_class_distribution(
        df=df,
        pitch_column=pitch_column,
        duration_column=duration_column,
    ).reset_index()
    if modin:
        x_col, y_col = 0, 1
    else:
        x_col, y_col = pitch_column, duration_column
    return fifths_bar_plot(
        bar_data=bar_data,
        x_col=x_col,
        y_col=y_col,
        labels=labels,
        title=title,
        fifth_transform=fifths_transform,
        width=width,
        height=height,
        output=output,
    )


def prepare_sunburst_data(sliced_harmonies_table: pd.DataFrame) -> pd.DataFrame:
    """"""
    chord_data = sliced_harmonies_table[
        sliced_harmonies_table.sd.str.len() == 1
    ].copy()  # scale degrees without
    # accidentals
    chord_data["interval"] = ms3.transform(
        chord_data.sd_progression, safe_interval
    ).fillna("∎")
    chord_data.figbass.fillna("3", inplace=True)
    chord_data["following_figbass"] = (
        chord_data.groupby(
            level=[0, 1, 2],
        )
        .figbass.shift(-1)
        .fillna("∎")
    )
    return chord_data


def prettify_counts(counter_object: Counter):
    N = counter_object.total()
    print(f"N = {N}")
    df = pd.DataFrame(
        counter_object.most_common(), columns=["progression", "count"]
    ).set_index("progression")
    df["%"] = (df["count"] * 100 / N).round(2)
    return df


def print_heading(heading: str, underline: chr = "-") -> None:
    """Underlines the given heading and prints it."""
    print(f"{heading}\n{underline * len(heading)}\n")


def remove_non_chord_labels(df):
    print(f"Length before: {len(df.index)}")
    non_chord = df.chord.isna()
    print(f"There are {non_chord.sum()} non-chord labels which we are going to delete:")
    display(df.loc[non_chord, "label"].value_counts())
    erroneous_chord = df.root.isna() & ~non_chord
    if erroneous_chord.sum() > 0:
        print(
            f"There are {erroneous_chord.sum()} labels with erroneous chord annotations which we are going to delete:"
        )
        display(df.loc[erroneous_chord, "label"].value_counts())
        non_chord |= erroneous_chord
    result = df.drop(df.index[non_chord])
    print(f"Length after: {len(result.index)}")
    return result


def remove_none_labels(df):
    print(f"Length before: {len(df.index)}")
    is_none = (df.chord == "@none").fillna(False)
    print(f"There are {is_none.sum()} @none labels which we are going to delete.")
    result = df.drop(df.index[is_none])
    print(f"Length after: {len(result.index)}")
    return result


def resolve_dir(directory: str):
    return os.path.realpath(os.path.expanduser(directory))


def rectangular_sunburst(
    sliced_harmonies_table: pd.DataFrame,
    path,
    height=1500,
    title="Sunburst",
) -> go.Figure:
    chord_data = prepare_sunburst_data(sliced_harmonies_table)
    title = f"{title} ({' - '.join(COLUMN2SUNBURST_TITLE[col] for col in path)})"
    fig = px.sunburst(
        chord_data,
        path=path,
        height=height,
        title=title,
    )
    return fig


def safe_interval(fifths):
    if pd.isnull(fifths):
        return "∎"
    return ms3.fifths2iv(fifths, smallest=True)


def sorted_gram_counts(lists_of_symbols, n=2, k=25):
    return prettify_counts(
        Counter(
            {
                t: count
                for t, count in sorted(
                    Counter(grams(lists_of_symbols, n=n)).items(),
                    key=lambda a: a[1],
                    reverse=True,
                )[:k]
            }
        )
    )


def tpc_bubbles(
    df: pd.Series | pd.DataFrame,
    normalize=True,
    width=1200,
    height=1500,
    title="Pitch class durations",
    duration_column="duration_qb",
    x_axis=None,
    y_axis=None,
    labels=None,
    output=None,
    flip=False,
    modin=False,
    **kwargs,
):
    """
    Expecting a long format DataFrame/Series with two index levels where the first level groups pitch class
    distributions: Pitch classes are the second index level and the distribution values are contained in the Series
    or the first column. Additional columns may serve, e.g. to add more hover_data fields (by passing the column name(s)
    as keyword argument 'hover_data'.
    """
    layout = dict(STD_LAYOUT)
    if flip:
        if modin:
            x, y = 1, 2
        else:
            *_, x, y = df.index.names
        xaxis_settings, yaxis_settings = dict(Y_AXIS), dict(X_AXIS)
        color_col = y
        x_axis, y_axis = y_axis, x_axis
        layout.update(dict(width=height, height=width))
    else:
        if modin:
            x, y = 2, 1
        else:
            *_, y, x = df.index.names
        xaxis_settings, yaxis_settings = dict(X_AXIS), dict(Y_AXIS)
        color_col = x
        layout.update(dict(height=height, width=width))
    if normalize:
        if isinstance(df, pd.Series):
            df = df.groupby(level=0, group_keys=False).apply(lambda S: S / S.sum())
        else:
            df.iloc[:, 0] = (
                df.iloc[:, 0]
                .groupby(level=0, group_keys=False)
                .apply(lambda S: S / S.sum())
            )
        title = "Normalized " + title
    df = df.reset_index()
    if modin:
        size_column = 2
    else:
        size_column = duration_column
    tpc_names = ms3.fifths2name(list(df.tpc))
    df["pitch class"] = tpc_names
    hover_data = kwargs.pop("hover_data", [])
    if isinstance(hover_data, str):
        hover_data = [hover_data]
    hover_data += ["pitch class"]
    fig = px.scatter(
        df,
        x=x,
        y=y,
        size=size_column,
        color=color_col,
        **COLOR_SCALE_SETTINGS,
        labels=labels,
        title=title,
        hover_data=hover_data,
        **kwargs,
    )
    fig.update_traces(TRACES_SETTINGS)

    if not flip:
        yaxis_settings["autorange"] = "reversed"
    if x_axis is not None:
        xaxis_settings.update(x_axis)
    if y_axis is not None:
        yaxis_settings.update(y_axis)
    fig.update_layout(xaxis=xaxis_settings, yaxis=yaxis_settings, **layout)
    fig.update_coloraxes(showscale=False)
    if output is not None:
        fig.write_image(output)
    return fig


def transition_matrix(
    l=None,
    gs=None,
    n=2,
    k=None,
    smooth=0,
    normalize=False,
    IC=False,
    filt=None,
    dist_only=False,
    sort=False,
    percent=False,
    decimals=None,
):
    """Returns a transition table from a list of symbols.

    Column index is the last item of grams, row index the n-1 preceding items.

    Parameters
    ----------

    l: list, optional
        List of elements between which the transitions are calculated.
        List can be nested.
    gs: list, optional
        List of tuples being n-grams
    n: int, optional
        get n-grams
    k: int, optional
        Number of rows and columns that you want to keep
    smooth: number, optional
        initial count value of all transitions
    normalize: bool, optional
        set True to divide every row by the sum of the row.
    IC: bool, optional
        Set True to calculate information content.
    filt: list, optional
        elements you want to exclude from the table. All ngrams containing at least one
        of the elements will be filtered out.
    dist_only: bool, optional
        if True, n-grams consisting only of identical elements are filtered out
    sort : bool, optional
        By default, the indices are ordered by gram frequency. Pass True to sort
        by bigram counts.
    percent : bool, optional
        Pass True to multiply the matrix by 100 before rounding to `decimals`
    decimals : int, optional
        To how many decimals you want to round the matrix.
    """
    if gs is None:
        assert n > 0, f"Cannot print {n}-grams"
        gs = grams(l, n=n)
    elif l is not None:
        assert True, "Specify either l or gs, not both."

    if filt:
        gs = list(filter(lambda n: not any(g in filt for g in n), gs))
    if dist_only:
        gs = list(filter(lambda tup: any(e != tup[0] for e in tup), gs))
    ngrams = pd.Series(gs).value_counts()
    ngrams.index = [(" ".join(t[:-1]), t[-1]) for t in ngrams.index.tolist()]
    context = pd.Index(set([ix[0] for ix in ngrams.index]))
    consequent = pd.Index(set([ix[1] for ix in ngrams.index]))
    df = pd.DataFrame(smooth, index=context, columns=consequent)

    for i, (cont, cons) in enumerate(ngrams.index):
        try:
            df.loc[cont, cons] += ngrams[i]
        except Exception:
            continue

    if k is not None:
        sort = True

    if sort:
        h_sort = list(df.max().sort_values(ascending=False).index.values)
        v_sort = list(df.max(axis=1).sort_values(ascending=False).index.values)
        df = df[h_sort].loc[v_sort]
    else:
        frequency = df.sum(axis=1).sort_values(ascending=False).index
        aux_index = frequency.intersection(df.columns, sort=False)
        aux_index = aux_index.union(
            df.columns.difference(frequency, sort=False), sort=False
        )
        df = df[aux_index].loc[frequency]

    SU = df.sum(axis=1)
    if normalize or IC:
        df = df.div(SU, axis=0)

    if IC:
        ic = np.log2(1 / df)
        ic["entropy"] = (ic * df).sum(axis=1)
        # ############# Identical calculations:
        # ic['entropy2'] = scipy.stats.entropy(df.transpose(),base=2)
        # ic['entropy3'] = -(df * np.log2(df)).sum(axis=1)
        df = ic
        if normalize:
            df["entropy"] = df["entropy"] / np.log2(len(df.columns) - 1)
    # else:
    #     df['total'] = SU

    if k is not None:
        df = df.iloc[:k, :k]

    if percent:
        df.iloc[:, :-1] *= 100

    if decimals is not None:
        df = df.round(decimals)

    return df


def value_count_df(S, thing=None, counts="counts"):
    """Value counts as DataFrame where the index has the name of the given Series or ``thing`` and where the counts
    are given in the column ``counts``.
    """
    thing = S.name if thing is None else thing
    vc = S.value_counts().rename(counts)
    normalized = vc / vc.sum()
    df = pd.concat([vc.to_frame(), normalized.rename("%")], axis=1)
    df.index.rename(thing, inplace=True)
    return df


def write_image(
    fig: go.Figure,
    filename: str,
    directory: Optional[str] = None,
    format=None,
    scale=None,
    width=2880,
    height=1620,
    validate=True,
):
    """
    Convert a figure to a static image and write it to a file.

    Args:
        fig:
            Figure object or dict representing a figure

        file: str or writeable
            A string representing a local file path or a writeable object
            (e.g. a pathlib.Path object or an open file descriptor)

        format: str or None
            The desired image format. One of
              - 'png'
              - 'jpg' or 'jpeg'
              - 'webp'
              - 'svg'
              - 'pdf'
              - 'eps' (Requires the poppler library to be installed and on the PATH)

            If not specified and `file` is a string then this will default to the
            file extension. If not specified and `file` is not a string then this
            will default to:
                - `plotly.io.kaleido.scope.default_format` if engine is "kaleido"
                - `plotly.io.orca.config.default_format` if engine is "orca"

        width: int or None
            The width of the exported image in layout pixels. If the `scale`
            property is 1.0, this will also be the width of the exported image
            in physical pixels.

            If not specified, will default to:
                - `plotly.io.kaleido.scope.default_width` if engine is "kaleido"
                - `plotly.io.orca.config.default_width` if engine is "orca"

        height: int or None
            The height of the exported image in layout pixels. If the `scale`
            property is 1.0, this will also be the height of the exported image
            in physical pixels.

            If not specified, will default to:
                - `plotly.io.kaleido.scope.default_height` if engine is "kaleido"
                - `plotly.io.orca.config.default_height` if engine is "orca"

        scale: int or float or None
            The scale factor to use when exporting the figure. A scale factor
            larger than 1.0 will increase the image resolution with respect
            to the figure's layout pixel dimensions. Whereas as scale factor of
            less than 1.0 will decrease the image resolution.

            If not specified, will default to:
                - `plotly.io.kaleido.scope.default_scale` if engine is "kaleido"
                - `plotly.io.orca.config.default_scale` if engine is "orca"

        validate: bool
            True if the figure should be validated before being converted to
            an image, False otherwise.
    """
    fname, fext = os.path.splitext(filename)
    if format is None:
        has_allowed_extension = fext.lstrip(".") in AVAILABLE_FIGURE_FORMATS
        output_filename = (
            filename
            if has_allowed_extension
            else f"{filename}.{DEFAULT_OUTPUT_FORMAT.lstrip('.')}"
        )
    else:
        output_filename = f"{filename}.{format.lstrip('.')}"
    if directory is None:
        output_filepath = os.path.join(OUTPUT_FOLDER, output_filename)
    else:
        output_filepath = os.path.join(directory, output_filename)
    fig.write_image(
        file=output_filepath,
        width=width,
        height=height,
        scale=scale,
        validate=validate,
    )
