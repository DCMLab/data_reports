from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from functools import cache
from typing import Iterable, List, Optional, Tuple

import colorlover
import frictionless as fl
import ms3
import numpy as np

# import modin.pandas as pd
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt, gridspec
from scipy.stats import entropy

from dimcat.utils import grams, make_transition_matrix
from git import Repo
from IPython.display import display
from matplotlib.figure import Figure as MatplotlibFigure
from plotly import graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.abspath(os.path.join(HERE, "..", "results"))
DEFAULT_OUTPUT_FORMAT = ".png"
DEFAULT_COLUMNS = ["mc", "mc_onset"]  # always added to bigram dataframes
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


def add_mode_column(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a copy of a DataFrame (which needs to have 'localkey_is_minor' boolean col) and adds a 'mode' column
    containing 'major' and 'minor'.
    """
    assert (
        "localkey_is_minor" in df.columns
    ), "df must have a 'localkey_is_minor' column"
    mode_col = df.localkey_is_minor.map({True: "minor", False: "major"}).rename("mode")
    return pd.concat([df, mode_col], axis=1)


def count_subsequent_occurrences(
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
    for ix, value in S.items():
        if not pd.isnull(value) and value in interval_list:
            current.append(ix)
            if value != 0:
                n += 1
        else:
            if not pd.isnull(value):
                current.append(ix)
            if n >= k_min:
                if df:
                    ix_chunks.loc[len(ix_chunks)] = (current, n)
                else:
                    ix_chunks.append((current, n))
            current = []
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


def get_repo_name(repo: Repo) -> str:
    """Gets the repo name from the origin's URL, or from the local path if there is None."""
    if isinstance(repo, str):
        repo = Repo(repo)
    if len(repo.remotes) == 0:
        return repo.git.rev_parse("--show-toplevel")
    remote = repo.remotes[0]
    return remote.url.split(".git")[0].split("/")[-1]


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


def make_sunburst(
    chords: pd.DataFrame,
    parent: str,
    filter_accidentals: bool = False,
    terminal_symbol: str = "⋉",
    inspect=False,
):
    """

    Args:
        chords: DataFrame containing columns "sd" and "sd_progression"
        parent: Label to be displayed in the middle.
        filter_accidentals:
            If set to True, scale degrees with accidentals (precisely: with a length > 1) are replaced by white space.
        inspect: Set to True to show a data sample instead of a sunburst plot.

    Returns:

    """
    in_scale = []
    sd2prog = defaultdict(Counter)
    for sd, sd_prog in chords[["sd", "sd_progression"]].itertuples(index=False):
        if not filter_accidentals or len(sd) == 1:
            in_scale.append(sd)
            sd2prog[sd].update(
                [terminal_symbol] if pd.isnull(sd_prog) else [str(sd_prog)]
            )
    label_counts = Counter(in_scale)
    labels, values = list(label_counts.keys()), list(label_counts.values())
    # labels, values = zip(*list((sd, label_counts[sd]) for sd in sorted(label_counts)))
    parents = [parent] * len(labels)
    labels = [parent] + labels
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


def plot_cum(
    S=None,
    cum=None,
    x_log: bool = True,
    markersize: int = 4,
    n_labels: int = 10,
    font_size: Optional[int] = None,
    left_range: Optional[Tuple[float, float]] = None,
    right_range: Optional[Tuple[float, float]] = None,
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
    scatter_args = dict(
        x=ix,
        y=cum.x,
        name="Absolute count",
        marker=dict(size=markersize),
    )
    if n_labels > 0:
        text_labels, text_positions = [], []
        for i, chrd in enumerate(cum.chord):
            if i < n_labels:
                text_labels.append(chrd)
                if i % 2:
                    text_positions.append("top center")
                else:
                    text_positions.append("bottom center")
            else:
                text_labels.append("")
                text_positions.append("top center")
        scatter_args["text"] = text_labels
        scatter_args["textposition"] = text_positions
        scatter_args["mode"] = "markers+text"
    else:
        scatter_args["mode"] = "markers"
    fig.add_trace(
        go.Scatter(**scatter_args),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=ix,
            y=cum.y,
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
    left_y_axis = dict(
        title_text="Absolute label count",
        secondary_y=False,
        type="log",
        gridcolor="lightgrey",
        zeroline=True,
        dtick=1,
    )
    right_y_axis = dict(
        title_text="Cumulative fraction",
        secondary_y=True,
        gridcolor="lightpink",
        zeroline=False,
        dtick=0.25,
    )
    if left_range is not None:
        left_y_axis["range"] = left_range
    if right_range is not None:
        right_y_axis["range"] = right_range
    layout_args = dict(
        kwargs, legend=dict(orientation="h", itemsizing="constant", x=-0.05)
    )
    if font_size is not None:
        layout_args["font"] = dict(size=font_size)
    fig.update_layout(**layout_args)
    fig.update_yaxes(**left_y_axis)
    fig.update_yaxes(**right_y_axis)
    return fig


def plot_transition_heatmaps(
    full_grams_left: List[tuple],
    full_grams_right: Optional[List[tuple]] = None,
    frequencies=True,
    remove_repeated: bool = False,
    sort_scale_degrees: bool = False,
    **kwargs,
) -> MatplotlibFigure:
    left_transition_matrix = make_transition_matrix(
        full_grams_left,
        distinct_only=remove_repeated,
        normalize=frequencies,
        percent=True,
    )
    left_unigrams = pd.Series(Counter(sum(full_grams_left, [])))
    if sort_scale_degrees:
        left_unigrams = left_unigrams.sort_index(key=scale_degree_order)
    else:
        left_unigrams = left_unigrams.sort_values(ascending=False)
    left_unigrams_norm = left_unigrams / left_unigrams.sum()
    ix_intersection = left_unigrams_norm.index.intersection(
        left_transition_matrix.index
    )
    col_intersection = left_unigrams_norm.index.intersection(
        left_transition_matrix.columns
    )
    left_transition_matrix = left_transition_matrix.loc[
        ix_intersection, col_intersection
    ]
    left_unigrams_norm = left_unigrams_norm.loc[ix_intersection]

    if full_grams_right is None:
        right_transition_matrix = None
        right_unigrams_norm = None
    else:
        right_transition_matrix = make_transition_matrix(
            full_grams_right,
            distinct_only=remove_repeated,
            normalize=frequencies,
            percent=True,
        )
        right_unigrams = pd.Series(Counter(sum(full_grams_right, [])))
        if sort_scale_degrees:
            right_unigrams = right_unigrams.sort_index(key=scale_degree_order)
        else:
            right_unigrams = right_unigrams.sort_values(ascending=False)
        right_unigrams_norm = right_unigrams / right_unigrams.sum()
        ix_intersection = right_unigrams_norm.index.intersection(
            right_transition_matrix.index
        )
        col_intersection = right_unigrams_norm.index.intersection(
            right_transition_matrix.columns
        )
        right_transition_matrix = right_transition_matrix.loc[
            ix_intersection, col_intersection
        ]
        right_unigrams_norm = right_unigrams_norm.loc[ix_intersection]

    return make_transition_heatmap_plots(
        left_transition_matrix,
        left_unigrams_norm,
        right_transition_matrix,
        right_unigrams_norm,
        frequencies=frequencies,
        **kwargs,
    )


def prepare_sunburst_data(
    sliced_harmonies_table: pd.DataFrame,
    filter_accidentals: bool = False,
    terminal_symbol: str = "⋉",
) -> pd.DataFrame:
    """

    Args:
        sliced_harmonies_table: DataFrame containing the columns "sd", "sd_progression", "figbass",

    Returns:

    """
    if filter_accidentals:
        chord_data = sliced_harmonies_table[
            sliced_harmonies_table.sd.str.len() == 1
        ].copy()  # scale degrees without
    else:
        chord_data = sliced_harmonies_table.copy()
    # accidentals
    chord_data["interval"] = ms3.transform(
        chord_data.sd_progression, safe_interval, terminal_symbol=terminal_symbol
    ).fillna(terminal_symbol)
    chord_data.figbass.fillna("3", inplace=True)
    chord_data["following_figbass"] = (
        chord_data.groupby(
            level=[0, 1, 2],
        )
        .figbass.shift(-1)
        .fillna(terminal_symbol)
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


def remove_non_chord_labels(
    df,
    remove_erroneous_chords: bool = True,
):
    print(f"Length before: {len(df.index)}")
    non_chord = df.chord.isna()
    print(f"There are {non_chord.sum()} non-chord labels which we are going to delete:")
    display(df.loc[non_chord, "label"].value_counts())
    if remove_erroneous_chords:
        erroneous_chord = df.root.isna() & ~non_chord
        if erroneous_chord.sum() > 0:
            print(
                f"There are {erroneous_chord.sum()} labels with erroneous chord annotations which we are going to "
                f"delete:"
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
    terminal_symbol: str = "⋉",
) -> go.Figure:
    chord_data = prepare_sunburst_data(
        sliced_harmonies_table, terminal_symbol=terminal_symbol
    )
    title = f"{title} ({' - '.join(COLUMN2SUNBURST_TITLE[col] for col in path)})"
    fig = px.sunburst(
        chord_data,
        path=path,
        height=height,
        title=title,
    )
    return fig


def scale_degree_order(
    scale_degree: str | Iterable[str],
) -> Tuple[int, int] | List[Tuple[int, int]]:
    """Can be used as key function for sorting scale degrees."""
    if not isinstance(scale_degree, str):
        return list(map(scale_degree_order, scale_degree))
    if scale_degree == "∅":
        return (10,)
    match = re.match(r"([#b]*)([1-7])", scale_degree)
    accidental, degree = match.groups()
    return int(degree), accidental.find("#") - accidental.find("b")


def safe_interval(fifths, terminal_symbol="⋉"):
    if pd.isnull(fifths):
        return terminal_symbol
    return ms3.fifths2iv(fifths, smallest=True)


def sorted_gram_counts(lists_of_symbols, n=2, k=25):
    return prettify_counts(
        Counter(
            {
                t: count
                for t, count in sorted(
                    Counter(grams(lists_of_symbols, n=n, to_string=True)).items(),
                    key=lambda a: a[1],
                    reverse=True,
                )[:k]
            }
        )
    )


def value_count_df(
    S: pd.Series,
    name: Optional[str] = None,
    counts_column: str = "counts",
    round: Optional[int] = 2,
):
    """Value counts as DataFrame where the index has the name of the given Series or ``name`` and where the counts
    are given in the column ``counts``.
    """
    name = S.name if name is None else name
    vc = S.value_counts().rename(counts_column)
    normalized = 100 * vc / vc.sum()
    if round is not None:
        normalized = normalized.round(round)
    df = pd.concat([vc.to_frame(), normalized.rename("%")], axis=1)
    df.index.rename(name, inplace=True)
    return df


def ix_segments2values(df, ix_segments, cols=["bass_degree", "chord"]):
    res = {col: [] for col in cols}
    for segment in ix_segments:
        col2list = get_cols(df, segment, cols)
        for col in cols:
            res[col].append(col2list[col])
    for col, list_of_lists in res.items():
        res[col] = [" ".join(val) for val in list_of_lists]
    return res


def get_cols(df, ix, cols):
    if isinstance(cols, str):
        cols = [cols]
    df = df.loc[ix]
    return {col: df[col].to_list() for col in cols}


def summarize(df):
    norepeat = (df.bass_note != df.bass_note.shift()).fillna(True)
    seconds_asc = count_subsequent_occurrences(df.bass_interval_pc, [1, 2])
    seconds_asc_vals = ix_segments2values(df, seconds_asc.ixs)
    seconds_desc = count_subsequent_occurrences(df.bass_interval_pc, [-1, -2])
    seconds_desc_vals = ix_segments2values(df, seconds_desc.ixs)
    both = count_subsequent_occurrences(df.bass_interval_pc, [1, 2, -1, -2])
    both_vals = ix_segments2values(df, both.ixs)
    n_stepwise = both.n.sum()
    length_norepeat = norepeat.sum()
    res = pd.Series(
        {
            "globalkey": df.globalkey.unique()[0],
            "localkey": df.localkey.unique()[0],
            "length": len(df),
            "length_norepeat": length_norepeat,
            "n_stepwise": n_stepwise,
            "%_stepwise": round(100 * n_stepwise / length_norepeat, 1),
            "n_ascending": seconds_asc.n.sum(),
            "n_descending": seconds_desc.n.sum(),
            "bd": " ".join(df.loc[norepeat, "bass_degree"].to_list()),
            "stepwise_bd": both_vals["bass_degree"],
            "stepwise_chords": both_vals["chord"],
            "ascending_bd": seconds_asc_vals[
                "bass_degree"
            ],  # ix_segments2list(df, seconds_asc.ixs),
            "ascending_chords": seconds_asc_vals["chord"],
            "descending_bd": seconds_desc_vals["bass_degree"],
            "descending_chords": seconds_desc_vals["chord"],
            "ixa": df.index[0],
            "ixb": df.index[-1],
        }
    )
    return res


def make_key_region_summary_table(
    df, mutate_dataframe: bool = True, *groupby_args, **groupby_kwargs
):
    """Takes an extended harmonies table that is segmented by local keys. The segments are iterated over using the
    *groupby_args and **groupby_kwargs arguments.
    """
    groupby_kwargs = dict(groupby_kwargs, group_keys=False)
    if mutate_dataframe:
        df = add_bass_degree_columns(
            df, mutate_dataframe=mutate_dataframe, *groupby_args, **groupby_kwargs
        )
    else:
        add_bass_degree_columns(
            df, mutate_dataframe=mutate_dataframe, *groupby_args, **groupby_kwargs
        )
    if "bass_interval" not in df.columns:
        bass_interval_column = df.groupby(
            *groupby_args, **groupby_kwargs
        ).bass_note.apply(lambda bd: bd.shift(-1) - bd)
        pc_interval_column = ms3.transform(bass_interval_column, ms3.fifths2pc)
        pc_interval_column = pc_interval_column.where(
            pc_interval_column <= 6, pc_interval_column % -6
        )
        if mutate_dataframe:
            df["bass_interval"] = bass_interval_column
            df["bass_interval_pc"] = pc_interval_column
        else:
            df = pd.concat(
                [
                    df,
                    bass_interval_column.rename("bass_interval"),
                    pc_interval_column.rename("bass_interval_pc"),
                ],
                axis=1,
            )
    return df.groupby(*groupby_args, **groupby_kwargs).apply(summarize)


def add_bass_degree_columns(
    df,
    mutate_dataframe: bool = True,
):
    if "bass_degree" not in df.columns:
        bass_degree_column = ms3.transform(
            df, ms3.fifths2sd, ["bass_note", "localkey_is_minor"]
        )
        if mutate_dataframe:
            df["bass_degree"] = bass_degree_column
        else:
            df = pd.concat([df, bass_degree_column.rename("bass_degree")], axis=1)
    if "intervals_over_bass" not in df.columns:
        intervals_over_bass_column = ms3.transform(
            df, chord_tones2interval_structure, ["chord_tones"]
        )
        intervals_over_root_column = ms3.transform(
            df, chord_tones2interval_structure, ["chord_tones", "root"]
        )
        if mutate_dataframe:
            df["intervals_over_bass"] = intervals_over_bass_column
            df["intervals_over_root"] = intervals_over_root_column
        else:
            df = pd.concat(
                [
                    df,
                    intervals_over_bass_column.rename("intervals_over_bass"),
                    intervals_over_root_column.rename("intervals_over_root"),
                ],
                axis=1,
            )
    if mutate_dataframe:
        return df


def chord_tones2interval_structure(
    fifths: Iterable[int], reference: Optional[int] = None
) -> Tuple[str]:
    """The fifth are interpreted as intervals expressing distances from the local tonic ("neutral degrees").
    The result will be a tuple of strings that express the same intervals but expressed with respect to the given
    reference (neutral degree), removing unisons.
    If no reference is specified, the first degree (usually, the bass note) is used as such.
    """
    try:
        fifths = tuple(fifths)
        if len(fifths) == 0:
            return ()
    except Exception:
        return ()
    if reference is None:
        reference = fifths[0]
    elif reference in fifths:
        position = fifths.index(reference)
        if position > 0:
            fifths = fifths[position:] + fifths[:position]
    adapted_intervals = [
        ms3.fifths2iv(adapted)
        for interval in fifths
        if (adapted := interval - reference) != 0
    ]
    return tuple(adapted_intervals)


def make_transition_heatmap_plots(
    left_transition_matrix: pd.DataFrame,
    left_unigrams: pd.Series,
    right_transition_matrix: Optional[pd.DataFrame] = None,
    right_unigrams: Optional[pd.Series] = None,
    top: int = 30,
    two_col_width=12,
    frequencies: bool = False,
    fontsize=8,
    labelsize=12,
    top_margin=0.99,
    bottom_margin=0.10,
    right_margin=0.005,
    left_margin=0.085,
) -> MatplotlibFigure:
    """
    Adapted from https://zenodo.org/records/2764889/files/reproduce_ABC.ipynb?download=1 which is the Jupyter notebook
    accompanying Moss FC, Neuwirth M, Harasim D, Rohrmeier M (2019) Statistical characteristics of tonal harmony: A
    corpus study of Beethoven’s string quartets. PLOS ONE 14(6): e0217242. https://doi.org/10.1371/journal.pone.0217242

    Args:
        left_unigrams:
        right_unigrams:
        left_transition_matrix:
        right_transition_matrix:
        top:
        two_col_width:
        frequencies: If set to True, the values of the unigram Series are interpreted as normalized frequencies and
            are multiplied with 100 for display on the y-axis.

    """
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

        def make_gridspec(
            left,
            right,
        ):
            gridspec_ratio = [0.25, 2.0]
            hspace = None
            wspace = 0.0
            gs = gridspec.GridSpec(1, 2, width_ratios=gridspec_ratio)
            gs.update(
                left=left,
                right=right,
                wspace=wspace,
                hspace=hspace,
                bottom=bottom_margin,
                top=top_margin,
            )
            return gs

        def add_entropy_bars(
            unigrams,
            bigrams,
            axis,
        ):
            # settings for margins etc.
            barsize = [0.0, 0.7]
            s_min = pd.Series(
                (
                    bigrams.apply(lambda x: entropy(x, base=2), axis=1)
                    / np.log2(bigrams.shape[0])
                )[:top].values,
                index=[
                    i + f" ({str(round(fr * 100, 1))})" if frequencies else i
                    for i, fr in zip(bigrams.index, unigrams[:top].values)
                ],
            )
            ax = s_min.plot(kind="barh", ax=axis, color="k")

            # create a list to collect the plt.patches data
            totals_min = []

            # find the values and append to list
            for i in ax.patches:
                totals_min.append(round(i.get_width(), 2))

            for i, p in enumerate(ax.patches):
                axis.text(
                    totals_min[i] - 0.01,
                    p.get_y() + 0.3,
                    f"${totals_min[i]}$",
                    color="w",
                    fontsize=fontsize,
                    verticalalignment="center",
                    horizontalalignment="left",
                )
            axis.set_xlim(barsize)

            axis.invert_yaxis()
            axis.invert_xaxis()
            axis.set_xticklabels([])
            axis.tick_params(
                axis="both",  # changes apply to the x-axis
                which="both",  # both major and minor ticks are affected
                left=False,  # ticks along the bottom edge are off
                right=False,
                bottom=False,
                labelleft=True,
                labelsize=labelsize,
            )

        def add_heatmap(transition_value_matrix, axis, colormap):
            sns.heatmap(
                transition_value_matrix,
                annot=True,
                fmt=".1f",
                cmap=colormap,
                ax=axis,
                # vmin=vmin,
                # vmax=vmax,
                annot_kws={"fontsize": fontsize, "rotation": 60},
                cbar=False,
            )
            axis.set_yticks([])
            axis.tick_params(bottom=False)

        single_col_width = two_col_width / 2
        plot_two_sides = right_transition_matrix is not None
        if plot_two_sides:
            assert (
                right_unigrams is not None
            ), "right_unigrams must be provided if right_bigrams is provided"
            fig = plt.figure(figsize=(two_col_width, single_col_width))
            gs1 = make_gridspec(
                left=left_margin,
                right=0.5 - right_margin,
            )
        else:
            fig = plt.figure(figsize=(single_col_width, single_col_width))
            gs1 = make_gridspec(
                left=left_margin,
                right=1.0 - right_margin,
            )

        # LEFT-HAND SIDE

        ax1 = plt.subplot(gs1[0, 0])

        add_entropy_bars(
            left_unigrams,
            left_transition_matrix,
            ax1,
        )

        ax2 = plt.subplot(gs1[0, 1])

        add_heatmap(
            left_transition_matrix[left_transition_matrix > 0].iloc[
                :top, :top
            ],  # only display non-zero values
            axis=ax2,
            colormap="Blues",
        )

        # RIGHT-HAND SIDE

        plot_two_sides = right_transition_matrix is not None
        if plot_two_sides:
            assert (
                right_unigrams is not None
            ), "right_unigrams must be provided if right_bigrams is provided"

            gs2 = make_gridspec(
                left=0.5 + left_margin,
                right=1.0 - right_margin,
            )

            ax3 = plt.subplot(gs2[0, 0])
            add_entropy_bars(
                right_unigrams,
                right_transition_matrix,
                ax3,
            )

            ax4 = plt.subplot(gs2[0, 1])
            add_heatmap(
                right_transition_matrix[right_transition_matrix > 0].iloc[:top, :top],
                axis=ax4,
                colormap="Reds",
            )

        fig.align_labels()
    return fig
