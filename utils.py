import os
import re
from functools import cache
from typing import List, Optional

import colorlover
import frictionless as fl
import ms3
import numpy as np

# import modin.pandas as pd
import pandas as pd
import plotly.express as px
from git import Repo
from plotly.colors import sample_colorscale

CORPUS_COLOR_SCALE = px.colors.qualitative.D3
COLOR_SCALE_SETTINGS = dict(
    color_continuous_scale="RdBu_r", color_continuous_midpoint=2
)
TPC_DISCRETE_COLOR_MAP = dict(zip(range(-15, 20), sample_colorscale("RdBu_r", 35)))
STD_LAYOUT = {
    "paper_bgcolor": "#FFFFFF",
    "plot_bgcolor": "#FFFFFF",
    "margin": {"l": 40, "r": 0, "b": 0, "t": 40, "pad": 0},
    "font": {"size": 15},
}

CADENCE_COLORS = dict(
    zip(("HC", "PAC", "PC", "IAC", "DC", "EC"), colorlover.scales["6"]["qual"]["Set1"])
)
CORPUS_COLOR_SCALE = px.colors.qualitative.D3

CORPUS_NAMES = dict(
    gastoldi_baletti="Gastoldi Baletti",
    peri_euridice="Peri Euridice",
    monteverdi_madrigals="Monteverdi Madrigals",
    sweelinck_keyboard="Sweelinck Keyboard",
    frescobaldi_fiori_musicali="Frescobaldi Fiori Musicali",
    kleine_geistliche_konzerte="Schütz Kleine Geistliche Konzerte",
    corelli="Corelli Trio Sonatas",
    couperin_clavecin="Couperin Clavecin",
    handel_keyboard="Handel Keyboard",
    bach_en_fr_suites="Bach Suites",
    bach_solo="Bach Solo",
    couperin_concerts="Couperin Concerts Royaux",
    pergolesi_stabat_mater="Pergolesi Stabat Mater",
    scarlatti_sonatas="Scarlatti Sonatas",
    wf_bach_sonatas="WF Bach Sonatas",
    jc_bach_sonatas="JC Bach Sonatas",
    mozart_piano_sonatas="Mozart Piano Sonatas",
    pleyel_quartets="Pleyel Quartets",
    beethoven_piano_sonatas="Beethoven Sonatas",
    kozeluh_sonatas="Kozeluh Sonatas",
    ABC="Beethoven String Quartets",
    schubert_dances="Schubert Dances",
    schubert_winterreise="Schubert Winterreise",
    mendelssohn_quartets="Mendelssohn Quartets",
    chopin_mazurkas="Chopin Mazurkas",
    schumann_kinderszenen="R Schumann Kinderszenen",
    schumann_liederkreis="R Schumann Liederkreis",
    c_schumann_lieder="C Schumann Lieder",
    liszt_pelerinage="Liszt Années",
    wagner_overtures="Wagner Overtures",
    tchaikovsky_seasons="Tchaikovsky Seasons",
    dvorak_silhouettes="Dvořák Silhouettes",
    grieg_lyric_pieces="Grieg Lyric Pieces",
    mahler_kindertotenlieder="Mahler Kindertotenlieder",
    ravel_piano="Ravel Piano",
    debussy_suite_bergamasque="Debussy Suite Bergamasque",
    bartok_bagatelles="Bartok Bagatelles",
    medtner_tales="Medtner Tales",
    poulenc_mouvements_perpetuels="Poulenc Mouvements Perpetuels",
    rachmaninoff_piano="Rachmaninoff Piano",
    schulhoff_suite_dansante_en_jazz="Schulhoff Suite Dansante En Jazz",
)

TRACES_SETTINGS = dict(marker_line_color="black")
TYPE_COLORS = dict(
    zip(
        ("Mm7", "M", "o7", "o", "mm7", "m", "%7", "MM7", "other"),
        colorlover.scales["9"]["qual"]["Paired"],
    )
)
X_AXIS = dict(gridcolor="lightgrey", zerolinecolor="grey")
Y_AXIS = dict()


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
    return df.groupby(level=0)[year_column].mean().sort_values()


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


@cache()
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
        bar_data, x_col, y_col, labels, title, fifths_transform, width, height, output
    )


def print_heading(heading: str, underline: chr = "-") -> None:
    """Underlines the given heading and prints it."""
    print(f"{heading}\n{underline * len(heading)}\n")


def resolve_dir(directory: str):
    return os.path.realpath(os.path.expanduser(directory))


def tpc_bubbles(
    df,
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
    modin=True,
):
    """
    Expecting a long format DataFrame with two index levels where the first level groups pitch class distributions.
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
        df = (
            df.groupby(level=0, group_keys=False)
            .apply(lambda S: S / S.sum())
            .reset_index()
        )
        title = "Normalized " + title
    else:
        df = df.reset_index()
    if modin:
        size_column = 2
    else:
        size_column = duration_column
    hover_data = ms3.fifths2name(list(df.tpc))
    df["pitch class"] = hover_data
    fig = px.scatter(
        df,
        x=x,
        y=y,
        size=size_column,
        color=color_col,
        **COLOR_SCALE_SETTINGS,
        labels=labels,
        title=title,
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
