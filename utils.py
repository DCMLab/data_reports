from __future__ import annotations

import colorsys
import os
import re
from collections import Counter, defaultdict
from fractions import Fraction
from functools import cache
from numbers import Number
from typing import Dict, Hashable, Iterable, Iterator, List, Literal, Optional, Tuple

import colorlover
import frictionless as fl
import ms3
import numpy as np
import numpy.typing as npt

# import modin.pandas as pd
import pandas as pd
import plotly.express as px
import seaborn as sns
from dimcat.base import FriendlyEnum
from dimcat.data import resources
from dimcat.data.resources.facets import add_chord_tone_intervals
from dimcat.data.resources.features import extend_bass_notes_feature
from dimcat.data.resources.results import TypeAlias, _entropy
from dimcat.data.resources.utils import merge_columns_into_one
from dimcat.plotting import make_bar_plot, update_figure_layout
from dimcat.utils import get_middle_composition_year, grams, make_transition_matrix
from docs.notebooks.create_gantt import create_gantt, fill_yaxis_gaps
from git import Repo
from IPython.display import display
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.figure import Figure as MatplotlibFigure
from plotly import graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from scipy.stats import entropy
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.abspath(os.path.join(HERE, "outputs"))
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


class TailwindBaseColor(FriendlyEnum):
    SLATE = "SLATE"
    GRAY = "GRAY"
    ZINC = "ZINC"
    NEUTRAL = "NEUTRAL"
    STONE = "STONE"
    RED = "RED"
    ORANGE = "ORANGE"
    AMBER = "AMBER"
    YELLOW = "YELLOW"
    LIME = "LIME"
    GREEN = "GREEN"
    EMERALD = "EMERALD"
    TEAL = "TEAL"
    CYAN = "CYAN"
    SKY = "SKY"
    BLUE = "BLUE"
    INDIGO = "INDIGO"
    VIOLET = "VIOLET"
    PURPLE = "PURPLE"
    FUCHSIA = "FUCHSIA"
    PINK = "PINK"
    ROSE = "ROSE"


shade_: TypeAlias = Literal[50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950]


class TailwindColors:
    """Color palette: look for tailwindcss_v3.3.3(.png|.svg)"""

    @classmethod
    def get_color(
        cls, name: TailwindBaseColor | str, shade: Optional[shade_] = None
    ) -> Tuple[int, int, int]:
        if shade is None:
            name_upper = name.upper()
            if hasattr(cls, name_upper):
                return getattr(cls, name_upper)
            raise ValueError(
                f"Shade has not been specified and name does not match any of the class members: {name_upper}"
            )
        tailwind_name = TailwindBaseColor(name)
        member = f"{tailwind_name.name}_{shade:03d}"
        return cls.get_color(member)

    @classmethod
    def iter_colors(
        cls,
        name: Optional[TailwindBaseColor | Iterable[TailwindBaseColor]] = None,
        shades: Optional[shade_ | Iterable[shade_]] = None,
        as_hsv: bool = False,
        names=True,
    ) -> Iterator[Tuple[str, Tuple[int, int, int]]] | Iterator[Tuple[int, int, int]]:
        if name is None:
            name_iterator = (name.name for name in TailwindBaseColor)
        else:
            if isinstance(name, str):
                name = [name]
            name_iterator = [TailwindBaseColor(name).name for name in name]
        if shades is None:
            shades = (50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950)
        elif isinstance(shades, int):
            shades = (shades,)
        for tailwind_name in name_iterator:
            for shade in shades:
                member = f"{tailwind_name}_{shade:03d}"
                result = cls.get_color(member)
                if as_hsv:
                    result = colorsys.rgb_to_hsv(*(round(c / 255.0, 1) for c in result))
                if names:
                    yield member, result
                else:
                    yield result


class TailwindColorsHex(TailwindColors):
    """
    Provides all colors from TailwindCSS as HTML strings.
    Copied from https://github.com/dostuffthatmatters/python-tailwind-colors/blob/
    3c1ac2359e3ae753875e06e68f5072586a0ae399/tailwind_colors/__init__.py

    Color palette: look for tailwindcss_v3.3.3(.png|.svg)

    ```python
    print(TAILWIND_COLORS.AMBER_600)
    # prints `#d97706`
    ```
    """

    SLATE_050: Literal["#f8fafc"] = "#f8fafc"
    SLATE_100: Literal["#f1f5f9"] = "#f1f5f9"
    SLATE_200: Literal["#e2e8f0"] = "#e2e8f0"
    SLATE_300: Literal["#cbd5e1"] = "#cbd5e1"
    SLATE_400: Literal["#94a3b8"] = "#94a3b8"
    SLATE_500: Literal["#64748b"] = "#64748b"
    SLATE_600: Literal["#475569"] = "#475569"
    SLATE_700: Literal["#334155"] = "#334155"
    SLATE_800: Literal["#1e293b"] = "#1e293b"
    SLATE_900: Literal["#0f172a"] = "#0f172a"
    SLATE_950: Literal["#020617"] = "#020617"

    GRAY_050: Literal["#f9fafb"] = "#f9fafb"
    GRAY_100: Literal["#f3f4f6"] = "#f3f4f6"
    GRAY_200: Literal["#e5e7eb"] = "#e5e7eb"
    GRAY_300: Literal["#d1d5db"] = "#d1d5db"
    GRAY_400: Literal["#9ca3af"] = "#9ca3af"
    GRAY_500: Literal["#6b7280"] = "#6b7280"
    GRAY_600: Literal["#4b5563"] = "#4b5563"
    GRAY_700: Literal["#374151"] = "#374151"
    GRAY_800: Literal["#1f2937"] = "#1f2937"
    GRAY_900: Literal["#111827"] = "#111827"
    GRAY_950: Literal["#030712"] = "#030712"

    ZINC_050: Literal["#fafafa"] = "#fafafa"
    ZINC_100: Literal["#f4f4f5"] = "#f4f4f5"
    ZINC_200: Literal["#e4e4e7"] = "#e4e4e7"
    ZINC_300: Literal["#d4d4d8"] = "#d4d4d8"
    ZINC_400: Literal["#a1a1aa"] = "#a1a1aa"
    ZINC_500: Literal["#71717a"] = "#71717a"
    ZINC_600: Literal["#52525b"] = "#52525b"
    ZINC_700: Literal["#3f3f46"] = "#3f3f46"
    ZINC_800: Literal["#27272a"] = "#27272a"
    ZINC_900: Literal["#18181b"] = "#18181b"
    ZINC_950: Literal["#09090b"] = "#09090b"

    NEUTRAL_050: Literal["#fafafa"] = "#fafafa"
    NEUTRAL_100: Literal["#f5f5f5"] = "#f5f5f5"
    NEUTRAL_200: Literal["#e5e5e5"] = "#e5e5e5"
    NEUTRAL_300: Literal["#d4d4d4"] = "#d4d4d4"
    NEUTRAL_400: Literal["#a3a3a3"] = "#a3a3a3"
    NEUTRAL_500: Literal["#737373"] = "#737373"
    NEUTRAL_600: Literal["#525252"] = "#525252"
    NEUTRAL_700: Literal["#404040"] = "#404040"
    NEUTRAL_800: Literal["#262626"] = "#262626"
    NEUTRAL_900: Literal["#171717"] = "#171717"
    NEUTRAL_950: Literal["#0a0a0a"] = "#0a0a0a"

    STONE_050: Literal["#fafaf9"] = "#fafaf9"
    STONE_100: Literal["#f5f5f4"] = "#f5f5f4"
    STONE_200: Literal["#e7e5e4"] = "#e7e5e4"
    STONE_300: Literal["#d6d3d1"] = "#d6d3d1"
    STONE_400: Literal["#a8a29e"] = "#a8a29e"
    STONE_500: Literal["#78716c"] = "#78716c"
    STONE_600: Literal["#57534e"] = "#57534e"
    STONE_700: Literal["#44403c"] = "#44403c"
    STONE_800: Literal["#292524"] = "#292524"
    STONE_900: Literal["#1c1917"] = "#1c1917"
    STONE_950: Literal["#0c0a09"] = "#0c0a09"

    RED_050: Literal["#fef2f2"] = "#fef2f2"
    RED_100: Literal["#fee2e2"] = "#fee2e2"
    RED_200: Literal["#fecaca"] = "#fecaca"
    RED_300: Literal["#fca5a5"] = "#fca5a5"
    RED_400: Literal["#f87171"] = "#f87171"
    RED_500: Literal["#ef4444"] = "#ef4444"
    RED_600: Literal["#dc2626"] = "#dc2626"
    RED_700: Literal["#b91c1c"] = "#b91c1c"
    RED_800: Literal["#991b1b"] = "#991b1b"
    RED_900: Literal["#7f1d1d"] = "#7f1d1d"
    RED_950: Literal["#450a0a"] = "#450a0a"

    ORANGE_050: Literal["#fff7ed"] = "#fff7ed"
    ORANGE_100: Literal["#ffedd5"] = "#ffedd5"
    ORANGE_200: Literal["#fed7aa"] = "#fed7aa"
    ORANGE_300: Literal["#fdba74"] = "#fdba74"
    ORANGE_400: Literal["#fb923c"] = "#fb923c"
    ORANGE_500: Literal["#f97316"] = "#f97316"
    ORANGE_600: Literal["#ea580c"] = "#ea580c"
    ORANGE_700: Literal["#c2410c"] = "#c2410c"
    ORANGE_800: Literal["#9a3412"] = "#9a3412"
    ORANGE_900: Literal["#7c2d12"] = "#7c2d12"
    ORANGE_950: Literal["#431407"] = "#431407"

    AMBER_050: Literal["#fffbeb"] = "#fffbeb"
    AMBER_100: Literal["#fef3c7"] = "#fef3c7"
    AMBER_200: Literal["#fde68a"] = "#fde68a"
    AMBER_300: Literal["#fcd34d"] = "#fcd34d"
    AMBER_400: Literal["#fbbf24"] = "#fbbf24"
    AMBER_500: Literal["#f59e0b"] = "#f59e0b"
    AMBER_600: Literal["#d97706"] = "#d97706"
    AMBER_700: Literal["#b45309"] = "#b45309"
    AMBER_800: Literal["#92400e"] = "#92400e"
    AMBER_900: Literal["#78350f"] = "#78350f"
    AMBER_950: Literal["#451a03"] = "#451a03"

    YELLOW_050: Literal["#fefce8"] = "#fefce8"
    YELLOW_100: Literal["#fef9c3"] = "#fef9c3"
    YELLOW_200: Literal["#fef08a"] = "#fef08a"
    YELLOW_300: Literal["#fde047"] = "#fde047"
    YELLOW_400: Literal["#facc15"] = "#facc15"
    YELLOW_500: Literal["#eab308"] = "#eab308"
    YELLOW_600: Literal["#ca8a04"] = "#ca8a04"
    YELLOW_700: Literal["#a16207"] = "#a16207"
    YELLOW_800: Literal["#854d0e"] = "#854d0e"
    YELLOW_900: Literal["#713f12"] = "#713f12"
    YELLOW_950: Literal["#422006"] = "#422006"

    LIME_050: Literal["#f7fee7"] = "#f7fee7"
    LIME_100: Literal["#ecfccb"] = "#ecfccb"
    LIME_200: Literal["#d9f99d"] = "#d9f99d"
    LIME_300: Literal["#bef264"] = "#bef264"
    LIME_400: Literal["#a3e635"] = "#a3e635"
    LIME_500: Literal["#84cc16"] = "#84cc16"
    LIME_600: Literal["#65a30d"] = "#65a30d"
    LIME_700: Literal["#4d7c0f"] = "#4d7c0f"
    LIME_800: Literal["#3f6212"] = "#3f6212"
    LIME_900: Literal["#365314"] = "#365314"
    LIME_950: Literal["#1a2e05"] = "#1a2e05"

    GREEN_050: Literal["#f0fdf4"] = "#f0fdf4"
    GREEN_100: Literal["#dcfce7"] = "#dcfce7"
    GREEN_200: Literal["#bbf7d0"] = "#bbf7d0"
    GREEN_300: Literal["#86efac"] = "#86efac"
    GREEN_400: Literal["#4ade80"] = "#4ade80"
    GREEN_500: Literal["#22c55e"] = "#22c55e"
    GREEN_600: Literal["#16a34a"] = "#16a34a"
    GREEN_700: Literal["#15803d"] = "#15803d"
    GREEN_800: Literal["#166534"] = "#166534"
    GREEN_900: Literal["#14532d"] = "#14532d"
    GREEN_950: Literal["#052e16"] = "#052e16"

    EMERALD_050: Literal["#ecfdf5"] = "#ecfdf5"
    EMERALD_100: Literal["#d1fae5"] = "#d1fae5"
    EMERALD_200: Literal["#a7f3d0"] = "#a7f3d0"
    EMERALD_300: Literal["#6ee7b7"] = "#6ee7b7"
    EMERALD_400: Literal["#34d399"] = "#34d399"
    EMERALD_500: Literal["#10b981"] = "#10b981"
    EMERALD_600: Literal["#059669"] = "#059669"
    EMERALD_700: Literal["#047857"] = "#047857"
    EMERALD_800: Literal["#065f46"] = "#065f46"
    EMERALD_900: Literal["#064e3b"] = "#064e3b"
    EMERALD_950: Literal["#022c22"] = "#022c22"

    TEAL_050: Literal["#f0fdfa"] = "#f0fdfa"
    TEAL_100: Literal["#ccfbf1"] = "#ccfbf1"
    TEAL_200: Literal["#99f6e4"] = "#99f6e4"
    TEAL_300: Literal["#5eead4"] = "#5eead4"
    TEAL_400: Literal["#2dd4bf"] = "#2dd4bf"
    TEAL_500: Literal["#14b8a6"] = "#14b8a6"
    TEAL_600: Literal["#0d9488"] = "#0d9488"
    TEAL_700: Literal["#0f766e"] = "#0f766e"
    TEAL_800: Literal["#115e59"] = "#115e59"
    TEAL_900: Literal["#134e4a"] = "#134e4a"
    TEAL_950: Literal["#042f2e"] = "#042f2e"

    CYAN_050: Literal["#ecfeff"] = "#ecfeff"
    CYAN_100: Literal["#cffafe"] = "#cffafe"
    CYAN_200: Literal["#a5f3fc"] = "#a5f3fc"
    CYAN_300: Literal["#67e8f9"] = "#67e8f9"
    CYAN_400: Literal["#22d3ee"] = "#22d3ee"
    CYAN_500: Literal["#06b6d4"] = "#06b6d4"
    CYAN_600: Literal["#0891b2"] = "#0891b2"
    CYAN_700: Literal["#0e7490"] = "#0e7490"
    CYAN_800: Literal["#155e75"] = "#155e75"
    CYAN_900: Literal["#164e63"] = "#164e63"
    CYAN_950: Literal["#083344"] = "#083344"

    SKY_050: Literal["#f0f9ff"] = "#f0f9ff"
    SKY_100: Literal["#e0f2fe"] = "#e0f2fe"
    SKY_200: Literal["#bae6fd"] = "#bae6fd"
    SKY_300: Literal["#7dd3fc"] = "#7dd3fc"
    SKY_400: Literal["#38bdf8"] = "#38bdf8"
    SKY_500: Literal["#0ea5e9"] = "#0ea5e9"
    SKY_600: Literal["#0284c7"] = "#0284c7"
    SKY_700: Literal["#0369a1"] = "#0369a1"
    SKY_800: Literal["#075985"] = "#075985"
    SKY_900: Literal["#0c4a6e"] = "#0c4a6e"
    SKY_950: Literal["#082f49"] = "#082f49"

    BLUE_050: Literal["#eff6ff"] = "#eff6ff"
    BLUE_100: Literal["#dbeafe"] = "#dbeafe"
    BLUE_200: Literal["#bfdbfe"] = "#bfdbfe"
    BLUE_300: Literal["#93c5fd"] = "#93c5fd"
    BLUE_400: Literal["#60a5fa"] = "#60a5fa"
    BLUE_500: Literal["#3b82f6"] = "#3b82f6"
    BLUE_600: Literal["#2563eb"] = "#2563eb"
    BLUE_700: Literal["#1d4ed8"] = "#1d4ed8"
    BLUE_800: Literal["#1e40af"] = "#1e40af"
    BLUE_900: Literal["#1e3a8a"] = "#1e3a8a"
    BLUE_950: Literal["#172554"] = "#172554"

    INDIGO_050: Literal["#eef2ff"] = "#eef2ff"
    INDIGO_100: Literal["#e0e7ff"] = "#e0e7ff"
    INDIGO_200: Literal["#c7d2fe"] = "#c7d2fe"
    INDIGO_300: Literal["#a5b4fc"] = "#a5b4fc"
    INDIGO_400: Literal["#818cf8"] = "#818cf8"
    INDIGO_500: Literal["#6366f1"] = "#6366f1"
    INDIGO_600: Literal["#4f46e5"] = "#4f46e5"
    INDIGO_700: Literal["#4338ca"] = "#4338ca"
    INDIGO_800: Literal["#3730a3"] = "#3730a3"
    INDIGO_900: Literal["#312e81"] = "#312e81"
    INDIGO_950: Literal["#1e1b4b"] = "#1e1b4b"

    VIOLET_050: Literal["#f5f3ff"] = "#f5f3ff"
    VIOLET_100: Literal["#ede9fe"] = "#ede9fe"
    VIOLET_200: Literal["#ddd6fe"] = "#ddd6fe"
    VIOLET_300: Literal["#c4b5fd"] = "#c4b5fd"
    VIOLET_400: Literal["#a78bfa"] = "#a78bfa"
    VIOLET_500: Literal["#8b5cf6"] = "#8b5cf6"
    VIOLET_600: Literal["#7c3aed"] = "#7c3aed"
    VIOLET_700: Literal["#6d28d9"] = "#6d28d9"
    VIOLET_800: Literal["#5b21b6"] = "#5b21b6"
    VIOLET_900: Literal["#4c1d95"] = "#4c1d95"
    VIOLET_950: Literal["#2e1065"] = "#2e1065"

    PURPLE_050: Literal["#faf5ff"] = "#faf5ff"
    PURPLE_100: Literal["#f3e8ff"] = "#f3e8ff"
    PURPLE_200: Literal["#e9d5ff"] = "#e9d5ff"
    PURPLE_300: Literal["#d8b4fe"] = "#d8b4fe"
    PURPLE_400: Literal["#c084fc"] = "#c084fc"
    PURPLE_500: Literal["#a855f7"] = "#a855f7"
    PURPLE_600: Literal["#9333ea"] = "#9333ea"
    PURPLE_700: Literal["#7e22ce"] = "#7e22ce"
    PURPLE_800: Literal["#6b21a8"] = "#6b21a8"
    PURPLE_900: Literal["#581c87"] = "#581c87"
    PURPLE_950: Literal["#3b0764"] = "#3b0764"

    FUCHSIA_050: Literal["#fdf4ff"] = "#fdf4ff"
    FUCHSIA_100: Literal["#fae8ff"] = "#fae8ff"
    FUCHSIA_200: Literal["#f5d0fe"] = "#f5d0fe"
    FUCHSIA_300: Literal["#f0abfc"] = "#f0abfc"
    FUCHSIA_400: Literal["#e879f9"] = "#e879f9"
    FUCHSIA_500: Literal["#d946ef"] = "#d946ef"
    FUCHSIA_600: Literal["#c026d3"] = "#c026d3"
    FUCHSIA_700: Literal["#a21caf"] = "#a21caf"
    FUCHSIA_800: Literal["#86198f"] = "#86198f"
    FUCHSIA_900: Literal["#701a75"] = "#701a75"
    FUCHSIA_950: Literal["#4a044e"] = "#4a044e"

    PINK_050: Literal["#fdf2f8"] = "#fdf2f8"
    PINK_100: Literal["#fce7f3"] = "#fce7f3"
    PINK_200: Literal["#fbcfe8"] = "#fbcfe8"
    PINK_300: Literal["#f9a8d4"] = "#f9a8d4"
    PINK_400: Literal["#f472b6"] = "#f472b6"
    PINK_500: Literal["#ec4899"] = "#ec4899"
    PINK_600: Literal["#db2777"] = "#db2777"
    PINK_700: Literal["#be185d"] = "#be185d"
    PINK_800: Literal["#9d174d"] = "#9d174d"
    PINK_900: Literal["#831843"] = "#831843"
    PINK_950: Literal["#500724"] = "#500724"

    ROSE_050: Literal["#fff1f2"] = "#fff1f2"
    ROSE_100: Literal["#ffe4e6"] = "#ffe4e6"
    ROSE_200: Literal["#fecdd3"] = "#fecdd3"
    ROSE_300: Literal["#fda4af"] = "#fda4af"
    ROSE_400: Literal["#fb7185"] = "#fb7185"
    ROSE_500: Literal["#f43f5e"] = "#f43f5e"
    ROSE_600: Literal["#e11d48"] = "#e11d48"
    ROSE_700: Literal["#be123c"] = "#be123c"
    ROSE_800: Literal["#9f1239"] = "#9f1239"
    ROSE_900: Literal["#881337"] = "#881337"
    ROSE_950: Literal["#4c0519"] = "#4c0519"


class TailwindColorsRgb(TailwindColors):
    """
    Provides all colors from TailwindCSS as RGB tuples.
    Copied from https://github.com/dostuffthatmatters/python-tailwind-colors/blob/
    3c1ac2359e3ae753875e06e68f5072586a0ae399/tailwind_colors/__init__.py

    Color palette: look for tailwindcss_v3.3.3(.png|.svg)

    ```python
    print(TAILWIND_COLORS_RGB.AMBER_600)
    # prints (217, 119, 6)
    ```
    """

    SLATE_050: tuple[Literal[248], Literal[250], Literal[252]] = (248, 250, 252)
    SLATE_100: tuple[Literal[241], Literal[245], Literal[249]] = (241, 245, 249)
    SLATE_200: tuple[Literal[226], Literal[232], Literal[240]] = (226, 232, 240)
    SLATE_300: tuple[Literal[203], Literal[213], Literal[225]] = (203, 213, 225)
    SLATE_400: tuple[Literal[148], Literal[163], Literal[184]] = (148, 163, 184)
    SLATE_500: tuple[Literal[100], Literal[116], Literal[139]] = (100, 116, 139)
    SLATE_600: tuple[Literal[71], Literal[85], Literal[105]] = (71, 85, 105)
    SLATE_700: tuple[Literal[51], Literal[65], Literal[85]] = (51, 65, 85)
    SLATE_800: tuple[Literal[30], Literal[41], Literal[59]] = (30, 41, 59)
    SLATE_900: tuple[Literal[15], Literal[23], Literal[42]] = (15, 23, 42)
    SLATE_950: tuple[Literal[2], Literal[6], Literal[23]] = (2, 6, 23)

    GRAY_050: tuple[Literal[249], Literal[250], Literal[251]] = (249, 250, 251)
    GRAY_100: tuple[Literal[243], Literal[244], Literal[246]] = (243, 244, 246)
    GRAY_200: tuple[Literal[229], Literal[231], Literal[235]] = (229, 231, 235)
    GRAY_300: tuple[Literal[209], Literal[213], Literal[219]] = (209, 213, 219)
    GRAY_400: tuple[Literal[156], Literal[163], Literal[175]] = (156, 163, 175)
    GRAY_500: tuple[Literal[107], Literal[114], Literal[128]] = (107, 114, 128)
    GRAY_600: tuple[Literal[75], Literal[85], Literal[99]] = (75, 85, 99)
    GRAY_700: tuple[Literal[55], Literal[65], Literal[81]] = (55, 65, 81)
    GRAY_800: tuple[Literal[31], Literal[41], Literal[55]] = (31, 41, 55)
    GRAY_900: tuple[Literal[17], Literal[24], Literal[39]] = (17, 24, 39)
    GRAY_950: tuple[Literal[3], Literal[7], Literal[18]] = (3, 7, 18)

    ZINC_050: tuple[Literal[250], Literal[250], Literal[250]] = (250, 250, 250)
    ZINC_100: tuple[Literal[244], Literal[244], Literal[245]] = (244, 244, 245)
    ZINC_200: tuple[Literal[228], Literal[228], Literal[231]] = (228, 228, 231)
    ZINC_300: tuple[Literal[212], Literal[212], Literal[216]] = (212, 212, 216)
    ZINC_400: tuple[Literal[161], Literal[161], Literal[170]] = (161, 161, 170)
    ZINC_500: tuple[Literal[113], Literal[113], Literal[122]] = (113, 113, 122)
    ZINC_600: tuple[Literal[82], Literal[82], Literal[91]] = (82, 82, 91)
    ZINC_700: tuple[Literal[63], Literal[63], Literal[70]] = (63, 63, 70)
    ZINC_800: tuple[Literal[39], Literal[39], Literal[42]] = (39, 39, 42)
    ZINC_900: tuple[Literal[24], Literal[24], Literal[27]] = (24, 24, 27)
    ZINC_950: tuple[Literal[9], Literal[9], Literal[11]] = (9, 9, 11)

    NEUTRAL_050: tuple[Literal[250], Literal[250], Literal[250]] = (250, 250, 250)
    NEUTRAL_100: tuple[Literal[245], Literal[245], Literal[245]] = (245, 245, 245)
    NEUTRAL_200: tuple[Literal[229], Literal[229], Literal[229]] = (229, 229, 229)
    NEUTRAL_300: tuple[Literal[212], Literal[212], Literal[212]] = (212, 212, 212)
    NEUTRAL_400: tuple[Literal[163], Literal[163], Literal[163]] = (163, 163, 163)
    NEUTRAL_500: tuple[Literal[115], Literal[115], Literal[115]] = (115, 115, 115)
    NEUTRAL_600: tuple[Literal[82], Literal[82], Literal[82]] = (82, 82, 82)
    NEUTRAL_700: tuple[Literal[64], Literal[64], Literal[64]] = (64, 64, 64)
    NEUTRAL_800: tuple[Literal[38], Literal[38], Literal[38]] = (38, 38, 38)
    NEUTRAL_900: tuple[Literal[23], Literal[23], Literal[23]] = (23, 23, 23)
    NEUTRAL_950: tuple[Literal[10], Literal[10], Literal[10]] = (10, 10, 10)

    STONE_050: tuple[Literal[250], Literal[250], Literal[249]] = (250, 250, 249)
    STONE_100: tuple[Literal[245], Literal[245], Literal[244]] = (245, 245, 244)
    STONE_200: tuple[Literal[231], Literal[229], Literal[228]] = (231, 229, 228)
    STONE_300: tuple[Literal[214], Literal[211], Literal[209]] = (214, 211, 209)
    STONE_400: tuple[Literal[168], Literal[162], Literal[158]] = (168, 162, 158)
    STONE_500: tuple[Literal[120], Literal[113], Literal[108]] = (120, 113, 108)
    STONE_600: tuple[Literal[87], Literal[83], Literal[78]] = (87, 83, 78)
    STONE_700: tuple[Literal[68], Literal[64], Literal[60]] = (68, 64, 60)
    STONE_800: tuple[Literal[41], Literal[37], Literal[36]] = (41, 37, 36)
    STONE_900: tuple[Literal[28], Literal[25], Literal[23]] = (28, 25, 23)
    STONE_950: tuple[Literal[12], Literal[10], Literal[9]] = (12, 10, 9)

    RED_050: tuple[Literal[254], Literal[242], Literal[242]] = (254, 242, 242)
    RED_100: tuple[Literal[254], Literal[226], Literal[226]] = (254, 226, 226)
    RED_200: tuple[Literal[254], Literal[202], Literal[202]] = (254, 202, 202)
    RED_300: tuple[Literal[252], Literal[165], Literal[165]] = (252, 165, 165)
    RED_400: tuple[Literal[248], Literal[113], Literal[113]] = (248, 113, 113)
    RED_500: tuple[Literal[239], Literal[68], Literal[68]] = (239, 68, 68)
    RED_600: tuple[Literal[220], Literal[38], Literal[38]] = (220, 38, 38)
    RED_700: tuple[Literal[185], Literal[28], Literal[28]] = (185, 28, 28)
    RED_800: tuple[Literal[153], Literal[27], Literal[27]] = (153, 27, 27)
    RED_900: tuple[Literal[127], Literal[29], Literal[29]] = (127, 29, 29)
    RED_950: tuple[Literal[69], Literal[10], Literal[10]] = (69, 10, 10)

    ORANGE_050: tuple[Literal[255], Literal[247], Literal[237]] = (255, 247, 237)
    ORANGE_100: tuple[Literal[255], Literal[237], Literal[213]] = (255, 237, 213)
    ORANGE_200: tuple[Literal[254], Literal[215], Literal[170]] = (254, 215, 170)
    ORANGE_300: tuple[Literal[253], Literal[186], Literal[116]] = (253, 186, 116)
    ORANGE_400: tuple[Literal[251], Literal[146], Literal[60]] = (251, 146, 60)
    ORANGE_500: tuple[Literal[249], Literal[115], Literal[22]] = (249, 115, 22)
    ORANGE_600: tuple[Literal[234], Literal[88], Literal[12]] = (234, 88, 12)
    ORANGE_700: tuple[Literal[194], Literal[65], Literal[12]] = (194, 65, 12)
    ORANGE_800: tuple[Literal[154], Literal[52], Literal[18]] = (154, 52, 18)
    ORANGE_900: tuple[Literal[124], Literal[45], Literal[18]] = (124, 45, 18)
    ORANGE_950: tuple[Literal[67], Literal[20], Literal[7]] = (67, 20, 7)

    AMBER_050: tuple[Literal[255], Literal[251], Literal[235]] = (255, 251, 235)
    AMBER_100: tuple[Literal[254], Literal[243], Literal[199]] = (254, 243, 199)
    AMBER_200: tuple[Literal[253], Literal[230], Literal[138]] = (253, 230, 138)
    AMBER_300: tuple[Literal[252], Literal[211], Literal[77]] = (252, 211, 77)
    AMBER_400: tuple[Literal[251], Literal[191], Literal[36]] = (251, 191, 36)
    AMBER_500: tuple[Literal[245], Literal[158], Literal[11]] = (245, 158, 11)
    AMBER_600: tuple[Literal[217], Literal[119], Literal[6]] = (217, 119, 6)
    AMBER_700: tuple[Literal[180], Literal[83], Literal[9]] = (180, 83, 9)
    AMBER_800: tuple[Literal[146], Literal[64], Literal[14]] = (146, 64, 14)
    AMBER_900: tuple[Literal[120], Literal[53], Literal[15]] = (120, 53, 15)
    AMBER_950: tuple[Literal[69], Literal[26], Literal[3]] = (69, 26, 3)

    YELLOW_050: tuple[Literal[254], Literal[252], Literal[232]] = (254, 252, 232)
    YELLOW_100: tuple[Literal[254], Literal[249], Literal[195]] = (254, 249, 195)
    YELLOW_200: tuple[Literal[254], Literal[240], Literal[138]] = (254, 240, 138)
    YELLOW_300: tuple[Literal[253], Literal[224], Literal[71]] = (253, 224, 71)
    YELLOW_400: tuple[Literal[250], Literal[204], Literal[21]] = (250, 204, 21)
    YELLOW_500: tuple[Literal[234], Literal[179], Literal[8]] = (234, 179, 8)
    YELLOW_600: tuple[Literal[202], Literal[138], Literal[4]] = (202, 138, 4)
    YELLOW_700: tuple[Literal[161], Literal[98], Literal[7]] = (161, 98, 7)
    YELLOW_800: tuple[Literal[133], Literal[77], Literal[14]] = (133, 77, 14)
    YELLOW_900: tuple[Literal[113], Literal[63], Literal[18]] = (113, 63, 18)
    YELLOW_950: tuple[Literal[66], Literal[32], Literal[6]] = (66, 32, 6)

    LIME_050: tuple[Literal[247], Literal[254], Literal[231]] = (247, 254, 231)
    LIME_100: tuple[Literal[236], Literal[252], Literal[203]] = (236, 252, 203)
    LIME_200: tuple[Literal[217], Literal[249], Literal[157]] = (217, 249, 157)
    LIME_300: tuple[Literal[190], Literal[242], Literal[100]] = (190, 242, 100)
    LIME_400: tuple[Literal[163], Literal[230], Literal[53]] = (163, 230, 53)
    LIME_500: tuple[Literal[132], Literal[204], Literal[22]] = (132, 204, 22)
    LIME_600: tuple[Literal[101], Literal[163], Literal[13]] = (101, 163, 13)
    LIME_700: tuple[Literal[77], Literal[124], Literal[15]] = (77, 124, 15)
    LIME_800: tuple[Literal[63], Literal[98], Literal[18]] = (63, 98, 18)
    LIME_900: tuple[Literal[54], Literal[83], Literal[20]] = (54, 83, 20)
    LIME_950: tuple[Literal[26], Literal[46], Literal[5]] = (26, 46, 5)

    GREEN_050: tuple[Literal[240], Literal[253], Literal[244]] = (240, 253, 244)
    GREEN_100: tuple[Literal[220], Literal[252], Literal[231]] = (220, 252, 231)
    GREEN_200: tuple[Literal[187], Literal[247], Literal[208]] = (187, 247, 208)
    GREEN_300: tuple[Literal[134], Literal[239], Literal[172]] = (134, 239, 172)
    GREEN_400: tuple[Literal[74], Literal[222], Literal[128]] = (74, 222, 128)
    GREEN_500: tuple[Literal[34], Literal[197], Literal[94]] = (34, 197, 94)
    GREEN_600: tuple[Literal[22], Literal[163], Literal[74]] = (22, 163, 74)
    GREEN_700: tuple[Literal[21], Literal[128], Literal[61]] = (21, 128, 61)
    GREEN_800: tuple[Literal[22], Literal[101], Literal[52]] = (22, 101, 52)
    GREEN_900: tuple[Literal[20], Literal[83], Literal[45]] = (20, 83, 45)
    GREEN_950: tuple[Literal[5], Literal[46], Literal[22]] = (5, 46, 22)

    EMERALD_050: tuple[Literal[236], Literal[253], Literal[245]] = (236, 253, 245)
    EMERALD_100: tuple[Literal[209], Literal[250], Literal[229]] = (209, 250, 229)
    EMERALD_200: tuple[Literal[167], Literal[243], Literal[208]] = (167, 243, 208)
    EMERALD_300: tuple[Literal[110], Literal[231], Literal[183]] = (110, 231, 183)
    EMERALD_400: tuple[Literal[52], Literal[211], Literal[153]] = (52, 211, 153)
    EMERALD_500: tuple[Literal[16], Literal[185], Literal[129]] = (16, 185, 129)
    EMERALD_600: tuple[Literal[5], Literal[150], Literal[105]] = (5, 150, 105)
    EMERALD_700: tuple[Literal[4], Literal[120], Literal[87]] = (4, 120, 87)
    EMERALD_800: tuple[Literal[6], Literal[95], Literal[70]] = (6, 95, 70)
    EMERALD_900: tuple[Literal[6], Literal[78], Literal[59]] = (6, 78, 59)
    EMERALD_950: tuple[Literal[2], Literal[44], Literal[34]] = (2, 44, 34)

    TEAL_050: tuple[Literal[240], Literal[253], Literal[250]] = (240, 253, 250)
    TEAL_100: tuple[Literal[204], Literal[251], Literal[241]] = (204, 251, 241)
    TEAL_200: tuple[Literal[153], Literal[246], Literal[228]] = (153, 246, 228)
    TEAL_300: tuple[Literal[94], Literal[234], Literal[212]] = (94, 234, 212)
    TEAL_400: tuple[Literal[45], Literal[212], Literal[191]] = (45, 212, 191)
    TEAL_500: tuple[Literal[20], Literal[184], Literal[166]] = (20, 184, 166)
    TEAL_600: tuple[Literal[13], Literal[148], Literal[136]] = (13, 148, 136)
    TEAL_700: tuple[Literal[15], Literal[118], Literal[110]] = (15, 118, 110)
    TEAL_800: tuple[Literal[17], Literal[94], Literal[89]] = (17, 94, 89)
    TEAL_900: tuple[Literal[19], Literal[78], Literal[74]] = (19, 78, 74)
    TEAL_950: tuple[Literal[4], Literal[47], Literal[46]] = (4, 47, 46)

    CYAN_050: tuple[Literal[236], Literal[254], Literal[255]] = (236, 254, 255)
    CYAN_100: tuple[Literal[207], Literal[250], Literal[254]] = (207, 250, 254)
    CYAN_200: tuple[Literal[165], Literal[243], Literal[252]] = (165, 243, 252)
    CYAN_300: tuple[Literal[103], Literal[232], Literal[249]] = (103, 232, 249)
    CYAN_400: tuple[Literal[34], Literal[211], Literal[238]] = (34, 211, 238)
    CYAN_500: tuple[Literal[6], Literal[182], Literal[212]] = (6, 182, 212)
    CYAN_600: tuple[Literal[8], Literal[145], Literal[178]] = (8, 145, 178)
    CYAN_700: tuple[Literal[14], Literal[116], Literal[144]] = (14, 116, 144)
    CYAN_800: tuple[Literal[21], Literal[94], Literal[117]] = (21, 94, 117)
    CYAN_900: tuple[Literal[22], Literal[78], Literal[99]] = (22, 78, 99)
    CYAN_950: tuple[Literal[8], Literal[51], Literal[68]] = (8, 51, 68)

    SKY_050: tuple[Literal[240], Literal[249], Literal[255]] = (240, 249, 255)
    SKY_100: tuple[Literal[224], Literal[242], Literal[254]] = (224, 242, 254)
    SKY_200: tuple[Literal[186], Literal[230], Literal[253]] = (186, 230, 253)
    SKY_300: tuple[Literal[125], Literal[211], Literal[252]] = (125, 211, 252)
    SKY_400: tuple[Literal[56], Literal[189], Literal[248]] = (56, 189, 248)
    SKY_500: tuple[Literal[14], Literal[165], Literal[233]] = (14, 165, 233)
    SKY_600: tuple[Literal[2], Literal[132], Literal[199]] = (2, 132, 199)
    SKY_700: tuple[Literal[3], Literal[105], Literal[161]] = (3, 105, 161)
    SKY_800: tuple[Literal[7], Literal[89], Literal[133]] = (7, 89, 133)
    SKY_900: tuple[Literal[12], Literal[74], Literal[110]] = (12, 74, 110)
    SKY_950: tuple[Literal[8], Literal[47], Literal[73]] = (8, 47, 73)

    BLUE_050: tuple[Literal[239], Literal[246], Literal[255]] = (239, 246, 255)
    BLUE_100: tuple[Literal[219], Literal[234], Literal[254]] = (219, 234, 254)
    BLUE_200: tuple[Literal[191], Literal[219], Literal[254]] = (191, 219, 254)
    BLUE_300: tuple[Literal[147], Literal[197], Literal[253]] = (147, 197, 253)
    BLUE_400: tuple[Literal[96], Literal[165], Literal[250]] = (96, 165, 250)
    BLUE_500: tuple[Literal[59], Literal[130], Literal[246]] = (59, 130, 246)
    BLUE_600: tuple[Literal[37], Literal[99], Literal[235]] = (37, 99, 235)
    BLUE_700: tuple[Literal[29], Literal[78], Literal[216]] = (29, 78, 216)
    BLUE_800: tuple[Literal[30], Literal[64], Literal[175]] = (30, 64, 175)
    BLUE_900: tuple[Literal[30], Literal[58], Literal[138]] = (30, 58, 138)
    BLUE_950: tuple[Literal[23], Literal[37], Literal[84]] = (23, 37, 84)

    INDIGO_050: tuple[Literal[238], Literal[242], Literal[255]] = (238, 242, 255)
    INDIGO_100: tuple[Literal[224], Literal[231], Literal[255]] = (224, 231, 255)
    INDIGO_200: tuple[Literal[199], Literal[210], Literal[254]] = (199, 210, 254)
    INDIGO_300: tuple[Literal[165], Literal[180], Literal[252]] = (165, 180, 252)
    INDIGO_400: tuple[Literal[129], Literal[140], Literal[248]] = (129, 140, 248)
    INDIGO_500: tuple[Literal[99], Literal[102], Literal[241]] = (99, 102, 241)
    INDIGO_600: tuple[Literal[79], Literal[70], Literal[229]] = (79, 70, 229)
    INDIGO_700: tuple[Literal[67], Literal[56], Literal[202]] = (67, 56, 202)
    INDIGO_800: tuple[Literal[55], Literal[48], Literal[163]] = (55, 48, 163)
    INDIGO_900: tuple[Literal[49], Literal[46], Literal[129]] = (49, 46, 129)
    INDIGO_950: tuple[Literal[30], Literal[27], Literal[75]] = (30, 27, 75)

    VIOLET_050: tuple[Literal[245], Literal[243], Literal[255]] = (245, 243, 255)
    VIOLET_100: tuple[Literal[237], Literal[233], Literal[254]] = (237, 233, 254)
    VIOLET_200: tuple[Literal[221], Literal[214], Literal[254]] = (221, 214, 254)
    VIOLET_300: tuple[Literal[196], Literal[181], Literal[253]] = (196, 181, 253)
    VIOLET_400: tuple[Literal[167], Literal[139], Literal[250]] = (167, 139, 250)
    VIOLET_500: tuple[Literal[139], Literal[92], Literal[246]] = (139, 92, 246)
    VIOLET_600: tuple[Literal[124], Literal[58], Literal[237]] = (124, 58, 237)
    VIOLET_700: tuple[Literal[109], Literal[40], Literal[217]] = (109, 40, 217)
    VIOLET_800: tuple[Literal[91], Literal[33], Literal[182]] = (91, 33, 182)
    VIOLET_900: tuple[Literal[76], Literal[29], Literal[149]] = (76, 29, 149)
    VIOLET_950: tuple[Literal[46], Literal[16], Literal[101]] = (46, 16, 101)

    PURPLE_050: tuple[Literal[250], Literal[245], Literal[255]] = (250, 245, 255)
    PURPLE_100: tuple[Literal[243], Literal[232], Literal[255]] = (243, 232, 255)
    PURPLE_200: tuple[Literal[233], Literal[213], Literal[255]] = (233, 213, 255)
    PURPLE_300: tuple[Literal[216], Literal[180], Literal[254]] = (216, 180, 254)
    PURPLE_400: tuple[Literal[192], Literal[132], Literal[252]] = (192, 132, 252)
    PURPLE_500: tuple[Literal[168], Literal[85], Literal[247]] = (168, 85, 247)
    PURPLE_600: tuple[Literal[147], Literal[51], Literal[234]] = (147, 51, 234)
    PURPLE_700: tuple[Literal[126], Literal[34], Literal[206]] = (126, 34, 206)
    PURPLE_800: tuple[Literal[107], Literal[33], Literal[168]] = (107, 33, 168)
    PURPLE_900: tuple[Literal[88], Literal[28], Literal[135]] = (88, 28, 135)
    PURPLE_950: tuple[Literal[59], Literal[7], Literal[100]] = (59, 7, 100)

    FUCHSIA_050: tuple[Literal[253], Literal[244], Literal[255]] = (253, 244, 255)
    FUCHSIA_100: tuple[Literal[250], Literal[232], Literal[255]] = (250, 232, 255)
    FUCHSIA_200: tuple[Literal[245], Literal[208], Literal[254]] = (245, 208, 254)
    FUCHSIA_300: tuple[Literal[240], Literal[171], Literal[252]] = (240, 171, 252)
    FUCHSIA_400: tuple[Literal[232], Literal[121], Literal[249]] = (232, 121, 249)
    FUCHSIA_500: tuple[Literal[217], Literal[70], Literal[239]] = (217, 70, 239)
    FUCHSIA_600: tuple[Literal[192], Literal[38], Literal[211]] = (192, 38, 211)
    FUCHSIA_700: tuple[Literal[162], Literal[28], Literal[175]] = (162, 28, 175)
    FUCHSIA_800: tuple[Literal[134], Literal[25], Literal[143]] = (134, 25, 143)
    FUCHSIA_900: tuple[Literal[112], Literal[26], Literal[117]] = (112, 26, 117)
    FUCHSIA_950: tuple[Literal[74], Literal[4], Literal[78]] = (74, 4, 78)

    PINK_050: tuple[Literal[253], Literal[242], Literal[248]] = (253, 242, 248)
    PINK_100: tuple[Literal[252], Literal[231], Literal[243]] = (252, 231, 243)
    PINK_200: tuple[Literal[251], Literal[207], Literal[232]] = (251, 207, 232)
    PINK_300: tuple[Literal[249], Literal[168], Literal[212]] = (249, 168, 212)
    PINK_400: tuple[Literal[244], Literal[114], Literal[182]] = (244, 114, 182)
    PINK_500: tuple[Literal[236], Literal[72], Literal[153]] = (236, 72, 153)
    PINK_600: tuple[Literal[219], Literal[39], Literal[119]] = (219, 39, 119)
    PINK_700: tuple[Literal[190], Literal[24], Literal[93]] = (190, 24, 93)
    PINK_800: tuple[Literal[157], Literal[23], Literal[77]] = (157, 23, 77)
    PINK_900: tuple[Literal[131], Literal[24], Literal[67]] = (131, 24, 67)
    PINK_950: tuple[Literal[80], Literal[7], Literal[36]] = (80, 7, 36)

    ROSE_050: tuple[Literal[255], Literal[241], Literal[242]] = (255, 241, 242)
    ROSE_100: tuple[Literal[255], Literal[228], Literal[230]] = (255, 228, 230)
    ROSE_200: tuple[Literal[254], Literal[205], Literal[211]] = (254, 205, 211)
    ROSE_300: tuple[Literal[253], Literal[164], Literal[175]] = (253, 164, 175)
    ROSE_400: tuple[Literal[251], Literal[113], Literal[133]] = (251, 113, 133)
    ROSE_500: tuple[Literal[244], Literal[63], Literal[94]] = (244, 63, 94)
    ROSE_600: tuple[Literal[225], Literal[29], Literal[72]] = (225, 29, 72)
    ROSE_700: tuple[Literal[190], Literal[18], Literal[60]] = (190, 18, 60)
    ROSE_800: tuple[Literal[159], Literal[18], Literal[57]] = (159, 18, 57)
    ROSE_900: tuple[Literal[136], Literal[19], Literal[55]] = (136, 19, 55)
    ROSE_950: tuple[Literal[76], Literal[5], Literal[25]] = (76, 5, 25)


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
    df: pd.DataFrame,
    composed_start_column: str = "composed_start",
    composed_end_column: str = "composed_end",
    name: str = "mean_composition_year",
) -> pd.Series:
    """Expects a dataframe containing ``year_column`` and computes its means by grouping on the first index level
    ('corpus' by default).
    Returns the result as a series where the index contains corpus names and the values are mean composition years.
    """
    years = get_middle_composition_year(df, composed_start_column, composed_end_column)
    return years.groupby(level=0).mean().sort_values().rename(name)


def chronological_corpus_order(
    df: pd.DataFrame, year_column: str = "composed_end"
) -> List[str]:
    """Expects a dataframe containing ``year_column`` and corpus names in the first index level.
    Returns the corpus names in chronological order
    """
    mean_composition_years = corpus_mean_composition_years(
        df=df, composed_end_column=year_column
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


def make_output_path(
    filename: str,
    extension=None,
    path=None,
) -> str:
    if extension:
        extension = "." + extension.lstrip(".")
    else:
        extension = DEFAULT_OUTPUT_FORMAT
    file = f"{filename}{extension}"
    if path:
        return resolve_dir(os.path.join(path, file))
    return file


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


def merge_index_levels(index, join_str=True):
    if index.nlevels > 1:
        return merge_columns_into_one(index.to_frame(index=False), join_str=join_str)
    return index


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


def plot_pca(
    data,
    info="data",
    show_features=20,
    color="corpus",
    symbol=None,
    size=None,
    **kwargs,
) -> Optional[go.Figure]:
    phrase_pca = PCA(3)
    decomposed_phrases = pd.DataFrame(
        phrase_pca.fit_transform(data), index=data.index, columns=["c1", "c2", "c3"]
    )
    print(
        f"Explained variance ratio: {phrase_pca.explained_variance_ratio_} "
        f"({phrase_pca.explained_variance_ratio_.sum():.1%})"
    )
    concatenate_this = [decomposed_phrases]
    hover_data = list(data.index.names)
    if color is not None:
        if isinstance(color, pd.Series):
            concatenate_this.append(color)
            color = color.name
        hover_data.append(color)
    if symbol is not None:
        if isinstance(symbol, pd.Series):
            concatenate_this.append(symbol)
            symbol = symbol.name
        hover_data.append(symbol)

    if size is None:
        constant_size = 3
    elif isinstance(size, Number):
        constant_size = size
    else:
        constant_size = 0
        if isinstance(size, pd.Series):
            concatenate_this.append(size)
            size = size.name
        hover_data.append(size)
    if len(concatenate_this) > 1:
        scatter_data = pd.concat(concatenate_this, axis=1).reset_index()
    else:
        scatter_data = decomposed_phrases
    fig = px.scatter_3d(
        scatter_data.reset_index(),
        x="c1",
        y="c2",
        z="c3",
        color=color,
        symbol=symbol,
        hover_data=hover_data,
        hover_name=hover_data[-1],
        title=f"3 principal components of the {info}",
        height=800,
        **kwargs,
    )
    marker_settings = dict(opacity=0.3)
    if constant_size:
        marker_settings["size"] = constant_size
    update_figure_layout(
        fig,
        legend={"itemsizing": "constant"},
        traces_settings=dict(marker=marker_settings),
    )
    if show_features < 1:
        return fig
    fig.show()
    for i in range(3):
        index = merge_index_levels(data.columns)
        component = pd.Series(
            phrase_pca.components_[i], index=index, name="coefficient"
        ).sort_values(ascending=False, key=abs)
        fig = px.bar(
            component.iloc[:show_features],
            labels=dict(index="feature", value="coefficient"),
            title=f"{show_features} most weighted features of component {i+1}",
        )
        fig.show()


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


def prepare_tf_idf_data(
    long_format_data: pd.DataFrame,
    index: str | List[str],
    columns: str | List[str],
    smooth=1e-20,
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    unigram_distribution = (
        long_format_data.groupby(columns).duration_qb.sum().sort_values(ascending=False)
    )
    absolute_frequency_matrix = long_format_data.pivot_table(
        index=index,
        columns=columns,
        values="duration_qb",
        aggfunc="sum",
    )
    tf = (
        absolute_frequency_matrix.fillna(0.0)
        .add(smooth)
        .div(absolute_frequency_matrix.sum(axis=1), axis=0)
    )  # term frequency
    (
        D,
        V,
    ) = absolute_frequency_matrix.shape  # D = number of documents, V = vocabulary size
    df = (
        absolute_frequency_matrix.notna().sum().sort_values(ascending=False)
    )  # absolute document frequency
    f = absolute_frequency_matrix.fillna(0.0)
    idf = pd.Series(np.log(D / df), index=df.index)  # inverse document frequency
    return unigram_distribution, f, tf, df, idf


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
    rank_index: bool = False,
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
    if rank_index:
        df = df.reset_index()
        df.index += 1
        df.index.rename("rank", inplace=True)
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


def add_bass_degree_columns(feature_df):
    feature_df = extend_bass_notes_feature(feature_df)
    feature_df = add_chord_tone_intervals(feature_df)
    return feature_df


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


# region phrase stage helpers


def make_stage_data(
    phrase_feature,
    columns="chord",
    components="body",
    drop_levels=3,
    reverse=True,
    level_name="stage",
    wide_format=True,
    query=None,
) -> resources.PhraseData:
    """Function sets the defaults for the stage TSVs produced in the following."""
    phrase_data = phrase_feature.get_phrase_data(
        columns=columns,
        components=components,
        drop_levels=drop_levels,
        reverse=reverse,
        level_name=level_name,
        wide_format=wide_format,
        query=query,
    )
    return phrase_data


def get_max_range(widths) -> Tuple[int, int, int]:
    """Index range capturing the first until last occurrence of the maximum value."""
    maximum, first_ix, last_ix = 0, 0, 0
    for i, width in enumerate(widths):
        if width > maximum:
            maximum = width
            first_ix = i
            last_ix = i
        elif width == maximum:
            last_ix = i
    return first_ix, last_ix + 1, maximum


def merge_up_to_max_width(
    lowest_tpc: npt.NDArray, tpc_width: npt.NDArray, largest: int
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Spans = lowest_tpc + tpc_width. Spans greater equal ``largest`` are left untouched. Smaller spans are merged as
    long as the merge does not result in a range larger than ``largest``.
    """
    lowest, highest = None, None
    merge_n = 0
    result_l, result_w = [], []

    def do_merge():
        """Add the readily merged section to the results, reset the counters."""
        nonlocal lowest, highest, merge_n
        if merge_n:
            result_l.extend([lowest] * merge_n)
            result_w.extend([highest - lowest] * merge_n)
            lowest, highest = None, None
            merge_n = 0

    for low, width in zip(lowest_tpc, tpc_width):
        if width > largest:
            do_merge()
            result_l.append(low)
            result_w.append(width)
            continue
        high = low + width
        if lowest is None:
            # start new merge range
            lowest = low
            highest = high
            merge_n += 1
            continue
        merge_low_point = min((low, lowest))
        merge_high_point = max((high, highest))
        merge_width = merge_high_point - merge_low_point
        if merge_width <= largest:
            # merge
            lowest = merge_low_point
            highest = merge_high_point
        else:
            do_merge()
            lowest = low
            highest = high
        merge_n += 1
    do_merge()
    return result_l, result_w


def _compute_smallest_fifth_ranges(
    lowest_tpc: npt.NDArray,
    tpc_width: npt.NDArray,
    smallest: int = 6,
    largest: int = 9,
    verbose: bool = False,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Recursively groups the given TPC "hull" into diatonic bands. Each entry of the two arrays represents a chord,
    in a way that lowest_tpc is the lowest tonal pitch class on the line of fifths, and tpc_width (by addition)
    represents the distance to the highest tonal pitch class on the line of fifths.

    Recursive mechanism:

    * Stop criterion: if ``max(tpc_width) ≤ smallest``, merge the whole hull into one band up to a
      range of ``smallest``.
    * Otherwise split the hull in three parts: left, middle, right: Middle spans the left_most to the
      right-most occurrence of max(tpc_width). Process the middle part by leaving all spans ``≥ largest`` untouched
      and merging smaller spans as long as the merge does not result in a range larger than ``largest``.
    * Process left and right recursively.

    Args:
        lowest_tpc: Lowest tonal pitch class on the line of fifths for each chord in the sequence.
        tpc_width: For each chord, the span of tonal pitch classes on the line of fifths.
        smallest:
            Stop criterion: if ``max(tpc_width) <= smallest``, merge the whole hull into one band
            up to a range of ``smallest``. Defaults to 6, which corresponds to 6 fifths, i.e., 7 tones of a diatonic.
        largest:
            Merge adjacent spans up to this range. Defaults to 9, which corresponds to 9 fifths, i.e.,
            10 tones of a major-minor extended diatonic.
        verbose:
            Print log messages.

    Returns:
        Hull representing merged spans of tonal pitch classes on the line of fifths.
    """
    if len(lowest_tpc) < 2:
        return lowest_tpc, tpc_width
    first_max_ix, last_max_ix, max_val = get_max_range(tpc_width)
    if verbose:
        print(f"max({tpc_width}) = {max_val}, [{first_max_ix}:{last_max_ix}]")
    if max_val <= smallest:
        if verbose:
            print(
                f"Calling merge_up_to_max_width({lowest_tpc}, {tpc_width}) because max_val {max_val} < {largest}"
            )
        return merge_up_to_max_width(lowest_tpc, tpc_width, largest=smallest)
    left_l, left_w = _compute_smallest_fifth_ranges(
        lowest_tpc[:first_max_ix],
        tpc_width[:first_max_ix],
        smallest=smallest,
        largest=largest,
        verbose=verbose,
    )
    middle_l, middle_w = merge_up_to_max_width(
        lowest_tpc[first_max_ix:last_max_ix],
        tpc_width[first_max_ix:last_max_ix],
        largest=largest,
    )
    right_l, right_w = _compute_smallest_fifth_ranges(
        lowest_tpc[last_max_ix:],
        tpc_width[last_max_ix:],
        smallest=smallest,
        largest=largest,
        verbose=verbose,
    )
    result_l = np.concatenate([left_l, middle_l, right_l])
    result_w = np.concatenate([left_w, middle_w, right_w])
    return result_l, result_w


def compute_smallest_diatonics(
    phrase_data: resources.PhraseData,
    smallest: int = 6,
    largest: int = 9,
    verbose: bool = False,
) -> pd.DataFrame:
    """Recursively computes diatonic bands based on a lower and an upper bound.

    Args:
        phrase_data: PhraseData for a single phrase (requires the columns 'lowest_tpc' and 'tpc_width').
        smallest:
            Stop criterion: if ``max(tpc_width) <= smallest``, merge the whole hull into
            bands spanning ``≤ smallest`` fifths. Defaults to 6, which corresponds to 6 fifths,
            i.e., 7 tones of a diatonic.
        largest:
            Merge adjacent spans up to this range. Defaults to 9, which corresponds to 9 fifths, i.e.,
            10 tones of a major-minor extended diatonic.
        verbose:
            Print log messages.

    Returns:
        A DataFrame with columns 'lowest_tpc' and 'tpc_width' representing the merged spans of tonal
        pitch classes for the given chord sequence.
    """
    lowest, widths = _compute_smallest_fifth_ranges(
        phrase_data.lowest_tpc.values,
        phrase_data.tpc_width.values,
        smallest=smallest,
        largest=largest,
        verbose=verbose,
    )
    return pd.DataFrame(
        dict(lowest_tpc=lowest, tpc_width=widths), index=phrase_data.index
    )


def make_criterion(
    phrase_feature: resources.PhraseAnnotations
    | resources.PhraseComponents
    | resources.PhraseLabels,
    criterion_name: Optional[str] = None,
    columns="chord",
    components="body",
    drop_levels=3,
    reverse=True,
    level_name="stage",
    query=None,
    join_str: Optional[str | bool] = None,
    fillna: Optional[Hashable] = None,
) -> pd.Series:
    """Convenience function for calling ``.get_phrase_data()`` with certain defaults and merging
    the resulting columns into one (when multiple).
    """
    phrase_data = phrase_feature.get_phrase_data(
        columns=columns,
        components=components,
        drop_levels=drop_levels,
        reverse=reverse,
        level_name=level_name,
        wide_format=False,
        query=query,
    )
    if not isinstance(columns, str) and len(columns) > 1:
        phrase_data = merge_columns_into_one(
            phrase_data, join_str=join_str, fillna=fillna
        )
        if criterion_name is None:
            criterion_name = "_and_".join(columns)
    else:
        phrase_data = phrase_data.iloc(axis=1)[0]
        if criterion_name is None:
            if isinstance(columns, str):
                criterion_name = columns
            else:
                criterion_name = columns[0]
    result = phrase_data.rename(criterion_name)
    return result


def make_criterion_stages(
    phrase_annotations: resources.PhraseAnnotations,
    criteria_dict: Dict[str, str | List[str]],
    join_str=True,
):
    """Takes a {name -> [columns]} dict."""
    uncompressed = make_stage_data(
        phrase_annotations,
        columns=["chord_and_mode", "duration_qb"],
        wide_format=False,
    )
    name2phrase_data = {"uncompressed": uncompressed}
    for name, columns in criteria_dict.items():
        criterion = make_criterion(
            phrase_annotations,
            columns=columns,
            criterion_name=name,
            join_str=join_str,
        )
        name2phrase_data[name] = uncompressed.regroup_phrases(criterion)
    return name2phrase_data


def get_stage_durations(phrase_data: resources.PhraseData):
    return phrase_data.groupby(
        ["corpus", "piece", "phrase_id", "stage"]
    ).duration_qb.sum()


def get_criterion_phrase_lengths(phrase_data: resources.PhraseData):
    """In terms of number of stages after merging."""
    stage_index = phrase_data.index.to_frame(index=False)
    phrase_id_col = stage_index.columns.get_loc("phrase_id")
    groupby = stage_index.columns.to_list()[: phrase_id_col + 1]
    stage_lengths = stage_index.groupby(groupby).stage.max() + 1
    return stage_lengths.rename("phrase_length")


def get_criterion_phrase_entropies(
    phrase_data: resources.PhraseData, criterion_name: Optional[str] = None
):
    if not criterion_name:
        criterion_name = phrase_data.columns.to_list()[0]
    criterion_distributions = phrase_data.groupby(
        ["corpus", criterion_name]
    ).duration_qb.sum()
    return criterion_distributions.groupby("corpus").agg(_entropy).rename("entropy")


def get_metrics_means(name2phrase_data: Dict[str, resources.PhraseData]):
    criterion_metric2value = {}
    for name, stages in name2phrase_data.items():
        stage_durations = get_stage_durations(stages)
        criterion_metric2value[
            (name, "mean stage duration", "mean")
        ] = stage_durations.mean()
        criterion_metric2value[
            (name, "mean stage duration", "sem")
        ] = stage_durations.sem()
        phrase_lengths = get_criterion_phrase_lengths(stages)
        criterion_metric2value[
            (name, "mean phrase length", "mean")
        ] = phrase_lengths.mean()
        criterion_metric2value[
            (name, "mean phrase length", "sem")
        ] = phrase_lengths.sem()
        phrase_entropies = get_criterion_phrase_entropies(stages)
        criterion_metric2value[
            (name, "mean phrase entropy", "mean")
        ] = phrase_entropies.mean()
        criterion_metric2value[
            (name, "mean phrase entropy", "sem")
        ] = phrase_entropies.sem()
    metrics = pd.Series(criterion_metric2value, name="value").unstack(sort=False)
    metrics.index.names = ["criterion", "metric"]
    return metrics


def compare_criteria_metrics(
    name2phrase_data: Dict[str, resources.PhraseData], **kwargs
):
    metrics = get_metrics_means(name2phrase_data).reset_index()
    return make_bar_plot(
        metrics,
        facet_row="metric",
        color="criterion",
        x_col="mean",
        y_col="criterion",
        x_axis=dict(matches=None, showticklabels=True),
        layout=dict(showlegend=False),
        error_x="sem",
        orientation="h",
        labels=dict(entropy="entropy of stage distributions in bits", corpus=""),
        **kwargs,
    )


def plot_corpuswise_criteria_means(
    criterion2values: Dict[str, pd.Series],
    category_title="stage_type",
    y_axis_label="mean duration of stages in ♩",
    chronological_corpus_names: Optional[List[str]] = None,
    **kwargs,
):
    """Takes a {trace_name -> values} dict where each entry will be turned into a bar plot trace for comparison."""
    aggregated = {
        name: durations.groupby("corpus").agg(["mean", "sem"])
        for name, durations in criterion2values.items()
    }
    df = pd.concat(aggregated, names=[category_title])
    corpora = df.index.get_level_values("corpus").unique()
    if chronological_corpus_names is not None:
        corpus_order = [
            corpus for corpus in chronological_corpus_names if corpus in corpora
        ]
    else:
        corpus_order = corpora
    return make_bar_plot(
        df,
        x_col="corpus",
        y_col="mean",
        error_y="sem",
        color=category_title,
        category_orders=dict(corpus=corpus_order),
        labels=dict(mean=y_axis_label, corpus=""),
        **kwargs,
    )


def plot_corpuswise_criteria(
    criterion2values,
    category_title="stage_type",
    y_axis_label="entropy of stage distributions in bits",
    chronological_corpus_names: Optional[List[str]] = None,
    **kwargs,
):
    """Takes a {trace_name -> PhraseData} dict where each entry will be turned into a bar plot trace for comparison."""
    df = pd.concat(criterion2values, names=[category_title])
    corpora = df.index.get_level_values("corpus").unique()
    if chronological_corpus_names is not None:
        corpus_order = [
            corpus for corpus in chronological_corpus_names if corpus in corpora
        ]
    else:
        corpus_order = corpora
    return make_bar_plot(
        df,
        x_col="corpus",
        y_col="entropy",
        color=category_title,
        category_orders=dict(corpus=corpus_order),
        labels=dict(entropy=y_axis_label, corpus=""),
        **kwargs,
    )


def _compare_criteria_stage_durations(
    name2phrase_data: Dict[str, resources.PhraseData],
    chronological_corpus_names: Optional[List[str]] = None,
):
    durations_dict = {
        name: get_stage_durations(stages) for name, stages in name2phrase_data.items()
    }
    return plot_corpuswise_criteria_means(
        durations_dict,
        chronological_corpus_names=chronological_corpus_names,
        height=800,
    )


def compare_criteria_stage_durations(
    phrase_annotations: resources.PhraseAnnotations,
    criteria_dict: Dict[str, str | List[str]],
    join_str=True,
    chronological_corpus_names: Optional[List[str]] = None,
):
    name2phrase_data = make_criterion_stages(
        phrase_annotations, criteria_dict, join_str=join_str
    )
    return _compare_criteria_stage_durations(
        name2phrase_data, chronological_corpus_names=chronological_corpus_names
    )


def _compare_criteria_phrase_lengths(
    name2phrase_data: Dict[str, resources.PhraseData],
    chronological_corpus_names: Optional[List[str]] = None,
):
    lengths_dict = {
        name: get_criterion_phrase_lengths(durations)
        for name, durations in name2phrase_data.items()
    }
    return plot_corpuswise_criteria_means(
        lengths_dict,
        y_axis_label="mean number of stages per phrase",
        height=800,
        chronological_corpus_names=chronological_corpus_names,
    )


def compare_criteria_phrase_lengths(
    phrase_annotations: resources.PhraseAnnotations,
    criteria_dict: Dict[str, str | List[str]],
    join_str=True,
    chronological_corpus_names: Optional[List[str]] = None,
):
    name2phrase_data = make_criterion_stages(
        phrase_annotations, criteria_dict, join_str=join_str
    )
    return _compare_criteria_phrase_lengths(
        name2phrase_data, chronological_corpus_names=chronological_corpus_names
    )


def _compare_criteria_entropies(
    name2phrase_data: Dict[str, resources.PhraseData],
    chronological_corpus_names: Optional[List[str]] = None,
):
    entropies = {
        name: get_criterion_phrase_entropies(durations)
        for name, durations in name2phrase_data.items()
    }
    return plot_corpuswise_criteria(
        entropies, chronological_corpus_names=chronological_corpus_names, height=800
    )


def compare_criteria_entropies(
    phrase_annotations,
    criteria_dict,
    join_str=True,
    chronological_corpus_names: Optional[List[str]] = None,
):
    name2phrase_data = make_criterion_stages(
        phrase_annotations, criteria_dict, join_str=join_str
    )
    return _compare_criteria_entropies(
        name2phrase_data, chronological_corpus_names=chronological_corpus_names
    )


def make_dominant_selector(phrase_data):
    """Phrase data must have columns 'numeral', 'chord_type', 'effective_localkey_is_minor'"""
    is_dominant = phrase_data.numeral.eq("V") & phrase_data.chord_type.isin(
        {"Mm7", "M", "Fr"}
    )
    leading_tone_is_root = (
        phrase_data.numeral.eq("#vii") & phrase_data.effective_localkey_is_minor
    ) | (phrase_data.numeral.eq("vii") & ~phrase_data.effective_localkey_is_minor)
    is_rootless_dominant = leading_tone_is_root & phrase_data.chord_type.isin(
        {"o", "o7", "%7", "Ger", "It"}
    )
    dominant_selector = is_dominant | is_rootless_dominant
    return dominant_selector


def get_phrase_chord_tones(
    phrase_annotations: resources.PhraseAnnotations,
    additional_columns: Optional[Iterable[str]] = None,
    query: Optional[str] = None,
) -> resources.PhraseData:
    """"""
    columns = [
        "label",
        "duration_qb",
        "chord",
        "localkey",
        "effective_localkey",
        "globalkey",
        "globalkey_is_minor",
        "chord_tones",
    ]
    if additional_columns is not None:
        column_extension = [c for c in additional_columns if c not in columns]
        add_relative_chord_tones = "chord_tones" in column_extension
        columns.extend(column_extension)
    else:
        add_relative_chord_tones = False
    chord_tones = phrase_annotations.get_phrase_data(
        reverse=True, columns=columns, drop_levels="phrase_component", query=query
    )
    df = chord_tones.df
    df.chord_tones.where(df.chord_tones != (), inplace=True)
    df.chord_tones.ffill(inplace=True)
    df = ms3.transpose_chord_tones_by_localkey(df, by_global=True).rename(
        columns=dict(chord_tones="chord_tone_tpcs")
    )
    if add_relative_chord_tones:
        df = pd.concat([df, chord_tones.df.chord_tones], axis=1)
    df["lowest_tpc"] = df.chord_tone_tpcs.map(min)
    df["highest_tpc"] = df.chord_tone_tpcs.map(max)
    df["tpc_width"] = df.highest_tpc - df.lowest_tpc
    return chord_tones.from_resource_and_dataframe(chord_tones, df)


# endregion phrase stage helpers
# region phrase Gantt helpers
def make_start_finish(duration_qb: pd.Series) -> pd.DataFrame:
    """Turns a duration_qb column into a dataframe with a Start and Finish column for use in a Gantt chart.
    The timestamps are negative quarterbeats leading up to 0, which is the end of the phrase. The ultima, which has
    duration 0 because its duration is part of the codetta, is assigned a duration of 1 for plotting.
    """
    starts = (-duration_qb.cumsum()).rename("Start")  # .astype("datetime64[s]")
    ends = starts.shift().fillna(1).rename("Finish")  # .astype("datetime64[s]")
    return pd.DataFrame({"Start": starts, "Finish": ends})


def plot_phrase(
    phrase_timeline_data,
    colorscale=None,
    shapes: Optional[List[dict]] = None,
    layout: Optional[dict] = None,
    font_size: Optional[int] = None,  # for everything
    textfont_size: Optional[
        int
    ] = None,  # for traces, independently of axis labels, legends, etc.
    x_axis: Optional[dict] = None,
    y_axis: Optional[dict] = None,
    color_axis: Optional[dict] = None,
    traces_settings: Optional[dict] = None,
) -> go.Figure:
    """Timeline (Gantt) data for a single phrase."""
    dummy_resource_value = phrase_timeline_data.Resource.iat[0]
    phrase_timeline_data = fill_yaxis_gaps(
        phrase_timeline_data, "chord_tone_tpc", Resource=dummy_resource_value
    )
    if phrase_timeline_data.Task.isna().any():
        names = ms3.transform(phrase_timeline_data.chord_tone_tpc, ms3.fifths2name)
        phrase_timeline_data.Task.fillna(names, inplace=True)
    corpus, piece, phrase_id, *_ = phrase_timeline_data.index[0]
    globalkey = phrase_timeline_data.globalkey.iat[0]
    title = f"Phrase {phrase_id} from {corpus}/{piece} ({globalkey})"
    kwargs = dict(title=title, colors=colorscale)
    if shapes:
        kwargs["shapes"] = shapes
    fig = create_gantt(
        phrase_timeline_data.sort_values("chord_tone_tpc", ascending=False), **kwargs
    )
    # fig.update_layout(hovermode="x", legend_traceorder="grouped")
    # fig.update_traces(hovertemplate="Task: %{text}<br>Start: %{x}<br>Finish: %{y}")
    if "timesig" in phrase_timeline_data.columns and "mn_onset" in phrase_timeline_data:
        timesigs = phrase_timeline_data.timesig.dropna().unique()
        if len(timesigs) == 1:
            timesig = timesigs[0]
            measure_duration = Fraction(timesig) * 4.0
            phrase_end_offset = -phrase_timeline_data.mn_onset.iat[0] * 4.0
            if x_axis is None:
                x_axis = dict(
                    dtick=measure_duration,
                    tick0=phrase_end_offset,
                )
            else:
                if "dtick" not in x_axis:
                    x_axis["dtick"] = measure_duration
                if "tick0" not in x_axis:
                    x_axis["tick0"] = phrase_end_offset
    update_figure_layout(
        fig,
        layout=layout,
        font_size=font_size,
        textfont_size=textfont_size,
        x_axis=x_axis,
        y_axis=y_axis,
        color_axis=color_axis,
        traces_settings=traces_settings,
    )
    return fig


def make_rectangle_shape(
    x0: Number,
    x1: Number,
    y0: Number,
    y1: Number,
    text: Optional[str] = None,
    textposition: str = "top left",
    layer: Literal["below", "above"] = "above",
    **kwargs,
) -> dict:
    result = dict(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, layer=layer, **kwargs)
    if text:
        label = result.get("label", dict())
        if "text" not in label:
            label["text"] = text
        if "textposition" not in label:
            label["textposition"] = textposition
        result["label"] = label
    return result


# endregion phrase Gantt helpers
