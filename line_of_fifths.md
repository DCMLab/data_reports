---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: revamp
  language: python
  name: revamp
---

## Line of Fifth plots

Notebook adapted from the one used for the presentation at HCI 2023 in Copenhagen

```{code-cell}
%load_ext autoreload
%autoreload 2
import os
from git import Repo
import plotly.express as px
import plotly.graph_objects as go
import dimcat as dc
import ms3

from utils import (
    get_pitch_class_distribution,
    plot_pitch_class_distribution,
    tpc_bubbles,
    resolve_dir,
    print_heading,
    get_repo_name,
    get_middle_composition_year
)
import pandas as pd
# import modin.pandas as pd
# import ray
# ray.init(runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}}, ignore_reinit_error=True)
```

```{code-cell}
from utils import OUTPUT_FOLDER, write_image
RESULTS_PATH = os.path.abspath(os.path.join(OUTPUT_FOLDER, "line_of_fifths"))
os.makedirs(RESULTS_PATH, exist_ok=True)
def save_figure_as(fig, filename, directory=RESULTS_PATH, **kwargs):
    write_image(fig, filename, directory, **kwargs)
```

**Loading data**

```{code-cell}
package_path = resolve_dir("~/distant_listening_corpus/couperin_concerts/couperin_concerts.datapackage.json")
repo = Repo(os.path.dirname(package_path))
print_heading("Data and software versions")
print(f"Data repo '{get_repo_name(repo)}' @ {repo.commit().hexsha[:7]}")
print(f"dimcat version {dc.__version__}")
print(f"ms3 version {ms3.__version__}")
D = dc.Dataset.from_package(package_path)
D
```

```{code-cell}
notes = D.get_feature("notes")
notes.df
```

```{code-cell}
annotations = D.get_feature("harmonylabels")
annotations.df
```

```{code-cell}
metadata = D.get_metadata()
metadata
```

## Pitch class distribution

```{code-cell}
tpc_distribution = get_pitch_class_distribution(notes)
tpc_distribution
```

```{code-cell}
la_mer_notes = ms3.load_tsv("La_Mer_1-84.notes.tsv")
la_mer_notes
```

```{code-cell}
fig = plot_pitch_class_distribution(
    df=la_mer_notes,
    modin=False,
    title="Pitch-class distribution in Claude Debussy's 'La Mer' (mm. 1-84)",
)
save_figure_as(fig, "debussy_la_mer_beginning_pitch_class_distribution_bars", height=800)
fig.show()
```

```{code-cell}
la_mer_mn_dist = la_mer_notes.groupby(['mn', 'tpc']).duration_qb.sum()
x_vals = sorted(la_mer_notes.tpc.unique())
x_names = ms3.fifths2name(x_vals)
x_axis = dict(tickvals=x_vals, ticktext=x_names)
fig = tpc_bubbles(
    la_mer_mn_dist,
    x_axis=x_axis,
    title="measure-wise pitch-class distribution in 'La Mer' (mm. 1-84)",
    labels=dict(mn="Measure number", tpc="Tonal pitch class"),
    modin=False
)
save_figure_as(fig, "debussy_la_mer_beginning_barwise_pitch_class_distributions_bubbles", width=1200)
fig.show()
```

```{code-cell}
sorted_composition_years = get_middle_composition_year(metadata).sort_values()
order_of_pieces = sorted_composition_years.index.to_list()
notes_piece_groups = notes.groupby(['corpus', 'piece'])
# aligned_integer_index = np.concatenate([notes_piece_groups.indices[piece_id] for piece_id in order_of_pieces])
# notes_chronological_piece_order = notes.take(aligned_integer_index)
id_notes = pd.concat({ID: notes_piece_groups.get_group(piece_id) for ID, piece_id in enumerate(order_of_pieces)})
id_notes.index.rename(['ID', 'corpus', 'piece',  'i'], inplace=True)
id_notes
```

```{code-cell}
piece_distributions = id_notes.groupby(['ID', 'corpus', 'piece', 'tpc']).duration_qb.sum()
piece_distributions
```

```{code-cell}
id_distributions = piece_distributions.copy()
id_distributions.index = piece_distributions.index.droplevel([1,2])
id_distributions
```

```{code-cell}
df = id_distributions.groupby(level=0, group_keys=False).apply(lambda S: S / S.sum()).reset_index()
hover_data = ms3.fifths2name(list(df.tpc))
df['pitch class'] = hover_data
df
```

```{code-cell}
fig = px.scatter(df, x=list(df.tpc), y=list(df.ID),
                     size=list(df.duration_qb),
                     color=list(df.tpc))
fig
```

```{code-cell}
fig = tpc_bubbles(
    id_distributions,
    title="piece-wise pitch-class distributions for the DLC",
    x_axis=x_axis,)
save_figure_as(fig, "all_pitch_class_distributions_piecewise_bubbles", width=1200)
fig.show()
```

```{code-cell}
year_notes = pd.concat({year: notes_piece_groups.get_group(piece_id)
                        for piece_id, year in sorted_composition_years.items()})
year_notes.index.rename(['year', 'corpus', 'piece',  'i'], inplace=True)
year_notes
```

```{code-cell}
year_groupby = year_notes.reset_index().groupby(['year', 'tpc'])
year_distributions = year_groupby.duration_qb.sum()
year_distributions = pd.concat([year_distributions, year_groupby.corpus.unique().rename('corpora')], axis=1)
fig = tpc_bubbles(
  year_distributions,
  title="year-wise pitch-class distributions for the DLC",
  x_axis=x_axis,
  hover_data="corpora",
  normalize=True,
  modin=False)
save_figure_as(fig, "all_pitch_class_distributions_yearwise_bubbles", width=1200)
fig.show()
```

```{code-cell}
fig = plot_pitch_class_distribution(
  notes,
  title="Pitch class distribution over the Distant Listening Corpus",
  modin=False)
save_figure_as(fig, "complete_pitch_class_distribution_absolute_bars", height=800)
fig.show()
```
