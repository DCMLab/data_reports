---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.7
kernelspec:
  display_name: ms3
  language: python
  name: ms3
---

```{code-cell} ipython3
Notes = ms3.load_tsv(os.path.join(PATH, 'all_subcorpora_annotated.notes.tsv'))
Notes
```

```{code-cell} ipython3
%load_ext autoreload
%autoreload 2
import os
import modin.pandas as pd
import ray

import plotly.express as px
#import pitchtypes as pt
import ms3

from utils import (
    load_facets,
    get_pitch_class_distribution,
    plot_pitch_class_distribution,
    tpc_bubbles
)
PATH = "all_subcorpora"
FIG_PATH = "figures"
```

```{code-cell} ipython3
ray.init(runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}}, ignore_reinit_error=True)
```

```{code-cell} ipython3
facets = load_facets(PATH, "_annotated")
```

```{code-cell} ipython3
notes = facets['notes']
notes
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
measures = facets['measures']
measures
```

```{code-cell} ipython3
annotations = facets['expanded']
annotations
```

```{code-cell} ipython3
metadata = facets['metadata']
metadata
```

+++ {"jupyter": {"outputs_hidden": false}}

## Pitch class distribution

```{code-cell} ipython3
tpc_distribution = get_pitch_class_distribution(notes)
tpc_distribution
```

```{code-cell} ipython3
la_mer_notes = ms3.load_tsv("debussy_la_mer/La_Mer_1-84.tsv")
la_mer_notes
```

```{code-cell} ipython3
plot_pitch_class_distribution(
    la_mer_notes,
    modin=False,
    title="Pitch-class distribution in Claude Debussy's 'La Mer' (mm. 1-84)",
    output=os.path.join(FIG_PATH, 'la_mer_distribution.png'))
```

```{code-cell} ipython3
la_mer_mn_dist = la_mer_notes.groupby(['mn', 'tpc']).duration_qb.sum()
x_vals = sorted(la_mer_notes.tpc.unique())
x_names = ms3.fifths2name(x_vals)
x_axis = dict(tickvals=x_vals, ticktext=x_names)
fig = tpc_bubbles(
    la_mer_mn_dist,
    x_axis=x_axis,
    title="measure-wise pitch-class distribution in Claude Debussy's 'La Mer' (mm. 1-84)",
    labels=dict(mn="Measure number", tpc="Tonal pitch class"),
    output=os.path.join(FIG_PATH, "la_mer.png"),
    modin=False
)
fig
```

```{code-cell} ipython3

```

```{code-cell} ipython3
id_notes = pd.concat({i: df for i, (_, df) in enumerate(notes.groupby(['corpus', 'piece']))})
id_notes.index.rename(['ID', 'i'], inplace=True)
id_notes
```

```{code-cell} ipython3
notes.groupby(['corpus', 'piece']).ngroups
```

```{code-cell} ipython3
all_distributions = notes.groupby(['corpus', 'piece','tpc']).duration_qb.sum().to_frame()
piece_distributions = pd.concat({i: df for i, (idx, df) in enumerate(all_distributions.groupby(['corpus', 'piece']))}, names=['ID', 'corpus', 'piece', 'tpc'])
piece_distributions
```

```{code-cell} ipython3
id_distributions = piece_distributions.copy()
id_distributions.index = piece_distributions.index.droplevel([1,2])
id_distributions
```

```{code-cell} ipython3
df = id_distributions.groupby(level=0, group_keys=False).apply(lambda S: S / S.sum()).reset_index()
hover_data = ms3.fifths2name(list(df.tpc))
df['pitch class'] = hover_data
df
```

```{code-cell} ipython3
fig = px.scatter(df, x=list(df.tpc), y=list(df.ID),
                     size=list(df.duration_qb),
                     color=list(df.tpc))
fig
```

```{code-cell} ipython3
test = id_distributions.loc[(slice(0,1),),]
tpc_bubbles(test, modin=True)
```

```{code-cell} ipython3
tpc_bubbles(id_distributions)
```

```{code-cell} ipython3
id_distributions = piece_distributions.droplevel([1,2])
id_distributions
```

```{code-cell} ipython3
import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(
    x=list(piece_distributions.tpc),
    y=list(piece_distributions.index),
    mode='markers',
    marker=dict(
        size=list(piece_distributions.duration_qb))))

fig.update_layout(
    autosize=False,
    width=500,
    height=500,
    margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
    paper_bgcolor="LightSteelBlue",
)
fig
```

```{code-cell} ipython3

```

```{code-cell} ipython3
x_vals = sorted(notes.tpc.unique())
x_names = ms3.fifths2name(x_vals)
x_axis = dict(tickvals=x_vals, ticktext=x_names)
fig = tpc_bubbles(
    piece_distributions,
    x_axis=x_axis,
    #title="measure-wise pitch-class distribution in Claude Debussy's 'La Mer' (mm. 1-84)",
    labels=dict(mn="Measure number", tpc="Tonal pitch class"),
    output=os.path.join(FIG_PATH, "all_tpc_distributions.png"),
)
```

```{code-cell} ipython3
fig = plot_pitch_class_distribution(notes, output=os.path.join(FIG_PATH, "dl_corpus_tpc.png"))
fig
```

```{code-cell} ipython3

```
