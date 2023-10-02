---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.6
kernelspec:
  display_name: dimcat
  language: python
  name: dimcat
---

```{code-cell} ipython3
import os
import modin.pandas as pd
import ray

from utils import load_facets
PATH = "all_subcorpora"
```

```{code-cell} ipython3
ray.init(runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}}, ignore_reinit_error=True)
```

```{code-cell} ipython3
facets = load_facets(PATH)
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

```{code-cell} ipython3
def make_piece_index(multi_index):
    if multi_index.nlevels < 2:
        return
    drop_levels = list(range(2, multi_index.nlevels))
    if drop_levels:
        multi_index = multi_index.droplevel(drop_levels)
    return multi_index.drop_duplicates()

annotated_pieces = make_piece_index(annotations.index)
print(f"{len(annotated_pieces)} pieces annotated")
```

```{code-cell} ipython3
annotated_pieces_df = pd.DataFrame(index=annotated_pieces)
notes_annotated = notes.join(annotated_pieces_df, how='right')
measures_annotated = measures.join(annotated_pieces_df, how='right')
metadata_annotated = metadata.join(annotated_pieces_df, how='right')
metadata_annotated
```

```{code-cell} ipython3
for facet_name, facet in [('notes', notes_annotated), ('measures', measures_annotated), ('metadata', metadata_annotated), ('expanded', annotations)]:
    filepath = os.path.join(PATH, f"all_subcorpora_annotated.{facet_name}.tsv")
    facet.to_csv(filepath, sep='\t')
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---

```
