---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: ms3
  language: python
  name: ms3
---

# Notes

```{code-cell} ipython3
import os
from collections import defaultdict, Counter

from git import Repo
import dimcat as dc
import ms3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import STD_LAYOUT, CADENCE_COLORS, CORPUS_COLOR_SCALE, chronological_corpus_order, color_background, get_corpus_display_name, get_repo_name, resolve_dir, value_count_df, get_repo_name, resolve_dir
```

```{code-cell} ipython3
CORPUS_PATH = os.environ.get('CORPUS_PATH', "~/debussy_piano")
print(f"CORPUS_PATH: '{CORPUS_PATH}'")
CORPUS_PATH = resolve_dir(CORPUS_PATH)
```

```{code-cell} ipython3
repo = Repo(CORPUS_PATH)
notebook_repo = Repo('.', search_parent_directories=True)
print(f"Notebook repository '{get_repo_name(notebook_repo)}' @ {notebook_repo.commit().hexsha[:7]}")
print(f"Data repo '{get_repo_name(CORPUS_PATH)}' @ {repo.commit().hexsha[:7]}")
print(f"dimcat version {dc.__version__}")
print(f"ms3 version {ms3.__version__}")
```

## Data loading

### Detected files

```{code-cell} ipython3
dataset = dc.Dataset()
dataset.load(directory=CORPUS_PATH)
dataset.data
```

```{code-cell} ipython3
print(f"N = {dataset.data.count_pieces()} annotated pieces.")
```

## Metadata

```{code-cell} ipython3
all_metadata = dataset.data.metadata()
print(f"Concatenated 'metadata.tsv' files cover {len(all_metadata)} of the {dataset.data.count_pieces()} scores.")
all_metadata.reset_index(level=1).groupby(level=0).nth(0).iloc[:,:20]
```

```{code-cell} ipython3
pcvs = dc.Pipeline([
    dc.NoteSlicer(quarters_per_slice=1.0),
    dc.PitchClassVectors(
        pitch_class_format="pc",
        weight_grace_durations=0.5)
]).process_data(dataset)
```

```{code-cell} ipython3
P = pcvs.get().fillna(0.0)
P
```

```{code-cell} ipython3
import numpy as np

def apply_dft_to_pitch_class_matrix(pc_mat, build_utm = True, long=False):
    """
    This functions takes a list of N pitch class distributions,
    modelised by a matrix of float numbers, and apply the
    DFT individually to all the pitch class distributions.

    Parameters
    ----------
    pc_mat: numpy matrix of shape (N, 12) (numpy.ndarray of numpy.float64)
        holds the pitch class distribution of all slice of a minimum temporal size.
    build_utm: bool, optional
        indicates whether the resulting list of DFT results need to be built into an upper
        triangle matrix representing all hierarchical levels possible from the original musical piece.
        As the DFT is linear, the computation of all hierarchical levels can be done at a later sate,
        thus saving some space (O(n) instead of O(n^2)).
        Default value is True.
    long : bool, optional
        By default, if `build_utm`, the upper triangle matrix will be returned as a square matrix
        where the lower left triangle beneath the diagonal is filled with zeros.
        Pass True to obtain the UTM in long format instead.

    Returns
    -------
    numpy matrix (numpy.ndarray of numpy.complex128)
        according to the parameters 'build_utm', either a Nx7 complex number matrix being
        the converted input matrix of pitch class distribution
        transformed into Fourier coefficient, or a NxNx7 complex number
        upper triangle matrix being the fourier coefficient obtained from all
        possible slices of the original musical piece.
    """
    pcv_nmb, pc_nmb = np.shape(pc_mat)
    #+1 to hold room for the 0th coefficient
    coeff_nmb = int(pc_nmb/2)+1
    res = np.fft.fft(pc_mat)[:, :coeff_nmb] #coeff 7 to 11 are uninteresting (conjugates of coeff 6 to 1).
    return res
```

```{code-cell} ipython3
df = pd.DataFrame(apply_dft_to_pitch_class_matrix(P), index=P.index).sort_index()
sample = df.sample(5)
sample
```

```{code-cell} ipython3
print(sample.style.format(precision=2).to_latex())
```

```{code-cell} ipython3
print(sample.style.to_latex())
```

```{code-cell} ipython3

```

```{code-cell} ipython3
def comp2str(c, dec=2):
    """Interpret a complex number as magnitude and phase and convert into a human-readable string."""
    magn = np.round(abs(c), dec)
    ang = -round(np.angle(c, True)) % 360
    return f"{magn}+{ang}°"


comp2str_vec = np.vectorize(comp2str)
```

```{code-cell} ipython3
comp2str_vec(apply_dft_to_pitch_class_matrix(P))
```
