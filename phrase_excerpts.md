---
jupytext:
  formats: md:myst,ipynb,py:percent
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: revamp
  language: python
  name: revamp
---

# Phrases in the DLC

```{code-cell} ipython3
---
mystnb:
  code_prompt_hide: Hide imports
  code_prompt_show: Show imports
tags: [hide-cell]
---
%load_ext autoreload
%autoreload 2

from pathlib import Path
import dimcat as dc
import ms3
import pandas as pd

from utils import (
    resolve_dir,
)

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 500)
```

```{code-cell} ipython3
:tags: [hide-input]

RESTART = False
dlc_path = Path(resolve_dir("~/distant_listening_corpus"))
excerpt_path = Path("~/Documents/phd/phrase_excerpts/231220_distant_listening_corpus/").expanduser()
DONE_FILE = excerpt_path / "extracted_piece_list"
package_path = dlc_path / "distant_listening_corpus.datapackage.json"
D = dc.Dataset.from_package(package_path)
D
```

```{code-cell} ipython3
phrases = D.get_feature("PhraseAnnotations")
phrases
```

```{code-cell} ipython3
def make_phrase_excerpts(
        dlc_path,
        excerpt_path,
        corpus,
        piece,
        phrase_annotations
):
    filepath = dlc_path / corpus / "MS3" / (piece + ".mscx")
    if not filepath.is_file():
        raise FileNotFoundError(filepath)
    score = ms3.Score(filepath)
    phrase_starts = phrase_annotations.groupby(level="phrase_id")[["mc", "mc_onset"]].nth(0)
    drop_levels = [name for name in phrase_starts.index.names if name != "phrase_id"]
    phrase_starts = phrase_starts.droplevel(drop_levels)
    phrase_starts = list(phrase_starts.itertuples(name=None))
    print(filepath)
    print("-" * len(str(filepath)))
    for (phrase_id, start_mc, start_mc_onset), (_, end_mc, end_mc_onset) in zip(
            phrase_starts, phrase_starts[1:] + [(None, None, None)]
            ):
        print(f"""  score.mscx.store_excerpt(
    start_mc={start_mc},
    start_mc_onset={start_mc_onset},
    end_mc={end_mc},
    end_mc_onset={end_mc_onset},
    exclude_end={True},
    directory={excerpt_path!r},
    suffix="_phrase{phrase_id}"
  )""")
        score.mscx.store_excerpt(
            start_mc=start_mc,
            start_mc_onset=start_mc_onset,
            end_mc=end_mc,
            end_mc_onset=end_mc_onset,
            exclude_end=True,
            directory=excerpt_path,
            suffix=f"_phrase{phrase_id}"
        )


def mark_as_done(item):
    global done
    with open(DONE_FILE, 'a') as f:
        f.write(item + "\n")


if RESTART:
    done = []
    if DONE_FILE.is_file():
        DONE_FILE.unlink()
elif DONE_FILE.is_file():
    done = open(DONE_FILE, 'r').read()
else:
    done = []

for (corpus, piece), df in phrases.groupby(level=["corpus", "piece"]):
    id_string = f"{corpus}, {piece}"
    if id_string in done:
        print(f"SKIPPED {id_string}")
        continue
    make_phrase_excerpts(
        dlc_path=dlc_path,
        excerpt_path=excerpt_path,
        corpus=corpus,
        piece=piece,
        phrase_annotations=df
    )
    mark_as_done(id_string)
```

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---

```
