# Data reports: Code for generating figures and tables

The version on this branch has been updated to use [Jupytext](https://jupytext.readthedocs.io/) which
lets you version-control Jupyter notebooks in the form of text files without creating diffs on the
outputs. Once you've installed Jupytext, you can open the `.py` or `.md` file in jupyter notebook/lab
which will create the `.ipynb` notebook for you which you can run and edit to your liking. 
Jupytext links it to the `.py` and `.md` representation (later we might opt for only one, or for
a different solution such as [Codebraid](https://codebraid.org/)).

In order to make sure that the text files are synchronized to your notebook the moment you commit,
you can [install add this pre-commit hook](https://jupytext.readthedocs.io/en/latest/faq.html#i-have-modified-a-text-file-but-git-reports-no-diff-for-the-paired-ipynb-file).

## Running the notebooks

* clone the corpus: `git clone --recurse-submodules -j8 git@github.com:DCMLab/romantic_piano_corpus.git`
* create new environment, make it visible to your Jupyter
  * for conda do `conda create --name {name} python=3.10`
  * activate it and install `pip install ipykernel`
  * `ipython kernel install --user --name={name}`
* within the new environment, install requirements, e.g. `pip install -r requirements.txt`
  * this currently involves installing dimcat from its `ms3_new` branch
* head into the clone of romantic_piano_corpus and run `ms3 extract -X -M -N`
* open up jupyter notebook or jupyter lab and open either the `.md` or the `.py` file ([documentation](https://jupytext.readthedocs.io/en/latest/paired-notebooks.html#how-to-open-scripts-with-either-the-text-or-notebook-view-in-jupyter))
* Set the `corpus_path` in the second cell to your local clone.

If the plots are not displayed and you are in JupyterLab, use [this guide](https://plotly.com/python/getting-started/#jupyterlab-support).