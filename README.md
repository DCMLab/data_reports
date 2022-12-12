# data_reports
## Code for generating figures and tables for data reports

The version on this branch has been updated to use [Jupytext](https://jupytext.readthedocs.io/) which
lets you version-control Jupyter notebooks in the form of text files without creating diffs on the
outputs. Once you've installed Jupytext, you can open the `.py` or `.md` file in jupyter notebook/lab
which will create the `.ipynb` notebook for you which you can run and edit to your liking. 
Jupytext links it to the `.py` and `.md` representation (later we might opt for only one, or for
a different solution such as [Codebraid](https://codebraid.org/)).

In order to make sure that the text files are synchronized to your notebook the moment you commit,
you can [install add this pre-commit hook](https://jupytext.readthedocs.io/en/latest/faq.html#i-have-modified-a-text-file-but-git-reports-no-diff-for-the-paired-ipynb-file).