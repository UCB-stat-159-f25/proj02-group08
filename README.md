# Project 02 - TODO - Next Steps: Cleanup README.md and Convert to GitHub Issues
Template repository for Project 2, Stat 159/259 Fall 2025


## 01. Environment Setup

### NOTE - TODO: Consolidate this in the Makefile - DONE!

- Create the environment called `sotu`
```bash
(base)$ conda env create -f environment.yml
```

- Check the available environments includes `sotu`
```bash
(base)> conda env list

...
sotu                     /home/jovyan/.local/share/envs/sotu
...
```

- Activate the environment, when you need to work 
```bash
(base)> conda activate sotu

(sotu)> 
```

- Deactivate the environment, when you are done working
```bash
(sotu)> conda deactivate
(base)>
```

## 02. Kernel Setup
### NOTE - TODO: Consolidate this in the Makefile - DONE!

- Create the `sotu` kernel
```bash
(sotu)> python -m ipykernel install --user --name sotu --display-name "IPython - sotu"
Installed kernelspec sotu in /home/jovyan/.local/share/jupyter/kernels/sotu
```

- Check kernel list
```bash
(sotu)> jupyter kernelspec list
  sotu       /home/jovyan/.local/share/jupyter/kernels/sotu
```

## 03. Project Structure 

### NOTE - TODO: Consolidate this in the Makefile - DONE!
```bash
(sotu)> tree -L 3 .
.
├── ai_documentation.txt
├── code
│   ├── __init__.py
│   ├── part01.py
│   ├── part02.py
│   ├── part03.py
│   └── part04.py
├── data
│   ├── 00_raw
│   │   └── SOTU.csv
│   ├── 01_processed
│   └── 02_vectorized
├── docs
├── environment.yml
├── jupytext.toml
├── Makefile
├── misc
│   ├── proj02-nlp-BACKUP.ipynb
│   ├── proj02-nlp.ipynb
│   ├── proj02-nlp.pdf
│   ├── proj02-nlp.py
│   ├── testing123.ipynb
│   └── testing123.py
├── notebooks
│   ├── nlp-P01.ipynb
│   ├── nlp-P01.py
│   ├── nlp-P02.ipynb
│   ├── nlp-P02.py
│   ├── nlp-P03.ipynb
│   ├── nlp-P03.py
│   ├── nlp-P04.ipynb
│   └── nlp-P04.py
├── outputs
├── project-description.md
├── project-description-todos.md
├── README.md
└── tests

11 directories, 27 files
```

## 04. Notebook Management with Jupytext

### NOTE - TODO: Consolidate this in the Makefile - DONE!

Testing:
- pairing `.ipynb` with `.py` (for easy diffing and merging)
```bash
(sotu)> jupytext --set-formats ipynb,py:percent testing123.ipynb
[jupytext] Reading testing123.ipynb in format ipynb
[jupytext] Updating notebook metadata with '{"jupytext": {"formats": "ipynb,py:percent"}}'
[jupytext] Updating testing123.ipynb
[jupytext] Updating testing123.py
```
- sync by `.ipynb` 
```bash
(sotu)> jupytext --sync testing123.ipynb
[jupytext] Reading testing123.ipynb in format ipynb
[jupytext] Loading testing123.py
[jupytext] Unchanged testing123.ipynb
[jupytext] Unchanged testing123.py
```

## 05. Project Documents and Documentation

### NOTE - TODO: Consolidate this in the Makefile - DONE!

```bash

```

--- 

Makefile presently

```bash
(sotu) (main)jupyter-dami[proj02-group08]> make help

01. Environment targets:
  env-create                 - Create the environment from environment.yml
  env-update                 - Update, install, and clean up packages
  env-run-jupyterlab         - Launch JupyterLab using the environment
  env-list                   - List all conda environments
  env-activate               - Activate the 'sotu' environment
  env-deactivate             - Deactivate the current environment

02. Kernel targets:
  ker-create                 - Create the IPython kernel
  ker-list                   - List all Jupyter kernels
  ker-remove                 - Remove the IPython kernel

03. Directory targets:
  dir-code-create            - Create the code/ directory and subdirectories
  dir-code-delete            - Delete the code/ directory
  dir-data-create            - Create the data/ directory and subdirectories
  dir-data-delete            - Delete the data/ directory and subdirectories
  dir-docs-create            - Create the docs/ directory
  dir-docs-delete            - Delete the docs/ directory
  dir-misc-create            - Create the misc/ directory
  dir-misc-delete            - Delete the misc/ directory
  dir-notebooks-create       - Create the notebooks/ directory
  dir-notebooks-delete       - Delete the notebooks/ directory
  dir-tests-create           - Create the tests/ directory
  dir-tests-delete           - Delete the tests/ directory

04. Jupytext targets:
  nb-pair-all-py             - Pair all from Python scripts (.py)       to Jupyter notebooks (.ipynb)
  nb-pair-all-ipynb          - Pair all from Jupyter notebooks (.ipynb) to Python scripts (.py)
  nb-pair-p01-py             - Pair P01 from .py    to .ipynb
  nb-pair-p01-ipynb          - Pair P01 from .ipynb to .py
  nb-pair-p02-py             - Pair P02 from .py    to .ipynb
  nb-pair-p02-ipynb          - Pair P02 from .ipynb to .py
  nb-pair-p03-py             - Pair P03 from .py    to .ipynb
  nb-pair-p03-ipynb          - Pair P03 from .ipynb to .py
  nb-pair-p04-py             - Pair P04 from .py    to .ipynb
  nb-pair-p04-ipynb          - Pair P04 from .ipynb to .py

05. Documentation targets:
  doc-ai-documentation       - Generate ai_documentation.txt from template
  doc-contribution-statement - Generate contribution_statement.md from template
  doc-myst-site-initialize    - Generate and Initialize MyST site

Appendix Aa. General E2E targets:
  all                        - Run all tasks
```
