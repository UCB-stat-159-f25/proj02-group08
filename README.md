# Project 02 - Composed of nlp-P0{1..4}.ipynb

Project 02 for Group 08

- Jocelyn Perez
- Claire Shimazaki
- Colby Zhang
- Olorundamilola Kazeem


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UCB-stat-159-f25/proj02-group08/HEAD)


## Available Make Directives 

```bash
❯ make help                                                   
01. Environment targets:
  env-create                    - Create the environment from environment.yml
  env-update                 - Update, install, and clean up packages INTO the environment
  env-update-environment-yml - Update and save the environment packages INTO the environment.yml file
  env-run-jupyterlab         - Launch JupyterLab using the environment
  env-list                   - List all conda environments
  env-package-install        - Install a package into an environment
  env-package-check          - Check if a package is installed in an environment
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
  doc-myst-site-init         - Generate and Initialize MyST site
  doc-myst-site-init-toc     - Generate and Initialize MyST site - Table of Contents
  doc-myst-site-init-ghpages - Generate and Initialize MyST site - Github Pages
```

## Project Structure

```bash
❯ tree -L 2 .
.
├── ai_documentation.txt
├── _build
│   ├── html
│   ├── site
│   └── templates
├── build
│   ├── bdist.linux-x86_64
│   └── lib
├── contribution_statement.md
├── data
│   ├── 00_raw
│   ├── 01_processed
│   ├── 02_vectorized
│   └── 03_processed_lda_bert
├── docs
├── environment_TESTING_123_HISTORY.yml
├── environment_TESTING_123_NOBUILDS.yml
├── environment.yml
├── img
│   ├── ProgressBar_BERTopic_Screenshot 2025-11-24 at 17.27.20.png
│   └── ProgressBar_Screenshot 2025-11-23 at 15.05.05.png
├── __init__.py
├── Makefile
├── misc
│   ├── proj02-nlp-BACKUP.ipynb
│   ├── proj02-nlp.ipynb
│   ├── proj02-nlp.pdf
│   ├── proj02-nlp.py
│   ├── project-description-todos.md
│   ├── testing123.ipynb
│   ├── testing123-playing.ipynb
│   └── testing123.py
├── myst.yml
├── notebooks
│   ├── __init__.py
│   ├── nlp-P01.ipynb
│   ├── nlp-P01.py
│   ├── nlp-P02.ipynb
│   ├── nlp-P02.py
│   ├── nlp-P03.ipynb
│   ├── nlp-P03.py
│   ├── nlp-P04.ipynb
│   ├── nlp-P04.py
│   ├── __pycache__
│   └── testing123
├── outputs
│   ├── AverageWordCount.png
│   ├── bertopic_distribution_first_doc.html
│   ├── bertopic_topics.html
│   ├── DistributionOfState.png
│   ├── lda_vis.html
│   ├── numberOfSpechesPerPresident.png
│   ├── numberOfSpechesPerYear.png
│   ├── sou_year_2000_and_above_text
│   ├── SpeechYearVersusCount.png
│   ├── top_words_2017_2023.png
│   ├── vectorized_speeches_heatmap.png
│   └── vectorized_speeches_scatterplot_PCA.png
├── proj02_group08.egg-info
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── requires.txt
│   ├── SOURCES.txt
│   └── top_level.txt
├── proj02group08.egg-info
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── requires.txt
│   ├── SOURCES.txt
│   └── top_level.txt
├── project-description.md
├── pyproject_BACKUP.toml
├── pyproject.toml
├── README.md
├── src
│   ├── __init__.py
│   ├── part00_utils_visuals.py
│   ├── part01.py
│   ├── part02.py
│   ├── part03.py
│   ├── part04.py
│   └── __pycache__
└── tests
    ├── __init__.py
    ├── test_part01.py
    ├── test_part02.py
    ├── test_part03.py
    └── test_part04.py

25 directories, 64 files
```