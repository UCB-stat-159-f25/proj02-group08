###
# 00. Reference
# - https://stat159.berkeley.edu/fall-2025/lectures/automation/make/#make-for-creating-a-new-environment
###

.ONESHELL:
SHELL = /bin/bash


###
# 01. Environment Management - TODO: Refactor to consolidate and work with .ONESHELL above
###
ENV_NAME=sotu

env-create:
	conda env create -f environment.yml

env-update:
	conda env update -f environment.yml --prune

env-run-jupyterlab:
	conda run -n $(ENV_NAME) jupyter lab

env-list:
	conda env list

env-activate:
	@echo "Run the following command manually to TURN ON the environment:"
	@echo "    conda activate sotu"
#	conda activate $(ENV_NAME)

env-deactivate:
	@echo "Run the following command manually to TURN OFF the environment:"
	@echo "    conda deactivate"
#	conda deactivate

.PHONY: env-create env-update env-run-jupyterlab env-list env-activate env-deactivate


###
# 02. Kernel Management
###

KERNEL = sotu
# KERNEL = $(ENV_NAME)

ker-create:
	python -m ipykernel install --user --name $(KERNEL) --display-name "IPython ($(KERNEL))"

ker-list:
	jupyter kernelspec list

ker-remove:
	jupyter kernelspec uninstall -y $(KERNEL)

.PHONY: ker-create ker-list ker-remove


###
# 03. Project, Directory, and File Structure
###

dir-code-create:
	mkdir -p code
	touch code/.gitkeep code/__init__.py code/part01.py code/part02.py code/part03.py code/part04.py

dir-code-delete:
	rm -rf code

dir-data-create:
	mkdir -p data/00_raw data/01_processed data/02_vectorized
	touch data/.gitkeep data/00_raw/.gitkeep data/01_processed/.gitkeep data/02_vectorized/.gitkeep

dir-data-delete:
	rm -rf data/00_raw data/01_processed data/02_vectorized

dir-docs-create:
	mkdir -p docs
	touch docs/.gitkeep docs/__init__.py

dir-docs-delete:
	rm -rf docs

dir-misc-create:
	mkdir -p misc
	touch misc/.gitkeep misc/__init__.py

dir-misc-delete:
	rm -rf misc

dir-notebooks-create:
	mkdir -p notebooks
	touch notebooks/.gitkeep notebooks/__init__.py

dir-notebooks-delete:
	rm -rf notebooks

dir-tests-create:
	mkdir -p tests
	touch tests/.gitkeep tests/__init__.py

dir-tests-delete:
	rm -rf tests

.PHONY: dir-code-create dir-code-delete dir-data-create dir-data-delete \
	dir-docs-create dir-docs-delete dir-misc-create dir-misc-delete \
	dir-notebooks-create dir-notebooks-delete dir-tests-create dir-tests-delete


###
# 04. Notebook Management with JupyText
# NOTE: TODO — Consolidate this in the Makefile
###

DIR_NOTEBOOKS = notebooks

nb-pair-all-py:
	jupytext --set-formats py:percent,ipynb $(DIR_NOTEBOOKS)/nlp-P0{1..4}.py
    # jupytext --set-formats py:percent,ipynb $(DIR_NOTEBOOKS)/nlp_P0{1..4}.py

nb-pair-all-ipynb:
	jupytext --set-formats ipynb:percent,py $(DIR_NOTEBOOKS)/nlp-P0{1..4}.ipynb
    # jupytext --set-formats ipynb:percent,py $(DIR_NOTEBOOKS)/nlp_P0{1..4}.ipynb

nb-pair-p01-py:
	jupytext --set-formats py:percent,ipynb $(DIR_NOTEBOOKS)/nlp-P01.py
    # jupytext --set-formats py:percent,ipynb $(DIR_NOTEBOOKS)/nlp_P01.py

nb-pair-p01-ipynb:
	jupytext --set-formats ipynb:percent,py $(DIR_NOTEBOOKS)/nlp-P01.ipynb
    # jupytext --set-formats ipynb:percent,py $(DIR_NOTEBOOKS)/nlp_P01.ipynb

nb-pair-p02-py:
	jupytext --set-formats py:percent,ipynb $(DIR_NOTEBOOKS)/nlp-P02.py
    # jupytext --set-formats py:percent,ipynb $(DIR_NOTEBOOKS)/nlp_P02.py

nb-pair-p02-ipynb:
	jupytext --set-formats ipynb:percent,py $(DIR_NOTEBOOKS)/nlp-P02.ipynb
    # jupytext --set-formats ipynb:percent,py $(DIR_NOTEBOOKS)/nlp_P02.ipynb

nb-pair-p03-py:
	jupytext --set-formats py:percent,ipynb $(DIR_NOTEBOOKS)/nlp-P03.py
    # jupytext --set-formats py:percent,ipynb $(DIR_NOTEBOOKS)/nlp_P03.py

nb-pair-p03-ipynb:
	jupytext --set-formats ipynb:percent,py $(DIR_NOTEBOOKS)/nlp-P03.ipynb
    # jupytext --set-formats ipynb:percent,py $(DIR_NOTEBOOKS)/nlp_P03.ipynb

nb-pair-p04-py:
	jupytext --set-formats py:percent,ipynb $(DIR_NOTEBOOKS)/nlp-P04.py
    # jupytext --set-formats py:percent,ipynb $(DIR_NOTEBOOKS)/nlp_P04.py

nb-pair-p04-ipynb:
	jupytext --set-formats ipynb:percent,py $(DIR_NOTEBOOKS)/nlp-P04.ipynb
    # jupytext --set-formats ipynb:percent,py $(DIR_NOTEBOOKS)/nlp_P04.ipynb

.PHONY: nb-pair-all-py nb-pair-all-ipynb nb-pair-p01-py nb-pair-p01-ipynb \
	nb-pair-p02-py nb-pair-p02-ipynb nb-pair-p03-py nb-pair-p03-ipynb \
	nb-pair-p04-py nb-pair-p04-ipynb


###
# 05. Project Documents and Documentation
###

doc-ai-documentation:
	cat <<'EOF' > ai_documentation.txt
		------------------------------------------------
		[nlp-P01.ipynb for Part 01 of proj02-nlp.ipynb]

		PROMPT:

		OUTPUT:


		------------------------------------------------
		[nlp-P02.ipynb for Part 02 of proj02-nlp.ipynb]

		PROMPT:

		OUTPUT:


		------------------------------------------------
		[nlp-P03.ipynb for Part 03 of proj02-nlp.ipynb]

		PROMPT:

		OUTPUT:


		------------------------------------------------
		[nlp-P04.ipynb for Part 04 of proj02-nlp.ipynb]

		PROMPT:

		OUTPUT:
	EOF

doc-contribution-statement:
	cat <<'EOF' > contribution_statement.md
		------------------------------------------------
		[nlp-P01.ipynb for Part 01 of proj02-nlp.ipynb]

		TEAM MEMBER NAME(S) - PERCENT CONTRIBUTION:


		------------------------------------------------
		[nlp-P02.ipynb for Part 02 of proj02-nlp.ipynb]

		TEAM MEMBER NAME(S) - PERCENT CONTRIBUTION:


		------------------------------------------------
		[nlp-P03.ipynb for Part 03 of proj02-nlp.ipynb]

		TEAM MEMBER NAME(S) - PERCENT CONTRIBUTION:


		------------------------------------------------
		[nlp-P04.ipynb for Part 04 of proj02-nlp.ipynb]

		TEAM MEMBER NAME(S) - PERCENT CONTRIBUTION:
	EOF

doc-myst-site-init:
	myst init

doc-myst-site-init-toc:
	myst init --write-toc

doc-myst-site-init-ghpages:
	myst init --gh-pages

.PHONY: doc-ai-documentation doc-contribution-statement \
	doc-myst-site-init doc-myst-site-init-toc doc-myst-site-init-ghpages


###
# Appendix Aa. General E2E - Phony and Default Target
###

all:
	@echo "TODO - Create an end-to-end pipeline — All tasks complete!"

.PHONY: all

###
# Appendix Zz. Help
###

help:
	@echo "01. Environment targets:"
	@echo "  env-create                	- Create the environment from environment.yml"
	@echo "  env-update                 - Update, install, and clean up packages"
	@echo "  env-run-jupyterlab         - Launch JupyterLab using the environment"
	@echo "  env-list                   - List all conda environments"
	@echo "  env-activate               - Activate the 'sotu' environment"
	@echo "  env-deactivate             - Deactivate the current environment"
	@echo ""
	@echo "02. Kernel targets:"
	@echo "  ker-create                 - Create the IPython kernel"
	@echo "  ker-list                   - List all Jupyter kernels"
	@echo "  ker-remove                 - Remove the IPython kernel"
	@echo ""
	@echo "03. Directory targets:"
	@echo "  dir-code-create            - Create the code/ directory and subdirectories"
	@echo "  dir-code-delete            - Delete the code/ directory"
	@echo "  dir-data-create            - Create the data/ directory and subdirectories"
	@echo "  dir-data-delete            - Delete the data/ directory and subdirectories"
	@echo "  dir-docs-create            - Create the docs/ directory"
	@echo "  dir-docs-delete            - Delete the docs/ directory"
	@echo "  dir-misc-create            - Create the misc/ directory"
	@echo "  dir-misc-delete            - Delete the misc/ directory"
	@echo "  dir-notebooks-create       - Create the notebooks/ directory"
	@echo "  dir-notebooks-delete       - Delete the notebooks/ directory"
	@echo "  dir-tests-create           - Create the tests/ directory"
	@echo "  dir-tests-delete           - Delete the tests/ directory"
	@echo ""
	@echo "04. Jupytext targets:"
	@echo "  nb-pair-all-py             - Pair all from Python scripts (.py)       to Jupyter notebooks (.ipynb)"
	@echo "  nb-pair-all-ipynb          - Pair all from Jupyter notebooks (.ipynb) to Python scripts (.py)"
	@echo "  nb-pair-p01-py             - Pair P01 from .py    to .ipynb"
	@echo "  nb-pair-p01-ipynb          - Pair P01 from .ipynb to .py"
	@echo "  nb-pair-p02-py             - Pair P02 from .py    to .ipynb"
	@echo "  nb-pair-p02-ipynb          - Pair P02 from .ipynb to .py"
	@echo "  nb-pair-p03-py             - Pair P03 from .py    to .ipynb"
	@echo "  nb-pair-p03-ipynb          - Pair P03 from .ipynb to .py"
	@echo "  nb-pair-p04-py             - Pair P04 from .py    to .ipynb"
	@echo "  nb-pair-p04-ipynb          - Pair P04 from .ipynb to .py"
	@echo ""
	@echo "05. Documentation targets:"
	@echo "  doc-ai-documentation       - Generate ai_documentation.txt from template"
	@echo "  doc-contribution-statement - Generate contribution_statement.md from template"
	@echo "  doc-myst-site-init         - Generate and Initialize MyST site"
	@echo "  doc-myst-site-init-toc     - Generate and Initialize MyST site - Table of Contents"
	@echo "  doc-myst-site-init-ghpages - Generate and Initialize MyST site - Github Pages"
	@echo ""
	@echo "Appendix Aa. General E2E targets:"
	@echo "  all                        - Run all tasks"

.PHONY: help
