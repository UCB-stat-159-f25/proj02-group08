###
# 01. Environment Setup
###

env-create:
	conda env create -f environment.yml

env-update:
	conda env update -f environment.yml --prune

env-run-jupyterlab:
	conda run -n sotu jupyter lab

env-list:
	conda env list

env-activate:
	@echo "Run the following command manually to TURN ON the environment:"
	@echo "    conda activate sotu"

env-deactivate:
	@echo "Run the following command manually to TURN OFF the environment:"
	@echo "    conda deactivate"

.PHONY: env-create env-update env-run-jupyterlab env-list env-activate env-deactivate


###
# 02. Kernel Management
###

KERNEL = sotu

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
	mkdir -p code/part01 code/part02 code/part03 code/part04
	touch code/.gitkeep code/part01/.gitkeep code/part02/.gitkeep code/part03/.gitkeep code/part04/.gitkeep
	touch code/__init__.py code/part01/__init__.py code/part02/__init__.py code/part03/__init__.py code/part04/__init__.py

dir-code-delete:
	rm -rf code/part01 code/part02 code/part03 code/part04

dir-data-create:
	mkdir -p data/00_raw data/01_processed data/02_vectorized
	touch data/.gitkeep data/00_raw/.gitkeep data/01_processed/.gitkeep data/02_vectorized/.gitkeep

dir-data-delete:
	rm -rf data/00_raw data/01_processed data/02_vectorized

dir-notebooks-create:
	mkdir -p notebooks
	touch notebooks/.gitkeep
	touch notebooks/.__init_.py

dir-notebooks-delete:
	rm -rf notebooks

dir-tests-create:
	mkdir -p tests
	touch tests/.gitkeep
	touch tests/.__init_.py

dir-tests-delete:
	rm -rf tests

.PHONY: dir-code-create dir-code-delete dir-data-create dir-data-delete dir-notebooks-create dir-notebooks-delete dir-tests-create dir-tests-delete


###
# 04. Notebook Management with JupyText 
#
# NOTE - TODO: Consolidate this in the Makefile
###

DIR_NOTEBOOKS = notebooks



###
# Appendix A. Help
###

help:
	@echo "Environment targets:"
	@echo "  env-create           - Create the environment from environment.yml"
	@echo "  env-update           - Update, install, and clean up packages"
	@echo "  env-run-jupyterlab   - Launch JupyterLab using the environment"
	@echo "  env-list             - List all conda environments"
	@echo "  env-activate         - Activate the 'sotu' environment"
	@echo "  env-deactivate       - Deactivate the current environment"
	@echo ""
	@echo "Kernel targets:"
	@echo "  ker-create           - Create the IPython kernel"
	@echo "  ker-list             - List all Jupyter kernels"
	@echo "  ker-remove           - Remove the IPython kernel"
	@echo ""
	@echo "Directory targets:"
	@echo "  dir-code-create      - Create the code/ directory and subdirectories"
	@echo "  dir-code-delete      - Delete the code/ directory and subdirectories"
	@echo "  dir-data-create      - Create the data/ directory and subdirectories"
	@echo "  dir-data-delete      - Delete the data/ directory and subdirectories"
	@echo "  dir-notebooks-create - Create the notebooks/ directory"
	@echo "  dir-notebooks-delete - Delete the notebooks/ directory"
	@echo "  dir-tests-create     - Create the tests/ directory"
	@echo "  dir-tests-delete     - Delete the tests/ directory"
	@echo ""
	@echo "General targets:"
	@echo "  all                  - Run all tasks"

.PHONY: help


###
# Appendix Z. Phony and Default Target
###

all:
	@echo "All tasks complete!"

.PHONY: all
