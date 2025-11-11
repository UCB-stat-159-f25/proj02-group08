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
# 03. Notebook Management with JupyText
###

DIR_NOTEBOOKS = notebooks


###
# Appendix A. Help
###

help:
	@echo "Environment targets:"
	@echo "  env-create          - Create the environment from environment.yml"
	@echo "  env-update          - Update, install, and clean up packages"
	@echo "  env-run-jupyterlab  - Launch JupyterLab using the environment"
	@echo "  env-list            - List all conda environments"
	@echo "  env-activate        - Activate the 'sotu' environment"
	@echo "  env-deactivate      - Deactivate the current environment"
	@echo ""
	@echo "Kernel targets:"
	@echo "  ker-create          - Create the IPython kernel"
	@echo "  ker-list            - List all Jupyter kernels"
	@echo "  ker-remove          - Remove the IPython kernel"
	@echo ""
	@echo "General targets:"
	@echo "  all                 - Run all tasks"

.PHONY: help


###
# Appendix Z. Phony and Default Target
###

all:
	@echo "All tasks complete!"

.PHONY: all
