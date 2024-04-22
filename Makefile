-include .env
.EXPORT_ALL_VARIABLES:
TOP                     := $(CURDIR)
PYTHON_SRC              := $(TOP)/src/python
DIST                    := $(PYTHON_SRC)/dist

# Function to remove directories and files
define rmrf
	@echo "Deleting ${1}"
	@rm -rf ${1}
endef

# PHONY targets
.PHONY: clean prepare-env requirements python-package

clean:
	$(call rmrf,.eggs)
	$(call rmrf,.tox)
	$(call rmrf,build)
	$(call rmrf,prof/)
	$(call rmrf,$(DIST))
	@find . -name '*.py,cover' | xargs rm -f
	@find . -name '*.pyc' | xargs rm -f
	@find . -name '__pycache__' | xargs rm -rf

prepare-env:
	python3 -m pip install -r $(PYTHON_SRC)/dev_requirements.txt
	@if [ -z `which pre-commit` ]; then \
		echo "Add \$HOME/.local/bin to your path (try source ~/.profile) and make prerequisites again."; exit 1; fi
	pre-commit install

requirements:
	pip install -r $(PYTHON_SRC)/requirements.txt

python-package:
	(cd $(PYTHON_SRC); python3 setup.py sdist)
