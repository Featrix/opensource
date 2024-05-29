-include .env
.EXPORT_ALL_VARIABLES:
TOP                     := $(CURDIR)
PYTHON_SRC              := $(TOP)/src/python
DIST                    := $(PYTHON_SRC)/dist
# Docs
SPHINXOPTS    			?=
SPHINXBUILD   			?= sphinx-build
SOURCEDIR     			= src/python/docs
BUILDDIR      			= src/python/docs_build


# Function to remove directories and files
define rmrf
	@echo "Deleting ${1}"
	@rm -rf ${1}
endef

# PHONY targets
.PHONY: clean prepare-env requirements python-package help Makefile


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

version:
	@python3 $(PYTHON_SRC)/versioning.py


clean:
	$(call rmrf,.eggs)
	$(call rmrf,.tox)
	$(call rmrf,build)
	$(call rmrf,prof/)
	$(call rmrf,$(DIST))
	$(call rmrf,$(BUILDDIR))
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

python-package: version
	(cd $(PYTHON_SRC); python3 setup.py sdist bdist_wheel)

pypi-test: python-package
	(cd $(PYTHON_SRC); python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*)

pypi: python-package
	(cd $(PYTHON_SRC); python3 -m twine upload dist/*)