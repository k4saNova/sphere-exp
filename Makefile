PYTHON = python3
VENV_DIR = venv
VPYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
INSTALL_PATH = $(VENV_DIR)/lib/$(PYTHON)/site-packages/

.PHONY: venv tests
venv:
	@ $(PYTHON) -m venv $(VENV_DIR)

tests:
	@ $(VPYTHON) tests/test.py

clean:
	rm $(PWD)/build/ $(PWD)/dist/ -rf

install:
	$(VPYTHON) setup.py bdist_wheel
	cp -r $(PWD)/build/lib/emil $(INSTALL_PATH)
