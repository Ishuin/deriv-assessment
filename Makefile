PYTHON ?= python

.PHONY: run validate test

run:
	$(PYTHON) run.py

validate:
	$(PYTHON) validate.py

test:
	$(PYTHON) -m pytest -q
