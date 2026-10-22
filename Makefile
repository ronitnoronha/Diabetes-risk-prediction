PY ?= python

.PHONY: all setup data pima cdc test

all: data pima cdc

setup:
	$(PY) -m pip install -r requirements.txt

data:   ## download, verify checksums, run data-quality checks
	$(PY) scripts/01_validate_data.py

pima:   ## ~20 min on 4 cores (50-fold CV, nested tuning)
	$(PY) scripts/02_pima_experiments.py

cdc:    ## ~25 min on 4 cores (grouped CV, random search, locked test set)
	$(PY) scripts/03_cdc_experiments.py

test:
	$(PY) -m pytest -q tests
