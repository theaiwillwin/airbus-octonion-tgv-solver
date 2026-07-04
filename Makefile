PY ?= python

.PHONY: install test run-single sweep figures phase1-pack

install:
	$(PY) -m pip install -e .

test:
	$(PY) -m pytest

run-single:
	$(PY) scripts/run_single.py --re 100 --nx 64 --t-final 0.5

sweep:
	$(PY) scripts/run_reynolds_sweep.py --config configs/reynolds_sweep.yaml

figures:
	$(PY) scripts/make_figures.py

phase1-pack:
	$(PY) scripts/generate_phase1_pack.py

trajectory:
	$(PY) scripts/run_trajectory_compression.py --re 100 --nx 64 --t-final 0.5

advantage:
	$(PY) scripts/run_advantage_study.py
