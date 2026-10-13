PY ?= python

.PHONY: help install check-data test test-fast notebooks app clean

help:
	@echo "make install      - pip install -r requirements.txt"
	@echo "make check-data   - verify both panels are the balanced grids the estimators assume"
	@echo "make test         - full suite, including the 39-donor placebo refits"
	@echo "make test-fast    - skip the placebo refits"
	@echo "make notebooks    - re-execute the four .ipynb files in place, committing fresh outputs"
	@echo "make app          - run the Streamlit estimator explorer in app/"
	@echo "make clean        - remove caches only; never touches data, figures or notebooks"

install:
	$(PY) -m pip install -r requirements.txt

check-data:
	$(PY) -m pytest tests/test_io.py -q

test:
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -q -m "not slow"

notebooks:
	cd notebooks && $(PY) -m nbconvert --to notebook --execute --inplace \
	  --ExecutePreprocessor.timeout=1800 \
	  01_design.ipynb 02_did_naive_vs_staggered.ipynb \
	  03_synthetic_control_prop99.ipynb 04_sensitivity_and_writeup.ipynb

app:
	$(PY) -m streamlit run app/streamlit_app.py

clean:
	rm -rf .pytest_cache **/__pycache__
	find . -name "*.pyc" -delete
