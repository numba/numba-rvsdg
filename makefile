.PHONY: build test docs env
build:
	python -m pip install -vv -e .
test:
	pytest --pyargs numba_rvsdg
docs:
	cd docs && make html
conda-env:
	conda create -n numba-rvsdg python=3.12 python-graphviz pyyaml pytest sphinx sphinx_rtd_theme
