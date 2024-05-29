.PHONY: build test docs env all
all:
	make lint && make build && make test
build:
	python -m pip install -vv -e .
test:
	# Activate using the sys.monitoring implementation of coverage.
	# Needs at least coverage veraion 7.4.0 to work.
	COVERAGE_CORE=sysmon coverage run -m pytest --pyargs numba_rvsdg
	coverage report
lint:
	pre-commit run --verbose --all-files
docs:
	cd docs && make html
conda-env:
	conda create -n numba-rvsdg
conda-install:
	conda install python=3.12 python-graphviz pyyaml pytest sphinx sphinx_rtd_theme coverage pre-commit
clean:
	git clean -dfX
