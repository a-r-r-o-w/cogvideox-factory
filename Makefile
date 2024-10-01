.PHONY: quality style

check_dirs := training tests

quality:
	ruff check $(check_dirs)
	ruff format --check $(check_dirs) setup.py

style:
	ruff check $(check_dirs) --fix
	ruff format $(check_dirs)
