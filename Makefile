.PHONY: quality style

check_dirs := src

quality:
	ruff check $(check_dirs)
	ruff format --check $(check_dirs) setup.py

style:
	ruff check $(check_dirs) --fix
	ruff format $(check_dirs)
