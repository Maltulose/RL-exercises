# These have been configured to only really run short tasks. Longer form tasks
# are usually completed in github actions.

NAME := RL_exercises
PACKAGE_NAME := rl_exercises

DIR := "${CURDIR}"
SOURCE_DIR := ${PACKAGE_NAME}
DIST := dist
TESTS_DIR := tests

.PHONY: help install-dev check format pre-commit clean clean-build build publish test

help:
	@echo "Makefile ${NAME}"
	@echo "* install-dev      to install all dev requirements and install pre-commit"
	@echo "* clean            to clean any doc or build files"
	@echo "* check            to check the source code for issues"
	@echo "* format           to format the code with black and isort"
	@echo "* pre-commit       to run the pre-commit check"
	@echo "* build            to build a dist"
	@echo "* publish          to help publish the current branch to pypi"
	@echo "* test             to run the tests"

PYTHON ?= python
PYTEST ?= python -m pytest
PIP ?= python -m pip
MAKE ?= make
BLACK ?= black
PYDOCSTYLE ?= pydocstyle
MYPY ?= mypy
PRECOMMIT ?= pre-commit
FLAKE8 ?= flake8

install-dev:
	$(PIP) install -e ".[dev]"
	pre-commit install

check-black:
	$(BLACK) ${SOURCE_DIR} --check || :
	$(BLACK) ${TESTS_DIR} --check || :

check-pydocstyle:
	$(PYDOCSTYLE) ${SOURCE_DIR} || :

check-mypy:
	$(MYPY) ${SOURCE_DIR} || :

check-flake8:
	$(FLAKE8) ${SOURCE_DIR} || :
	$(FLAKE8) ${TESTS_DIR} || :

check: check-black check-mypy check-flake8 check-pydocstyle

pre-commit:
	$(PRECOMMIT) run --all-files

format-black:
	$(BLACK) ${SOURCE_DIR}
	$(BLACK) ${TESTS_DIR}

format: format-black

test:
	$(PYTEST) ${TESTS_DIR}

test-week-1:
	$(PYTEST) ${TESTS_DIR}/week_1

test-week-2:
	$(PYTEST) ${TESTS_DIR}/week_2

test-week-3:
	$(PYTEST) ${TESTS_DIR}/week_3

test-week-5:
	$(PYTEST) ${TESTS_DIR}/week_5

test-week-6:
	$(PYTEST) ${TESTS_DIR}/week_6

test-week-8:
	$(PYTEST) ${TESTS_DIR}/week_8

clean-build:
	$(PYTHON) setup.py clean
	rm -rf ${DIST}

# Build a distribution in ./dist
build:
	$(PYTHON) setup.py sdist

# Clean up any builds in ./dist as well as doc, if present
clean: clean-build 