.PHONY: setup dev docs test test-coverage

VENV ?= venv

default: setup


setup:
	pip3 install --user virtualenv
	virtualenv $(VENV)
	eval ". $(VENV)/bin/activate && pip install -r requirements/common.txt"
	eval ". $(VENV)/bin/activate && pip install -r requirements/docs.txt"
	eval ". $(VENV)/bin/activate && pip install -r requirements/tests.txt"

lint:
	flake8 --max-line-length=80 graphdot/ --exclude=basekernel.py
	flake8 --max-line-length=80 example/
	flake8 --max-line-length=80 --ignore=E121,E123,E126,E226,E24,E704,F401,W503,W504 test/
	flake8 --max-line-length=80 --ignore=E121,E123,E126,E226,E24,E704,F401,W503,W504 benchmark/

test:
	tox -e py37

test-coverage:
	tox -e coverage

docs:
	cd docs && make html