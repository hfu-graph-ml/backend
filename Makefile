.PHONY: run init install

# Virtual enviroment directory
VENV = .venv

ifeq ($(OS),Windows_NT)
	PIP = $(VENV)/Scripts/pip.exe
else
	PIP = $(VENV)/bin/pip
endif

run:
	uvicorn backend.main:app --reload

install:
# Check if venv is activated
ifneq ("$(wildcard $(VIRTUAL_ENV))","")
	$(PIP) install -r requirements.txt
endif

init:
ifeq ("$(wildcard $(VENV))","")
	python -m venv $(VENV)
endif