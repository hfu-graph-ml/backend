# Virtual enviroment directory
VENV = .venv

# Binaries / Scripts
DEACTIVATE = deactivate

ifeq ($(OS),Windows_NT)
	ACTIVATE = $(VENV)/Scripts/Activate.ps1
	PYTHON = $(VENV)/Scripts/python.exe
	PIP = $(VENV)/Scripts/pip.exe
else
	ACTIVATE = $(VENV)/bin/activate
	PYTHON = $(VENV)/bin/python
	PIP = $(VENV)/bin/pip
endif

run: init
	$(PYTHON) src/main.py

init: requirements.txt
	python -m venv $(VENV)
	$(PIP) install -r requirements.txt

.PHONY: run init