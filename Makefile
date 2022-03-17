# Virtual enviroment directory
VENV = .venv

ifeq ($(OS),Windows_NT)
	PIP = $(VENV)/Scripts/pip.exe
else
	PIP = $(VENV)/bin/pip
endif

run:
	uvicorn backend.main:app --reload

init: requirements.txt
	python -m venv $(VENV)
	$(PIP) install -r requirements.txt

.PHONY: run init