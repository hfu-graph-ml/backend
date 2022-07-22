---
hide:
  - navigation
---

# Backend

## Tech stack

- [FastAPI](https://github.com/tiangolo/fastapi) is used for HTTP routing

## Development

The following tools are needed:

- Python > 3.8
- Pip
- Make

### Initialization

These steps are required to setup the development enviroment:

```shell
git clone git@github.com:hfu-graph-ml/backend.git
make init
```

### Activation

On Linux the VENV activation is straight forward:

```shell
source .venv/Scripts/activate
```

There are slight differences between CMD and PowerShell in Windows (as always). It is **recommended** to use PS Core, as
it is more modern and resembles a Linux shell more closely (cd, rm, ...)

```shell
# On CMD
.venv/Scripts/activate.bat

# On PS (Core)
.venv/Scripts/Activate.ps1
```

### Install dependencies and run server

To install dependencies run:

```shell
make install
```

After that we can run the server via `make run`.

### Adding new dependencies

First make sure you activated the VENV. Then run:

```shell
pip install <package>
pip freeze > requirements.txt
```