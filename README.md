# Backend

## Development

The following tools are needed:

- Python > 3.8
- Pip
- Make

### Usage on Windows

There are slight differences between CMD and PowerShell in Windows (as always). It is **recommended** to use PS Core, as
it is more modern and resembles a Linux shell more closely (cd, rm, ...)

```shell
# On CMD
.venv/Scripts/activate.bat

# On PS (Core)
.venv/Scripts/Activate.ps1
```

The Makefile and this README uses PowerShell.

### Initialization

These steps are required to setup the development enviroment

```shell
git clone git@github.com:hfu-graph-ml/backend.git
cd backend
make run
.venv/Scripts/Activate.ps1
```

Sadly the activation cannot be automated under Windows due to multiple reasons. Believe me, I tried...

### Adding new dependencies

```shell
.venv/Scripts/Activate.ps1
pip install <package>
pip freeze > requirements.txt
```