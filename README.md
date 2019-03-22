olfactory
---


# Introduction

The module **olfactory** provide common tools for (pre)processing and analysis.



## Installation

(not published yet)


## Usage

(tbd)



# For developers

## Download the project

Clone repository:

```bash
git clone https://github.com/OctaveLauby/olfactory.git
cd olfactory
```

One can make an editable code installation:

```bash
pip olfactory -e .
```


## Virtual Environment

Using new pipenv feature (`pip install pipenv`)

```bash
pipenv install --dev
pipenv shell
...
exit
```


**Comments**:

1. Matplotlib does not have to be imported: plotting submodule is not loaded in that case

2. One can alternatively use classic virtual environment:

```bash
python -m venv venv
source venv/Scripts/activate
python -m pip install -r requirements.txt
...
deactivate
```


## Distribution


1. Building manifest file:

```bash
check-manifest --create
```

2. Building the wheel:

```bash
python setup.py bdist_wheel
```

3. Building the source distribution:

```bash
python setup.py sdist
```

4. Publishing:

```bash
python setup.py bdist_wheel sdist
twine upload dist/*
```

> For TestPyPi publication:  `twine upload --repository-url https://test.pypi.org/legacy/ dist/* `


> [Not working on Git terminal](https://github.com/pypa/packaging-problems/issues/197) for some reason



## Testing

```bash
python -m pytest olfactory -vv
python -m pylint olfactory --ignore-patterns=test*
```



