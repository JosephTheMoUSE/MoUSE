# MoUSE

**MoUSE** - **Mo**use **U**ltrasonic **S**queak **E**xplorer

Toolkit for processing, localisation and classification of rodent ultrasonic squeaks written in Python.

I you want to install MoUSE Desktop App see [MoUSE Desktop App repository](https://github.com/JosephTheMoUSE/MoUSE-GUI#mouse-desktop-app). If you are looking for a MoUSE Desktop App usage tutorial,
see [MoUSE wiki page](https://github.com/JosephTheMoUSE/MoUSE-docs/wiki).

## Install

#### Developers

Use this commands to create link to this package in python's `site-packages`:
```bash
$ git clone https://github.com/JosephTheMoUSE/MoUSE.git
$ cd MoUSE
$ python -m pip install -e .
```
All changes will be available without reinstallation (`site-packages` for this package just points to `src` directory).

#### Users (if any)

```bash
$ python -m pip install git+https://github.com/JosephTheMoUSE/MoUSE.git
```

PyPI package may be created in the future ¯\_(ツ)_/¯

## Project structure 

### Dealing with imports

Working with this project you would probably like to propagate your implemented modules one(or two) level higher,
to do so you should follow examples in `__init__.py` in each folder.

Basically put this code into `__init__.py` of package you are working on:
```python
from .my_awesome_module import awesome_f1, awesome_f2
``` 
After this your can import `awesome_f1` like this (second option is still available):
```python
from mause.package import awesome_f1
``` 
instead of:
```python
from mause.package.my_awesome_module import awesome_f1
``` 

### Important files

`src/mouse/utils/constants.py` - constants useful for method development, e.g. 
column names, shared data paths.
