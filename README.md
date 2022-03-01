# MoUSE

MoUSE - Mouse Ultrasonic Squeak Explorer

Toolkit for processing, localisation and classification of rodent ultrasonic squeaks written in python

## Install

#### Developers

use this commands to create link to this package in python's `site-packages`:
```bash
$ git clone https://github.com/JosephTheMoUSE/MoUSE.git
$ cd MoUSE
$ python -m pip install -e .
```
all changes will be available without re installation (`site-packages` for this package just points to `src` directory)

#### Users (if any)

```bash
$ python -m pip install git+https://github.com/JosephTheMoUSE/MoUSE.git
```

PyPI package may be created  ¯\_(ツ)_/¯

## Project structure 

### Dealing with imports

Working with this project you would probably like to propagate your implemented modules one(or two) level higher,
to do so you should follow examples in `__init__.py` in each folder

Basically put this code into `__init__.py` of package you are working on:
```python
from .my_awesome_module import awesome_f1, awesome_f2
``` 
after this your can import `awesome_f1` like this (second option is still available):
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
