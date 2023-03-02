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

#### Users

```bash
$ python -m pip install git+https://github.com/JosephTheMoUSE/MoUSE.git
```
