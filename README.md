# MoUSE backend overview

**MoUSE** - **Mo**use **U**ltrasonic **S**queak **E**xplorer

This repo contains the backend for our whole JosephTheMoUSE project. This is toolkit for processing, localisation and classification of rodent ultrasonic squeaks written in Python. 

## Backend MoUSE istallation 

#### For Developers

If you are familiar with programming, use below instructions for this MoUSE backend installation. 
Firstly, it is available to run it on windows/linux/macOS. You just need to have installed on your computer Python in version ^3.11 and Poetry compatible with your Python version. In the Python installation add it to the environment variable PATH on your computer. The same do with the Poetry. Add it to the environment variable PATH. We are using Poetry for dependency management as it is nowadays very popular and efficient tool for this purpose.

Now, please clone this repository to your destination folder as follows:

```bash
$ git clone https://github.com/JosephTheMoUSE/MoUSE.git
$ cd MoUSE
```

```bash
$ poetry install
```

After that, you are ready to go!

#### For Users

```bash
$ python -m pip install git+https://github.com/JosephTheMoUSE/MoUSE.git
```

I you you want to install MoUSE Desktop App (frontend) see [MoUSE Desktop App repository](https://github.com/JosephTheMoUSE/MoUSE-GUI#mouse-desktop-app). If you are looking for a MoUSE Desktop App usage tutorial, see [MoUSE wiki page](https://github.com/JosephTheMoUSE/MoUSE-docs/wiki).

If you have some problems with this MoUSE backend installation, post an issue on github and we will try to help you.
