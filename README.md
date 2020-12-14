# Reactify

General-purpose reactivity detection using proton NMR spectra.

## Installation

You can install the _Reactify_ package using `pip`:

`pip install git+https://github.com/croningp/Reactify`

For development we recommend cloning the repository and installing it in development mode:

```shell
git clone https://github.com/croningp/Reactify
cd Reactify
pip install -e .
```

## Getting started

### Training the model

You can train the _Reactify_ model using the dataset from the original publication. To do
so, [download] the manuscript dataset and extract it to the _Reactify_ installation
folder:

```shell
cd <installation directory or directory where you cloned Reactify>
# alternatively download the dataset and place it in this folder
wget ???
tar xvf data.tar.xz
```

You can then launch the training script from anywhere:

```shell
# To see the script's full list of command line parameters
python -m Reactify.training -h
```

[download]: https://? 