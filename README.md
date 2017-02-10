
# PyMut

PyMut is a Python3 module that fills the gap between machine learning and
bioinformatics, providing methods that help in the prediction of pathology in
protein mutations. Using PyMut you can compute features, train predictors,
evaluate them, predict the pathology of mutations, select the best features...

## Installation

We recommend using Anaconda, a free distribution of the SciPy stack

1. [Download anaconda with Python 3.5](https://www.continuum.io/downloads) for your platform (Windows, Linux or OSX), and install (double-click in Windows), or run in Linux of OSX:

    `bash Anaconda3-4.1.1-Linux-x86_64.sh`

2. Create an anaconda Python 3 environment (we will name it pymut), and activate the+
environment:

    `conda create python=3 --name pymut`
    `source activate pymut`

3. Install PyMut:

    `pip install pymut`

## Tutorial

Visit the [PyMut tutorial](http://mmb.pcb.ub.es/pmut2017/PyMut-tutorial) to see
a full example of usage of PyMut. In the tutorial we show how to:

* Compute features that describe mutations and plot their distribution.
* Train classifiers, evaluate them using cross-validation and plot their ROC curves.
* Select the best features.
* Train a pathology predictor.
* Predict the pathology of mutations using our newly trained predictor.

