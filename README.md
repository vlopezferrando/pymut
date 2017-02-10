
# PyMut

## Introduction

Discerning wether protein mutations are linked to pathological effects is a
question that has been tackled by many using machine learning approaches.

PyMut is a Python library that eases the data collection, training and evaluation
of machine learning methods.

## Installation

### Install python dependencies

#### Option 1: Anaconda

Anaconda is a bundle of precompiled python packages
that includes `numpy` and `scipy` among others. It is the fastest way
to start using `pymut`.

1. [Download anaconda with Python 3.5](https://www.continuum.io/downloads) for your platform (Windows, Linux or OSX).
2. Install anaconda (double-click in Windows), or run in Linux of OSX:

    `bash Anaconda3-4.1.1-Linux-x86_64.sh`

3. Install the only dependency that is missing from anaconda:

    `conda install seaborn`

#### Option 2: Pip

You can also install all the dependencies using pip:

    pip install scipy numpy pandas scikit-learn matplotlib seaborn jupyter

#### Future

In the future `pymut` will be in the pypy repository and the installation process
will be as straightforward as `pip install pymut`.

### Download pymut

Clone from the mmb gitlab repository:

    git clone http://mmb.pcb.ub.es/gitlab/victorlf/pymut.git

Start using it by:

    cd pymut
    ipython
    In [1]: import pymut

## Tutorial

In this tutorial we will see how we can use `pymut` to build a pathology predictor
given an annotated set of mutations.

The basic data structure used in `pymut` are [Pandas](http://pandas.pydata.org/) DataFrames. In the DataFrame, each row represents a **mutation** and has the following fields (columns):

* `protein_id`: a protein identifier, preferrably UniProt. It will be used to retrieve the alignments from disk.
* `sequence`: a string with the protein sequence, such as `'MPLNRTLSMSSLPGLEDWEDE'`.
* `position`: an integer representing the position of the mutation in the sequence (the first aminoacid has position `1`).
* `wt`: one character (eg. `'A'`), the wild-type amino acid in the given position (must match the sequence).
* `mt`: one character (eg. `'P'`), the mutated amino acid.
* `disease`: a bool, `True` means the mutation causes a disease, `False` means it is neutral.

To create our initial DataFrame we can use many methods provided py `pandas`. It can be imported from json, csv, pickle, etc. In the tutorial we will just use the following 22 mutations:


    import sys; sys.path.insert(1, '/home/victor/git/pymut')
    from pymut import *
    %matplotlib inline
    import pandas as pd

    variants = pd.DataFrame.from_records([
            ('Q3LHN2', 'MCYGYGCGCGSFCRLGYGCGYEGCRYGCGHRGCGDGCCCPSCYRRYRFTGFY', 5, 'Y', 'H', False),
            ('Q3LHN2', 'MCYGYGCGCGSFCRLGYGCGYEGCRYGCGHRGCGDGCCCPSCYRRYRFTGFY', 32, 'G', 'C', False),
            ('P62861', 'KVHGSLARAGKVRGQTPKVAKQEKKKKKTGRAKRRMQYNRRFVNVVPTFGKKKGPNANS', 19, 'V', 'M', False),
            ('P63313', 'MADKPDMGEIASFDKAKLKKTETQEKNTLPTKETIEQEKRSEIS', 7, 'M', 'R', False),
            ('Q00LT1', 'MCTTLFLLSTLAMLWRRRFANRVQPEPSDVDGAARGSSLDADPQSSGREKEPLK', 2, 'C', 'Y', True),
            ('Q00LT1', 'MCTTLFLLSTLAMLWRRRFANRVQPEPSDVDGAARGSSLDADPQSSGREKEPLK', 30, 'V', 'M', True),
            ('Q00LT1', 'MCTTLFLLSTLAMLWRRRFANRVQPEPSDVDGAARGSSLDADPQSSGREKEPLK', 17, 'R', 'C', False),
            ('P0CJ72', 'MATPGFSCLLLSTSEIDLPMKRRV', 13, 'T', 'I', False),
            ('P15516', 'MKFFVFALILALMLSMTGADSHAKRHHGYKRKFHEKHHSHRGYRSNYLYDN', 41, 'R', 'Q', False),
            ('P56378', 'MLQSIIKNIWIPMKPYYTKVYQEIWIGMGLMGFIVYKIRAADKRSKALKASAPAPGHH', 9, 'I', 'V', False),
            ('P56381', 'MVAYWRQAGLSYIRYSQICAKAVRDALKTEFKANAEKTSGSNVKIVKVKKE', 12, 'Y', 'C', True),
            ('Q96KF2', 'MLCAHFSDQGPAHLTTSKSAFLSNKKTSTLKHLLGETRSDGSACNSGISGGRGRKIP', 4, 'A', 'V', False),
            ('Q3MIV0', 'MSFDNNYHGGQGYAKGGLGCSYGCGLSGYGYACYCPWCYERSWFSGCF', 26, 'L', 'H', False),
            ('Q3MIV0', 'MSFDNNYHGGQGYAKGGLGCSYGCGLSGYGYACYCPWCYERSWFSGCF', 29, 'Y', 'C', False),
            ('P26678', 'MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICLLLICIIVMLL', 9, 'R', 'L', True),
            ('P26678', 'MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICLLLICIIVMLL', 9, 'R', 'C', True),
            ('P26678', 'MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICLLLICIIVMLL', 9, 'R', 'H', True),
            ('P58511', 'MNWKVLEHVPLLLYILAAKTLILCLTFAGVKMYQRKRLEAKQQKLEAERKKQSEKKDN', 51, 'K', 'R', False),
            ('P62273', 'MGHQQLYWSHPRKFGQGSRSCRVCSNRHGLIRKYGLNMCRQCFRQYAKDIGFIKLD', 31, 'I', 'F', True),
            ('P62273', 'MGHQQLYWSHPRKFGQGSRSCRVCSNRHGLIRKYGLNMCRQCFRQYAKDIGFIKLD', 50, 'I', 'T', True),
            ('P84101', 'MTRGNQRELARQKNMKKQSDSVKGKRRDDGLSAAARKQRDSEIMQQKQKKANEKKEEPK', 40, 'D', 'Y', False),
            ('O97980', 'MEEQPECREEKRGSLHVWKSELVEVEDDVYLRHSSSLTYRL', 16, 'H', 'Y', False),
        ], columns=('protein_id', 'sequence', 'position', 'wt', 'mt', 'disease'))

## Feature computation

The first step in order to get predictions is to compute features that describe each mutation. PyMut will use every column named `'feature_*'` as a feature that can be used for training and prediction. So if you already have some features for your data, you can add them this way:

    variants['feature_blosum62'] = [2, -3, 1, -1, -2, 1, -3, -1, 1, 3, -2, 0, -3, -2, -2, -3, 0, 2, 0, -1, -3, 2]
    variants['feature_kite_doolitle'] = [1.056, -0.322, 0.256, 0.711, -0.633, 0.256, -0.778, -0.644, -0.111, 0.033, 0.422, -0.3, 0.778, 0.422, -0.922, -0.778, -0.144, 0.067, 0.189, 0.644, -1.089, -1.056]

PyMut offers the method `features_distribution` to plot the distribution of disease and neutral mutations for each feature:

    features_distribution(variants[['disease', 'feature_blosum62', 'feature_kite_doolitle']])

PyMut also handles the computation of 170 features, derived mostly from BLAST searches and Multiple Sequence Alignments.
PyMut will find these alignments in a `data` directory next to `pymut`, using the following format:

* `data/blast_90/UniRef100_O97980.90.xml.gz`
* `data/blast_100/UniRef100_O97980.100.xml.gz`
* `data/kalign_90/UniRef100_O97980.90.afa.gz`
* `data/kalign_100/UniRef100_O97980.100.afa.gz`

For each file, `UniRef100_O97980` is the UniRef100 identifier of the protein, `90` or `100` point that the alignment is made over UniRef90 or UniRef100.

To compute all these features, run:

    compute_features(variants)

### Other methods (continue tutorial)

* `cross_validate` -> cross validate using a given classifier and different fold strategies
* `feature_selection` -> iteratively add the best features to the selected ensemble
* `train_classifier` -> train a classifier with annotated data and return a scikit-learn pipeline
* `predict` -> use the predictor to annotate variants
* `evaluate_prediction` -> compute a table with different evaluation metrics (accuracy, sensitivity, specificity, mcc...)
* `get_learning_curve` -> compute the learning curve for a given classifier
