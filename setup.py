# -*- coding: utf-8 -*-

from distutils.core import setup

setup(
    name = 'pymut',
    packages = ['pymut'],
    version = '0.1.1',
    description = 'Machine learning methods for the prediction of pathology in protein mutations, as in the PMut predictor (http://mmb.pcb.ub.es/pmut2017/)',
    author = 'Víctor López Ferrando',
    author_email = 'victorlopez90@gmail.com',
    license = 'MIT',
    url = 'https://github.com/vlopezferrando/pymut',
    download_url = 'https://github.com/vlopezferrando/pymut/tarball/0.1.1',
    keywords = ['protein', 'mutations', 'machine learning'],
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'imbalanced-learn',
    ],
)
