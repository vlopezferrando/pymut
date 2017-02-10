import logging
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns

from sklearn import (
    cross_validation, ensemble, grid_search, learning_curve, linear_model,
    metrics, naive_bayes, pipeline, preprocessing)

from imblearn.metrics import specificity_score


###############
# Classifiers #
###############

CLASSIFIERS = {
    'ExtraTrees': {
        'name': 'Extra Randomized Trees',
        'class': ensemble.ExtraTreesClassifier,
        'best_params': {  # MCC: 0.582
            'max_depth': 9,
            'criterion': 'entropy',
            'min_samples_leaf': 7,
            'bootstrap': True,
            'max_features': 0.47,
            'min_samples_split': 4,
            'n_jobs': 2
        },
        'param_distributions': {
            'max_depth': sp.stats.randint(5, 10),
            'max_features': sp.stats.uniform(0.05, 0.5),
            'min_samples_split': sp.stats.randint(1, 8),
            'min_samples_leaf': sp.stats.randint(6, 15),
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy'],
            'n_jobs': [2]
        }
    },
    'RandomForest': {
        'name': 'Random Forest',
        'best_params': {  # MCC: 0.638
            'bootstrap': True,
            'criterion': 'gini',
            'max_depth': 9,
            'max_features': 0.49,
            'min_samples_leaf': 10,
            'min_samples_split': 2,
            'n_estimators': 13,
            'n_jobs': 1,
        },
        'class': ensemble.RandomForestClassifier,
        'param_distributions': {
            'n_estimators': sp.stats.randint(5, 20),
            'criterion': ['gini', 'entropy'],
            'max_features': sp.stats.uniform(0.05, 0.5),
            'min_samples_split': sp.stats.randint(1, 8),
            'min_samples_leaf': sp.stats.randint(6, 15),
            'bootstrap': [True, False],
            'n_jobs': [2]
        }
    },
    'AdaBoost': {
        'name': 'Ada Boost',
        'class': ensemble.AdaBoostClassifier,
        'best_params': {  # MCC: 0.641
            'n_estimators': 258
        },
        'param_distributions': {
            'n_estimators': sp.stats.randint(50, 300)
        }
    },
    'LogisticRegression': {
        'name': 'Logistic Regression',
        'class': linear_model.LogisticRegression,
        'best_params': {  # MCC: 0.487
            'C': 1.425494495402806,
            'dual': False,
            'fit_intercept': True,
            'max_iter': 146,
            'n_jobs': 2,
            'penalty': 'l2',
            'solver': 'lbfgs'
        },
        'param_distributions': {
            'penalty': ['l2'],  # (default)
            'dual': [False],  # (default), better when # samples > # features
            'C': sp.stats.uniform(0.5, 1.5),  # (default: 1.0)
            'fit_intercept': [True],  # (default), add biast to decision
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],  # liblinear
            'max_iter': sp.stats.randint(100, 250),  # (default: 100)
            'n_jobs': [2]
        }
    },
    'SGD': {
        'name': 'Stochastic Gradient Descent',
        'class': linear_model.SGDClassifier,
        'best_params': {  # MCC: 0.488
            'loss': 'log',
            'n_iter': 11,
            'penalty': 'l1'
        },
        'param_distributions': {
            'loss': ['hinge', 'log', 'modified_huber', 'perceptron', 'huber'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'n_iter': sp.stats.randint(6, 15)
        }
    },
    'NaiveBayes': {
        'name': 'Gaussian Naive Bayes',
        'class': naive_bayes.GaussianNB,
        'best_params': {},  # No parameters
        'param_distributions': {}
    }
}

# Set classifier colors
for classifier_name, color in zip(sorted(CLASSIFIERS), sns.color_palette()):
    CLASSIFIERS[classifier_name]['color'] = color

# Swap nicer colors
_red = CLASSIFIERS['LogisticRegression']['color']
_gold = CLASSIFIERS['RandomForest']['color']
CLASSIFIERS['LogisticRegression']['color'] = _gold
CLASSIFIERS['RandomForest']['color'] = _red


def _get_classifier(classifier_name, classifier_params=None):
    if classifier_params:
        return CLASSIFIERS[classifier_name]['class'](**classifier_params)
    else:
        return CLASSIFIERS[classifier_name]['class'](
            **CLASSIFIERS[classifier_name]['best_params'])


def _get_estimator(classifier_name, classifier_params=None):
    classifier = _get_classifier(classifier_name, classifier_params)
    return pipeline.Pipeline([
        ("imputer", preprocessing.Imputer(strategy="median")),
        ("scaler", preprocessing.StandardScaler()),
        ("classifier", classifier)])


#########
# Folds #
#########

FOLDS = {
    'kfold': {
        'name': 'k-fold',
        'class': cross_validation.KFold,
        'params': {
            'n_folds': 10,
            'shuffle': True,
        }
    },
    'stratified_kfold': {
        'name': 'Stratified k-fold',
        'class': cross_validation.StratifiedKFold,
        'params': {
            'n_folds': 10,
            'shuffle': True,
        }
    },
    'label_kfold': {
        'name': 'Label k-fold',
        'class': cross_validation.LabelKFold,
        'params': {
            'n_folds': 10,
        }
    },
    'predefined': {
        'name': 'Predefined split',
        'class': cross_validation.PredefinedSplit,
        'params': {}
    }
}


def _get_folds(folds_name, y, folds_params=None, labels=None):
    # Get default params
    params = FOLDS[folds_name]['params']

    # Update params if given
    if folds_params:
        params.update(folds_params)

    # Set default params depending on y and labels
    if folds_name == 'kfold':
        params['n'] = len(y)
    elif folds_name == 'stratified_kfold':
        params['y'] = y
    elif folds_name == 'label_kfold':
        params['labels'] = labels

    # Get folds
    return FOLDS[folds_name]['class'](**params)


def _evaluate(X, y, estimator, train, test, percentiles=None, scores=None):
    if not percentiles:
        percentiles = [100]

    estimator.fit(X[train], y[train])
    prob = estimator.predict_proba(X[test])[:, 1]
    pred = prob > 0.5

    results = pd.DataFrame()

    for percentile in percentiles:
        # Get confidence percentiles
        #low_score = np.percentile(prob, percentile/2.0)
        #high_score = np.percentile(prob, 100 - percentile/2.0)
        low_score = np.percentile(prob[prob < 0.5], percentile)
        high_score = np.percentile(prob[prob > 0.5], 100 - percentile)

        # Filter predictions with higher confidence
        fil = np.logical_or(prob <= low_score, high_score <= prob)

        # Initialize evaluation series
        evaluation = pd.Series((
            percentile,
            len(X),
            len(train),
            100.*len(train)/len(X),
            len(test),
            100.*sum(fil)/len(X),
            low_score,
            high_score,
            y,
            train,
            test,
            prob,
            pred,
        ), index=(
            'percentile', 'n_variants', 'n_train', 'p_train', 'n_test',
            'p_test', 'low_score', 'high_score',
            'y', 'train', 'test', 'prob', 'pred')
        )

        # Append evaluation
        evaluation = evaluation.append(
            evaluate_prediction(y[test[fil]], pred[fil]))

        # Append ROC curve
        if not (y[test[fil]].all() or (~y[test[fil]]).all()):
            evaluation = evaluation.append(
                pd.Series(metrics.roc_curve(y[test[fil]], prob[fil]),
                          index=('fpr', 'tpr', 'thresholds')))

        results = results.append(evaluation, ignore_index=True)

    return results


##################
# Cross-validate #
##################

def cross_validate(
        variants,
        classifiers=[('RandomForest', None)],
        folds=[('kfold', None)],
        label_name=None,
        scores=None,
        percentiles=None):
    # Get X, y, labels and estimators
    X, y = _get_X_y(variants)
    labels = variants[label_name] if label_name else None
    estimators = [
        _get_estimator(classifier_name, classifier_params)
        for classifier_name, classifier_params in classifiers]

    results = []
    for folds_name, folds_params in folds:
        logging.info('Generating %s (%s)...', folds_name, str(folds_params))
        for i, (train, test) in enumerate(
                _get_folds(folds_name, y, folds_params, labels)):
            print('.', end=' ')
            for (classifier_name, _), estimator in \
                    zip(classifiers, estimators):
                result = _evaluate(
                    X, y, estimator, train, test, percentiles, scores)
                result['classifier'] = CLASSIFIERS[classifier_name]['name']
                result['folds'] = FOLDS[folds_name]['name'] if folds_name != 'label_kfold' else label_name + ' k-fold'
                results.append(result)

    return pd.concat(results)


def roc_curve(cv_results):
    sns.set_style("whitegrid")
    sns.set_context("poster")
    figure = plt.figure()

    for cv_i, ((classifier, folds, percentile), cv_group) in enumerate(cv_results.groupby(
            ['classifier', 'folds', 'percentile'])):

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)

        for i, result in cv_group.iterrows():
            mean_tpr += sp.interp(mean_fpr, result.fpr, result.tpr)

        mean_tpr /= len(cv_group)
        mean_tpr[0], mean_tpr[-1] = 0.0, 1.0

        color = next(v['color'] for k, v in CLASSIFIERS.items()
                     if v['name'] == classifier)
        # QUICKFIX: get always different colors
        color = sns.color_palette()[cv_i % len(sns.color_palette())]

        if percentile == 100:
            line_style = '-'
        elif percentile == 66:
            line_style = '--'
        else:
            line_style = ':'

        plt.plot(
            mean_fpr,
            mean_tpr,
            lw=1.8,
            color=color,
            ls=line_style,
            label='%s. %d%%, %s. AUC = %0.3f, MCC = %0.3f' % (
                classifier,
                percentile,
                folds,
                cv_group.auc.mean(),
                cv_group.mcc.mean()))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6, 0.7), label='Luck')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc='lower right', prop={'size': 16})

    return figure


######################
# Features selection #
######################

def new_feature_score(params):
    indices, folds, clf, X, y = params
    return cross_validation.cross_val_score(
        clf, X[:, indices], y, cv=folds,
        scoring=metrics.make_scorer(metrics.matthews_corrcoef)
    ).mean()


def add_feature(indices, folds, estimator, X, y, n_jobs=4):
    new_indices = list(set(range(X[0].size)) - set(indices))
    params_list = [
        (indices + [i], folds, estimator, X, y)
        for i in new_indices
    ]

    pool = multiprocessing.Pool(n_jobs)
    scores = pool.map(new_feature_score, params_list)

    indices.append(new_indices[scores.index(max(scores))])
    return max(scores)


def remove_feature(indices, folds, estimator, X, y):
    estimator.fit(X[:, indices], y)
    importances = list(
        estimator.named_steps['classifier'].feature_importances_)
    del indices[importances.index(min(importances))]
    return indices


def iterative_features_selection(
        variants, max_features, folds_name='kfold', label_name=None,
        features=[], n_jobs=1):
    X, y = _get_X_y(variants)
    estimator = _get_estimator('RandomForest')
    labels = variants[label_name] if label_name else None

    # Best feature indices
    all_features = np.array(variants.filter(like='feature_').columns)
    indices = [list(all_features).index(feature) for feature in features]

    mcc = metrics.make_scorer(metrics.matthews_corrcoef)

    for i in range(max_features):
        # Folds
        folds = _get_folds(folds_name, y, labels=labels)

        # All features
        score = cross_validation.cross_val_score(
            estimator, X, y, scoring=mcc, cv=folds, n_jobs=n_jobs).mean()
        print('MCC = %.3f. All features (%d)' % (score, len(all_features)))

        # Currently selected features
        if indices:
            score = cross_validation.cross_val_score(
                estimator, X[:, indices], y, scoring=mcc, cv=folds,
                n_jobs=n_jobs).mean()
            print('MCC = %.3f. Current features (%d: %s)' % (
                score, len(indices), str(all_features[indices])))

        # Add three features
        score = add_feature(indices, folds, estimator, X, y, n_jobs)
        print('MCC = %.3f. Added 1 feature (%d: %s)' % (
            score, len(indices), str(all_features[indices])))
        score = add_feature(indices, folds, estimator, X, y, n_jobs)
        print('MCC = %.3f. Added 2 features (%d: %s)' % (
            score, len(indices), str(all_features[indices])))
        score = add_feature(indices, folds, estimator, X, y, n_jobs)
        print('MCC = %.3f. Added 3 features (%d: %s)' % (
            score, len(indices), str(all_features[indices])))

        # Remove one feature
        indices = remove_feature(indices, folds, estimator, X, y)
        score = cross_validation.cross_val_score(
            estimator, X[:, indices], y, scoring=mcc, cv=folds,
            n_jobs=n_jobs).mean()
        print('MCC = %.3f. Removed one feature (%d: %s)' % (
            score, len(indices), str(all_features[indices])))


##################
# Learning curve #
##################

def get_learning_curve(
        variants,
        classifier_name='RandomForest', classifier_params=None,
        folds_name='kfold', folds_params=None, label_name=None):
    # Get X, y, labels, estimator, folds and scorer
    X, y = _get_X_y(variants)
    labels = variants[label_name] if label_name else None
    estimator = _get_estimator(classifier_name, classifier_params)
    cv = _get_folds(folds_name, y, folds_params, labels)
    mcc_scorer = metrics.make_scorer(metrics.matthews_corrcoef)

    train_sizes, train_scores, test_scores = learning_curve.learning_curve(
        estimator, X, y, cv=cv, scoring=mcc_scorer)

    figure = plt.figure()
    plt.title('Learning curve')
    plt.xlabel('Training examples')
    plt.ylabel('MCC score')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean,
             'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean,
             'o-', color='g', label='Cross-validation score')
    plt.legend(loc='best')

    return figure


############
# Evaluate #
############

def evaluate(variants, estimator):
    """
    Evaluate the predictions of an estimator over an annotated set of variants

    variants must have a disease column
    """
    predict(variants, estimator)
    return evaluate_prediction(
        variants.disease, variants.pred_disease)


def evaluate_prediction(y, pred):
    # Convert to numpy array
    pred = np.array(pred, dtype=np.float64)

    # Count null predictions
    n_null = np.isnan(pred).sum()
    #pred.isnull().sum()

    # Remove null predictions and convert to np.bool
    y = y[~np.isnan(pred)].astype(np.bool)
    pred = pred[~np.isnan(pred)].astype(np.bool)

    if y.all() or (~y).all():
        return pd.Series((
            1.0 - n_null/(n_null + len(y)),
            metrics.accuracy_score(y, pred),
            np.nan, np.nan, np.nan, np.nan,
        ), index=('coverage', 'accuracy', 'specificity', 'sensitivity', 'auc', 'mcc'))
    return pd.Series((
        1.0 - n_null/(n_null + len(y)),
        metrics.accuracy_score(y, pred),
        specificity_score(y, pred),
        metrics.recall_score(y, pred),
        metrics.roc_auc_score(y, pred),
        metrics.matthews_corrcoef(y, pred),
    ), index=('coverage', 'accuracy', 'specificity', 'sensitivity', 'auc', 'mcc'))


################################
# Classifiers parameter search #
################################

def random_parameter_search(
        variants,
        n_iter=10,
        classifier_name='RandomForest',
        folds_name='kfold',
        folds_params=None,
        label_name=None):
    # Get X, y and classifier
    X, y = _get_X_y(variants)
    estimator = _get_estimator(classifier_name)

    # Get labels if given
    labels = variants[label_name] if label_name else None

    # Get folds
    logging.info('Generating %s folds, params: %s',
                 folds_name, str(folds_params))
    folds = _get_folds(folds_name, y, folds_params, labels)

    # User Mathews Correlation Coefficient scorer
    mcc_scorer = metrics.make_scorer(metrics.matthews_corrcoef)

    # Get parameters distribution for estimator
    param_distributions = {
        'classifier__' + k: v
        for k, v in CLASSIFIERS[classifier_name]['param_distributions'].items()
    }

    search = grid_search.RandomizedSearchCV(
        estimator,
        param_distributions,
        cv=folds,
        n_iter=n_iter,
        scoring=mcc_scorer,
        verbose=1)

    search.fit(X, y)
    return search


#########
# Train #
#########

def train(variants, classifier_name='RandomForest', classifier_params=None):
    """
    Train an estimator given the variants training set,
    the classifier name and its parameters
    """
    # Get X, y, estimator, fit and return
    X, y = _get_X_y(variants)
    estimator = _get_estimator(classifier_name, classifier_params)
    estimator.fit(X, y)
    return estimator


###########
# Predict #
###########

def predict(variants, estimator):
    """Get the prediction for variants using the given estimator"""
    X = _get_X(variants)
    variants['pred_score'] = estimator.predict_proba(X)[:, 1]
    variants['pred_disease'] = variants['pred_score'] > 0.5


def _get_X(variants):
    """Get features matrix X from variants"""
    X = variants.filter(like='feature_').as_matrix()
    X[X == -np.inf] = np.nan
    X[X == np.inf] = np.nan
    return X


def _get_X_y(variants):
    """Get features matrix X and disease vector y from variants"""
    X = _get_X(variants)
    y = variants.disease.as_matrix().astype(bool)
    return X, y
