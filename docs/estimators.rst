.. _estimators:

Estimators (classification and regression)
==========================================

This module contains wrappers with :class:`sklearn` interface for different machine learning libraries:

* TMVA

* sklearn

* XGBoost

* pybrain

* neurolab

* theanets.

We defined some interface for classifiers' and regressors' wrappers, so new wrappers can be added for another libraries
following the same interface. Notably the interface has backward compatibility with scikit-learn library.

Sklearn wrapper is the same sklearn model, but it operates with :class:`pandas.DataFrame` data (though supports :class:`numpy.ndarray` as well)
and can use only those features user pointed in constructor (:class:`pandas.DataFrame` provides named columns).


Estimators interfaces (for classification and regression)
---------------------------------------------------------
.. automodule:: rep.estimators.interface
    :members:
    :inherited-members:
    :undoc-members:
    :show-inheritance:


Sklearn classifier and regressor
--------------------------------
.. automodule:: rep.estimators.sklearn
    :members:
    :show-inheritance:
    :undoc-members:

TMVA classifier and regressor
-----------------------------
.. automodule:: rep.estimators.tmva
    :members:
    :show-inheritance:
    :undoc-members:


XGBoost classifier and regressor
--------------------------------
.. automodule:: rep.estimators.xgboost
    :members:
    :show-inheritance:
    :undoc-members:

Theanets classifier and regressor
---------------------------------
.. automodule:: rep.estimators.theanets
    :members:
    :show-inheritance:
    :undoc-members:

Neurolab classifier and regressor
---------------------------------
.. automodule:: rep.estimators.neurolab
    :members:
    :show-inheritance:
    :undoc-members:

Pybrain classifier and regressor
--------------------------------
.. automodule:: rep.estimators.pybrain
    :members:
    :show-inheritance:
    :undoc-members:

Examples
--------

Classification
**************

* Prepare dataset
    >>> from sklearn import datasets
    >>> import pandas, numpy
    >>> from rep.utils import train_test_split
    >>> from sklearn.metrics import roc_auc_score
    >>> # iris data
    >>> iris = datasets.load_iris()
    >>> data = pandas.DataFrame(iris.data, columns=['a', 'b', 'c', 'd'])
    >>> labels = iris.target
    >>> # Take just two classes instead of three
    >>> data = data[labels != 2]
    >>> labels = labels[labels != 2]
    >>> train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=0.7)

* Sklearn classification
    >>> from rep.estimators import SklearnClassifier
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> # Using gradient boosting with default settings
    >>> sk = SklearnClassifier(GradientBoostingClassifier(), features=['a', 'b'])
    >>> # Training classifier
    >>> sk.fit(train_data, train_labels)
    >>> pred = sk.predict_proba(test_data)
    >>> print pred
    [[  9.99842983e-01   1.57016893e-04]
     [  1.45163843e-04   9.99854836e-01]
     [  9.99842983e-01   1.57016893e-04]
     [  9.99827693e-01   1.72306607e-04], ..]
    >>> roc_auc_score(test_labels, pred[:, 1])
    0.99768518518518523


* TMVA classification
    >>> from rep.estimators import TMVAClassifier
    >>> tmva = TMVAClassifier(method='kBDT', NTrees=100, Shrinkage=0.1, nCuts=-1, BoostType='Grad', features=['a', 'b'])
    >>> tmva.fit(train_data, train_labels)
    >>> pred = tmva.predict_proba(test_data)
    >>> print pred
    [[  9.99991025e-01   8.97546346e-06]
     [  1.14084636e-04   9.99885915e-01]
     [  9.99991009e-01   8.99060302e-06]
     [  9.99798700e-01   2.01300452e-04], ..]
    >>> roc_auc_score(test_labels, pred[:, 1])
    0.99999999999999989

* XGBoost classification
    >>> from rep.estimators import XGBoostClassifier
    >>> # XGBoost with default parameters
    >>> xgb = XGBoostClassifier(features=['a', 'b'])
    >>> xgb.fit(train_data, train_labels, sample_weight=numpy.ones(len(train_labels)))
    >>> pred = xgb.predict_proba(test_data)
    >>> print pred
    [[ 0.9983651   0.00163494]
     [ 0.00170585  0.99829417]
     [ 0.99845636  0.00154361]
     [ 0.96618336  0.03381656], ..]
    >>> roc_auc_score(test_labels, pred[:, 1])
    0.99768518518518512


Regression
**********

* Prepare dataset
    >>> from sklearn import datasets
    >>> from sklearn.metrics import mean_squared_error
    >>> from rep.utils import train_test_split
    >>> import pandas, numpy
    >>> # diabetes data
    >>> diabetes = datasets.load_diabetes()
    >>> features = ['feature_%d' % number for number in range(diabetes.data.shape[1])]
    >>> data = pandas.DataFrame(diabetes.data, columns=features)
    >>> labels = diabetes.target
    >>> train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=0.7)

* Sklearn regression
    >>> from rep.estimators import SklearnRegressor
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> # Using gradient boosting with default settings
    >>> sk = SklearnRegressor(GradientBoostingRegressor(), features=features[:8])
    >>> # Training classifier
    >>> sk.fit(train_data, train_labels)
    >>> pred = sk.predict(train_data)
    >>> numpy.sqrt(mean_squared_error(train_labels, pred))
    60.666009962879265

* TMVA regression
    >>> from rep.estimators import TMVARegressor
    >>> tmva = TMVARegressor(method='kBDT', NTrees=100, Shrinkage=0.1, nCuts=-1, BoostType='Grad', features=features[:8])
    >>> tmva.fit(train_data, train_labels)
    >>> pred = tmva.predict(test_data)
    >>> numpy.sqrt(mean_squared_error(test_labels, pred))
    73.74191838418254

* XGBoost regression
    >>> from rep.estimators import XGBoostRegressor
    >>> # XGBoost with default parameters
    >>> xgb = XGBoostRegressor(features=features[:8])
    >>> xgb.fit(train_data, train_labels, sample_weight=numpy.ones(len(train_labels)))
    >>> pred = xgb.predict(test_data)
    >>> numpy.sqrt(mean_squared_error(test_labels, pred))
    65.557743652940133
