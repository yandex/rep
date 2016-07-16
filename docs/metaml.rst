.. _metaml:

Meta Machine Learning
=====================

Meta machine learning contains specific ML-algorithms, that are taking some classification/regression model as an input.

Also there is a Factory which allows set of models training and comparing them very simply.

Factory
-------
.. automodule:: rep.metaml.factory
    :members:
    :show-inheritance:
    :undoc-members:

Factory Examples
----------------

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

* Train factory of classifiers
    >>> from rep.metaml import ClassifiersFactory
    >>> from rep.estimators import TMVAClassifier, SklearnClassifier, XGBoostClassifier
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> factory = ClassifiersFactory()
    >>> estimators
    >>> factory.add_classifier('tmva', TMVAClassifier(method='kBDT', NTrees=100, Shrinkage=0.1, nCuts=-1, BoostType='Grad', features=['a', 'b']))
    >>> factory.add_classifier('ada', GradientBoostingClassifier())
    >>> factory['xgb'] = XGBoostClassifier(features=['a', 'b'])
    >>> factory.fit(train_data, train_labels)
    model ef           was trained in 0.22 seconds
    model tmva         was trained in 2.47 seconds
    model ada          was trained in 0.02 seconds
    model xgb          was trained in 0.01 seconds
    Totally spent 2.71 seconds on training
    >>> pred = factory.predict_proba(test_data)
    data was predicted by tmva         in 0.02 seconds
    data was predicted by ada          in 0.00 seconds
    data was predicted by xgb          in 0.00 seconds
    Totally spent 0.05 seconds on prediction
    >>> print pred
    OrderedDict([('tmva', array([[  9.98732217e-01,   1.26778255e-03], [  9.99649503e-01,   3.50497149e-04], ..])),
                 ('ada', array([[  9.99705117e-01,   2.94883265e-04], [  9.99705117e-01,   2.94883265e-04], ..])),
                 ('xgb', array([[  9.91589248e-01,   8.41078255e-03], ..], dtype=float32))])
    >>> for key in pred:
    >>>    print key, roc_auc_score(test_labels, pred[key][:, 1])
    tmva 0.933035714286
    ada 1.0
    xgb 0.995535714286

Grid Search
-----------

.. automodule:: rep.metaml.gridsearch


Folding
-------
.. automodule:: rep.metaml.folding
    :members:
    :show-inheritance:
    :undoc-members:

Cache
-----
.. automodule:: rep.metaml.cache
    :members:
    :show-inheritance:
    :undoc-members:

Stacking
--------
.. automodule:: rep.metaml.stacking
    :members:
    :show-inheritance:
    :undoc-members:
