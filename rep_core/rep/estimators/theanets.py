from __future__ import division, print_function, absolute_import
import numpy

from .interface import Classifier
from .utils import check_inputs

try:
    import theanets as tnt
except ImportError as e:
    raise ImportError("Install theanets before")

__author__ = 'Ignatyeva Lisa'

class TheanetsClassifier(Classifier):

    def __init__(self, 
                 layers,
                 features=None,
                 **kwargs):
        # TODO: make printing TheanetsClassifier informative (include training method & its parameters)
        # TODO: support more parameters in creating an experiment
        self.layers = layers
        self.kwargs = kwargs
        self.exp = tnt.Experiment(tnt.Classifier, 
                                  layers=layers)
        Classifier.__init__(self, features=features)

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            # TODO: implement sample weight
            raise ValueError('sample_weight is not supported yet')
        X, y, sample_weight = check_inputs(X, y, sample_weight)
        X = self._get_train_features(X)
        self.exp.train((X.values.astype(numpy.float32), y.astype(numpy.int32)),
                       **self.kwargs)

    def predict_proba(self, X):
        X = self._get_train_features(X)
        return self.exp.network.predict(X.values.astype(numpy.float32))

    def staged_predict_proba(self, X):
        # TODO: implement it!
        raise AttributeError('staged_predict_proba is not supported for theanets')