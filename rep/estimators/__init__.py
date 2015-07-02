from __future__ import division, print_function, absolute_import

from .interface import Classifier, Regressor
from .sklearn import SklearnClassifier, SklearnRegressor
from .theanets import TheanetsClassifier, TheanetsRegressor
from .neurolab import NeurolabClassifier, NeurolabRegressor
from .pybrain import PyBrainClassifier, PyBrainRegressor

try:
    from .tmva import TMVAClassifier, TMVARegressor
except:
    pass

try:
    from .xgboost import XGBoostClassifier, XGBoostRegressor
except:
    pass