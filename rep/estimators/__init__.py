from __future__ import division, print_function, absolute_import

from .interface import Classifier, Regressor
from .sklearn import SklearnClassifier, SklearnRegressor

try:
    from .theanets import TheanetsClassifier, TheanetsRegressor
except:
    pass

try:
    from .neurolab import NeurolabClassifier, NeurolabRegressor
except:
    pass

try:
    from .pybrain import PyBrainClassifier, PyBrainRegressor
except:
    pass

try:
    from .tmva import TMVAClassifier, TMVARegressor
except:
    pass

try:
    from .xgboost import XGBoostClassifier, XGBoostRegressor
except:
    pass