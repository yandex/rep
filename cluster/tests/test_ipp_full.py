import ipyparallel as ipp
c = ipp.Client(profile="default")

import numpy, pandas
import os
from rep.utils import train_test_split
from sklearn.metrics import roc_auc_score
import subprocess
columns = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'g']
if not os.path.exists("toy_datasets/magic04.data"):
	os.makedirs("toy_datasets")
	p = subprocess.Popen("wget -O toy_datasets/magic04.data -nc --no-check-certificate https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data", shell=True)
	p.wait()

print "Downloaded magic04.data"
data = pandas.read_csv('toy_datasets/magic04.data', names=columns)
labels = numpy.array(data['g'] == 'g', dtype=int)
data = data.drop('g', axis=1)
import numpy
import numexpr
import pandas
from rep import utils
from sklearn.ensemble import GradientBoostingClassifier
from rep.report.metrics import RocAuc
from rep.metaml import GridOptimalSearchCV, FoldingScorer, RandomParameterOptimizer
from rep.estimators import SklearnClassifier, TMVAClassifier, XGBoostRegressor
# define grid parameters
grid_param = {}
grid_param['learning_rate'] = [0.2, 0.1, 0.05, 0.02, 0.01]
grid_param['max_depth'] = [2, 3, 4, 5]
# use random hyperparameter optimization algorithm
generator = RandomParameterOptimizer(grid_param)
# define folding scorer
scorer = FoldingScorer(RocAuc(), folds=3, fold_checks=3)
estimator = SklearnClassifier(GradientBoostingClassifier(n_estimators=30))
#grid_finder = GridOptimalSearchCV(estimator, generator, scorer)
#% time grid_finder.fit(data, labels)
grid_finder = GridOptimalSearchCV(estimator, generator, scorer, parallel_profile="default")
print "start grid search"
grid_finder.fit(data, labels)

grid_finder.params_generator.print_results()

assert 10 == grid_finder.params_generator.n_evaluations, "oops"
