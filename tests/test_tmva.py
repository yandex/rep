from __future__ import division, print_function, absolute_import
from rep.test.test_estimators import check_classifier, check_regression
from rep.estimators import TMVAClassifier, TMVARegressor

__author__ = 'Alex Rogozhnikov'


def test_tmva():
    # check classifier
    factory_options = "Silent=True:V=False:DrawProgressBar=False"
    cl = TMVAClassifier(factory_options=factory_options, method='kBDT', NTrees=10)
    check_classifier(cl, check_instance=True, has_staged_pp=False, has_importances=False)

    cl = TMVAClassifier(factory_options=factory_options, method='kSVM', Gamma=0.25, Tol=0.001,
                        sigmoid_function='identity')
    check_classifier(cl, check_instance=True, has_staged_pp=False, has_importances=False)

    cl = TMVAClassifier(factory_options=factory_options, method='kCuts',
                        FitMethod='GA', EffMethod='EffSel', sigmoid_function='sig_eff=0.9')
    check_classifier(cl, check_instance=True, has_staged_pp=False, has_importances=False)
    # check regressor, need to run twice to check for memory leak.
    for i in range(2):
        check_regression(TMVARegressor(factory_options=factory_options, method='kBDT', NTrees=10), check_instance=True,
                         has_staged_predictions=False, has_importances=False)
