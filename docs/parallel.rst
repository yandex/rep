.. _parallel:

Parallel computing
==================

Many problems in machine learning require training several or many estimators.
At the same time in applications single model training can take from minutes to hours.

In order to get things done faster we can use parallel computing.

Such meta-algorithms as:

 * :class:`rep.metaml.folding.FoldingClassifier` and :class:`rep.metaml.folding.FoldingRegressor`
 * :class:`rep.metaml.factory.ClassifiersFactory` and :class:`rep.metaml.factory.RegressorsFactory`
 * :class:`rep.metaml.gridsearch.GridOptimalSearchCV`

support `parallel_profile` option. The following options exist:

 * :code:`parallel_profile=None` to use single thread (default)
 * :code:`parallel_profile='threads-3'` to run in 3 threads on a single machine (you can use any, not necessarily 3)
 * :code:`parallel_profile='my_ipython_cluster_name'` to use IPython cluster.

More details about IPython cluster (`ipyparallel`) is available `here <http://ipyparallel.readthedocs.org/en/latest/>`_.

Important remark: since algorithms inside **REP** interplays with each other, you can combine parallel computations.
You can run `GridOptimalSearchCV` (using IPython cluster) to optimize `FoldingClassifier` (using threads)
which runs over `RandomForest` (using again threads).

This way you can use available resources more effectively.




