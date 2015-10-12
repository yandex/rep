from fabric.api import *
import fabtools
from fabtools import require

# env variables should be changed
env.hosts = ['192.168.1.1']
env.user = "root"
env.disable_known_hosts = True

REP_DIR_PATH = '/home/'
# full path ending with / is required; ~/ is not working
# ROOT and XGBoost will be installed there as well
REP_DIR_NAME = 'rep_dir'


# Probably you don't want to change anything below


def prepare():
    fabtools.deb.update_index()
    require.deb.packages([
        'python-dev',
        'libblas-dev',
        'libatlas-dev',
        'liblapack-dev',
        'gfortran',
        'g++',
        'python-setuptools',
        'libpng-dev',
        'libjpeg8-dev',
        'libfreetype6-dev',
        'libxft-dev',
        'cmake',
        'git',
        'python-numpy',
        'python-scipy',
        'ipython',
        'ipython-notebook',
        'python-pandas',
        'python-sympy',
        'python-nose'
    ])

    if not fabtools.python.is_pip_installed():
        fabtools.python.install_pip()

    run('cd {} && mkdir {}'.format(REP_DIR_PATH, REP_DIR_NAME))


def install_xgboost():
    with cd('{}{}'.format(REP_DIR_PATH, REP_DIR_NAME)):
        fabtools.git.clone('https://github.com/dmlc/xgboost')
        with cd('xgboost'):
            run('make')
            with cd('python-package'):
                run('python setup.py install')


def install_rep():
    # install REP itself
    with shell_env(LC_ALL='C'):
        fabtools.python.install_requirements('https://github.com/yandex/rep/raw/master/requirements.txt', use_sudo=True)
        fabtools.python.install('rep', use_sudo=True)

    # install hep_ml
    fabtools.python.install('hep_ml', use_sudo=True)

    # testing
    with cd(REP_DIR_PATH + REP_DIR_NAME):
        fabtools.git.clone('https://github.com/yandex/rep')
        with cd('rep/tests'):
            tests = run('ls').split()
            for test in [filename for filename in tests if filename.startswith('test')]:
                run('python {}'.format(test))


def cleanup():
    run('rm -r {}'.format(REP_DIR_PATH + REP_DIR_NAME))


def install_root():
    with cd('{}{}'.format(REP_DIR_PATH, REP_DIR_NAME)):
        run('wget https://root.cern.ch/download/root_v5.34.34.Linux-ubuntu14-x86_64-gcc4.8.tar.gz')
        run('tar -zxf root_v5.34.34.Linux-ubuntu14-x86_64-gcc4.8.tar.gz')
        run('rm root_v5.34.34.Linux-ubuntu14-x86_64-gcc4.8.tar.gz')

        with cd('root/bin'):
            run('source thisroot.sh')
        with shell_env(ROOTSYS='{}{}/root'.format(REP_DIR_PATH, REP_DIR_NAME), PATH='$ROOTSYS/bin:$PATH'):
            fabtools.python.install(['rootpy==0.7.1', 'root_numpy==4.1.2'])


def setup_rep():
    prepare()
    install_xgboost()
    install_root()
    install_rep()
