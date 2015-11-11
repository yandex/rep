from setuptools import setup
import codecs
import os
import re

here = os.path.abspath(os.path.dirname(__file__))


def find_version(*file_paths):
    # Open in Latin-1 so that we avoid encoding errors.
    # Use codecs.open for Python 2 compatibility
    with codecs.open(os.path.join(here, *file_paths), 'r') as f:
        version_file = f.read()

    # The version line must have the form
    # __version__ = 'ver'
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    # TODO better selection
    assert str.startswith(version_match.group(1), '0.6')
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


with open(os.path.join(here, 'README.md')) as f:
    long_description = f.read()

with open(os.path.join(here, 'AUTHORS')) as f:
    authors = f.read()

with open(os.path.join(here, 'requirements.txt')) as f:
    requirements = f.read().split('\n')

setup(
    name="rep",
    version=find_version('rep', '__init__.py'),
    description="infrastructure for computational experiments on shared big datasets",
    long_description="Reproducible Experiment Platform is a collaborative software infrastructure for computational " \
                     "experiments on shared big datasets, which allows obtaining reproducible, repeatable results " \
                     "and consistent comparisons of the obtained results.",
    url='https://github.com/yandex/rep',

    # Author details
    author=authors,
    author_email='axelr@yandex-team.ru, antares@yandex-team.ru',

    # Choose your license
    license='Apache-2.0 License',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages.
    # packages=find_packages(exclude=["cern_utils", "docs", "tests*"]),
    packages=['rep', 'rep.estimators', 'rep.data', 'rep.metaml', 'rep.report', 'rep.test'],
    package_dir={'rep': 'rep'},
    classifiers=[
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7 ',
        'Programming Language :: Python :: 3.4 ',
    ],

    # What does your project relate to?
    keywords='machine learning, ydf, high energy physics, particle physics, data analysis, reproducible experiment',


    # List run-time dependencies here. These will be installed by pip when your
    # project is installed.
    install_requires=requirements,

)
