"""
Generates environment files separately for python 2 and python 3. (rep_py2, rep_py3)
SImply substitutes fields in the template


"""
from __future__ import division, print_function, absolute_import

__author__ = 'Alex Rogozhnikov'

with open('environment-rep-template.yaml', 'r') as template_file:
    content = template_file.read()

for python_version in ["2.7", "3.5"]:
    python_major_version = python_version[:1]
    new_content = content.replace('{PYTHON_MAJOR_VERSION}', python_major_version)\
                         .replace('{PYTHON_VERSION}', python_version)
    with open('environment-rep' + python_major_version + '.yaml', 'w') as new_file:
        new_file.write(new_content)