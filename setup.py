# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='QIP',
    version='0.1',
    description='Quantum Computing Library',
    long_description=readme,
    author='Sumner Hearth',
    author_email='sumnernh@gmail.com',
    url='https://github.com/Renmusxd/QIP',
    license=license,
    packages=find_packages(exclude=('tests',)),
    ext_modules=cythonize(
        Extension("*", ["qip/ext/*.pyx"], extra_compile_args=["-O3"], include_dirs=[numpy.get_include()])
    )
)
