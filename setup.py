# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

readme = "QIP: A quantum computing simulation library."

with open('LICENSE') as f:
    license = f.read()

setup(
    name='QIP',
    version='0.2.3',
    description='Quantum Computing Library',
    long_description=readme,
    author='Sumner Hearth',
    author_email='sumnernh@gmail.com',
    url='https://github.com/Renmusxd/QIP',
    license=license,
    packages=find_packages(exclude=('tests','benchmark')),
    ext_modules=cythonize([
        Extension("qip.ext.*", ["qip/ext/*.pyx"], extra_compile_args=["-O3"], include_dirs=[numpy.get_include()])
    ])
)
