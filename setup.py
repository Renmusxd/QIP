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
    version='0.2.4',
    description='Quantum Computing Library',
    long_description=readme,
    author='Sumner Hearth',
    author_email='sumnernh@gmail.com',
    url='https://github.com/Renmusxd/QIP',
    license=license,
    packages=find_packages(exclude=('tests','benchmark')),
    ext_modules=cythonize([
        Extension("qip.ext.*", ["qip/ext/*.pyx"], extra_compile_args=["-O3"], include_dirs=[numpy.get_include()])
    ]),
    install_requires=['numpy>=1.13.1', 'scipy>=0.19.1', 'cython>=0.27.3']
)
