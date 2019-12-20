# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


setup(
    name='QIP',
    version='0.6.0',
    python_requires='>3.4',
    description='Quantum Computing Library',
    long_description='QIP: A quantum computing simulation library.',
    author='Sumner Hearth',
    author_email='sumnernh@gmail.com',
    url='https://github.com/Renmusxd/QIP',
    license='MIT',
    packages=find_packages(exclude=('tests', 'benchmark')),
    requires=['numpy', 'protobuf', 'qip_backend'],
)
