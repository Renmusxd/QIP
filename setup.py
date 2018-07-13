# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from distutils.extension import Extension
from setuptools.command.build_ext import build_ext
from sys import platform
import os

try:
    from Cython.Build import cythonize
except ImportError:
    # I can't find a nicer way to do this. Pull requests graciously accepted.

    # Make fake cythonize call for use now, Extensions will fail and restart
    # due to setup_requires call.
    def cythonize(module_list):
        return module_list
    print("==============================================================\n"
          "| Cython not available, compilation will fail and restart    |\n"
          "| after installing cython.                                   |\n"
          "==============================================================")


# Delay import of numpy until after it has been installed. Same doesn't quite
# work for cython for some reason.
# Taken from https://stackoverflow.com/questions/2379898 user R_Beagrie
class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""
    def run(self):
        import numpy
        self.include_dirs.append(numpy.get_include())
        build_ext.run(self)


with open('LICENSE') as f:
    license = f.read()

if platform.startswith('linux') or 'OPENMP' in os.environ:
    print("="*10 + " Attempting install with OpenMP support " + "="*10)
    has_openmp = True
else:
    print("="*10 + " OSX and Windows may not support OpenMP! Set the environment variable OPENMP to enable! " + "="*10)
    has_openmp = False

parallel_flags = ['-fopenmp']
extra_compile_flags = ["-O3", "-ffast-math", "-march=native"] + (parallel_flags if has_openmp else [])
extra_link_args = parallel_flags if has_openmp else []

setup(
    name='QIP',
    version='0.3.5',
    python_requires='>3.4',
    description='Quantum Computing Library',
    long_description='QIP: A quantum computing simulation library.',
    author='Sumner Hearth',
    author_email='sumnernh@gmail.com',
    url='https://github.com/Renmusxd/QIP',
    license=license,
    packages=find_packages(exclude=('benchmark')),
    package_data={'': ['LICENSE', 'requirements.txt']},
    cmdclass={'build_ext': CustomBuildExtCommand},
    requires=['numpy', 'cython'],
    install_requires=['numpy', 'cython'],
    setup_requires=['setuptools>=18.0', 'numpy', 'cython'],
    ext_modules=cythonize([Extension('qip.ext.*',
                           sources=['qip/ext/*.pyx'],
                           libraries=["m"],  # Allows dynamic linking of the math library on some Unix systems.
                           extra_compile_args=extra_compile_flags,
                           extra_link_args=extra_link_args)])
)
