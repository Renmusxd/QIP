# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from distutils.extension import Extension
from setuptools.command.build_ext import build_ext
from sys import platform
import glob
import os


# Delay import of numpy until after it has been installed. Same doesn't quite
# work for cython for some reason.
# Taken from https://stackoverflow.com/questions/2379898 user R_Beagrie
class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""
    def run(self):
        import numpy
        self.include_dirs.append(numpy.get_include())
        build_ext.run(self)


has_openmp = platform.lower().startswith('linux') or 'OPENMP' in os.environ

if has_openmp:
    print("="*10 + " Attempting install with OpenMP support " + "="*10)
    if platform.lower().startswith('darwin'):
        # OSX is hard
        print("!!! OpenMP on OSX can be tricky, make sure to install it using [brew install libomp] !!!")
        print("!!! if compilation/runtime fails you can unset OPENMP environment variable to        !!!")
        print("!!! disable openmp, or else specify an install location with 'export OPENMP=...'     !!!")
        parallel_flags = ['-Xclang', '-fopenmp', '-lomp']
        if 'OPENMP_PATH' in os.environ:
            parallel_flags += ['-I'+os.path.join(os.environ['OPENMP'],'include')]
            parallel_flags += ['-L' + os.path.join(os.environ['OPENMP'], 'lib/libomp.dylib')]
    else:
        # Linux is easy
        parallel_flags = ['-fopenmp']
else:
    print("="*10 + " OSX and Windows may not support OpenMP!       " + "="*10)
    print("="*10 + " export OPENMP environment variable to enable. " + "="*10)
    parallel_flags = []

extra_compile_flags = ["-O3", "-ffast-math", "-march=native"] + (parallel_flags if has_openmp else [])
extra_link_args = parallel_flags if has_openmp else []

USE_CYTHON = "CYTHON" in os.environ
if USE_CYTHON:
    try:
        from Cython.Build import cythonize
        print("Cythonizing source files...")
        # Cython handles creating extensions
        extensions = cythonize([Extension('qip.ext.*',
                               sources=['qip/ext/*.pyx'],
                               libraries=["m"],  # Allows dynamic linking of the math library on some Unix systems.
                               extra_compile_args=extra_compile_flags,
                               extra_link_args=extra_link_args)])
    except ImportError as e:
        print("Cython not installed, please install cython and try again")
        raise e
else:
    print("Not cythonizing c files. To cythonize install cython and set CYTHON env variable")
    # Make each of the c file extensions
    EXT = '.c'
    extensions = [Extension('qip.ext.{}'.format(os.path.basename(filename)[:-len(EXT)]),
                            sources=[filename],
                            libraries=["m"],  # Allows dynamic linking of the math library on some Unix systems.
                            extra_compile_args=extra_compile_flags,
                            extra_link_args=extra_link_args)
                  for filename in glob.glob('qip/ext/*'+EXT)]

setup(
    name='QIP',
    version='0.5.0',
    python_requires='>3.4',
    description='Quantum Computing Library',
    long_description='QIP: A quantum computing simulation library.',
    author='Sumner Hearth',
    author_email='sumnernh@gmail.com',
    url='https://github.com/Renmusxd/QIP',
    license='MIT',
    packages=find_packages(exclude=('tests','benchmark')),
    cmdclass={'build_ext': CustomBuildExtCommand},
    requires=['numpy', 'protobuf'],
    install_requires=['numpy', 'protobuf'],
    setup_requires=['numpy', 'protobuf'],
    ext_modules=extensions
)
