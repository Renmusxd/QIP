# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from distutils.extension import Extension
from setuptools.command.build_ext import build_ext

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


setup(
    name='QIP',
    version='0.3.3',
    python_requires='>3.4',
    description='Quantum Computing Library',
    long_description='QIP: A quantum computing simulation library.',
    author='Sumner Hearth',
    author_email='sumnernh@gmail.com',
    url='https://github.com/Renmusxd/QIP',
    license=license,
    packages=find_packages(exclude=('test','benchmark')),
    package_data={'': ['LICENSE', 'requirements.txt']},
    cmdclass={'build_ext': CustomBuildExtCommand},
    requires=['numpy', 'cython'],
    install_requires=['numpy', 'cython'],
    setup_requires=['setuptools>=18.0', 'numpy', 'cython'],
    ext_modules=cythonize([Extension('qip.ext.*',
                           sources=['qip/ext/*.pyx', 'qip/ext/*.pxd'],
                           extra_compile_args=['-O3'])])
)
