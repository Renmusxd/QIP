# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from distutils.extension import Extension
from setuptools.command.build_ext import build_ext

# This awful setup is because cython is required to run the file responsible to fetching cython, likewise numpy
# needs to provide header information prior to install. Pull requests welcome!
try:
    from Cython.Build import cythonize
except ModuleNotFoundError:
    raise Exception('Cython must be installed before attempting to install QIP. Setuptools + Pip technically supports '
                    'cython at install but has no clear documentation to make it work the same way as out-of-the-box '
                    'cython. Just run:\npip install cython\nthen reinstall QIP as usual.')
else:
    # Taken from https://stackoverflow.com/questions/2379898 user R_Beagrie
    class CustomBuildExtCommand(build_ext):
        """build_ext command for use when numpy headers are needed."""
        def run(self):
            import numpy
            from Cython.Build import cythonize

            self.include_dirs.append(numpy.get_include())
            build_ext.run(self)

    with open('LICENSE') as f:
        license = f.read()

    setup(
        name='QIP',
        version='0.3',
        python_requires='>3.4',
        description='Quantum Computing Library',
        long_description='QIP: A quantum computing simulation library.',
        author='Sumner Hearth',
        author_email='sumnernh@gmail.com',
        url='https://github.com/Renmusxd/QIP',
        license=license,
        packages=find_packages(exclude=('tests','benchmark')),
        package_data={'': ['LICENSE']},
        cmdclass={'build_ext': CustomBuildExtCommand},
        requires=['numpy', 'cython'],
        install_requires=['numpy', 'cython'],
        setup_requires=['setuptools>=18.0', 'numpy', 'cython'],
        ext_modules=cythonize([Extension('qip.ext.*',
                               sources=['qip/ext/*.pyx'],
                               extra_compile_args=['-O3'])])
    )
