import io
import sys
import re
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

with open('graphdot/__init__.py') as fd:
    __version__ = re.search("__version__ = '(.*)'", fd.read()).group(1)

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


long_description = read('README.md')


class Tox(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import tox  # import here, cause outside the eggs aren't loaded
        errcode = tox.cmdline(self.test_args)
        sys.exit(errcode)


setup(
    name='graphdot',
    version=__version__,
    python_requires='>=3.6',
    url='https://gitlab.com/yhtang/graphdot',
    license='BSD',
    author='Yu-Hang Tang',
    tests_require=['tox'],
    install_requires=[
        'numpy>=1.16',
        'scipy>=1.3.0',
        'sympy>=1.3',
        'pandas>=0.24',
        'networkx>=2.4',
        'pycuda>=2019',
        'treelib>=1.6.1',
        'kahypar>=1.1.4',
    ] + [
        'ase>=3.17',
        'pymatgen>=2019',
        'mendeleev'
    ],
    extras_require={
        'docs': ['sphinx', 'sphinx-rtd-theme'],
    },
    cmdclass={'test': Tox},
    author_email='Tang.Maxin@gmail.com',
    description='GPU-accelerated graph similarity algorithm library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude='test'),
    package_data={
        'graphdot': ['*/*.h', '*/*/*.h', '*/*/*.cu'],
    },
    # include_package_data=True,
    platforms='any',
    classifiers=[
        'Programming Language :: Python',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: C++',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]
)
