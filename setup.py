from setuptools import setup

# https://betterscientificsoftware.github.io/python-for-hpc/tutorials/python-pypi-packaging/#creating-a-python-package

setup(
    name='PerplexityLab',
    version='0.0.1b',
    description='Python package for Reproducible research, File management, Explore multiple conditions, Parallel, '
                'Remember, Avoid re-doing, Analysis, Make reports, Explorable reports, Carbon and energy footprint',
    url='https://github.com/agussomacal/PerplexityLab',
    author='Somacal Agustin',
    author_email='saogmuas@gmail.com',
    license='GNU General Public License v3.0',
    packages=['PerplexityLab'],
    install_requires=["joblib",
                      "pathos",
                      "numpy",
                      "pandas",
                      "matplotlib",
                      "seaborn",
                      "tqdm",
                      "makefun",
                      "eco2ai",
                      ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
    ],
)
