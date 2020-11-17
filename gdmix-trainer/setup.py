from pathlib import Path
from setuptools import find_namespace_packages, setup
from sys import platform as _platform

import sys


VERSION="0.3.0"
current_dir = Path(__file__).resolve().parent
with open(current_dir.joinpath('README.md'), encoding='utf-8') as f:
    long_description = f.read()

if _platform not in ["linux", "linux2", "darwin"]:
    print("ERROR: platform {} isn't supported".format(_platform))
    sys.exit(1)

setup(
    name="gdmix-trainer",
    python_requires='>=3.7',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=["Programming Language :: Python :: 3.7",
                 "Intended Audience :: Science/Research",
                 "Intended Audience :: Developers",
                 "License :: OSI Approved"],
    license='BSD-2-CLAUSE',
    version=f'{VERSION}',
    package_dir={'': 'src'},
    packages=find_namespace_packages(where='src'),
    include_package_data=True,
    install_requires=[
        "setuptools>=41.0.0",
        "tensorflow==1.15.2",
        "tensorflow_ranking==0.1.4",
        "fastavro==0.21.22",
        "decorator==4.4.2",
        "detext-nodep==2.0.9",
        "psutil==5.7.0",
        "scipy==1.3.2",
        "scikit-learn==0.21.2",
        "smart-arg==0.2.12"
    ],
    tests_require=['pytest']
)
