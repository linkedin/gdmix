from pathlib import Path
from setuptools import find_namespace_packages, setup
from sys import platform as _platform

import sys

VERSION = "0.4.0"
current_dir = Path(__file__).resolve().parent
with open(current_dir.joinpath('README.md'), encoding='utf-8') as f:
    long_description = f.read()

if _platform not in ["linux", "linux2", "darwin"]:
    print("ERROR: platform {} isn't supported".format(_platform))
    sys.exit(1)

TF_VERSION_QUANTIFIER = '>=2.4,<2.5'

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
    version=VERSION,
    package_dir={'': 'src'},
    packages=find_namespace_packages(where='src'),
    include_package_data=True,
    install_requires=[
        "absl-py==0.10",
        "decorator==4.4.2",
        "detext-nodep==3.0.0",
        "fastavro==1.4.7",
        "google-auth==1.25.0",
        "google-cloud-bigquery==2.18.0",
        "grpcio==1.32.0",
        "numpy==1.19.5",
        "protobuf==3.19",
        "psutil==5.7.0",
        "scikit-learn==1.0",
        "setuptools>=41.0.0",
        "six==1.15.0",
        "smart-arg==0.4",
        "statsmodels==0.13.1",
        f"tensorflow{TF_VERSION_QUANTIFIER}",
        f"tensorflow-text{TF_VERSION_QUANTIFIER}",
        f"tensorflow-serving-api{TF_VERSION_QUANTIFIER}",
        "tensorflow_ranking",
        f"tf-models-official{TF_VERSION_QUANTIFIER}",
        "tomli==1.2.2"
    ],
    tests_require=['pytest']
)
