#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_namespace_packages, setup


with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

requirements = [
    "f90nml>=1.1.0",
    "numpy==1.26.4",
    "xarray",
]

test_requirements = [
    "coverage",
    "pytest",
    "pytest-subtests",
    "serialbox",
]

ndsl_requirements = ["ndsl @ git+https://github.com/NOAA-GFDL/NDSL.git@2024.09.00"]

develop_requirements = [
    *ndsl_requirements,
    *test_requirements,
    "pre-commit",
]

extras_requires = {
    "develop": develop_requirements,
    "ndsl": ndsl_requirements,
    "test": test_requirements,
}

setup(
    author="NOAA - Geophysical Fluid Dynamics Laboratory",
    author_email="oliver.elbert@noaa.gov",
    python_requires=">=3.11,<3.12",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GPLv3 License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    description="PyFV3 is a NDSL-based FV3 dynamical core for atmospheric models.",
    install_requires=requirements,
    extras_require=extras_requires,
    license="GPLv3 license",
    long_description=readme,
    include_package_data=True,
    name="pyFV3",
    packages=find_namespace_packages(include=["pyFV3", "pyFV3.*"]),
    setup_requires=[],
    test_suite="tests",
    url="https://github.com/NOAA-GFDL/pyFV3",
    version="0.2.0",
    zip_safe=False,
)
