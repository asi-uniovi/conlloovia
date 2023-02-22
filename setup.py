#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=7.0",
    "pint",
    "pulp",
    "rich",
    "cloudmodel @ git+https://jentrialgo@github.com/jldiaz-uniovi/cloudmodel.git#egg=cloudmodel",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Joaquín Entrialgo",
    author_email="joaquin@uniovi.es",
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    description="Use linear programming to allocate containers to cloud infrastructure",
    entry_points={
        "console_scripts": [
            "conlloovia=conlloovia.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="conlloovia",
    name="conlloovia",
    packages=find_packages(include=["conlloovia", "conlloovia.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/jentrialgo/conlloovia",
    version="0.2.0",
    zip_safe=False,
)
