#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name='verse',
    version='0.1',
    description='AutoVerse',
    author='Yangge Li, Katherine Braught, Haoqing Zhu',
    maintainer='Yangge Li, Katherine Braught, Haoqing Zhu',
    maintainer_email='{li213, braught2, haoqing3}@illinois.edu',
    packages=find_packages(exclude=["tests", "demo"]),
    python_requires='>=3.8',
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "polytope",
        "pyvista",
        "networkx",
        "sympy",
        "six",
        "astunparse",
        "treelib",
        "z3-solver",
        "igraph",
        "plotly",
        "beautifulsoup4",
        "lxml",
        "torch",
        "tqdm",
        "intervaltree",
        "Pympler"
        "nbformat",
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.8',
    ]
)
