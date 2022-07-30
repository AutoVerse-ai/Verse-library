#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name='verse',
    version='0.1',
    description='AutoVerse',
    author='Yangge Li, Katherine Braught, Haoqing Zhu',
    maintainer='Yangge Li, Katherine Braught, Haoqing Zhu',
    maintainer_email='{li213, braught2, haoqing3}@illinois.edu',
    license='Apache-2.0',
    packages=find_packages(exclude=["tests", "demo"]),
    python_requires='>=3.8',
    install_requires=[
        "numpy~=1.22.1",
        "scipy~=1.8.1",
        "matplotlib~=3.4.2",
        "polytope~=0.2.3",
        "pyvista~=0.32.1",
        "networkx~=2.2",
        "sympy~=1.6.2",
        "six~=1.14.0",
        "astunparse~=1.6.3",
        "treelib~=1.6.1",
        "z3-solver~=4.8.17.0",
        "igraph~=0.9.10",
        "plotly~=5.8.0",
        "beautifulsoup4~=4.11.1",
        "lxml~=4.9.1"
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache License 2.0',
        'Programming Language :: Python :: 3.8',
    ]
)
