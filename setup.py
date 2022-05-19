#!/usr/bin/env python3
from setuptools import setup

setup(
    name='dryvr_plus_plus',
    version='0.1',
    description='DryVR++',
    author='MaybeShewill-CV',
    maintainer='Chiao Hsieh',
    maintainer_email='chsieh16@illinois.edu',
    license='Apache-2.0',
    packages=["dryvr_plus_plus"],
    python_requires='>=3.6',
    install_requires=[
        "numpy~=1.22.1",
        "scipy~=1.8.0",
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
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache License 2.0',
        'Programming Language :: Python :: 3.8',
    ]
)