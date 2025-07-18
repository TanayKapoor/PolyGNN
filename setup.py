#!/usr/bin/env python
"""
Setup script for PolyGNN - Polymer Graph Neural Network package
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="polygnn",
    version="0.1.0",
    description="Graph Neural Networks for Polymer Property Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="PolyGNN Development Team",
    author_email="",
    url="https://github.com/user/polygnn",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=22.0',
            'flake8>=4.0',
            'isort>=5.0',
        ],
        'notebook': [
            'jupyter>=1.0',
            'matplotlib>=3.0',
            'seaborn>=0.11',
            'plotly>=5.0',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    keywords="polymer, graph neural networks, materials science, machine learning, pytorch geometric",
    project_urls={
        "Bug Reports": "https://github.com/user/polygnn/issues",
        "Source": "https://github.com/user/polygnn",
        "Documentation": "https://github.com/user/polygnn/docs",
    },
    entry_points={
        'console_scripts': [
            'polygnn-setup=dataset_setup:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 