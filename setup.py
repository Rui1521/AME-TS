#!/usr/bin/env python3
"""
Setup script for AME-TS.

AME-TS is a time series foundation model with structure-aware routing.
Interpretable structural descriptors form a prior over the experts of a
Mixture-of-Experts architecture and anchor what each expert specializes in.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README next to this file.
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements.txt, with a fallback list if it is missing.
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split('\n')
    requirements = [r.strip() for r in requirements if r.strip() and not r.startswith('#')]
else:
    requirements = [
        "torch>=2.0.0",
        "lightning>=2.0.0",
        "hydra-core>=1.3.0",
        "einops>=0.6.0",
        "jaxtyping>=0.2.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "gluonts>=0.13.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scipy>=1.10.0",
        "statsmodels>=0.14.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ]

setup(
    name="amets",
    version="0.0.1.dev0",
    author="AME-TS Authors",
    description="Structure-aware expert routing for time series foundation models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Rui1521/AME-TS",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "plotly>=5.14.0",
        ],
    },
    # No console_scripts: training is Hydra-driven and evaluation is
    # argparse-driven, both invoked as scripts rather than installed commands.
    #   python scripts/train.py --config-name ame_base
    #   python scripts/evaluate.py --checkpoint <path.ckpt> ...
    include_package_data=True,
    zip_safe=False,
)
