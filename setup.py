#!/usr/bin/env python
"""
Setup script for cNVAE-PSG: Cross-Modal PSG-to-ECG Reconstruction
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Read version from __init__.py
def get_version():
    init_file = Path(__file__).parent / "src" / "cnvae_psg" / "__init__.py"
    if init_file.exists():
        with open(init_file, "r") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

# Read README for long description
def get_long_description():
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        with open(readme_file, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Read requirements
def get_requirements():
    req_file = Path(__file__).parent / "requirements.txt"
    if req_file.exists():
        with open(req_file, "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

setup(
    name="cnvae-psg",
    version=get_version(),
    author="T-CAIREM Research Team",
    author_email="tcairem@utoronto.ca",
    description="Cross-Modal PSG-to-ECG Reconstruction using Conditional Neural Vector Quantized VAE",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/T-CAIREM/cNVAE-PSG",
    
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    python_requires=">=3.8",
    install_requires=get_requirements(),
    
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0", 
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "gpu": [
            "nvidia-ml-py3>=7.352.0",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "cnvae-train=cnvae_psg.train:main",
            "cnvae-evaluate=cnvae_psg.evaluate:main",
            "cnvae-preprocess=cnvae_psg.data.preprocess:main",
        ],
    },
    
    include_package_data=True,
    package_data={
        "cnvae_psg": ["configs/*.yaml", "configs/*.json"],
    },
    
    project_urls={
        "Bug Reports": "https://github.com/T-CAIREM/cNVAE-PSG/issues",
        "Source": "https://github.com/T-CAIREM/cNVAE-PSG",
        "Documentation": "https://github.com/T-CAIREM/cNVAE-PSG/docs",
        "Dataset": "https://doi.org/10.57764/tvsv-y363",
        "T-CAIREM": "https://tcairem.utoronto.ca/",
    },
    
    keywords=[
        "deep learning", 
        "sleep medicine", 
        "signal processing", 
        "ECG reconstruction",
        "polysomnography",
        "variational autoencoder",
        "cross-modal learning",
        "healthcare AI"
    ],
    
    zip_safe=False,
)
