"""
cNVAE-PSG: Cross-Modal PSG-to-ECG Reconstruction

A research package for investigating cross-modal reconstruction from 
polysomnography (PSG) signals to electrocardiography (ECG) using 
conditional Neural Vector Quantized Variational Autoencoders.

This package is part of the T-CAIREM research initiative at the 
University of Toronto in collaboration with Sunnybrook Health Sciences Centre.
"""

__version__ = "0.1.0"
__author__ = "T-CAIREM Research Team"
__email__ = "tcairem@utoronto.ca"
__license__ = "Research and Academic Use License"

# Import main components
try:
    from .models import cNVAE
    from .data import PSGDataset, PSGDataLoader
    from .utils import load_config, setup_logging
except ImportError:
    # Handle case where dependencies aren't installed yet
    pass

__all__ = [
    "cNVAE",
    "PSGDataset", 
    "PSGDataLoader",
    "load_config",
    "setup_logging",
]
