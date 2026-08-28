"""
Neural Network Engine - High-performance pure-Rust deep learning engine with Python bindings.
"""

from .neural_network_engine import *

__doc__ = neural_network_engine.__doc__
if hasattr(neural_network_engine, "__all__"):
    __all__ = neural_network_engine.__all__
