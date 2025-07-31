"""
Scresonators: A namespace package for measuring and fitting superconducting resonator data.

This package provides tools for:
- Fitting resonator data using various methods (DCM, INV, CPZM, etc.)
- Measuring resonator properties with VNA and other instruments
- Plotting and visualizing resonator data

Subpackages:
- scresonators.fit_resonator: Resonator fitting and analysis tools
- scresonators.measurement: Measurement and data acquisition tools
- scresonators.plotting: Plotting and visualization utilities
"""

__version__ = "0.1.0"
__author__ = "Boulder Cryogenic Quantum Testbed"

# This file is intentionally minimal to make scresonators a namespace package
# Import key classes and functions for convenience
try:
    from .fit_resonator.resonator import Resonator, FitMethod
    from .fit_resonator import cavity_functions
except ImportError:
    # Handle case where submodules might not be available
    pass

__all__ = [
    "Resonator",
    "FitMethod",
    "cavity_functions",
]
