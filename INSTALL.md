# Installation Guide for Scresonators

This guide provides instructions for installing the scresonators package and its dependencies.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation Methods

### Method 1: Development Installation (Recommended)

For development or if you want to modify the code:

```bash
# Clone the repository (if not already done)
git clone https://github.com/Boulder-Cryogenic-Quantum-Testbed/scresonators.git
cd scresonators

# Install in development mode
pip install -e .
```

This installs the package in "editable" mode, so changes to the source code are immediately reflected without reinstalling.

### Method 2: Standard Installation

For regular use:

```bash
# From the package directory
pip install .
```

### Method 3: Install with Optional Dependencies

To install with additional optional dependencies:

```bash
# Install with development tools
pip install -e ".[dev]"

# Install with documentation tools
pip install -e ".[docs]"

# Install with Jupyter notebook support
pip install -e ".[jupyter]"

# Install with all optional dependencies
pip install -e ".[dev,docs,jupyter]"
```

## Dependency Installation

### Core Dependencies

The following packages will be automatically installed:

- `numpy>=1.20.0` - Numerical computing
- `scipy>=1.7.0` - Scientific computing
- `matplotlib>=3.3.0` - Plotting
- `lmfit>=1.0.0` - Non-linear fitting
- `attrs>=21.0.0` - Data classes
- `inflect>=5.0.0` - Text processing
- `pyvisa>=1.11.0` - Instrument communication
- `pyvisa-py>=0.5.0` - Pure Python VISA backend

### Manual Dependency Installation

If you prefer to install dependencies manually:

```bash
pip install -r requirements.txt
```

## Verification

To verify the installation works correctly:

```python
import scresonators
from scresonators.fit_resonator import Resonator, FitMethod
from scresonators.fit_resonator import cavity_functions

print(f"Scresonators version: {scresonators.__version__}")
print("Installation successful!")
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed correctly
2. **VISA errors**: For instrument communication, you may need to install additional VISA drivers
3. **Matplotlib backend issues**: On some systems, you may need to configure the matplotlib backend

### VISA Backend Setup

For instrument communication, you may need to install a VISA backend:

```bash
# For National Instruments VISA (recommended for hardware)
# Download and install from: https://www.ni.com/en-us/support/downloads/drivers/download.ni-visa.html

# Or use the pure Python backend (included with pyvisa-py)
# This is automatically installed with the package
```

### Development Setup

For development work:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests (if available)
pytest

# Format code
black .

# Check code style
flake8 .
```

## Usage Examples

See the README.md file for usage examples and the `examples/` directory for more detailed examples.

## Support

For issues and questions:
- Check the GitHub issues: https://github.com/Boulder-Cryogenic-Quantum-Testbed/scresonators/issues
- Review the documentation and examples
