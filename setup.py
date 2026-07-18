"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages

# Get the long description from the README file
long_description = open('README.md').read()
# Get the version number from the VERSION file
version_number = open('VERSION').read().strip()

setup(
    name="scresonators",
    version=version_number,
    author="CU Boulder Cryogenic Quantum Testbed",
    url="https://github.com/Boulder-Cryogenic-Quantum-Testbed/scresonators",
    license = 'MIT',
    description="Python library for measuring and fitting superconducting resonator data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="scresonators, superconducting resonators, resonator fit",
    package_dir={"scresonators": "src", "scresonators.plotting": "plotting"},
    packages=["scresonators", "scresonators.fit_methods", "scresonators.plotting"],
    python_requires = ">=3.9",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.6",
        "lmfit>=1.0.2",
        "matplotlib>=3.4",
        "scikit-rf>=0.17",
        "emcee>=3.0",
    ],
    classifiers = [
        'Development Status :: Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Visualization'
    ],
    project_urls={
        "Bug Reports": "https://github.com/Boulder-Cryogenic-Quantum-Testbed/scresonators/issues",
        "Testbed": "https://www.nist.gov/programs-projects/boulder-cryogenic-quantum-testbed",
        "Contributors": "https://github.com/Boulder-Cryogenic-Quantum-Testbed/scresonators/blob/master/contrib/HELLO.md",
    }
)
