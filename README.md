scresonators: fitting superconducting resonator data using Python
===========================================

[![CodeFactor](https://www.codefactor.io/repository/github/boulder-cryogenic-quantum-testbed/scresonators/badge/master)](https://www.codefactor.io/repository/github/boulder-cryogenic-quantum-testbed/scresonators/overview/master)
![Tests](https://github.com/boulder-cryogenic-quantum-testbed/scresonators/actions/workflows/tests.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

scresonators fits complex S21 data for hanger-mode superconducting resonators. Supported fit methods:

- **DCM** — Diameter Correction Method, with an optional nonlinear (Kerr / two-photon loss) model
- **LCE** — Linear Combination of Eigenvalues
- **ReflectionMode** — reflection-geometry resonators

Preprocessing (electrical-delay removal, linear-baseline removal, and circle anchoring of the off-resonant point) is applied automatically before fitting.

Installation
-------------------------

We ***strongly*** recommend using [virtual environments](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) for managing your dependencies.

Install from a clone of this repository:

```
pip install .
```

Usage
-------------------------

Data conventions: frequencies in **GHz**, scattering data as a 1-D **complex** array (this removes ambiguity between linear/dB amplitudes and degree/radian phases).

```python
from scresonators import Resonator, DCM

res = Resonator()
res.load_data(fdata=freqs, sdata=s21)   # freqs in GHz, s21 complex
res.set_fitting_strategy(DCM())
result = res.fit(verbose=True)

print(result['Qi'].value, result['Qi'].stderr)
```

Touchstone files and `skrf` networks are supported directly:

```python
res.load_data_from_touchstone('my_resonator.s2p', port=(1, 0))  # fit S21
```

For a complete example with plotting, see [examples/example_fit_dcm.py](examples/example_fit_dcm.py).

Documentation
-------------

For scresonators documentation visit our [readthedocs](https://boulder-cryogenic-quantum-testbed.github.io/scresonators-docs/) (documentation has not yet been checked for outdated information).

The documentation source code is located at https://github.com/Boulder-Cryogenic-Quantum-Testbed/scresonators-docs

Citation
--------
If you use scresonators to fit data or generate plots used in a publication you can cite it as follows:

```latex
@misc{scresonators,
  title        = {{boulder-cryogenic-quantum-testbed/scresonators}},
  year         = {2023},
  howpublished = {\url{https://github.com/Boulder-Cryogenic-Quantum-Testbed/scresonators}}
}
```

Contribution
----------
We invite you to contribute to the continued development of scresonators by forking this repository and sending pull requests.

Even if you do not contribute to the code yourself, please support us by alerting us to problems you encounter or improvements we can make by making a new issue [here](https://github.com/Boulder-Cryogenic-Quantum-Testbed/scresonators/issues).

All contributions are expected to be consistent with [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/).

License
-------
scresonators uses the MIT license. Details are described in [the LICENSE file](https://github.com/Boulder-Cryogenic-Quantum-Testbed/scresonators/blob/master/LICENSE), but restrictions are very loose so feel free to use and modify our code to meet your requirements.

References
-------
1. DCM: M. S. Khalil, M. J. A. Stoutimore, F. C. Wellstood, and K. D. Osborn    Journal of Applied Physics 111, 054510 (2012); doi: 10.1063/1.3692073
2. INV: A. Megrant et al.     APPLIED PHYSICS LETTERS 100, 113510 (2012)
3. CPZM: Chunqing Deng, Martin Otto, and Adrian Lupascu University of Waterloo, Waterloo, Ontario N2L 3G1, Canada (9 August 2013)
4. PHI: J. Gao, "The Physics of Superconducting Microwave Resonators" Cal-tech Ph.D. thesis, May 2008
