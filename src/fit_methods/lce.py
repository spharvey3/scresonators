import numpy as np

from .fit_method import FitMethod


class LCE(FitMethod):
    """Linear Combination of Eigenvalues model."""

    fc_index_offset = 3
    phase_orientation = -1

    @staticmethod
    def func(f, Q, Qc, f0, phi):
        """LCE fit function."""
        #TODO: separate preprocessing methods by fit method, then add prefactor to func
        return 1-((Q/Qc)*(1-1j*np.tan(phi)))/(1-2j*Q*(f-f0)/f0)

    def phi_guess(self, phase_guess):
        return 0

    # NOTE: LCE inherits the base extractQi (1/Qi = 1/Q - 1/Qc) with no phi
    # correction, unlike DCM which uses cos(phi)/Qc. If phi here represents an
    # impedance mismatch as in DCM, Qi is biased -- needs a physics decision.
