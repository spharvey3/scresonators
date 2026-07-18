import numpy as np
import lmfit

from .fit_method import FitMethod


class LCE(FitMethod):
    """Linear Combination of Eigenvalues model."""

    # uncertainties can't be calculated when the f0 guess is too good!!!
    fc_index_offset = 3

    @staticmethod
    def func(f, Q, Qc, f0, phi):
        """LCE fit function."""
        #TODO: separate preprocessing methods by fit method, then add prefactor to func
        return 1-((Q/Qc)*(1-1j*np.tan(phi)))/(1-2j*Q*(f-f0)/f0)

    def initial_params(self, f_c, linewidth, r, phase_guess) -> lmfit.Parameters:
        Q_guess = f_c/linewidth
        Qc_guess = Q_guess/(2*r)  # the circle diameter is 2*r = Q/Qc

        params = lmfit.Parameters()
        params.add('Q', value=Q_guess)
        params.add('Qc', value=Qc_guess)
        params.add('f0', value=f_c, min=f_c*0.9, max=f_c*1.1)
        params.add('phi', value=0, min=-np.pi, max=np.pi)

        return params
