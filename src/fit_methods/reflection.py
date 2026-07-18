import lmfit

from .fit_method import FitMethod


class ReflectionMode(FitMethod):
    """Reflection mode fit."""

    # uncertainties can't be calculated when the f0 guess is too good!!!
    fc_index_offset = 3

    @staticmethod
    def func(f, Q, Qc, f0):
        """Reflection mode fit function."""
        return 1-(2*Q/Qc)/(1-2j*Q*(f-f0)/f0)

    def initial_params(self, f_c, linewidth, r, phase_guess) -> lmfit.Parameters:
        Q_guess = 2*f_c/linewidth
        Qc_guess = Q_guess/r  # the circle diameter is 2*r = 2*Q/Qc

        params = lmfit.Parameters()
        params.add('Q', value=Q_guess)
        params.add('Qc', value=Qc_guess)
        params.add('f0', value=f_c, min=f_c*0.9, max=f_c*1.1)

        return params
