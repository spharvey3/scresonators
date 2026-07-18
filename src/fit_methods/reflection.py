from .fit_method import FitMethod


class ReflectionMode(FitMethod):
    """Reflection mode fit."""

    fc_index_offset = 3
    phase_orientation = -1
    Q_guess_factor = 2
    diameter_factor = 2  # the circle diameter is 2*r = 2*Q/Qc

    @staticmethod
    def func(f, Q, Qc, f0):
        """Reflection mode fit function."""
        return 1-(2*Q/Qc)/(1-2j*Q*(f-f0)/f0)
