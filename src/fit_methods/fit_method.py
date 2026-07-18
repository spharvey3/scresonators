from abc import ABC, abstractmethod
import inspect

import numpy as np
import lmfit
from scipy.ndimage import gaussian_filter

from ..utils import find_circle, partitionFrequencyBand


class FitMethod(ABC):
    """
    Base class for resonator fit methods.

    Subclasses supply the physics: the S21 model (`func`) and the mapping from
    the shared resonance estimate to initial fit parameters (`initial_params`).
    Everything else -- model construction, parameter unpacking, the linewidth
    estimation heuristics, and Qi extraction -- lives here.
    """

    #: whether the method provides a nonlinear model (see DCM)
    supports_nonlinear = False
    #: index offset applied to argmax|dS/df| when picking the f0 guess;
    #: a slightly imperfect guess keeps lmfit's uncertainty estimation happy
    fc_index_offset = 0

    @staticmethod
    @abstractmethod
    def func(f, *params) -> np.ndarray:
        """The model S21 as a function of frequency and the fit parameters."""

    @abstractmethod
    def initial_params(self, f_c, linewidth, r, phase_guess) -> lmfit.Parameters:
        """
        Builds the initial lmfit.Parameters from the shared resonance estimate.

        Args:
            f_c: estimated resonant frequency
            linewidth: estimated linewidth in frequency units
            r: radius of the best-fit circle of the scattering data
            phase_guess: angle of the circle center in the complex plane

        Returns:
            params: lmfit.Parameters with initial guesses for this method
        """

    def create_model(self) -> lmfit.Model:
        """Creates an lmfit Model using the method's func."""
        return lmfit.Model(self.func, independent_vars=['f'])

    def fit_function(self, f, params):
        """Same as func but unpacks the fit parameters for you."""
        names = list(inspect.signature(self.func).parameters)[1:]
        return self.func(f, *[params[name].value for name in names])

    def estimate_resonance(self, fdata: np.ndarray, sdata: np.ndarray):
        """
        Estimates the resonant frequency, linewidth, and circle geometry.

        In order to robustly estimate the linewidth we smooth the data to eliminate noise.
        Then we calculate |dS/df| to find the region in frequency space where the resonator
        response is changing rapidly. Taking the derivative eliminates the need to consider
        phi for estimating the linewidth. Once we have |dS/df| we count up the segments in
        the frequency data where |dS/df| > cutoff. This is implemented with the dot product,
        and that length in frequency space is estimated to be the linewidth.

        Args:
            fdata: numpy array of the frequency data
            sdata: numpy array of the scattering data

        Returns:
            tuple: (f_c, linewidth, r, phase_guess)
        """
        filtered_data = gaussian_filter(sdata, sigma=3)  # sigma may need to be changed for noisy data
        gradS = np.gradient(filtered_data, fdata)

        chiFunction = partitionFrequencyBand(fdata, gradS)
        linewidth = np.dot(chiFunction[:-1], np.diff(fdata))

        x_c, y_c, r = find_circle(np.real(sdata), np.imag(sdata))
        phase_guess = np.angle(x_c + 1j * y_c)

        f_c = fdata[np.argmax(np.abs(gradS)) + self.fc_index_offset]
        return f_c, linewidth, r, phase_guess

    def find_initial_guess(self, fdata: np.ndarray, sdata: np.ndarray) -> lmfit.Parameters:
        """
        Finds the initial guess of the fitting parameters.

        Args:
            fdata: numpy array of the frequency data
            sdata: numpy array of the scattering data

        Returns:
            params: lmfit.Parameters object which stores the initial guesses of the fitting parameters
        """
        return self.initial_params(*self.estimate_resonance(fdata, sdata))

    def extractQi(self, params: lmfit.Parameters) -> lmfit.Parameters:
        """
        Calculates the fit value & stddev of Qi from 1/Qi = 1/Q - 1/Qc.
        """
        Q = params['Q'].value
        Qc = params['Qc'].value
        params.add('inverseQi', value=1/Q - 1/Qc)
        inverseQi = params['inverseQi'].value
        params.add('Qi', value=1/inverseQi)
        #if you get an error that points here check that all your parameters varied during the fit, set verbose = True
        if (
            params['Q'].stderr is None or
            params['Qc'].stderr is None or
            np.isinf(params['Q'].stderr) or
            np.isinf(params['Qc'].stderr)
        ):
            params['inverseQi'].stderr = np.inf
        else:
            params['inverseQi'].stderr = np.sqrt(
                (params['Q'].stderr/Q**2)**2 +
                (params['Qc'].stderr/Qc**2)**2
            )
        params['Qi'].stderr = params['inverseQi'].stderr/inverseQi**2
        return params

    def find_initial_guess_nonlinear(self, fdata: np.ndarray, sdata: np.ndarray) -> lmfit.Parameters:
        raise NotImplementedError(f"{type(self).__name__} has no nonlinear model")

    def create_model_nonlinear(self, nonlinear=False) -> lmfit.Model:
        raise NotImplementedError(f"{type(self).__name__} has no nonlinear model")
