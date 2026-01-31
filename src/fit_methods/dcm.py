import numpy as np
import lmfit
from functools import partial

from ..utils import *
from .fit_method import FitMethod
from scipy.ndimage import gaussian_filter

class DCM(FitMethod):
    def __init__(self):
        pass

    @staticmethod
    def func(f, Q, Qc, f0, phi):
        """DCM fit function."""
        return 1-(Q*np.exp(-1j*phi)/Qc)/(1+2j*Q*(f-f0)/f0)

    @staticmethod
    def fit_function(f, params):
        '''
        same as func but unpacks the fit parameters for you.
        '''
        Q = params['Q'].value
        Qc = params['Qc'].value
        f0 = params['f0'].value
        phi = params['phi'].value

        return 1-(Q*np.exp(-1j*phi)/Qc)/(1+2j*Q*(f-f0)/f0)

    
    def create_model(self):
        """Creates an lmfit Model using the static func method."""
        model = lmfit.Model(self.func, independent_vars=['f'])
        return model
    
    def simple_initial_guess(self, fdata: np.ndarray, sdata: np.ndarray):
        f_c = fdata[np.argmin(np.abs(sdata))]

        params = lmfit.Parameters()
        params.add('Q', value=1e6)
        params.add('Qc', value=5e5)
        params.add('f0', value=f_c, min=f_c * 0.9, max=f_c * 1.1)
        params.add('phi', value=0, min=-np.pi, max=np.pi)

        return params


    def find_initial_guess(self, fdata: np.ndarray, sdata: np.ndarray) -> lmfit.Parameters:
        
        """
        Finds the initial guess of the fitting parameters.
        
        In order to robustly estimate the linewidth we smooth the data to eliminate noise.
        Then we calculate |dS/df| to find the region in frequency space where the resonator response is changing rapidly.
        taking the derivative eliminates the need to consider phi for estimating the linewidth.
        Once we have |dS/df| we count up the segments in the frequency data where |dS/df| > cutoff
        This is implemented with the dot product, and that length in frequency space is estimated to be the linewidth.
        The linewidth & f0 determine Q, and the circle diameter is Q/Qc.
        
        Args:
            fdata: numpy array of the frequency data
            sdata: numpy array of the scattering data
            
        Returns:
            params: lmfit.Parameters object which stores the initial guesses of the fitting parameters
        """
        filtered_data = gaussian_filter(sdata, sigma = 3)#sigma may need to be changed for noisy data
        gradS = np.gradient(filtered_data, fdata)

        chiFunction = partitionFrequencyBand(fdata, gradS)
        x_c, y_c, r = find_circle(np.real(sdata), np.imag(sdata))#the circle diameter is 2*r = Q/Qc
        phase_guess = np.angle(x_c + 1j*y_c)
        #print(f'phi guess: {phase_guess}')

        # chiFunction = np.zeros(len(gradSmagnitude))
        # cutoff = 0.5*(np.min(gradSmagnitude)+np.max(gradSmagnitude))
        # for n in range(len(gradSmagnitude)):
        #     if gradSmagnitude[n] > cutoff:
        #         chiFunction[n] = 1#set to one if |dS/df| is above the cutoff at this point
        linewidth = np.dot(chiFunction[:-1], np.diff(fdata))


        gradSmagnitude = np.abs(gradS)
        f_c = fdata[np.argmax(gradSmagnitude)-1]#uncertainties can't be calculated when this guess is too good!!!
        Q_guess = 2*f_c/(linewidth) # This is getting a value 2x larger than it should be 
        Q_guess = Q_guess/2
        #print(f'Q_guess: {Q_guess}')
        Qc_guess = Q_guess/(2*r)

        # Create an lmfit.Parameters object to store initial guesses
        params = lmfit.Parameters()
        params.add('Q', value=Q_guess, min=0, max=1e8)
        params.add('Qc', value=Qc_guess, min=0, max=1e8)
        params.add('f0', value=f_c, min=f_c*0.9, max=f_c*1.1)
        params.add('phi', value=phase_guess, min=-np.pi, max=np.pi)

        return params

    def extractQi(self, params):
        """
        Calculates the fit value & stddev of Qi.
        """
        Q = params['Q'].value
        Qc = params['Qc'].value
        phi = params['phi'].value
        params.add('inverseQi', value =1/Q - np.cos(phi)/Qc)
        inverseQi = params['inverseQi'].value
        params.add('Qi', value = 1/inverseQi)
        params.add('Qc_real', value = Qc/np.cos(phi))
        params['Qc_real'].stderr = params['Qc'].stderr/np.cos(phi) if params['Qc'].stderr is not None else None
        #if you get an error that points here check that all your parameters varied during the fit, set verbose = True
        # Check if any of the standard errors are infinity
        if (
            params['Q'].stderr is None or
            params['Qc'].stderr is None or
            params['phi'].stderr is None or
            np.isinf(params['Q'].stderr) or
            np.isinf(params['Qc'].stderr) or
            np.isinf(params['phi'].stderr)
        ):
            params['inverseQi'].stderr = np.inf
        else:
            params['inverseQi'].stderr = np.sqrt(
            (params['Q'].stderr/Q**2)**2 +
            (np.sin(phi)*params['phi'].stderr/Qc)**2 +
            (np.cos(phi)*params['Qc'].stderr/Qc**2)**2
            )
        params['Qi'].stderr = params['inverseQi'].stderr/inverseQi**2
        return params

    #TODO: add function that converts standard errors to 95% confidence intervals

    @staticmethod
    def solve_photon_number(delta_norm, xi, eta):
        """
        Solve the normalized cubic equation (Eq. 7 from the paper) for photon number.

        The equation is:
        1/2 = n³(ξ² + η²/4) + 2n²(η/4 - ξΔ) + n(1/4 + Δ²)

        Rearranged to standard form: an³ + bn² + cn + d = 0

        Args:
            delta_norm: normalized frequency detuning Δ = (ω - ωr)/(κi + κc)
            xi: normalized Kerr parameter ξ = |ain|² * Knl / (κi + κc)
            eta: normalized two-photon loss η = |ain|² * γnl / (κi + κc)

        Returns:
            n: normalized photon number (real, positive solution)
        """
        # Coefficients for cubic: a*n³ + b*n² + c*n + d = 0
        a = xi**2 + eta**2 / 4
        b = 2 * (eta / 4 - xi * delta_norm)
        c = 0.25 + delta_norm**2
        d = -0.5

        if np.abs(a) < 1e-15:
            # Linear or quadratic case when nonlinearity is negligible
            if np.abs(b) < 1e-15:
                # Linear case: cn + d = 0
                return -d / c
            else:
                # Quadratic case: bn² + cn + d = 0
                discriminant = c**2 - 4*b*d
                if discriminant < 0:
                    return -d / c  # fallback
                n1 = (-c + np.sqrt(discriminant)) / (2*b)
                n2 = (-c - np.sqrt(discriminant)) / (2*b)
                # Return positive real solution
                for n in [n1, n2]:
                    if np.real(n) > 0:
                        return np.real(n)
                return np.abs(n1)

        # Solve cubic equation using numpy roots
        coeffs = [a, b, c, d]
        roots = np.roots(coeffs)

        # Find the physically meaningful solution (real and positive)
        real_positive_roots = []
        for root in roots:
            if np.abs(np.imag(root)) < 1e-10 and np.real(root) > 0:
                real_positive_roots.append(np.real(root))

        if len(real_positive_roots) == 0:
            # Fallback: return the root with smallest imaginary part
            idx = np.argmin(np.abs(np.imag(roots)))
            return np.abs(np.real(roots[idx]))
        elif len(real_positive_roots) == 1:
            return real_positive_roots[0]
        else:
            # Multiple solutions (bistability region) - return the smallest stable one
            return min(real_positive_roots)

    @staticmethod
    def func_nonlinear(f, Q, Qc, f0, phi, Knl_eff, gamma_nl_eff, nonlinear=False):
        """
        Nonlinear DCM fit function including Kerr effect and two-photon loss.

        This implements Eq. (8) from the supplementary material:
        S21 = A * exp(i*(ω*td + φ)) * [1 - (δc/(δc+δi)) * (1-iα) / (1 + ηn + 2i(Δ - ξn))]

        For simplicity, we set A=1, td=0, and α=0 (asymmetry factor).

        Args:
            f: frequency array
            Q: total quality factor
            Qc: coupling quality factor
            f0: resonant frequency
            phi: impedance mismatch angle
            Knl_eff: effective Kerr parameter = Knl * |ain|² (Hz)
            gamma_nl_eff: effective two-photon loss = gamma_nl * |ain|² (Hz)
            nonlinear: if False, returns the linear DCM function

        Returns:
            S21: complex scattering parameter
        """
        if not nonlinear or (Knl_eff == 0 and gamma_nl_eff == 0):
            # Return linear DCM function
            return 1 - (Q * np.exp(-1j * phi) / Qc) / (1 + 2j * Q * (f - f0) / f0)

        # Calculate total loss rate from Q factor
        # κi + κc = ωr/Q (total loss rate)
        omega_r = 2 * np.pi * f0
        kappa_total = omega_r / Q  # κi + κc

        # Normalized parameters (using effective values that already include ain_sq)
        xi = Knl_eff / kappa_total  # normalized Kerr
        eta = gamma_nl_eff / kappa_total  # normalized two-photon loss

        S21 = np.zeros(len(f), dtype=complex)

        for i, freq in enumerate(f):
            # Normalized detuning
            delta = 2 * np.pi * (freq - f0)
            delta_norm = delta / kappa_total

            # Solve for photon number
            n = DCM.solve_photon_number(delta_norm, xi, eta)

            # Nonlinear frequency shift and loss
            xi_n = Knl_eff * n / kappa_total  # ξn term
            eta_n = gamma_nl_eff * n / kappa_total  # ηn term

            # Nonlinear S21 (Eq. 8, simplified with A=1, td=0, α=0)
            denominator = 1 + eta_n + 2j * (delta_norm - xi_n)
            S21[i] = 1 - (Q * np.exp(-1j * phi) / Qc) / denominator

        return S21

    @staticmethod
    def fit_function_nonlinear(f, params, nonlinear=False):
        """
        Nonlinear fit function that unpacks the fit parameters.

        Args:
            f: frequency array
            params: lmfit.Parameters object
            nonlinear: if False, returns linear DCM function

        Returns:
            S21: complex scattering parameter
        """
        Q = params['Q'].value
        Qc = params['Qc'].value
        f0 = params['f0'].value
        phi = params['phi'].value

        if nonlinear and 'Knl_eff' in params:
            Knl_eff = params['Knl_eff'].value
            gamma_nl_eff = params['gamma_nl_eff'].value
            return DCM.func_nonlinear(f, Q, Qc, f0, phi, Knl_eff, gamma_nl_eff, nonlinear=True)
        else:
            return 1 - (Q * np.exp(-1j * phi) / Qc) / (1 + 2j * Q * (f - f0) / f0)

    def find_initial_guess_nonlinear(self, fdata: np.ndarray, sdata: np.ndarray) -> lmfit.Parameters:
        """
        Finds initial guess for nonlinear fitting parameters.

        Starts with the linear initial guess and adds nonlinear parameters.

        Args:
            fdata: numpy array of frequency data
            sdata: numpy array of scattering data

        Returns:
            params: lmfit.Parameters object with initial guesses including nonlinear params
        """
        # Get linear initial guess first
        params = self.find_initial_guess(self,fdata, sdata)

        # Add effective nonlinear parameters
        # Knl_eff = Knl * |ain|² and gamma_nl_eff = gamma_nl * |ain|²
        # These are power-dependent and fitted directly from the data
        # To extract actual Knl and gamma_nl, fit at multiple powers and
        # plot (Knl_eff * n) vs n - the slope gives Knl
        params.add('Knl_eff', value=1e2, min=0, max=1e5)  # Effective Kerr (Hz)
        params.add('gamma_nl_eff', value=1e2, min=0, max=1e5)  # Effective two-photon loss (Hz)

        return params

    def create_model_nonlinear(self, nonlinear=False):
        """
        Creates an lmfit Model for nonlinear fitting.

        Args:
            nonlinear: if True, creates nonlinear model; if False, creates linear model

        Returns:
            model: lmfit.Model object
        """
        if nonlinear:
            # Use partial to fix nonlinear=True so the nonlinear formula is actually used
            # This prevents lmfit from treating 'nonlinear' as a fittable parameter
            func_with_nonlinear = partial(self.func_nonlinear, nonlinear=True)
            # lmfit expects __name__ attribute on the function
            func_with_nonlinear.__name__ = 'func_nonlinear'
            model = lmfit.Model(func_with_nonlinear, independent_vars=['f'])
        else:
            model = lmfit.Model(self.func, independent_vars=['f'])
        return model
