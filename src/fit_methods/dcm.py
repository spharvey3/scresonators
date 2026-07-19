import numpy as np
import lmfit
from functools import partial

from .fit_method import FitMethod


class DCM(FitMethod):
    """Diameter Correction Method, with an optional nonlinear (Kerr / two-photon loss) model."""

    supports_nonlinear = True
    fc_index_offset = -1

    @staticmethod
    def func(f, Q, Qc, f0, phi):
        """DCM fit function."""
        return 1-(Q*np.exp(-1j*phi)/Qc)/(1+2j*Q*(f-f0)/f0)

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
        params['Qc_real'].vary = params['Qc'].vary or params['phi'].vary
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

    @staticmethod
    def qi_fano_band(params, b):
        """Fano-interference Qi uncertainty band for a hanger/notch DCM fit.

        Bounds the systematic error on Qi from coherent leakage of the signal
        with a background path (Rieger et al., arXiv:2209.03036). The measured
        signal is S21^Sigma = (1-b)*S21 + b*exp(i*phi_fano): the background path
        Moebius-transforms the resonance circle, tilting its centre off the real
        axis. The standard diameter-corrected Qi (1/Qi = 1/Q - cos(phi)/Qc) is
        exactly the paper's Qi,mid - the real-axis projection of the measured
        circle centre. With the leakage amplitude ``b`` measured but its phase
        unknown, the true centre lies anywhere on a circle of radius |R'|*b_tilde
        about the measured centre, where |R'| = Q/(2*Qc) (hanger factor of two)
        and b_tilde = b/(1-b). Because 1/Qi is linear in the centre's real part,
        the worst-case real displacement +/-|R'|*b_tilde maps to a symmetric band
        on the loss, Delta(1/Qi) = +/- b_tilde/Qc (the factor of two cancels).

        This is the b_tilde_min -> 0 limit of the paper's Eq. 12 (valid because we
        *measure* b rather than only bounding it from above). It is a HANGER/notch
        result; reflection geometry would give +/- 2*b_tilde/Qc.

        Args:
            params: an lmfit.Parameters from a DCM fit (uses 'Q', 'Qc', 'phi').
            b: measured background-leakage amplitude in [0, 1). ``None`` or <= 0
               collapses the band to the diameter-corrected Qi (perfect no-op).

        Returns:
            (Qi_min, Qi_mid, Qi_max). Qi_mid equals the standard DCM Qi. Qi_max is
            ``np.inf`` when the leakage ambiguity is large enough that the upper
            edge of the loss band reaches zero (the upper Qi bound is unbounded) -
            the regime, highlighted by the paper, where strong coupling plus a few
            percent leakage makes Qi un-upper-boundable.
        """
        Q = params['Q'].value
        Qc = params['Qc'].value
        phi = params['phi'].value
        inv_mid = 1.0/Q - np.cos(phi)/Qc
        if b is None or b <= 0:
            qi = 1.0/inv_mid
            return qi, qi, qi
        bt = b/(1.0 - b)                      # b-tilde
        inv_min = inv_mid + bt/Qc             # more loss  -> smaller (finite) Qi
        inv_max = inv_mid - bt/Qc             # less loss  -> larger Qi
        qi_max = 1.0/inv_max if inv_max > 0 else np.inf
        return 1.0/inv_min, 1.0/inv_mid, qi_max

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

        Unlike the linear model (which only depends on frequency ratios), this
        model mixes f0 with Knl_eff/gamma_nl_eff in Hz, so f and f0 MUST be in
        GHz. Fitter raises if the frequency data does not look like GHz.

        Args:
            f: frequency array (GHz)
            Q: total quality factor
            Qc: coupling quality factor
            f0: resonant frequency (GHz)
            phi: impedance mismatch angle
            Knl_eff: effective Kerr parameter = Knl * |ain|² (Hz)
            gamma_nl_eff: effective two-photon loss = gamma_nl * |ain|² (Hz)
            nonlinear: if False, returns the linear DCM function

        Returns:
            S21: complex scattering parameter
        """
        if not nonlinear or (Knl_eff == 0 and gamma_nl_eff == 0):
            return DCM.func(f, Q, Qc, f0, phi)

        # Calculate total loss rate from Q factor
        # Note: f0 is in GHz, so convert to Hz for consistent units with Knl_eff/gamma_nl_eff
        f0_hz = f0 * 1e9
        omega_r = 2 * np.pi * f0_hz
        kappa_total = omega_r / Q  # κi + κc in Hz

        # Normalized parameters (using effective values that already include ain_sq)
        # Knl_eff and gamma_nl_eff are in Hz, kappa_total is in Hz
        xi = Knl_eff / kappa_total  # normalized Kerr
        eta = gamma_nl_eff / kappa_total  # normalized two-photon loss

        S21 = np.zeros(len(f), dtype=complex)

        for i, freq in enumerate(f):
            # Normalized detuning (freq is in GHz, convert to Hz)
            delta = 2 * np.pi * (freq - f0) * 1e9
            delta_norm = delta / kappa_total

            # Solve for photon number
            n = DCM.solve_photon_number(delta_norm, xi, eta)

            # Nonlinear frequency shift and loss
            xi_n = xi * n  # ξn term
            eta_n = eta * n  # ηn term

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
            return DCM.func(f, Q, Qc, f0, phi)

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
        params = self.find_initial_guess(fdata, sdata)

        # Add effective nonlinear parameters
        # Knl_eff = Knl * |ain|² and gamma_nl_eff = gamma_nl * |ain|²
        # These are power-dependent and fitted directly from the data
        # To extract actual Knl and gamma_nl, fit at multiple powers and
        # plot (Knl_eff * n) vs n - the slope gives Knl
        params.add('Knl_eff', value=3e2, min=-1e7, max=1e7)  # Effective Kerr (Hz)
        params.add('gamma_nl_eff', value=3e2, min=0, max=1e8)  # Effective two-photon loss (Hz)

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
