import numpy as np
import lmfit
from scipy.ndimage import gaussian_filter
from scipy.stats import linregress
from scipy.interpolate import interp1d

from .utils import remove_delay, find_circle, partitionFrequencyBand


def _within(value, lo, hi, fallback):
    """Return value clipped into [lo, hi], or fallback if value is not finite.

    lmfit clips a finite start into its bounds by itself, but a NaN start
    passes through and scipy's least_squares then raises 'Initial guess is
    outside of provided bounds' (NaN fails every comparison, even against
    infinite bounds). A bad linewidth estimate can give such a NaN, or an
    infinite Q that would land exactly on the bound.
    """
    if not np.isfinite(value):
        value = fallback
    return float(np.clip(value, lo, hi))


def _principal_angle(angle):
    """Wrap an angle to (-pi, pi]."""
    return float(np.pi - np.mod(np.pi - angle, 2*np.pi))


class Preprocessor:
    """
    Prepares raw scattering data for fitting: background removal, electrical
    delay removal, and rotation/scaling of the off-resonant point to an anchor.

    Holds the preprocessing state (delay, theta_0, phi, off-resonant point)
    so it can be inspected after a fit or reused for a theta refinement pass.
    """

    #: fewest points a side of the resonance must keep outside the linewidth
    #: for initial_guess_delay to trust its phase slope; below this it uses the
    #: outer quarter of the band on that side instead
    min_delay_points = 3
    #: bounds on Ql in the delay fit
    Ql_bounds = (0, 1e8)

    def __init__(self, fit_method, remove_delay=True, fit_delay=True, delay_guess=None,
                 databg=None, linear_baseline=False, normalize=4, anchor_circle=True,
                 fit_theta=False, off_res_point=None, plot_diagnostics=False, fname=''):
        """
        Args:
            fit_method: FitMethod instance, used to estimate Q/f0 for the delay fit
            remove_delay: whether to remove electrical delay from the data
            fit_delay: whether to fit the delay (otherwise delay_guess is used directly)
            delay_guess: initial guess for the delay; None lets the preprocessor estimate it
            databg: optional background data object with freqs/linear_amps/phases attributes
            linear_baseline: whether to remove a linear baseline from phase and magnitude
            normalize: number of endpoints used for the linear baseline fit
            anchor_circle: whether to rotate/scale the off-resonant point to 1+0j
            fit_theta: whether find_delay refines theta_0 around a previous pass
            off_res_point: optionally prescribe the off-resonant point
            plot_diagnostics: save diagnostic plots of the delay and circle fits
            fname: filename prefix for diagnostic plots
        """
        self.fit_method = fit_method
        self.remove_elec_delay = remove_delay
        self.fit_delay = fit_delay
        self.delay_guess = delay_guess
        self.databg = databg
        self.linear_baseline = linear_baseline
        self.normalize = normalize
        self.anchor_circle = anchor_circle
        self.fit_theta = fit_theta
        self.off_res_point = off_res_point
        self.plot_diagnostics = plot_diagnostics
        self.fname = fname

        self.delay = None
        self.theta_0 = None
        self.phi = None
        self.Ql_guess = None
        self.fr_guess = None
        self.delay_chi = 0
        self.sdata_predelay = None

    def run(self, fdata: np.ndarray, sdata: np.ndarray, verbose=False) -> np.ndarray:
        """
        Runs the full preprocessing chain: background removal, linear baseline
        removal, delay removal, and circle anchoring.
        """
        if self.databg is not None:
            #this feature is untested
            sdata = self.remove_background(fdata, sdata)
        if self.linear_baseline:
            sdata, _, _, _, _ = self.remove_linear_baseline(fdata, sdata, self.normalize)

        # This removes electrical delay, creating parameters delay and theta_0
        if self.remove_elec_delay:
            self.sdata_predelay = sdata
            if self.fit_delay:
                self.delay = self.find_delay(fdata, sdata, verbose=verbose)
            else:
                self.delay = self.delay_guess if self.delay_guess is not None else 0
            sdata = remove_delay(fdata, sdata, self.delay)

        # Rotate and scale the off-resonant point to a prescribed anchor point
        if self.anchor_circle:
            sdata = self.anchor_to_point(fdata, sdata, verbose=verbose)
        return sdata

    def refine_with_theta(self, fdata: np.ndarray, Ql: float, fr: float, verbose=False) -> np.ndarray:
        """
        Second preprocessing pass: refits the delay with Q and f0 fixed to the
        values from a first fit, allowing theta_0 to refine, then re-anchors.
        """
        self.fr_guess = fr
        self.Ql_guess = Ql
        sdata = self.sdata_predelay
        self.delay = self.find_delay(fdata, sdata, verbose=verbose)
        self.off_res_point = None
        sdata = remove_delay(fdata, sdata, self.delay)
        return self.anchor_to_point(fdata, sdata, verbose=verbose)

    def remove_linear_baseline(self, xdata: np.ndarray, ydata: np.ndarray, normalize: int):
        """
        Preprocesses S21 data linearly. Removes cable delay and normalizes
        phase/magnitude of S21 by linear fit of a specified number of endpoints.

        Args:
            xdata (np.ndarray): The frequency data.
            ydata (np.ndarray): The complex S21 data to preprocess.
            normalize (int): Number of endpoints to use for normalization.

        Returns:
            tuple: Preprocessed S21 data, phase slope, phase intercept,
                   magnitude slope, and magnitude intercept.
        """
        if normalize * 2 > len(ydata):
            raise ValueError(
                "Not enough points to normalize. Please decrease the 'normalize' value or include more data points near resonance.")

        # Unwrap phase for linear preprocessing
        phase = np.unwrap(np.angle(ydata))

        # Normalize phase using linear fit
        slope, intercept, _, _, _ = linregress(
            np.append(xdata[:normalize], xdata[-normalize:]),
            np.append(phase[:normalize], phase[-normalize:])
        )

        # Adjust phase to remove cable delay and rotate off-resonant point to (1, 0i)
        adjusted_phase = phase - (slope * xdata + intercept)

        # Normalize magnitude using linear fit
        y_db = 20 * np.log10(np.abs(ydata))
        mag_slope, mag_intercept, _, _, _ = linregress(
            np.append(xdata[:normalize], xdata[-normalize:]),
            np.append(y_db[:normalize], y_db[-normalize:])
        )
        adjusted_magnitude = 10 ** ((y_db - (mag_slope * xdata + mag_intercept)) / 20)

        preprocessed_data = adjusted_magnitude * np.exp(1j * adjusted_phase)

        return preprocessed_data, slope, intercept, mag_slope, mag_intercept

    def remove_background(self, fdata: np.ndarray, sdata: np.ndarray):
        """
        Removes background signal by interpolating and adjusting amplitude and phase,
        using stored background data.

        Untested

        Args:
            fdata (np.ndarray): Frequency points of the measured data.
            sdata (np.ndarray): Complex S21 data to be corrected.

        Returns:
            np.ndarray: Corrected complex S21 data with background removed.
        """
        if self.databg is None:
            raise ValueError("Background data ('databg') not provided.")

        linear_amps = np.abs(sdata)
        phases = np.angle(sdata)

        # Create interpolation functions for background amplitude and phase
        fmag = interp1d(self.databg.freqs, self.databg.linear_amps, kind='cubic', fill_value="extrapolate")
        fang = interp1d(self.databg.freqs, self.databg.phases, kind='cubic', fill_value="extrapolate")

        # Correct measured data using the background interpolated at the measurement frequencies
        linear_amps_corrected = linear_amps / fmag(fdata)
        phases_corrected = phases - fang(fdata)

        # Return corrected data as complex S21 values
        return linear_amps_corrected * np.exp(1j * phases_corrected)

    def find_delay(self, fdata: np.ndarray, sdata: np.ndarray, verbose=False):
        """
        Modified version of the circle fit method described in Probst.

        Adjusts electrical delay such that the offset phase most closely fits an arctangent.

        Args:
            fdata: numpy array of the frequency data
            sdata: numpy array of the scattering data
            verbose: print the delay fit results

        Returns:
            delay: float representing the electrical delay
        """
        if self.delay_guess is None:
            delay_guess = self.initial_guess_delay(fdata, sdata)
        else:
            delay_guess = self.delay_guess
        if not np.isfinite(delay_guess):
            delay_guess = 0.0

        if self.Ql_guess is None or self.fr_guess is None:
            # Estimate Q and f0 with the delay guess removed: the delay adds a
            # constant 2*pi*delay*|S| to |dS/df|, which for a shallow dip swamps
            # the resonance and inflates the linewidth estimate.
            guess_params = self.fit_method.find_initial_guess(fdata, remove_delay(fdata, sdata, delay_guess))

        params = lmfit.Parameters()
        if self.Ql_guess is not None:
            params.add(name = 'Ql', value = self.Ql_guess, min=0.9*self.Ql_guess, max=1.1*self.Ql_guess)
            params['Ql'].vary = False
        if self.fr_guess is not None:
            params.add(name = 'fr', value = self.fr_guess, min=0.99*self.fr_guess, max=1.01*self.fr_guess)
            params['fr'].vary = False
        if self.Ql_guess is None or self.fr_guess is None:
            self._add_free_Ql_fr(params, fdata, guess_params)

        if self.fit_theta and self.theta_0 is not None:
            params.add(name = 'delay', value = self.delay, vary=False)
            params.add(name = 'theta_0', value = self.theta_0, min=self.theta_0-0.05, max=self.theta_0+0.05)
        else:
            # delay is a time (ns for GHz data), not an angle -- leave it unbounded
            params.add(name = 'delay', value = delay_guess)
            # theta_0 is left unbounded. remove_delay rotates the whole circle by
            # 2*pi*delay*f, so near fr a delay change d_tau moves theta_0 by
            # 2*pi*fr*d_tau (31 rad/ns at 5 GHz): the two are nearly degenerate,
            # and any finite bound on theta_0 caps how far the delay can move.
            # A shallow dip, whose delay guess is the least accurate, then pins
            # theta_0 at the bound and biases delay and Q. arctan_deviation
            # already snaps the data to the model's 2*pi branch, so an unbounded
            # theta_0 costs nothing; it is wrapped to (-pi, pi] afterwards.
            params.add(name = 'theta_0', value = 0)

        min_result = self._fit_offset_phase(params, fdata, sdata)
        params = min_result.params
        electrical_delay = params['delay'].value
        self.Ql_guess = params['Ql'].value
        self.fr_guess = params['fr'].value
        self.theta_0 = _principal_angle(params['theta_0'].value)
        if verbose:
            print(f'Fit delay. Chi sq: {min_result.redchi:.3g}, Q: {self.Ql_guess:.0f}, fr: {params["fr"].value:.4f}, theta_0: {self.theta_0:.4f}, delay: {electrical_delay:.4f}')
        self.delay_chi = min_result.redchi

        if self.plot_diagnostics:
            self._plot_delay_fit(fdata, sdata, params, electrical_delay)

        return electrical_delay

    def _add_free_Ql_fr(self, params, fdata, guess_params):
        """Add whichever of Ql and fr params lacks, free, started from guess_params."""
        f_lo, f_hi = np.min(fdata), np.max(fdata)
        if 'Ql' not in params:
            # A linewidth estimate of zero (or a dip lost in the noise) gives an
            # infinite or NaN Q; fall back to a Q whose linewidth is a tenth of
            # the span.
            Ql_fallback = 10 * 0.5 * (f_lo + f_hi) / (f_hi - f_lo)
            params.add(name = 'Ql', value = _within(guess_params['Q'].value, *self.Ql_bounds, Ql_fallback),
                       min=self.Ql_bounds[0], max=self.Ql_bounds[1])
        if 'fr' not in params:
            # the resonance lies inside the measured band; bounding fr there
            # (rather than at a fixed 1e10) also keeps data taken in Hz in bounds
            params.add(name = 'fr', value = _within(guess_params['f0'].value, f_lo, f_hi, 0.5*(f_lo + f_hi)),
                       min=f_lo, max=f_hi)

    def _fit_offset_phase(self, params, fdata, sdata, rounds=10):
        """Minimize arctan_deviation over params; returns the last lmfit MinimizerResult."""
        #running the minimizer several times also improves results
        for _ in range(rounds):
            min_result = lmfit.minimize(fcn=self.arctan_deviation, params=params, args=(fdata, sdata),
                                        method='least_squares', max_nfev = 10000000)
            params = min_result.params
            #TODO: we should have a condition on the residuals to break this loop early
        return min_result

    def _plot_delay_fit(self, fdata, sdata, params, electrical_delay):
        """Plot the sloped arctan as a verification step."""
        import matplotlib.pyplot as plt
        sdata_new = remove_delay(fdata, sdata, electrical_delay)
        xc, yc, r = find_circle(np.real(sdata_new), np.imag(sdata_new))

        offset_phase = params['theta_0'].value + self.fit_method.phase_orientation*2*np.arctan(2*params['Ql'].value*(1-fdata/params['fr'].value))
        new_phase = np.unwrap(np.angle(sdata_new-(xc+1j*yc)))
        # Match the branch-snap done in arctan_deviation so the plotted data and
        # model line up instead of differing by a multiple of 2π.
        new_phase = new_phase - 2*np.pi*np.round(np.mean(new_phase - offset_phase)/(2*np.pi))
        plt.figure(layout='tight', figsize=(5.5,3.5))
        plt.plot(fdata*1e3, new_phase, '.', label = 'Data', color="#0869c8")
        plt.plot(fdata*1e3, offset_phase, label = 'Fit', color="#b51d14")
        plt.ylabel('Offset phase (rad.)')
        plt.xlabel('Frequency (MHz)')
        plt.legend()
        plt.savefig(self.fname + '_delay_fit.png')
        plt.close()

    def initial_guess_delay(self, fdata: np.ndarray, sdata: np.ndarray):
        '''
        Just fit the phase while discarding data in the linewidth.

        Args:
            fdata: numpy array of the frequency data
            sdata: numpy array of the scattering data

        Returns:
            delay: float representing the electrical delay
        '''
        filtered_data = gaussian_filter(sdata, sigma=3)  # sigma may need to be changed for noisy data
        gradS = np.gradient(filtered_data, fdata)
        gradSmagnitude = np.abs(gradS)

        chiFunction = partitionFrequencyBand(fdata, gradS, keep = 'above', cutoff = 0.05)
        fc_index = np.argmax(gradSmagnitude)

        freq_arrays = np.split(fdata, [fc_index])
        data_arrays = np.split(sdata, [fc_index])
        chi_arrays = np.split(chiFunction, [fc_index])

        # fit the phase slope on either side of the resonance, excluding the
        # linewidth, and average the two delays
        n_points = len(fdata)
        delay_fit = []
        for n in range(2):
            #trim the data
            trim = np.nonzero(chi_arrays[n])
            trimmed_freq = np.delete(freq_arrays[n], trim)
            trimmed_data = np.delete(data_arrays[n], trim)

            # The cutoff is set relative to the smallest |dS/df|. With a cable
            # delay, |dS/df| off resonance sits on a floor of 2*pi*delay*|S|, and
            # for a shallow dip (Ql/Qc << 1) the resonance barely rises above it,
            # so almost every point is classed as inside the linewidth and a side
            # can keep 0-2 points; linregress then returns NaN. Fall back to the
            # outer quarter of the band on that side, where the resonance adds a
            # phase of only ~(Ql/Qc)/(detuning in linewidths).
            if len(trimmed_freq) < self.min_delay_points:
                outer = np.arange(n_points // 4) if n == 0 else np.arange(n_points - n_points // 4, n_points)
                trimmed_freq = fdata[outer]
                trimmed_data = sdata[outer]
            if len(trimmed_freq) < 2:
                continue

            #fit trimmed freq and data
            trimmed_phase = np.unwrap(np.angle(trimmed_data))
            lrg_result = linregress(trimmed_freq, trimmed_phase)
            delay_fit.append(lrg_result.slope / (-2 * np.pi))
        delay_fit = [d for d in delay_fit if np.isfinite(d)]
        if not delay_fit:
            # last resort: the phase slope of the whole band
            delay_fit = [linregress(fdata, np.unwrap(np.angle(sdata))).slope / (-2 * np.pi)]
        avg_delay_guess = np.mean(delay_fit)
        if not np.isfinite(avg_delay_guess):
            avg_delay_guess = 0.0

        return avg_delay_guess

    def arctan_deviation(self, params: lmfit.Parameters, fdata: np.ndarray, sdata: np.ndarray):
        '''
        An objective function to be minimized to find the electrical delay.

        The delay value passed through params is removed from the scattering data. Then the best fit circle is found for
        the resulting data. Its center is translated to the origin. If the delay removed corresponds to the actual
        electrical delay then the phase of this transformed data is an arctangent.

        Args:
            params: lmfit.Parameters object to store parameters for the arctangent function and the delay
            fdata: numpy array of the frequency data
            sdata: numpy array of the scattering data

        Returns:
            residuals: numpy array of the residuals comparing the transformed data to the arctangent described by params
        '''
        fr = params['fr'].value
        Ql = params['Ql'].value
        t_delay = params['delay'].value
        theta_0 = params['theta_0'].value

        sdata = remove_delay(fdata, sdata, t_delay)
        xc, yc, r = find_circle(np.real(sdata), np.imag(sdata))
        sdata = sdata - (xc+1j*yc)

        offset_phase = np.unwrap(np.angle(sdata))

        orientation = self.fit_method.phase_orientation
        # np.unwrap anchors to the first sample's principal branch, which can
        # land 2π away from the model at low power. Snap to the model's branch.
        model = theta_0 + orientation*2*np.arctan(2*Ql*(1-fdata/fr))
        offset_phase = offset_phase - 2*np.pi*np.round(np.mean(offset_phase - model)/(2*np.pi))

        residuals = np.abs(offset_phase - model)
        return residuals

    def find_off_res_point(self, fdata, sdata, verbose=False):
        """
        This replaces the poorly named calibration function
        Args:
            fdata: numpy array of the frequency data
            sdata: numpy array of the scattering data
            verbose: print the circle fit results

        Returns:
            the off-resonant point as a complex float
            additionally sets self.phi
        """
        if self.theta_0 is not None:
            theta_0 = self.theta_0
        else:
            # No delay fit ran (remove_delay or fit_delay off), so theta_0 is
            # unknown. Fit it the same way find_delay does -- the phase of the
            # data about the circle center against the arctan model -- with the
            # delay held at zero, since run() has already removed any delay.
            # (Fitting the raw phase of S21 instead is wrong: for a hanger the
            # circle does not enclose the origin, so that phase never winds.)
            params = lmfit.Parameters()
            self._add_free_Ql_fr(params, fdata, self.fit_method.find_initial_guess(fdata, sdata))
            params.add(name = 'delay', value = 0, vary = False)
            params.add(name = 'theta_0', value = 0)
            theta_0 = _principal_angle(self._fit_offset_phase(params, fdata, sdata).params['theta_0'].value)

        beta = theta_0 + np.pi
        xc, yc, r = find_circle(np.real(sdata), np.imag(sdata))
        self.phi = _principal_angle(np.pi - theta_0 + np.angle(xc+1j*yc))
        if verbose:
            print(f'Fit circle: Phi (preprocessing): {self.phi:0.3g}, theta_0: {theta_0:0.3g}, xc: {xc:0.3g}, yc: {yc:0.3g}')
        off_res_point = xc+ 1j*yc+ r*np.exp(1j*beta)
        if self.plot_diagnostics:
            self._plot_circle_fit(sdata, xc, yc, r, off_res_point)

        return off_res_point

    def _plot_circle_fit(self, sdata, xc, yc, r, off_res_point):
        import matplotlib.pyplot as plt
        fig=plt.figure(figsize=(4,4), layout='tight')
        plt.plot(np.real(sdata), np.imag(sdata), color="#0869c8")
        circle = plt.Circle((xc, yc), r, color='#cacaca', fill=False, label='fit circle', linewidth=1)
        plt.gca().add_patch(circle)
        plt.axis('equal')
        plt.plot(np.real(off_res_point), np.imag(off_res_point), 'ro', label = 'Off Res Point')
        plt.legend()
        plt.savefig(self.fname + '_circle_fit.png')
        plt.close(fig)

    def anchor_to_point(self, fdata: np.ndarray, sdata: np.ndarray, anchor_point = None, verbose=False):
        """
        Rotates and scales the scattering data to send the off-resonant point to an anchor point (e.g. 1+0*1j).

        Args:
            fdata: numpy array of the frequency data
            sdata: numpy array of the scattering data
            anchor_point: optional complex float to specify the desired off-resonant point (defaults to 1+0j)

        Returns:
            sdata: numpy array of the transformed scattering data
        """
        if anchor_point is None:
            anchor_point = 1+0*1j
        if self.off_res_point is None:
            self.off_res_point = self.find_off_res_point(fdata, sdata, verbose=verbose)

        new_sdata = anchor_point*sdata/self.off_res_point
        return new_sdata
