import numpy as np
import lmfit
from scipy.ndimage import gaussian_filter
from scipy.stats import linregress
from scipy.interpolate import interp1d

from .utils import remove_delay, find_circle, sloped_arctan, partitionFrequencyBand


class Preprocessor:
    """
    Prepares raw scattering data for fitting: background removal, electrical
    delay removal, and rotation/scaling of the off-resonant point to an anchor.

    Holds the preprocessing state (delay, theta_0, phi, off-resonant point)
    so it can be inspected after a fit or reused for a theta refinement pass.
    """

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
        params = lmfit.Parameters()
        if self.Ql_guess is None and self.fr_guess is None:
            guess_params = self.fit_method.find_initial_guess(fdata, sdata)

        if self.delay_guess is None:
            delay_guess = self.initial_guess_delay(fdata, sdata)
        else:
            delay_guess = self.delay_guess

        if self.Ql_guess is not None:
            params.add(name = 'Ql', value = self.Ql_guess, min=0.9*self.Ql_guess, max=1.1*self.Ql_guess)
            params['Ql'].vary = False
        else:
            params.add(name = 'Ql', value = guess_params['Q'].value, min=0, max=1e8)
        if self.fr_guess is not None:
            params.add(name = 'fr', value = self.fr_guess, min=0.99*self.fr_guess, max=1.01*self.fr_guess)
            params['fr'].vary = False
        else:
            params.add(name = 'fr', value = guess_params['f0'].value, min=0, max=1e10)

        if self.fit_theta and self.theta_0 is not None:
            params.add(name = 'delay', value = self.delay, vary=False)
            params.add(name = 'theta_0', value = self.theta_0, min=self.theta_0-0.05, max=self.theta_0+0.05)
        else:
            # delay is a time (ns for GHz data), not an angle -- leave it unbounded
            params.add(name = 'delay', value = delay_guess)
            params.add(name = 'theta_0', value = 0, min=-np.pi, max=np.pi)

        #running the minimizer several times also improves results
        for _ in range(10):
            min_result = lmfit.minimize(fcn=self.arctan_deviation, params=params, args=(fdata, sdata),
                                    method='least_squares', max_nfev = 10000000)

            params = min_result.params

            #TODO: we should have a condition on the residuals to break this loop early
        electrical_delay = params['delay'].value
        self.Ql_guess = params['Ql'].value
        self.fr_guess = params['fr'].value
        self.theta_0 = params['theta_0'].value
        if verbose:
            print(f'Fit delay. Chi sq: {min_result.redchi:.3g}, Q: {self.Ql_guess:.0f}, fr: {params["fr"].value:.4f}, theta_0: {self.theta_0:.4f}, delay: {electrical_delay:.4f}')
        self.delay_chi = min_result.redchi

        if self.plot_diagnostics:
            self._plot_delay_fit(fdata, sdata, params, electrical_delay)

        return electrical_delay

    def _plot_delay_fit(self, fdata, sdata, params, electrical_delay):
        """Plot the sloped arctan as a verification step."""
        import matplotlib.pyplot as plt
        sdata_new = remove_delay(fdata, sdata, electrical_delay)
        xc, yc, r = find_circle(np.real(sdata_new), np.imag(sdata_new))

        offset_phase = params['theta_0'].value + self.fit_method.phase_orientation*2*np.arctan(2*params['Ql'].value*(1-fdata/params['fr'].value))
        new_phase = np.unwrap(np.angle(sdata_new-(xc+1j*yc)))
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
        delay_fit = []
        for n in range(2):
            #trim the data
            trim = np.nonzero(chi_arrays[n])
            trimmed_freq = np.delete(freq_arrays[n], trim)
            trimmed_data = np.delete(data_arrays[n], trim)

            #fit trimmed freq and data
            trimmed_phase = np.unwrap(np.angle(trimmed_data))
            lrg_result = linregress(trimmed_freq, trimmed_phase)
            delay_fit.append(lrg_result.slope / (-2 * np.pi))
        avg_delay_guess = (delay_fit[0]+delay_fit[1])/2

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
        residuals = np.abs(offset_phase - (theta_0 + orientation*2*np.arctan(2*Ql*(1-fdata/fr))))
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
            beta = self.theta_0 + np.pi
            xc, yc, r = find_circle(np.real(sdata), np.imag(sdata))
            self.phi = np.pi - self.theta_0 + np.angle(xc+1j*yc)
            if verbose:
                print(f'Fit circle: Phi (preprocessing): {self.phi:0.3g}, theta_0: {self.theta_0:0.3g}, xc: {xc:0.3g}, yc: {yc:0.3g}')
            off_res_point = xc+ 1j*yc+ r*np.exp(1j*beta)
            if self.plot_diagnostics:
                self._plot_circle_fit(sdata, xc, yc, r, off_res_point)

            return off_res_point
        else:
            #this block is untested
            #TODO: fit the offset phase to an arctan (without delay, that is not the place for this function)
            orpmodel = lmfit.Model(sloped_arctan)
            guess_params = self.fit_method.find_initial_guess(fdata, sdata)

            orparams = lmfit.Parameters()
            orparams.add(name = 'Ql', value = guess_params['Q'].value)
            orparams.add(name = 'fr', value = guess_params['f0'].value)
            orparams.add(name = 'delay', value = 0, vary = False)
            orparams.add(name = 'theta_0', value = 0)
            orparams.add(name = 'orientation', value = self.fit_method.phase_orientation, vary = False)

            orp_results = orpmodel.fit(np.unwrap(np.angle(sdata)), orparams, f=fdata, fit_kws={'max_nfev': 10000000})
            theta_0 = orp_results.params['theta_0'].value
            # same geometry as the theta_0 branch above
            beta = theta_0 + np.pi
            xc, yc, r = find_circle(np.real(sdata), np.imag(sdata))
            self.phi = np.pi - theta_0 + np.angle(xc + 1j * yc)
            if verbose:
                print(f'phi (preprocessing): {self.phi}')
            return xc + 1j * yc + r * np.exp(1j * beta)

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
