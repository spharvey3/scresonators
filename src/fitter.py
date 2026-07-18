import numpy as np

from .fit_methods.fit_method import FitMethod
from .preprocessing import Preprocessor


class Fitter:
    #: recognized keyword options (includes legacy aliases preprocess_linear/preprocess_circle)
    _KNOWN_OPTIONS = {
        'remove_delay', 'fit_delay', 'delay_guess', 'fit_theta', 'databg',
        'linear_baseline', 'preprocess_linear', 'normalize', 'anchor_circle',
        'preprocess_circle', 'off_res_point', 'nonlinear', 'Qc_set', 'f0_set',
        'MC_weight', 'MC_rounds', 'plot_diagnostics', 'fname',
    }

    def __init__(self, fit_method=None, **kwargs):
        """Initializes the Fitter with a fitting method that includes the fitting function.

        Args:
            fit_method (FitMethod): An instance of a fitting method class, e.g. DCM().

        Keyword Args:
            remove_delay (bool): remove electrical delay during preprocessing (default True)
            fit_delay (bool): fit the delay instead of using delay_guess directly (default True)
            delay_guess (float): initial guess for the electrical delay (default None: estimate it)
            fit_theta (bool): refine theta_0 with a second delay fit after the first fit (default False)
            databg: background data object with freqs/linear_amps/phases attributes (default None)
            linear_baseline (bool): remove a linear baseline from phase and magnitude (default False)
            normalize (int): number of endpoints for the linear baseline fit (default 4)
            anchor_circle (bool): rotate/scale the off-resonant point to 1+0j (default True)
            off_res_point (complex): prescribe the off-resonant point (default None: find it)
            nonlinear (bool): use the fit method's nonlinear model (default False)
            Qc_set (tuple): fix (Qc, phi) during the fit (default None)
            f0_set (float): fix the resonant frequency during the fit (default None)
            MC_weight (bool): refine the fit with emcee Monte Carlo sampling (default False)
            MC_rounds (int): number of emcee steps (default 5000)
            plot_diagnostics (bool): save diagnostic plots of the delay and circle fits (default False)
            fname (str): filename prefix for diagnostic plots (default '')
        """
        if isinstance(fit_method, type):
            raise TypeError(f"pass an instance, e.g. {fit_method.__name__}(), not the class {fit_method.__name__}")
        if not isinstance(fit_method, FitMethod):
            raise TypeError("fit_method must be a FitMethod instance, e.g. DCM()")
        unknown = set(kwargs) - self._KNOWN_OPTIONS
        if unknown:
            raise TypeError(f"unknown Fitter option(s): {sorted(unknown)}")

        self.fit_method = fit_method
        self.fit_theta = kwargs.get('fit_theta', False)
        self.MC_rounds = kwargs.get('MC_rounds', 5000)
        self.MC_weight = kwargs.get('MC_weight', False)
        self.nonlinear = kwargs.get('nonlinear', False)
        if self.nonlinear and not fit_method.supports_nonlinear:
            raise ValueError(f"{type(fit_method).__name__} does not support nonlinear fitting")
        self.Qc_set = kwargs.get('Qc_set', None)
        self.f0_set = kwargs.get('f0_set', None)
        self.res_chi = None

        self.preprocessor = Preprocessor(
            fit_method,
            remove_delay=kwargs.get('remove_delay', True),
            fit_delay=kwargs.get('fit_delay', True),
            delay_guess=kwargs.get('delay_guess', None),
            databg=kwargs.get('databg', None),
            linear_baseline=kwargs.get('linear_baseline', kwargs.get('preprocess_linear', False)),
            normalize=kwargs.get('normalize', 4),
            anchor_circle=kwargs.get('anchor_circle', kwargs.get('preprocess_circle', True)),
            fit_theta=self.fit_theta,
            off_res_point=kwargs.get('off_res_point', None),
            plot_diagnostics=kwargs.get('plot_diagnostics', False),
            fname=kwargs.get('fname', ''),
        )

    # preprocessing state, exposed for convenience
    @property
    def delay(self):
        return self.preprocessor.delay

    @property
    def theta_0(self):
        return self.preprocessor.theta_0

    @property
    def phi(self):
        return self.preprocessor.phi

    @property
    def delay_chi(self):
        return self.preprocessor.delay_chi

    def fit(self, fdata, sdata, manual_init=None, verbose=False):
        """
        Fit resonator data using the provided method and lmfit's Model fit.

        Args:
            fdata: numpy array of the frequency data
            sdata: complex valued numpy array of the scattering parameter data
            manual_init: optional lmfit.Parameters to use instead of the automatic initial guess
            verbose: print fit reports

        Returns:
            tuple: (preprocessed sdata, fit parameters with Qi extracted, initial guess)
        """
        if self.nonlinear and np.max(fdata) > 1e3:
            raise ValueError("the nonlinear model assumes frequencies in GHz; "
                             f"fdata max {np.max(fdata):.3g} looks like a different unit")

        sdata = self.preprocessor.run(fdata, sdata, verbose=verbose)

        # Setup the initial parameters or use provided manual_init
        if manual_init:
            params = manual_init
        elif self.nonlinear:
            params = self.fit_method.find_initial_guess_nonlinear(fdata, sdata)
        else:
            params = self.fit_method.find_initial_guess(fdata, sdata)
        init_guess = params
        self._apply_constraints(params)

        # Fit only the data within two linewidths of the resonance
        kappa = params['f0'].value / params['Q'].value
        mask = np.abs(fdata - params['f0'].value) < 2*kappa
        if not np.isfinite(kappa) or np.count_nonzero(mask) < len(params):
            # the linewidth estimate failed (e.g. no points above the gradient
            # cutoff gives Q_guess=inf, kappa=0); fall back to the full span
            mask = np.ones(len(fdata), dtype=bool)
        fdata_fit = fdata[mask]

        if self.nonlinear:
            model = self.fit_method.create_model_nonlinear(nonlinear=True)
        else:
            model = self.fit_method.create_model()
        result = model.fit(sdata[mask], params, f=fdata_fit, method='least_squares')
        og_chi = result.redchi

        if self.fit_theta:
            # Redo the preprocessing with Q and f0 fixed to the fitted values,
            # letting theta_0 refine, then fit again
            sdata = self.preprocessor.refine_with_theta(
                fdata, Ql=result.params['Q'].value, fr=result.params['f0'].value, verbose=verbose)
            result = model.fit(sdata[mask], result.params, f=fdata_fit, method='least_squares')

        self.res_chi = result.redchi
        if verbose:
            print(result.fit_report())
            print(f'Resonator fit chi sq: {result.redchi:.3g}, before fitting theta: {og_chi:.3g}')

        # Using Monte Carlo to explore parameter space if enabled
        # This seems to consistently give worse results than the basic least squares fit
        if self.MC_weight:
            emcee_kwargs = {
                'steps': self.MC_rounds,
                'thin': 10,
                'burn': int(self.MC_rounds * 0.3),
                'is_weighted': False,  # the residuals carry no measurement uncertainties
                'workers': 1
            }
            # use the same masked data as the least-squares fit
            result = model.fit(data=sdata[mask], params=result.params, f=fdata_fit, method='emcee', fit_kws=emcee_kwargs)
            if verbose:
                print(result.fit_report())

        params = self.fit_method.extractQi(result.params)

        return sdata, params, init_guess

    def _apply_constraints(self, params):
        """Fixes parameters the caller prescribed via Qc_set / f0_set."""
        if self.Qc_set is not None:
            params['Qc'].value = self.Qc_set[0]
            params['phi'].value = self.Qc_set[1]
            params['Qc'].vary = False
            params['phi'].vary = False
        if self.f0_set is not None:
            params['f0'].value = self.f0_set
            params['f0'].vary = False
