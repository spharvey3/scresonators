import numpy as np
from skrf import Network
from .fitter import Fitter
from .fit_methods.fit_method import FitMethod


class Resonator:
    """
    Holds resonator scattering data and orchestrates fitting.

    Data conventions:
        fdata: 1-D array of frequencies in GHz.
        sdata: 1-D complex array of the scattering parameter. Complex data
            removes ambiguity from amplitude/phase data (linear or dB, degrees
            or radians). An skrf-style (M, N, N) array is also accepted, with
            `port` selecting which S-parameter to fit.

    The raw data is kept in `fdata`/`sdata`; `fit()` stores the preprocessed
    trace in `sdata_preprocessed`, so fitting is repeatable.
    """

    def __init__(self, fdata=None, sdata=None):
        self.fdata = fdata
        self.sdata = sdata
        self.fitter = None
        self.fit_result = None
        self.init_guess = None
        self.sdata_preprocessed = None

    def load_data(self, fdata, sdata, port=(0, 0)):
        """Load data into the resonator object.

        Args:
            fdata: 1-D array of frequencies in GHz
            sdata: 1-D complex array of the scattering parameter, or an
                skrf-style (M, N, N) array
            port: (out, in) indices selecting the S-parameter when sdata is
                (M, N, N); defaults to S11
        """
        fdata = np.asarray(fdata)
        sdata = np.asarray(sdata)
        if sdata.ndim == 3:
            sdata = sdata[:, port[0], port[1]]
        elif sdata.ndim != 1:
            raise ValueError(f"sdata must have shape (M,) or (M, N, N), got {sdata.shape}")
        if fdata.shape != sdata.shape:
            raise ValueError(f"fdata {fdata.shape} and sdata {sdata.shape} must have the same length")
        self.fdata = fdata
        self.sdata = sdata

    def load_data_from_network(self, network: Network, port=(0, 0)):
        """Load frequencies (converted to GHz) and one S-parameter from an skrf Network."""
        # Network.f is always in Hz, regardless of network.frequency.units
        self.load_data(network.f / 1e9, network.s, port=port)

    def load_data_from_touchstone(self, touchstone_file: str, port=(0, 0)):
        """Load frequencies (converted to GHz) and one S-parameter from a touchstone file."""
        self.load_data_from_network(Network(touchstone_file), port=port)

    def set_fitting_strategy(self, strategy: FitMethod, **kwargs):
        """Set the fitting strategy with a FitMethod instance; kwargs are forwarded to Fitter."""
        if isinstance(strategy, type):
            raise TypeError(f"pass an instance, e.g. {strategy.__name__}(), not the class {strategy.__name__}")
        if not isinstance(strategy, FitMethod):
            raise TypeError("strategy must be a FitMethod instance, e.g. DCM()")
        self.fitter = Fitter(fit_method=strategy, **kwargs)

    def fit(self, manual_init=None, verbose=False):
        """Perform fitting using the selected fitting strategy."""
        if not self.fitter:
            raise ValueError("Fitting strategy not set.")
        if self.fdata is None or self.sdata is None:
            raise ValueError("Data not loaded")

        self.sdata_preprocessed, self.fit_result, self.init_guess = self.fitter.fit(
            self.fdata, self.sdata, manual_init=manual_init, verbose=verbose)
        return self.fit_result

    # Additional resonator functionalities can be added here
