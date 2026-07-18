from skrf import Network
from .fitter import Fitter
from .fit_methods.fit_method import FitMethod



"""
only take data in appropriately shaped np arrays
fdata: np shape (,M)
sdata: np shape (M,N,N) for an N-Port device with M frequency points, shape (,M) also works for a 1-Port device
this removes ambiguity from amplitude/phase data (linear or db & degrees or radians)
and naturally generalizes for Effective Reflection Mode measurements (Two Port required)
"""

class Resonator:
    def __init__(self, fdata=None, sdata = None):
        self.fdata = fdata
        self.sdata = sdata
        self.fitter = None
        self.fit_result = None

    # TODO: Can we use *args to combine the following three functions into one?
    def load_data(self, fdata, sdata):
        """Load data into the resonator object."""
        self.fdata = fdata
        self.sdata = sdata

    def load_data_from_network(self, network: Network):
        self.sdata = network.s
        network.frequency.units = 'GHz'
        self.fdata = network.f


    def load_data_from_touchstone(self, touchstone_file: str):
        network = Network(touchstone_file)
        self.sdata = network.s
        network.frequency.units = 'GHz'
        self.fdata = network.f

    def set_fitting_strategy(self, strategy: FitMethod, **kwargs):
        """Set the fitting strategy with a FitMethod instance; kwargs are forwarded to Fitter."""
        if isinstance(strategy, type):
            raise TypeError(f"pass an instance, e.g. {strategy.__name__}(), not the class {strategy.__name__}")
        if not isinstance(strategy, FitMethod):
            raise TypeError("strategy must be a FitMethod instance, e.g. DCM()")
        self.fitter = Fitter(fit_method=strategy, **kwargs)

    def fit(self, manual_init=None, verbose = False):
        """Perform fitting using the selected fitting strategy."""
        if not self.fitter:
            raise ValueError("Fitting strategy not set.")
        if self.fdata is None or self.sdata is None:
            raise ValueError("Data not loaded")

        output = self.fitter.fit(self.fdata, self.sdata, manual_init=manual_init, verbose = verbose)
        self.sdata = output[0]
        self.fit_result = output[1]
        self.init_guess = output[2]
        return self.fit_result

    #Additional resonator functionalities can be added here
