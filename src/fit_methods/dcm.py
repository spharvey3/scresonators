import numpy as np
import lmfit

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
        Q_guess = 2*f_c/(linewidth)
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
