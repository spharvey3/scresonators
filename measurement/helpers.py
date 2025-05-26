import numpy as np
import scipy.constants as cs
import seaborn as sns
import matplotlib.pyplot as plt

def n(p, f, q, qc):
    return pow_res(p) * q**2 / qc / (cs.h * f**2 * np.pi)


def pow_res(p):
    return 10 ** (p / 10) * 1e-3

def concat_scans(data): 
    dd = {'freqs': np.array([]), 'amps': np.array([]),'phases': np.array([])}
    for i in range(len(data)):
        dd['freqs'] = np.concatenate((dd['freqs'], data[i]['freqs'][1:]))
        dd['amps'] = np.concatenate((dd['amps'], data[i]['amps'][1:]))
        dd['phases'] = np.concatenate((dd['phases'], data[i]['phases'][1:]))
    return dd

def find_peaks(data, prom=0.1): 
    # Peak finding parameters
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d

    min_dist = 10e6  # minimum distance between peaks, may need to be edited if things are really close
    max_width = 15e6  # maximum width of peaks in MHz, may need to be edited if peaks are off
    freq_sigma = 3  # sigma for gaussian filter

    
    # Convert parameters to indices
    df = data['freqs'][1] - data['freqs'][0]
    min_dist_inds = int(min_dist / df)
    max_width_inds = int(max_width / df)
    filt_sigma = int(np.ceil(freq_sigma / df))
    ydata = data["amps"][1:-1]
    # Apply Gaussian filter to smooth data
    ydata_smooth = gaussian_filter1d(ydata, sigma=filt_sigma)
    ydata = ydata / ydata_smooth
    
    # Show debug plots if requested
    
    fig, ax = plt.subplots(2, 1, figsize=(8, 7))
    ax[0].plot(data['freqs'][1:-1], data["amps"][1:-1])
    ax[0].plot(data['freqs'][1:-1], ydata_smooth)
    ax[1].plot(data['freqs'][1:-1], ydata)
    
    min_dist_inds = int(np.max((1, min_dist_inds)))
    # Find peaks in the data
    coarse_peaks, props = find_peaks(
        -ydata,
        distance=min_dist_inds,
        prominence=prom,
        width=[0, max_width_inds],
    )

    # Store peak information
    data["coarse_peaks_index"] = coarse_peaks
    data["coarse_peaks"] = data['freqs'][coarse_peaks]
    data["coarse_props"] = props

    for i in range(len(coarse_peaks)):
        peak = coarse_peaks[i]
        ax[0].axvline(data["freqs"][peak], linestyle="--", color="0.2", linewidth=0.5)
        ax[1].axvline(data["freqs"][peak], linestyle="--", color="0.2", linewidth=0.5)
    return data 
    

def get_homophase(config):
    """
    Calculate the list of frequencies that gives you equal phase spacing
    Parameters:
    config (dict): A dictionary containing the following keys:
        - "npoints" (int): Number of points in the frequency list.
        - "span" (float): Frequency span.
        - "kappa" (float): linewidth
        - "kappa_inc" (float): expected linewidth fudge factor (fix me).
        - "center_freq" (float): Center frequency.
    Returns:
    numpy.ndarray: An array containing the calculated frequency list.
    """
    nlin = 2

    N = config["npoints"] - nlin * 2
    df = config["span"]
    w = df / config["kappa"] * config["kappa_inc"]
    at = np.arctan(2 * w / (1 - w**2)) + np.pi
    R = w / np.tan(at / 2)
    fr = config["freq_center"]
    print(fr)
    n = np.arange(N) - N / 2 + 1 / 2
    flist = fr + R * df / (2 * w) * np.tan(n / (N - 1) * at)
    flist_lin = (
        -np.arange(nlin, 0, -1) * df / N * 3
        + config["freq_center"]
        - config["span"] / 2
    )
    flist_linp = (
        np.arange(1, nlin + 1) * df / N * 3 + config["freq_center"] + config["span"] / 2
    )
    flist = np.concatenate([flist_lin, flist, flist_linp])
    return flist

def make_lin(d):
    return 10**(d/20)

def config_figs():

    # Set seaborn color palette
    colors = ["#0869c8", "#b51d14", '#ddb310', '#658b38', '#7e1e9c', '#75bbfd', '#cacaca']
    sns.set_palette(sns.color_palette(colors))

    # Figure parameters
    plt.rcParams['figure.figsize'] = [8, 4]
    plt.rcParams.update({'font.size': 13})