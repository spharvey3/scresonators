import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scresonators.measurement.fitting as fitter


def plot_scan(freq, amps, phase, pars=None, pinit=None, power=None, slope=None):
    """
    Plot the scan data with amplitude, phase, and IQ plots.

    Parameters:
    -----------
    freq : array
        Frequency points
    amps : array
        Amplitude data
    phase : array
        Phase data
    pars : list, optional
        Fit parameters
    pinit : list, optional
        Initial fit parameters
    power : float, optional
        Power level in dBm
    slope : float, optional
        Phase slope for unwrapping

    Returns:
    --------
    None
    """
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].plot(freq / 1e6, amps, "k.", markersize=3)
    ax[0].set_xlabel("Frequency (MHz)")
    ax[0].set_ylabel("Amplitude")
    if pars is not None:
        q = 1 / (1 / pars[1] + 1 / pars[2]) * 1e4
        qi = pars[1] * 1e4
        qc = pars[2] * 1e4
        lab = f"$Q$={q:.3g}\n $Q_i$={qi:.3g} \n $Q_c$={qc:.3g}"
        ax[0].plot(freq / 1e6, fitter.hangerS21func(freq, *pars))
        # ax[0].plot(freq / 1e6, fitter.hangerS21func(freq, *pinit))
    ax[0].set_title(f"Power: {power:.1f} dB")
    if slope is None:
        phase = np.unwrap(phase)
        slope, of = np.polyfit(freq, phase, 1)
        phase_sub = phase - slope * freq - of
    else:
        phase_sub = phase - slope * freq
    ax[1].plot(
        freq / 1e6,
        np.unwrap(phase_sub) - np.mean(np.unwrap(phase_sub)),
        "k.-",
        markersize=3,
    )
    ax[1].set_xlabel("Frequency (MHz)")
    ax[1].set_ylabel("Phase")
    # ax[0].legend(loc="lower right")
    ax[0].text(
        0.95,
        0.05,
        lab,
        transform=ax[0].transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="black"),
    )
    ax2 = ax[0].twinx()
    if pars is not None:
        ax2.set_ylim(ax[0].get_ylim()[0] / pars[4], ax[0].get_ylim()[1] / pars[4])
    ax2.set_ylabel("Normalized Amplitude")

    ax[2].plot(amps * np.cos(phase_sub), amps * np.sin(phase_sub), "k.-")
    fig.tight_layout()
    # plt.show()


def fit_resonator(data, power, fitparams=None, qc=None, plot=False):

    # Convert amplitude from dB to linear scale
    amps_linear = 10 ** (data["amps"] / 20)
    # f0, Qi, Qe, phi, scale, a0, slope # Qi/Qe in units of 10k
    if qc is not None:
        hangerfit = lambda f, f0, qi, phi, scale: fitter.hangerS21func(
            f, f0, qi, qc / 1e4, phi, scale
        )

        pars, err = curve_fit(hangerfit, data["freqs"], amps_linear, p0=fitparams)
        freq_center = pars[0]
        q = 1 / (1 / pars[1] / 1e4 + 1 / qc)
        kappa = freq_center / q
        r2 = fitter.get_r2(data["freqs"], amps_linear, hangerfit, pars)
        err = np.sqrt(np.diag(err))
        # print('f error: ', err[0], 'Qi error: ', err[1], 'phi error: ', err[2], 'scale error: ', err[3])
        print(f"Qi err: {err[1]/pars[1]}")
        pars = [pars[0], pars[1], qc / 1e4, pars[2], pars[3]]
        fitparams = [pars[0], pars[1], qc / 1e4, pars[3], pars[4]]
    else:
        if fitparams is None:
            min_freq = data["freqs"][np.argmin(amps_linear)]
            fitparams = [min_freq, 100, 100, 0, np.max(amps_linear)]

        pars, err, pinit = fitter.fithanger(
            data["freqs"], amps_linear, fitparams=fitparams
        )

        pars, err, pinit = fitter.fithanger(data["freqs"], amps_linear, fitparams=pars)

        freq_center = pars[0]
        q = 1 / (1 / pars[1] + 1 / pars[2]) * 1e4
        kappa = freq_center / q
    if plot:
        plot_scan(
            data["freqs"],
            amps_linear,
            data["phases"],
            pars,
            fitparams,
            power,
        )
    plt.show()

    return freq_center, q, kappa, pars
