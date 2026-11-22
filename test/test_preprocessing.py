import numpy as np
from pytest import approx

from src.preprocessing import Preprocessor
from src.fit_methods import DCM

F0 = 4.5  # GHz
QI = 400_000
QC = 200_000
QTOT = 1 / (1/QI + 1/QC)


def make_dcm_s21(freqs, f0=F0, Q=QTOT, Qc=QC, phi=0.0, delay=0.0):
    s21 = 1 - (Q*np.exp(-1j*phi)/Qc) / (1 + 2j*Q*(freqs-f0)/f0)
    return np.exp(-2j*np.pi*freqs*delay) * s21


def make_freqs(f0=F0, Q=QTOT, npoints=501, span_linewidths=15):
    linewidth = 2*f0/Q
    return np.linspace(f0 - span_linewidths*linewidth, f0 + span_linewidths*linewidth, npoints)


def test_find_delay_recovers_known_delay():
    delay = 80
    freqs = make_freqs()
    s21 = make_dcm_s21(freqs, delay=delay)

    preprocessor = Preprocessor(DCM())
    fitted_delay = preprocessor.find_delay(freqs, s21)

    # the arctan objective trades a little delay against the circle geometry,
    # so recovery is approximate
    assert fitted_delay == approx(delay, rel=0.15)


def test_run_anchors_off_res_point():
    freqs = make_freqs()
    # scaled and rotated data with delay: preprocessing should undo all of it
    s21 = 0.7 * np.exp(0.4j) * make_dcm_s21(freqs, delay=80)

    preprocessed = Preprocessor(DCM()).run(freqs, s21)

    # far from resonance S21 should sit at the anchor point 1+0j
    assert preprocessed[0] == approx(1 + 0j, abs=0.1)
    assert preprocessed[-1] == approx(1 + 0j, abs=0.1)


def test_remove_linear_baseline():
    freqs = make_freqs()
    clean = make_dcm_s21(freqs)
    # impose linear phase and dB-magnitude baselines
    phase_slope, mag_slope_db = 3.0, 10.0
    baseline = 10**((mag_slope_db*(freqs-freqs[0]))/20) * np.exp(1j*phase_slope*(freqs-freqs[0]))
    dirty = clean * baseline

    corrected, *_ = Preprocessor(DCM()).remove_linear_baseline(freqs, dirty, normalize=4)

    assert np.allclose(np.abs(corrected), np.abs(clean), atol=0.02)


def test_remove_background():
    class Background:
        pass

    freqs = make_freqs()
    clean = make_dcm_s21(freqs)
    background = 0.5 * np.exp(0.3j)
    dirty = clean * background

    databg = Background()
    databg.freqs = freqs
    databg.linear_amps = np.full(len(freqs), np.abs(background))
    databg.phases = np.full(len(freqs), np.angle(background))

    corrected = Preprocessor(DCM(), databg=databg).remove_background(freqs, dirty)

    assert np.allclose(corrected, clean, atol=1e-10)


# A very shallow dip: undercoupled (Qc = 1e7 >> Qi = 1e5, a 0.09 dB dip), 80
# points over +-8 linewidths, 70 ns of cable delay. Off resonance |dS/df| sits
# on the 2*pi*delay floor and the resonance barely rises above it, which used
# to leave initial_guess_delay ~1 point per side and a NaN delay guess.
SHALLOW_F0 = 5.0
SHALLOW_QC = 1e7
SHALLOW_QI = 1e5
SHALLOW_PHI = 0.1
SHALLOW_QTOT = 1 / (1/SHALLOW_QI + np.cos(SHALLOW_PHI)/SHALLOW_QC)
SHALLOW_DELAY = 70


def make_shallow_s21(noise=0.0, seed=0):
    linewidth = SHALLOW_F0/SHALLOW_QTOT
    freqs = np.linspace(SHALLOW_F0 - 8*linewidth, SHALLOW_F0 + 8*linewidth, 80)
    s21 = make_dcm_s21(freqs, f0=SHALLOW_F0, Q=SHALLOW_QTOT, Qc=SHALLOW_QC,
                       phi=SHALLOW_PHI, delay=SHALLOW_DELAY)
    rng = np.random.default_rng(seed)
    s21 = s21 + noise*(rng.normal(size=len(freqs)) + 1j*rng.normal(size=len(freqs)))/np.sqrt(2)
    return freqs, s21


def test_initial_guess_delay_shallow_dip_is_finite():
    freqs, s21 = make_shallow_s21(noise=1e-3)

    delay_guess = Preprocessor(DCM()).initial_guess_delay(freqs, s21)

    assert np.isfinite(delay_guess)
    assert delay_guess == approx(SHALLOW_DELAY, rel=0.02)


def test_find_delay_shallow_dip_theta_0_not_pinned():
    # theta_0 and delay are nearly degenerate (a delay change d_tau rotates the
    # circle by 2*pi*f0*d_tau); a bound on theta_0 used to stop the delay fit
    # short of the true delay and bias Q by ~10% even without noise
    freqs, s21 = make_shallow_s21()

    preprocessor = Preprocessor(DCM())
    delay = preprocessor.find_delay(freqs, s21)

    assert delay == approx(SHALLOW_DELAY, abs=0.01)
    assert preprocessor.Ql_guess == approx(SHALLOW_QTOT, rel=0.01)
    assert -np.pi < preprocessor.theta_0 <= np.pi
    # for a hanger the offset phase at resonance is pi - phi
    assert np.angle(np.exp(1j*(preprocessor.theta_0 - (np.pi - SHALLOW_PHI)))) == approx(0, abs=0.01)


def test_find_delay_survives_nan_linewidth_estimate():
    # a dip lost in the noise can give a NaN Q estimate; that must fall back to
    # a finite start instead of raising 'Initial guess is outside of provided bounds'
    freqs, s21 = make_shallow_s21(noise=1e-3)
    method = DCM()
    true_guess = method.find_initial_guess

    def nan_guess(fdata, sdata):
        params = true_guess(fdata, sdata)
        params['Q'].set(value=np.nan)
        return params

    method.find_initial_guess = nan_guess
    delay = Preprocessor(method).find_delay(freqs, s21)

    assert np.isfinite(delay)
