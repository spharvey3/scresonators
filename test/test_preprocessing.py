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
