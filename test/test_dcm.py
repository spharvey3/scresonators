import numpy as np
import pytest
from pytest import approx

from src.resonator import Resonator
from src.fitter import Fitter
from src.fit_methods import DCM, LCE

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


def fit_dcm(freqs, s21, manual_init=None, **kwargs):
    res = Resonator()
    res.load_data(fdata=freqs, sdata=s21)
    res.set_fitting_strategy(DCM(), **kwargs)
    return res.fit(manual_init=manual_init)


def test_dcm_roundtrip_with_delay():
    freqs = make_freqs()
    s21 = make_dcm_s21(freqs, delay=80)

    result = fit_dcm(freqs, s21)

    assert result['Q'].value == approx(QTOT, rel=0.05)
    assert result['f0'].value == approx(F0, rel=1e-4)
    assert result['Qi'].value == approx(QI, rel=0.10)


def test_dcm_roundtrip_phi_no_delay():
    freqs = make_freqs()
    s21 = make_dcm_s21(freqs, phi=0.2)

    result = fit_dcm(freqs, s21)

    assert result['Q'].value == approx(QTOT, rel=0.05)
    assert result['f0'].value == approx(F0, rel=1e-4)
    assert result['Qi'].value == approx(QI, rel=0.10)


def test_f0_set_fixes_resonant_frequency():
    freqs = make_freqs()
    s21 = make_dcm_s21(freqs, delay=80)

    result = fit_dcm(freqs, s21, f0_set=F0)

    assert result['f0'].value == F0
    assert not result['f0'].vary


def test_manual_init():
    freqs = make_freqs()
    s21 = make_dcm_s21(freqs, delay=80)

    manual_init = DCM().find_initial_guess(freqs, s21)
    result = fit_dcm(freqs, s21, manual_init=manual_init)

    assert result['Q'].value == approx(QTOT, rel=0.10)


def test_strategy_requires_instance():
    res = Resonator()
    with pytest.raises(TypeError, match=r"DCM\(\)"):
        res.set_fitting_strategy(DCM)


def test_fitter_requires_instance():
    with pytest.raises(TypeError, match=r"DCM\(\)"):
        Fitter(fit_method=DCM)


def test_nonlinear_requires_support():
    Fitter(fit_method=DCM(), nonlinear=True)  # supported: constructs fine
    with pytest.raises(ValueError, match="LCE"):
        Fitter(fit_method=LCE(), nonlinear=True)
