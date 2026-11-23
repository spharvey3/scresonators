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


def make_sparse_s21(Qc, Qi, f0=5.0, phi=0.1, delay=70, noise=1e-3, npoints=80, span_linewidths=8, seed=0):
    """DCM hanger trace with cable delay and complex Gaussian noise."""
    Q = 1 / (1/Qi + np.cos(phi)/Qc)
    linewidth = f0/Q
    freqs = np.linspace(f0 - span_linewidths*linewidth, f0 + span_linewidths*linewidth, npoints)
    s21 = make_dcm_s21(freqs, f0=f0, Q=Q, Qc=Qc, phi=phi, delay=delay)
    rng = np.random.default_rng(seed)
    s21 = s21 + noise*(rng.normal(size=npoints) + 1j*rng.normal(size=npoints))/np.sqrt(2)
    return freqs, s21, Q


def test_dcm_shallow_dip_fits():
    # Qc = 1e7, Qi = 1e5: a 0.09 dB dip, ten times the noise. This used to
    # raise 'Initial guess is outside of provided bounds' from the delay fit.
    freqs, s21, Q = make_sparse_s21(Qc=1e7, Qi=1e5)

    result = fit_dcm(freqs, s21)

    assert result['Q'].value == approx(Q, rel=0.10)
    assert result['f0'].value == approx(5.0, rel=1e-5)


def test_dcm_moderately_shallow_dip_fits():
    # Qc = 1e6, Qi = 1e5: a 0.8 dB dip, which failed the same way
    freqs, s21, Q = make_sparse_s21(Qc=1e6, Qi=1e5)

    result = fit_dcm(freqs, s21)

    assert result['Q'].value == approx(Q, rel=0.02)


def test_dcm_deep_dip_sparse_with_delay():
    # a deep dip on the same sparse grid still recovers Q closely
    freqs, s21, Q = make_sparse_s21(Qc=3e5, Qi=4e5)

    result = fit_dcm(freqs, s21)

    assert result['Q'].value == approx(Q, rel=0.02)
    assert result['f0'].value == approx(5.0, rel=1e-6)


def test_dcm_fit_delay_false_uses_given_delay():
    # fit_delay=False skips the delay fit, so theta_0 comes from the circle
    # anchoring step; this path used to raise a TypeError inside lmfit
    freqs, s21, Q = make_sparse_s21(Qc=2e5, Qi=4e5, phi=-0.4, npoints=201, span_linewidths=10)
    s21 = 0.5*np.exp(-2j) * s21

    result = fit_dcm(freqs, s21, fit_delay=False, delay_guess=70)

    assert result['Q'].value == approx(Q, rel=0.01)
    assert result['phi'].value == approx(-0.4, abs=0.01)


def test_dcm_remove_delay_false():
    freqs = make_freqs()
    s21 = 0.7*np.exp(0.4j) * make_dcm_s21(freqs, phi=0.1)

    result = fit_dcm(freqs, s21, remove_delay=False)

    assert result['Q'].value == approx(QTOT, rel=0.01)
    assert result['phi'].value == approx(0.1, abs=0.01)


def make_featureless_s21(seed):
    rng = np.random.default_rng(seed)
    freqs = np.linspace(5.0, 5.001, 401)
    s21 = np.exp(-2j*np.pi*freqs*50) * (1 + 1e-3*(rng.normal(size=401) + 1j*rng.normal(size=401)))
    return freqs, s21


def test_unphysical_fit_is_flagged():
    # no resonance at all: this trace fits to Qc ~ 3e3 and Qi < 0 with a small
    # reported stderr; the fitter must not return that as a measurement
    from src import UnphysicalFitWarning
    freqs, s21 = make_featureless_s21(seed=25)
    res = Resonator()
    res.load_data(fdata=freqs, sdata=s21)
    res.set_fitting_strategy(DCM())

    with pytest.warns(UnphysicalFitWarning, match='Qi'):
        result = res.fit()

    assert result['Qi'].value < 0
    assert any('Qi' in reason for reason in res.fitter.unphysical)
    for name in ('Q', 'Qc', 'Qi', 'phi'):
        assert np.isinf(result[name].stderr)


def test_unphysical_fit_raises_when_asked():
    from src import UnphysicalFitError
    freqs, s21 = make_featureless_s21(seed=25)

    with pytest.raises(UnphysicalFitError, match='Qi'):
        fit_dcm(freqs, s21, raise_unphysical=True)


def test_physical_fit_not_flagged():
    import warnings
    from src import UnphysicalFitWarning
    freqs, s21, Q = make_sparse_s21(Qc=1e7, Qi=1e5)
    res = Resonator()
    res.load_data(fdata=freqs, sdata=s21)
    res.set_fitting_strategy(DCM(), raise_unphysical=True)

    with warnings.catch_warnings():
        warnings.simplefilter('error', UnphysicalFitWarning)
        result = res.fit()

    assert res.fitter.unphysical == []
    assert np.isfinite(result['Qi'].stderr)


def test_unphysical_reasons():
    import lmfit
    params = lmfit.Parameters()
    for name, value in dict(Q=1e5, Qc=2e5, Qi=2e5, phi=0.3).items():
        params.add(name, value=value)
    assert DCM().unphysical_reasons(params) == []

    params['Qi'].value = -5e4
    params['phi'].value = 2.0
    reasons = DCM().unphysical_reasons(params)
    assert len(reasons) == 2
    assert any('Qi' in r for r in reasons) and any('phi' in r for r in reasons)
