import numpy as np
import pytest
from pytest import approx
import skrf

from src.resonator import Resonator
from src.fit_methods import DCM, LCE, ReflectionMode

F0 = 4.5  # GHz
QI = 400_000
QC = 200_000
QTOT = 1 / (1/QI + 1/QC)


def make_freqs(f0=F0, Q=QTOT, npoints=501, span_linewidths=15):
    linewidth = 2*f0/Q
    return np.linspace(f0 - span_linewidths*linewidth, f0 + span_linewidths*linewidth, npoints)


def make_dcm_s21(freqs, f0=F0, Q=QTOT, Qc=QC, phi=0.0, delay=0.0):
    s21 = 1 - (Q*np.exp(-1j*phi)/Qc) / (1 + 2j*Q*(freqs-f0)/f0)
    return np.exp(-2j*np.pi*freqs*delay) * s21


# ---------------------------------------------------------------- data loading

def test_load_data_selects_port_from_3d():
    freqs = make_freqs()
    s21 = make_dcm_s21(freqs)

    res = Resonator()
    res.load_data(freqs, s21.reshape(-1, 1, 1))

    assert res.sdata.ndim == 1
    assert np.allclose(res.sdata, s21)


def test_load_data_rejects_bad_shapes():
    freqs = make_freqs()
    s21 = make_dcm_s21(freqs)

    with pytest.raises(ValueError, match="shape"):
        Resonator().load_data(freqs, s21.reshape(-1, 1))
    with pytest.raises(ValueError, match="length"):
        Resonator().load_data(freqs[:-1], s21)


def test_load_data_from_network_converts_to_ghz():
    freqs = make_freqs()
    s21 = make_dcm_s21(freqs)
    network = skrf.Network(frequency=skrf.Frequency.from_f(freqs, unit='GHz'),
                           s=s21.reshape(-1, 1, 1))

    res = Resonator()
    res.load_data_from_network(network)

    assert res.fdata == approx(freqs)
    assert res.sdata.ndim == 1

    res.set_fitting_strategy(DCM())
    result = res.fit()
    assert result['Q'].value == approx(QTOT, rel=0.05)
    assert result['f0'].value == approx(F0, rel=1e-4)


def test_fit_preserves_raw_data_and_is_repeatable():
    freqs = make_freqs()
    s21 = make_dcm_s21(freqs, delay=80)

    res = Resonator()
    res.load_data(freqs, s21)
    res.set_fitting_strategy(DCM())
    first = res.fit()['Qi'].value

    assert np.allclose(res.sdata, s21)  # raw data untouched
    assert res.sdata_preprocessed is not None

    res.set_fitting_strategy(DCM())
    assert res.fit()['Qi'].value == approx(first, rel=1e-6)


def test_unknown_fitter_option_raises():
    res = Resonator()
    with pytest.raises(TypeError, match="MC_weihgt"):
        res.set_fitting_strategy(DCM(), MC_weihgt=True)


# ---------------------------------------------------------- method round-trips

def test_lce_roundtrip():
    freqs = make_freqs()
    s21 = LCE.func(freqs, QTOT, QC, F0, 0.0)

    res = Resonator()
    res.load_data(freqs, s21)
    res.set_fitting_strategy(LCE())
    result = res.fit()

    assert result['Q'].value == approx(QTOT, rel=0.05)
    assert result['f0'].value == approx(F0, rel=1e-4)
    assert result['Qi'].value == approx(QI, rel=0.10)


def test_reflection_roundtrip():
    freqs = make_freqs()
    s21 = ReflectionMode.func(freqs, QTOT, QC, F0)

    res = Resonator()
    res.load_data(freqs, s21)
    res.set_fitting_strategy(ReflectionMode())
    result = res.fit()

    assert result['Q'].value == approx(QTOT, rel=0.05)
    assert result['f0'].value == approx(F0, rel=1e-4)


def test_dcm_roundtrip_fit_theta():
    freqs = make_freqs()
    s21 = make_dcm_s21(freqs, delay=80)

    res = Resonator()
    res.load_data(freqs, s21)
    res.set_fitting_strategy(DCM(), fit_theta=True)
    result = res.fit()

    assert result['Q'].value == approx(QTOT, rel=0.05)
    assert result['Qi'].value == approx(QI, rel=0.10)


# ------------------------------------------------------------------- nonlinear

def test_nonlinear_roundtrip():
    knl_eff = 5e4  # Hz
    freqs = make_freqs(npoints=201)
    s21 = DCM.func_nonlinear(freqs, QTOT, QC, F0, 0.0,
                             Knl_eff=knl_eff, gamma_nl_eff=0.0, nonlinear=True)

    res = Resonator()
    res.load_data(freqs, s21)
    res.set_fitting_strategy(DCM(), nonlinear=True)
    result = res.fit()

    assert result['Q'].value == approx(QTOT, rel=0.10)
    assert result['Knl_eff'].value == approx(knl_eff, rel=0.25)


def test_nonlinear_rejects_non_ghz_frequencies():
    freqs = make_freqs() * 1e9  # Hz
    s21 = make_dcm_s21(make_freqs())

    res = Resonator()
    res.load_data(freqs, s21)
    res.set_fitting_strategy(DCM(), nonlinear=True)
    with pytest.raises(ValueError, match="GHz"):
        res.fit()


def test_solve_photon_number_linear_limit():
    # with xi = eta = 0 the cubic reduces to n = 0.5 / (0.25 + delta^2)
    for delta in [0.0, 0.5, -2.0]:
        n = DCM.solve_photon_number(delta, 0.0, 0.0)
        assert n == approx(0.5 / (0.25 + delta**2))
