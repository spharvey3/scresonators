import numpy as np
import lmfit
import pytest
from pytest import approx

from src.fit_methods import DCM, LCE, ReflectionMode

F0 = 4.5  # GHz
Q = 130_000
QC = 200_000


def make_freqs(f0=F0, Q=Q, npoints=501, span_linewidths=15):
    linewidth = 2*f0/Q
    return np.linspace(f0 - span_linewidths*linewidth, f0 + span_linewidths*linewidth, npoints)


@pytest.mark.parametrize("method", [DCM(), LCE(), ReflectionMode()])
def test_find_initial_guess_keys_and_scale(method):
    freqs = make_freqs()
    names = list(lmfit.Model(method.func, independent_vars=['f']).param_names)
    sdata = method.func(freqs, **{'Q': Q, 'Qc': QC, 'f0': F0, 'phi': 0.0} if 'phi' in names
                        else {'Q': Q, 'Qc': QC, 'f0': F0})

    params = method.find_initial_guess(freqs, sdata)

    assert set(names) <= set(params.keys())
    # the guess is a heuristic; observed errors are <=45%, allow up to 100%
    assert params['Q'].value == approx(Q, rel=1.0)
    assert params['f0'].value == approx(F0, rel=0.01)


@pytest.mark.parametrize("method", [DCM(), LCE(), ReflectionMode()])
def test_fit_function_matches_func(method):
    freqs = make_freqs()
    params = lmfit.Parameters()
    params.add('Q', value=Q)
    params.add('Qc', value=QC)
    params.add('f0', value=F0)
    if 'phi' in lmfit.Model(method.func, independent_vars=['f']).param_names:
        params.add('phi', value=0.1)
        direct = method.func(freqs, Q, QC, F0, 0.1)
    else:
        direct = method.func(freqs, Q, QC, F0)

    assert np.allclose(method.fit_function(freqs, params), direct)


def _params_with_stderr(Q_err=100.0, Qc_err=200.0, phi=0.0, phi_err=0.01):
    params = lmfit.Parameters()
    params.add('Q', value=Q)
    params.add('Qc', value=QC)
    params.add('phi', value=phi)
    params['Q'].stderr = Q_err
    params['Qc'].stderr = Qc_err
    params['phi'].stderr = phi_err
    return params


def test_extract_qi_base():
    params = _params_with_stderr()
    result = LCE().extractQi(params)

    expected_inverse_qi = 1/Q - 1/QC
    assert result['Qi'].value == approx(1/expected_inverse_qi)
    expected_stderr = np.sqrt((100.0/Q**2)**2 + (200.0/QC**2)**2)
    assert result['inverseQi'].stderr == approx(expected_stderr)
    assert result['Qi'].stderr == approx(expected_stderr/expected_inverse_qi**2)


def test_extract_qi_base_missing_stderr():
    params = _params_with_stderr()
    params['Q'].stderr = None
    result = LCE().extractQi(params)

    assert np.isinf(result['inverseQi'].stderr)
    assert np.isinf(result['Qi'].stderr)


def test_extract_qi_dcm_cos_phi():
    phi = 0.2
    params = _params_with_stderr(phi=phi)
    result = DCM().extractQi(params)

    expected_inverse_qi = 1/Q - np.cos(phi)/QC
    assert result['Qi'].value == approx(1/expected_inverse_qi)
    assert result['Qc_real'].value == approx(QC/np.cos(phi))
    assert result['Qc_real'].stderr == approx(200.0/np.cos(phi))


def test_extract_qi_dcm_infinite_stderr():
    params = _params_with_stderr()
    params['Qc'].stderr = np.inf
    result = DCM().extractQi(params)

    assert np.isinf(result['inverseQi'].stderr)


def test_nonlinear_func_linear_limit():
    freqs = make_freqs()
    dcm = DCM()
    linear = dcm.func(freqs, Q, QC, F0, 0.1)
    nonlinear = dcm.func_nonlinear(freqs, Q, QC, F0, 0.1, Knl_eff=0, gamma_nl_eff=0, nonlinear=True)

    assert np.allclose(nonlinear, linear)


def test_nonlinear_initial_guess_adds_params():
    freqs = make_freqs()
    sdata = DCM.func(freqs, Q, QC, F0, 0.0)
    params = DCM().find_initial_guess_nonlinear(freqs, sdata)

    assert 'Knl_eff' in params
    assert 'gamma_nl_eff' in params


def test_nonlinear_not_supported_raises():
    with pytest.raises(NotImplementedError):
        LCE().create_model_nonlinear()
    with pytest.raises(NotImplementedError):
        ReflectionMode().find_initial_guess_nonlinear(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
