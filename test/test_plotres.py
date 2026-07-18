import matplotlib
matplotlib.use('Agg')

import lmfit
import numpy as np
import pytest

from plotting import plotres


def test_round_measured_value():
    value, err = plotres.round_measured_value(133333.0, 4337.0)
    assert value == 133000.0
    assert err == 4300.0


def test_round_measured_value_no_stderr():
    assert plotres.round_measured_value(1.234, None) == (1.234, 0)
    assert plotres.round_measured_value(1.234, np.inf) == (1.234, 0)
    assert plotres.round_measured_value(1.234, 0) == (1.234, 0)


def test_format_param_text_scientific():
    text = plotres.format_param_text('Q', 133333.0, 4337.0)
    assert text == 'Q: (1.330 $\\pm$ 0.043) $\\times$ 10$^{5}$'


def test_format_param_text_small_value():
    text = plotres.format_param_text('phi', 0.123, 0.01, units='rad')
    assert 'phi:' in text
    assert '$\\pm$' in text
    assert text.endswith('rad')


def test_format_param_text_missing_stderr():
    # stderr None rounds to 0; must not raise
    text = plotres.format_param_text('f0', 4.5, None)
    assert text.startswith('f0:')


def _fit_params():
    params = lmfit.Parameters()
    params.add('Q', value=130_000)
    params.add('Qc', value=200_000)
    params.add('f0', value=4.5)
    params.add('phi', value=0.01)
    for name in params:
        params[name].stderr = abs(params[name].value) * 0.01
    return params


def test_display_all_params_smoke():
    import matplotlib.pyplot as plt
    fig, ax = plotres.makeSummaryFigure()
    try:
        plotres.displayAllParams(_fit_params(), fig=fig)
        assert fig._scres_annotations['objects']
    finally:
        plt.close(fig)


def test_annotate_param_defaults_to_param_name():
    import matplotlib.pyplot as plt
    fig, ax = plotres.makeSummaryFigure()
    try:
        params = _fit_params()
        plotres.annotateParam(params['Q'], fig=fig)
        texts = [obj.get_text() for obj in fig._scres_annotations['objects']]
        assert any('Q:' in text for text in texts)
    finally:
        plt.close(fig)
