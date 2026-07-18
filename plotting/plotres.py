import matplotlib.pyplot as plt
import numpy as np


#TODO: need to improve label placement when specified by user

def makeSummaryFigure():
    fig = plt.figure(layout='tight', figsize=(8, 6.5))
    # Modified layout: circle gets 2/3 width, mag/phase share 1/3 width
    ax = fig.subplot_mosaic([['circle', 'circle', 'mag'], ['circle', 'circle', 'phase']])
    fig._scres_annotations = {'text': None, 'columns': {'col1': [], 'col2': []}, 'objects': []}

    ax['mag'].sharex(ax['phase'])
    ax['phase'].set_xlabel('$f-f_0$ (kHz)')
    ax['mag'].tick_params(labelbottom = False)
    ax['mag'].set_aspect('auto')
    ax['circle'].set_aspect('equal')
    ax['phase'].set_aspect('auto')
    return fig, ax


def _figure_axes(fig=None):
    if fig is None:
        fig = plt.gcf()
    return fig, AxesListToDict(fig.get_axes())


#TODO: add support for linear magnitude
def magnitude(fdata, sdata, fig=None, **kwargs):
    fig, ax = _figure_axes(fig)

    ax['mag'].plot(fdata, 20 * np.log10(np.abs(sdata)), **kwargs)
    ax['mag'].set_ylabel('Magnitude (dB)')
    return fig, ax

#TODO: add support for degrees
def phase(fdata, sdata, fig=None, **kwargs):
    fig, ax = _figure_axes(fig)

    ax['phase'].plot(fdata, np.unwrap(np.angle(sdata)), **kwargs)
    ax['phase'].set_ylabel('phase (rad)')
    return fig, ax


def summaryPlot(fdata, sdata, fig=None, **kwargs):
    '''
    Plots the scattering data on the circle, magnitude, and phase axes of a
    summary figure, passing **kwargs to the matplotlib plot calls.
    '''
    fig, ax = _figure_axes(fig)
    ax['circle'].plot(np.real(sdata), np.imag(sdata), **kwargs)
    ax['circle'].set_xlabel('Re($S_{21}$)')
    ax['circle'].set_ylabel('Im($S_{21}$)')
    ax['mag'].plot(fdata, 20*np.log10(np.abs(sdata)), **kwargs)
    ax['mag'].set_ylabel('Magnitude (dB)')
    ax['phase'].plot(fdata, np.unwrap(np.angle(sdata)), **kwargs)
    ax['phase'].set_ylabel('Phase (rad)')
    ax['circle'].axhline(0, color='gray', linestyle='-', linewidth=1)
    ax['circle'].axvline(1, color='gray', linestyle='-', linewidth=1)

    return fig, ax


def annotate(annotation_text: str, fig=None):
    fig, ax = _figure_axes(fig)
    annotations = _annotation_state(fig)

    if annotations['text'] is None:
        annotations['text'] = ax['circle'].annotate(str(annotation_text), (-1, -1.2), annotation_clip=False)
    else:
        text = annotations['text'].get_text()
        annotations['text'].set_text(text + str(annotation_text))

        x_pos, y_pos = annotations['text'].get_position()
        annotations['text'].set_position((x_pos, y_pos - 0.125))
        # TODO: query the font height & line spacing for the y-position adjustment


def format_param_text(name, value, stderr, units=''):
    '''
    Formats a measured value and its error as "name: value ± stderr" (LaTeX),
    rounding to the error's significant figures and using scientific notation
    for large values.
    '''
    val, stderr = round_measured_value(value, stderr)

    # Format based on the ratio of stderr to val
    if val != 0 and abs(stderr/val) >= 1/500:  # stderr is within a factor of 500 of val
        if abs(val) < 1000:  # Don't use scientific notation for numbers < 1000
            param_text = f'{name}: {val:.3f} $\\pm$ {stderr:.3f}'
        else:
            # Format as (val ± stderr) × 10^n
            exponent = int(np.floor(np.log10(abs(val))))
            val_scaled = val / (10**exponent)
            stderr_scaled = stderr / (10**exponent)
            param_text = f'{name}: ({val_scaled:.3f} $\\pm$ {stderr_scaled:.3f}) $\\times$ 10$^{{{exponent}}}$'
    else:  # stderr is much smaller than val
        if abs(val) < 1000 and abs(stderr) < 1000:  # Don't use scientific notation for numbers < 1000
            param_text = f'{name}: {val:.3f} $\\pm$ {stderr:.3f}'
        else:
            # Format as val × 10^n1 ± stderr × 10^n2, but check each component
            if abs(val) < 1000:
                val_part = f'{val:.3f}'
            else:
                val_exp = int(np.floor(np.log10(abs(val))))
                val_scaled = val / (10**val_exp)
                val_part = f'{val_scaled:.3f} $\\times$ 10$^{{{val_exp}}}$'

            if abs(stderr) < 1000:
                stderr_part = f'{stderr:.3f}'
            else:
                stderr_exp = int(np.floor(np.log10(abs(stderr))))
                stderr_scaled = stderr / (10**stderr_exp)
                stderr_part = f'{stderr_scaled:.3f} $\\times$ 10$^{{{stderr_exp}}}$'

            param_text = f'{name}: {val_part} $\\pm$ {stderr_part}'

    if units:
        param_text = param_text + ' ' + units

    return param_text


_PARAM_UNITS = {'phi': 'rad', 'f0': 'GHz'}


def _annotation_state(fig):
    if not hasattr(fig, '_scres_annotations'):
        fig._scres_annotations = {'text': None, 'columns': {'col1': [], 'col2': []}, 'objects': []}
    return fig._scres_annotations


def annotateParam(param, name=None, fig=None):
    '''
    Adds a parameter annotation below the summary figure, alternating between
    two columns. `name` defaults to the lmfit parameter's own name.
    '''
    fig, ax = _figure_axes(fig)
    annotations = _annotation_state(fig)
    if name is None:
        name = param.name

    param_text = format_param_text(name, param.value, param.stderr,
                                   units=_PARAM_UNITS.get(param.name, ''))

    # Add parameter to appropriate column (alternating between the columns)
    columns = annotations['columns']
    count = sum(len(col) for col in columns.values())
    col_key = f'col{(count % 2) + 1}'
    columns[col_key].append(param_text)

    # Clear existing annotations and re-lay-out both columns
    for ann in annotations['objects']:
        ann.remove()
    annotations['objects'] = []

    col_positions = np.array([0.16, 0.5])+0.04  # x positions in figure coordinates

    for col_key, col_pos in zip(['col1', 'col2'], col_positions):
        if columns[col_key]:  # Only create annotation if column has content
            col_text = '\n'.join(columns[col_key])
            ann = fig.text(col_pos, 0.03, col_text,  # Use figure coordinates
                          ha='center', va='bottom',
                          fontsize=11,
                          transform=fig.transFigure)
            annotations['objects'].append(ann)


def displayAllParams(parameters, names=None, fig=None):
    '''
    Annotates the summary figure with a set of fit parameters.

    Args:
        parameters: lmfit.Parameters from a fit
        names: which parameters to display; defaults to the interesting subset
               (Q, Qi, Qc_real or Qc, f0, phi) of those present
    '''
    if names is None:
        names = ['Q', 'Qi', 'Qc_real' if 'Qc_real' in parameters else 'Qc', 'f0', 'phi']
        names = [name for name in names if name in parameters]
    for name in names:
        annotateParam(parameters[name], name=name, fig=fig)


def AxesListToDict(ax_list):
    '''
    utility function to convert a list of matplotlib axes to a dictionary of them indexed by their label
    '''
    ax_dict = dict()
    for n in range(len(ax_list)):
        ax_dict.update({ax_list[n]._label: ax_list[n]})
    return ax_dict

#TODO: more careful verification of this function -- Google's AI gave it to me quicker than stackexchange
def round_measured_value(value, stdev):
    '''
    Rounding for measured quantities
    Two significant figures for the error
    value rounded to line up with first digit in the error
    '''
    if stdev is not None and not np.isinf(stdev) and stdev != 0:
        place = int(np.floor(np.log10(stdev)))
        rounded_value = round(value, -place)
        rounded_err = round(stdev, -(place-1))
    else:
        rounded_value = value
        rounded_err = 0
    return rounded_value, rounded_err
