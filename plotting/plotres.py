import matplotlib.pyplot as plt
import skrf as rf
import numpy as np


#It might make sense to package all this into a subclass of matplotlib.Figure
#TODO: need to improve label placement when specified by user

def makeSummaryFigure():
    fig = plt.figure(layout='tight', figsize=(8, 6.5))
    # Modified layout: smith gets 2/3 width, mag/phase share 1/3 width
    ax = fig.subplot_mosaic([['circle', 'circle', 'mag'], ['circle', 'circle', 'phase']])
    fig.parameterAnnotation = None

    ax['mag'].sharex(ax['phase'])
    ax['phase'].set_xlabel('$f-f_0$ (kHz)')
    ax['mag'].tick_params(labelbottom = False)
    ax['mag'].set_aspect('auto')
    ax['circle'].set_aspect('equal')
    ax['phase'].set_aspect('auto')
    #fig.subplots_adjust(hspace=0)#overridden by using the layout=constrain option in plt.figure()
    return fig, ax

def smith(sdata, **kwargs):
    fig = plt.gcf()
    ax_list = fig.get_axes()
    ax = AxesListToDict(ax_list)

    rf.plotting.plot_smith(sdata, ax=ax['smith'], x_label=None, y_label=None, title='Smith Chart', **kwargs)
    return fig, ax

#TODO: add support for linear magnitude
def magnitude(fdata, sdata, **kwargs):
    fig = plt.gcf()
    ax_list = fig.get_axes()
    ax = AxesListToDict(ax_list)

    ax['mag'].plot(fdata, 20 * np.log10(np.abs(sdata)), **kwargs)
    ax['mag'].set_ylabel('Magnitude (dB)')
    return fig, ax

#TODO: add support for degrees
def phase(fdata, sdata, **kwargs):
    fig = plt.gcf()
    ax_list = fig.get_axes()
    ax = AxesListToDict(ax_list)

    ax['phase'].plot(fdata, np.unwrap(np.angle(sdata)), **kwargs)
    ax['phase'].set_ylabel('phase (rad)')
    return fig, ax


def summaryPlot(fdata, sdata, **kwargs):
    '''
    This function combines plotres.circle() (not added yet), .magnitude(), and .phase() functionality, passing **kwargs to
    the relevant matplotlib function
    '''
    fig = plt.gcf()
    ax_list = fig.get_axes()
    ax = AxesListToDict(ax_list)
    #fdata = 1e6*(fdata - np.mean(fdata)) #center frequency data around 0 for better smith chart plotting
    #rf.plotting.plot_smith(sdata, ax=ax['smith'], x_label=None, y_label=None, title='Smith Chart', **kwargs)
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


def annotate(annotation_text: str):
    fig = plt.gcf()
    ax_list = fig.get_axes()
    ax = AxesListToDict(ax_list)

    if fig.parameterAnnotation == None:
        fig.parameterAnnotation = ax['circle'].annotate(str(annotation_text), (-1, -1.2), annotation_clip=False)
    else:
        text = fig.parameterAnnotation.get_text()
        text = text + str(annotation_text)
        fig.parameterAnnotation.set_text(text)

        x_pos, y_pos = fig.parameterAnnotation.get_position()
        fig.parameterAnnotation.set_position((x_pos, y_pos - 0.125))
        # TODO: query the font height & line spacing for the y-position adjustment


def annotateParam(param, name):
    fig = plt.gcf()
    ax_list = fig.get_axes()
    ax = AxesListToDict(ax_list)
    val = param.value
    stderr = param.stderr
    val, stderr = round_measured_value(val, stderr)

    # Format based on the ratio of stderr to val
        
    if val!=0 and abs(stderr/val) >= 1/500:  # stderr is within a factor of 100 of val
        if abs(val) < 1000:  # Don't use scientific notation for numbers < 1000
            param_text = f'{name}: {val:.3f} $\\pm$ {stderr:.3f}'
        else:
            # Format as (val ± stderr) × 10^n
            exponent = int(np.floor(np.log10(abs(val))))
            val_scaled = val / (10**exponent)
            stderr_scaled = stderr / (10**exponent)
            param_text = f'{name}: ({val_scaled:.3f} $\\pm$ {stderr_scaled:.3f}) $\\times$ 10$^{{{exponent}}}$'
    else:  # stderr is more than factor of 100 smaller
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

    if param.name=='phi':
        param_text = param_text + ' rad'
    elif param.name=='f0':
        param_text = param_text + ' GHz'
    
    # Initialize parameter tracking if not exists
    if not hasattr(fig, 'parameterAnnotations'):
        fig.parameterAnnotations = {'col1': [], 'col2': [], 'col3': []}
        fig.parameterCount = 0
    
    # Add parameter to appropriate column (cycling through 3 columns)
    col_num = (fig.parameterCount % 2) + 1
    col_key = f'col{col_num}'
    fig.parameterAnnotations[col_key].append(param_text)
    fig.parameterCount += 1
    
    # Clear existing annotations
    if hasattr(fig, 'annotationObjects'):
        for ann in fig.annotationObjects:
            ann.remove()
    fig.annotationObjects = []
    
    # Create 3-column layout spanning both smith and phase axes
    # Column positions: left (smith left), center (smith center), right (smith right/phase)
    # col_positions = np.array([0.19, 0.48, 0.76])+0.04  # x positions in figure coordinates
    
    # for i, (col_key, col_pos) in enumerate(zip(['col1', 'col2', 'col3'], col_positions)):
    #     if fig.parameterAnnotations[col_key]:  # Only create annotation if column has content
    #         col_text = '\n'.join(fig.parameterAnnotations[col_key])
    #         ann = fig.text(col_pos, -0.0, col_text,  # Use figure coordinates
    #                       ha='center', va='bottom',
    #                       fontsize=11,
    #                       transform=fig.transFigure)
    #         fig.annotationObjects.append(ann)

   
    col_positions = np.array([0.16, 0.5])+0.04  # x positions in figure coordinates

    for i, (col_key, col_pos) in enumerate(zip(['col1', 'col2'], col_positions)):
        if fig.parameterAnnotations[col_key]:  # Only create annotation if column has content
            col_text = '\n'.join(fig.parameterAnnotations[col_key])
            ann = fig.text(col_pos, 0.03, col_text,  # Use figure coordinates
                          ha='center', va='bottom',
                          fontsize=11,
                          transform=fig.transFigure)
            fig.annotationObjects.append(ann)
            

def displayAllParams(parameters):
    for key in parameters:
        annotateParam(parameters[key])


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

