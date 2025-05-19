from scresonators.measurement.ZNB import ZNB20
from scresonators.measurement.VNA_funcs import *
from scresonators.measurement.datamanagement import SlabFile
from scresonators.measurement.helpers import get_homophase
import numpy as np
import os
import datetime



def do_vna_scan(VNA, file_name, expt_path, cfg, spar="s21", att=0, plot=True):
    """
    Perform a VNA scan and save the data to a file.

    Parameters:
    -----------
    VNA : ZNB object
        The VNA instrument object
    file_name : str
        Name for saving the data
    expt_path : str
        Path for saving the data
    cfg : dict
        Configuration dictionary with measurement parameters:
        - freq_center: Center frequency in MHz
        - span: Frequency span in MHz
        - npoints: Number of frequency points
        - power: VNA output power in dBm
        - bandwidth: Measurement bandwidth in Hz
        - averages: Number of averages
    spar : str
        Scattering parameter (e.g., 'S21')
    att : float, optional
        Attenuation value in dB, default is 0
    plot : bool, optional
        Whether to plot the data, default is True

    Returns:
    --------
    dict
        Dictionary containing the measurement data
    """
    try:
        # Calculate frequency range
        freq_center = cfg["freq_center"]
        freq_span = cfg["span"]
        freq_start = freq_center - 0.5 * freq_span
        freq_stop = freq_center + 0.5 * freq_span

        # Generate frequency sweep points
        freq_sweep = np.linspace(freq_start, freq_stop, cfg["npoints"])

        # Get power setting
        power = cfg["power"]

        # Prepare trace and scattering parameter
        trace_name = ("trace1",)
        scattering_parameter = (spar,)

        # Configure VNA
        VNA.initialize_one_tone_spectroscopy(trace_name, scattering_parameter)
        VNA.set_startfrequency(freq_start)
        VNA.set_stopfrequency(freq_stop)
        VNA.set_points(cfg["npoints"])
        VNA.set_power(power)
        VNA.set_measBW(cfg["bandwidth"])
        VNA.set_sweeps(cfg["averages"])
        VNA.set_averages(cfg["averages"])
        VNA.set_averagestatus(status="on")

        # Calculate actual power at device
        power_at_device = power - att

        # Perform measurement
        VNA.measure()

        # Get measurement data
        [amps, phases] = VNA.get_traces(trace_name)[0]

        # Create data dictionary with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            "series": timestamp,
            "amps": amps,
            "phases": phases,
            "freqs": freq_sweep,
            "vna_power": power,
            "power_at_device": power_at_device,
            "bandwidth": cfg["bandwidth"],
            "averages": cfg["averages"],
            "npoints": cfg["npoints"],
        }
        tfinish = datetime.datetime.now()
        # print(f"Time elapsed: {(tfinish-tstart)/60} min, expected time: {time_expected/60} min")
        # Save data to file using native SlabFile methods
        file_path = os.path.join(expt_path, file_name)
        with SlabFile(file_path, "w") as f:
            # Save arrays using add_data
            f.add_data(f, "fpts", freq_sweep)
            f.add_data(f, "mags", amps)
            f.add_data(f, "phases", phases)

            # Save scalar values as a dictionary
            metadata = {
                "vna_power": power,
                "power_at_device": power_at_device,
                "averages": cfg["averages"],
                "bandwidth": cfg["bandwidth"],
                "npoints": cfg["npoints"],
                "timestamp": timestamp,
            }
            f.save_dict(metadata)

        # Plot data if requested
        if plot:
            plot_all(data, filepath=expt_path)

        return data

    except Exception as e:
        print(f"Error in do_vna_scan: {str(e)}")
        raise

def do_vna_scan_segments(
    VNA, file_name, expt_path, cfg, spar="s21", warm_att=0, cold_att=0, plot=True
):
    """
    Perform a VNA scan with segmented frequency ranges and save the data to a file.

    This function divides the frequency span into three segments:
    1. Lower frequency range with fewer points
    2. Center frequency range with more points (higher resolution)
    3. Upper frequency range with fewer points

    Parameters:
    -----------
    VNA : ZNB object
        The VNA instrument object
    file_name : str
        Name for saving the data
    expt_path : str
        Path for saving the data
    cfg : dict
        Configuration dictionary with measurement parameters:
        - freq_center: Center frequency in MHz
        - span: Frequency span in MHz
        - npoints1: Number of frequency points for outer segments
        - npoints2: Number of frequency points for center segment
        - power: VNA output power in dBm
        - bandwidth: Measurement bandwidth in Hz
        - averages: Number of averages
        - npoints: Total number of points (for metadata)
    warm_att : float, optional
        Warm attenuation value in dB, default is 0
    cold_att : float, optional
        Cold attenuation value in dB, default is 0
    plot : bool, optional
        Whether to plot the data, default is True

    Returns:
    --------
    dict
        Dictionary containing the measurement data
    """
    try:
        # Get frequency center and bandwidth
        freq_center = cfg["freq_center"]
        bandwidth = cfg["bandwidth"]
        power = cfg["power"]

        # Prepare trace and scattering parameter
        scattering_parameter = (spar,)
        trace_name = ("trace1",)

        # Define segment width ratio (center segment width / total span)
        wid = 0.2

        # Calculate point spacing for smooth transitions between segments
        pt_spc = cfg["span"] * (1 - wid) / 2 / cfg["npoints1"]

        # Generate frequency points for each segment
        freq_sweep0 = np.linspace(
            freq_center - cfg["span"] / 2,
            freq_center - wid / 2 * cfg["span"] - pt_spc,
            cfg["npoints1"],
        )
        freq_sweep1 = np.linspace(
            freq_center - wid * cfg["span"] / 2,
            freq_center + wid * cfg["span"] / 2,
            cfg["npoints2"],
        )
        freq_sweep2 = np.linspace(
            freq_center + wid / 2 * cfg["span"] + pt_spc,
            freq_center + cfg["span"] / 2,
            cfg["npoints1"],
        )

        # Combine all frequency points
        freq_sweep = np.concatenate((freq_sweep0, freq_sweep1, freq_sweep2))

        # Initialize VNA
        VNA.initialize_one_tone_spectroscopy_seg(trace_name, scattering_parameter)

        # Define sweep type
        swp_typ = "dwell"

        # Configure VNA segments
        # Segment 0: Lower frequency range
        npoints = cfg["npoints1"] + cfg["npoints2"] + cfg["npoints1"]
        time_expected = 1 / cfg["bandwidth"] * npoints * cfg["averages"]
        print(f"expected time: {time_expected / 60} min")
        t = 1 / cfg["bandwidth"] / 6
        tstart = datetime.datetime.now()
        VNA.define_segment(
            1,
            freq_center - cfg["span"] / 2,
            freq_center - wid / 2 * cfg["span"] - pt_spc,
            cfg["npoints1"],
            cfg["power"],
            t,
            cfg["bandwidth"],
            set_time=swp_typ,
        )

        # Segment 1: Center frequency range (higher resolution)
        VNA.define_segment(
            2,
            freq_center - wid * cfg["span"] / 2,
            freq_center + wid * cfg["span"] / 2,
            cfg["npoints2"],
            cfg["power"],
            t,
            cfg["bandwidth"],
            set_time=swp_typ,
        )

        # Segment 2: Upper frequency range
        VNA.define_segment(
            3,
            freq_center + wid / 2 * cfg["span"] + pt_spc,
            freq_center + cfg["span"] / 2,
            cfg["npoints1"],
            cfg["power"],
            t,
            cfg["bandwidth"],
            set_time=swp_typ,
        )

        # Set averaging parameters
        VNA.set_averages(cfg["averages"])
        VNA.set_averagestatus(status="on")
        VNA.set_sweeps(cfg["averages"])

        # Calculate actual power at device
        power_at_device = cfg["power"] - warm_att - cold_att

        # Perform measurement
        VNA.measure()

        # Get measurement data
        [amps, phases] = VNA.get_traces(trace_name)[0]
        tfinish = datetime.datetime.now()
        elapsed_time = (tfinish - tstart).total_seconds()
        print(f"Time elapsed: {elapsed_time / 60} min")

        # Create data dictionary with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            "series": timestamp,
            "amps": amps,
            "phases": phases,
            "freqs": freq_sweep,
            "vna_power": power,
            "power_at_device": power_at_device,
            "bandwidth": bandwidth,
            "averages": cfg["averages"],
            "npoints": cfg["npoints"],
        }

        # Save data to file using native SlabFile methods
        file_path = os.path.join(expt_path, file_name)
        with SlabFile(file_path, "w") as f:
            # Save arrays using add_data
            f.add_data(f, "freqs", freq_sweep)
            f.add_data(f, "amps", amps)
            f.add_data(f, "phases", phases)

            # Save scalar values as a dictionary
            metadata = {
                "vna_power": power,
                "power_at_device": power_at_device,
                "averages": cfg["averages"],
                "bandwidth": bandwidth,
                "npoints": cfg["npoints"],
                "timestamp": timestamp,
            }
            f.save_dict(metadata)

        # Plot data if requested
        if plot:
            plot_all(data, filepath=expt_path)

        return data

    except Exception as e:
        print(f"Error in do_vna_scan_segments: {str(e)}")
        raise


def do_vna_scan_single_point(
    VNA, file_name, expt_path, cfg, spar="s21", warm_att=0, cold_att=0, plot=True
):
    """
    Perform a VNA scan with segmented frequency ranges and save the data to a file.

    This function divides the frequency span into three segments:
    1. Lower frequency range with fewer points
    2. Center frequency range with more points (higher resolution)
    3. Upper frequency range with fewer points

    Parameters:
    -----------
    VNA : ZNB object
        The VNA instrument object
    file_name : str
        Name for saving the data
    expt_path : str
        Path for saving the data
    cfg : dict
        Configuration dictionary with measurement parameters:
        - freq_center: Center frequency in MHz
        - span: Frequency span in MHz
        - npoints: Number of frequency points for outer segments
        - power: VNA output power in dBm
        - bandwidth: Measurement bandwidth in Hz
        - averages: Number of averages
        - npoints: Total number of points (for metadata)
    warm_att : float, optional
        Warm attenuation value in dB, default is 0
    cold_att : float, optional
        Cold attenuation value in dB, default is 0
    plot : bool, optional
        Whether to plot the data, default is True

    Returns:
    --------
    dict
        Dictionary containing the measurement data
    """
    try:
        # Get frequency center and bandwidth

        bandwidth = cfg["bandwidth"]
        power = cfg["power"]
        cfg["kappa_inc"] = 1.1
        cfg["kappa"] = cfg['kappa']*1e6

        freq_list = get_homophase(cfg)

        # Prepare trace and scattering parameter
        scattering_parameter = (spar,)
        trace_name = ("trace1",)

        # Initialize VNA
        # start by deleting all previous segments.
        VNA.delete_segments()

        # Define sweep type
        swp_typ = "dwell"

        # Configure VNA segments
        # Segment 0: Lower frequency range

        t = 1 / cfg["bandwidth"] / 6

        for i, freq in enumerate(freq_list):
            VNA.define_segment(
                i + 1,
                freq,
                freq,
                1,
                cfg["power"],
                t,
                cfg["bandwidth"],
                set_time=swp_typ,
            )
        VNA.initialize_one_tone_spectroscopy_seg(trace_name, scattering_parameter)
        # Set averaging parameters
        VNA.set_averages(cfg["averages"])
        VNA.set_averagestatus(status="on")
        VNA.set_sweeps(cfg["averages"])

        # Calculate actual power at device
        power_at_device = cfg["power"] - warm_att - cold_att

        # Perform measurement
        VNA.measure()

        # Get measurement data
        [amps, phases] = VNA.get_traces(trace_name)[0]

        # Create data dictionary with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            "series": timestamp,
            "amps": amps,
            "phases": phases,
            "freqs": freq_list,
            "vna_power": power,
            "power_at_device": power_at_device,
            "bandwidth": bandwidth,
            "averages": cfg["averages"],
            "npoints": cfg["npoints"],
        }

        # Save data to file
        file_path = os.path.join(expt_path, file_name)

        with SlabFile(file_path, "w") as f:
            # Save arrays using add_data
            f.add_data(f, "fpts", freq_list)
            f.add_data(f, "mags", amps)
            f.add_data(f, "phases", phases)

            # Save scalar values as a dictionary
            metadata = {
                "vna_power": power,
                "averages": cfg["averages"],
                "bandwidth": bandwidth,
                "npoints": cfg["npoints"],
                "timestamp": timestamp,
            }
            f.save_dict(metadata)
        #           f.append_pt["attenuation"] = {"warm": warm_att, "cold": cold_att}

        # Plot data if requested
        if plot:
            plot_all(data, filepath=expt_path)

        return data

    except Exception as e:
        print(f"Error in do_vna_scan_segments: {str(e)}")
        raise