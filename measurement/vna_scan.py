from scresonators.measurement.ZNB import ZNB20
from scresonators.measurement.VNA_funcs import plot_all
from scresonators.measurement.datamanagement import SlabFile
from scresonators.measurement.helpers import get_homophase
import numpy as np
import os
import datetime


def do_vna_scan_consolidated(
    VNA,
    file_name,
    expt_path,
    cfg,
    scan_type="standard",
    spar="s21",
    warm_att=0,
    cold_att=0,
    plot=True,
):
    """
    Perform a VNA scan and save the data to a file.
    This consolidated function supports three scan types:
    - "standard": Linear frequency sweep
    - "segments": Three segments with different point densities
    - "single_point": Specific frequency points determined by get_homophase

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
        - npoints: Number of frequency points (for standard scan)
        - npoints1: Number of frequency points for outer segments (for segments scan)
        - npoints2: Number of frequency points for center segment (for segments scan)
        - power: VNA output power in dBm
        - bandwidth: Measurement bandwidth in Hz
        - averages: Number of averages
        - kappa: Linewidth in MHz (for single_point scan)
    scan_type : str, optional
        Type of scan to perform: "standard", "segments", or "single_point"
    spar : str, optional
        Scattering parameter (e.g., 'S21'), default is "s21"
    att : float, optional
        Attenuation value in dB for standard scan, default is 0
    warm_att : float, optional
        Warm attenuation value in dB for segments and single_point scans, default is 0
    cold_att : float, optional
        Cold attenuation value in dB for segments and single_point scans, default is 0
    plot : bool, optional
        Whether to plot the data, default is True

    Returns:
    --------
    dict
        Dictionary containing the measurement data
    """
    try:
        # Common variables
        power = cfg["power"]
        bandwidth = cfg["bandwidth"]
        trace_name = ("trace1",)
        scattering_parameter = (spar,)

        power_at_device = power - warm_att - cold_att
        tstart = datetime.datetime.now()

        # Generate frequency points based on scan type
        if scan_type == "standard":
            # Calculate frequency range
            freq_start = cfg["freq_center"] - 0.5 * cfg["span"]
            freq_stop = cfg["freq_center"] + 0.5 * cfg["span"]

            # Generate frequency sweep points
            freq_sweep = np.linspace(freq_start, freq_stop, cfg["npoints"])

            # Configure VNA for standard scan
            VNA.initialize_one_tone_spectroscopy(
                trace_name, scattering_parameter, spec_type="lin"
            )
            VNA.set_startfrequency(freq_start)
            VNA.set_stopfrequency(freq_stop)
            VNA.set_points(cfg["npoints"])
            VNA.set_power(power)
            VNA.set_measBW(bandwidth)

        elif scan_type == "segments":
            freq_center = cfg["freq_center"]

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

            # Define sweep type
            swp_typ = "dwell"

            # Configure VNA segments
            npoints = cfg["npoints1"] + cfg["npoints2"] + cfg["npoints1"]
            time_expected = 1 / bandwidth * npoints * cfg["averages"]
            print(f"expected time: {time_expected / 60} min")
            t = 1 / bandwidth / 6

            # Initialize VNA for segmented scan
            VNA.initialize_one_tone_spectroscopy(
                trace_name, scattering_parameter, spec_type="segm"
            )
            # Segment 0: Lower frequency range
            VNA.define_segment(
                1,
                freq_center - cfg["span"] / 2,
                freq_center - wid / 2 * cfg["span"] - pt_spc,
                cfg["npoints1"],
                power,
                t,
                bandwidth,
                set_time=swp_typ,
            )

            # Segment 1: Center frequency range (higher resolution)
            VNA.define_segment(
                2,
                freq_center - wid * cfg["span"] / 2,
                freq_center + wid * cfg["span"] / 2,
                cfg["npoints2"],
                power,
                t,
                bandwidth,
                set_time=swp_typ,
            )

            # Segment 2: Upper frequency range
            VNA.define_segment(
                3,
                freq_center + wid / 2 * cfg["span"] + pt_spc,
                freq_center + cfg["span"] / 2,
                cfg["npoints1"],
                power,
                t,
                bandwidth,
                set_time=swp_typ,
            )
        elif scan_type == "single_point":
            # Set required parameters for get_homophase
            if "kappa_inc" not in cfg:
                cfg["kappa_inc"] = 1.1
            if "kappa" in cfg and not isinstance(cfg["kappa"], float):
                cfg["kappa"] = float(cfg["kappa"])
            if "kappa" in cfg:
                cfg["kappa"] = cfg["kappa"] * 1e6

            # Get frequency list from homophase function
            freq_sweep = get_homophase(cfg)

            # Initialize VNA for single point scan
            # Start by deleting all previous segments
            VNA.delete_segments()

            # Define sweep type
            swp_typ = "dwell"

            # Configure VNA segments for each frequency point
            t = 1 / bandwidth / 12

            for i, freq in enumerate(freq_sweep):
                VNA.define_segment(
                    i + 1,
                    freq,
                    freq,
                    1,
                    power,
                    t,
                    bandwidth,
                    set_time=swp_typ,
                )

            VNA.initialize_one_tone_spectroscopy(
                trace_name, scattering_parameter, spec_type="segm"
            )

        # Set averaging parameters
        VNA.set_averages(cfg["averages"])
        VNA.set_averagestatus(status="on")
        VNA.set_sweeps(cfg["averages"])
        # Perform measurement (common for all scan types)
        VNA.measure()

        # Get measurement data (common for all scan types)
        [amps, phases] = VNA.get_traces(trace_name)[0]

        # Record finish time for segments scan
        if scan_type == "segments":
            tfinish = datetime.datetime.now()
            elapsed_time = (tfinish - tstart).total_seconds()
            print(f"Time elapsed: {elapsed_time / 60} min")

        # Create data dictionary with timestamp (common for all scan types)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if "slope" in cfg:
            phase_corr = np.unwrap(phases) - cfg["slope"] * freq_sweep
            phase_corr = phase_corr - np.mean(phase_corr)
        else:
            phase_corr = np.unwrap(phases)
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

        # Save data to file (common for all scan types)
        file_path = os.path.join(expt_path, file_name)
        with SlabFile(file_path, "w") as f:
            # Save arrays using add_data
            # Note: standard scan uses "fpts", segments and single_point use "freqs"
            if scan_type == "standard":
                f.add_data(f, "fpts", freq_sweep)
            else:
                f.add_data(
                    f, "freqs" if scan_type == "segments" else "fpts", freq_sweep
                )

            f.add_data(f, "mags", amps)
            f.add_data(f, "phases", phase_corr)
            f.add_data(f, "phases_raw", phases)

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

        # Plot data if requested (common for all scan types)
        if plot:
            plot_all(data, filepath=expt_path)

        return data

    except Exception as e:
        print(f"Error in do_vna_scan_consolidated ({scan_type}): {str(e)}")
        raise


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
    return do_vna_scan_consolidated(
        VNA,
        file_name,
        expt_path,
        cfg,
        scan_type="standard",
        spar=spar,
        plot=plot,
    )


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
    return do_vna_scan_consolidated(
        VNA,
        file_name,
        expt_path,
        cfg,
        scan_type="segments",
        spar=spar,
        warm_att=warm_att,
        cold_att=cold_att,
        plot=plot,
    )


def do_vna_scan_single_point(
    VNA, file_name, expt_path, cfg, spar="s21", warm_att=0, cold_att=0, plot=True
):
    """
    Perform a VNA scan with specific frequency points determined by get_homophase.

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
        - kappa: Linewidth in MHz
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
    return do_vna_scan_consolidated(
        VNA,
        file_name,
        expt_path,
        cfg,
        scan_type="single_point",
        spar=spar,
        warm_att=warm_att,
        cold_att=cold_att,
        plot=plot,
    )
