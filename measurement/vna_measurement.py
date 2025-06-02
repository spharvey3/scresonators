import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import csv
import copy
from scipy.optimize import curve_fit
from scresonators.fit_resonator import ana_tls
from scresonators.fit_resonator.ana_resonator import ResonatorFitter
from scresonators.fit_resonator.ana_resonator import ResonatorData
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from scresonators.measurement.helpers import n, config_figs
import seaborn as sns
from scresonators.fit_resonator.basic_fit import fit_resonator


@dataclass
class ResonatorMeasurement:
    """
    Data class to store all measurements for a single resonator at a specific power level.
    """

    # Basic measurement parameters
    frequency_amp: float  # Resonance frequency in Hz
    power: float  # Power in dBm
    power_at_device: float  # Power at device in dBm

    # Quality factors
    q_total_amp: float  # Total quality factor
    q_internal_amp: float  # Internal quality factor
    q_coupling_amp: float  # Coupling quality factor

    # Alternative fit results (if available)
    q_total: Optional[float] = None
    q_internal: Optional[float] = None
    q_coupling: Optional[float] = None
    frequency: Optional[float] = None

    # Error parameters for alternative fit (if available)
    q_total_err: Optional[float] = None
    q_internal_err: Optional[float] = None
    q_coupling_err: Optional[float] = None
    frequency_err: Optional[float] = None
    phase_err: Optional[float] = None

    # Measurement details
    kappa: float = 0.0  # Linewidth in Hz
    photon_number: float = 0.0  # Average photon number
    averages: int = 1  # Number of averages used
    fit_parameters: List[float] = field(default_factory=list)  # Raw fit parameters

    # Measurement data
    raw_data: Dict[str, Any] = field(default_factory=dict)  # Raw measurement data


@dataclass
class PowerSweepResult:
    """
    Data class to store results from a power sweep for multiple resonators.
    """

    # Measurement results organized by frequency index and power index
    # Dictionary structure: {freq_idx: {power_idx: ResonatorMeasurement}}
    measurements: Dict[int, Dict[int, ResonatorMeasurement]]

    # Original and current frequencies
    frequencies: List[float]  # Original frequencies
    current_frequencies: List[
        float
    ]  # Current frequencies (may be updated during sweep)
    powers: List[float]  # Power values

    # Configuration used for the sweep
    config: Dict[str, Any]

    # Tracking parameters
    spans: List[float]  # Frequency spans for each resonator
    averaging_factors: List[float]  # Averaging factors for each resonator
    q_adjustment_factors: List[float]  # Q adjustment factors for each resonator
    keep_measuring: List[bool]  # Whether to continue measuring each resonator


def _perform_initial_scan(hw, expt_path, result, freq_idx, power, att, fname, config):
    """
    Perform an initial scan to find the resonance frequency and linewidth.

    Parameters:
    -----------
    hw : ZNB object
        The VNA instrument object
    expt_path : str
        Path for saving data
    result : PowerSweepResult
        Current result object
    freq_idx : int
        Index of the frequency being measured
    power : float
        Power level in dBm
    att : float
        Attenuation value
    fname : str
        Base filename

    Returns:
    --------
    ResonatorMeasurement
        Measurement result for the initial scan
    """
    # Configure VNA scan with wider span
    scan_config = {
        "freq_center": float(result.current_frequencies[freq_idx]),
        "span": float(result.spans[freq_idx]) * 1.3,
        "npoints": 800,
        "power": power,
        "bandwidth": 1000,  # result.config["bandwidth"],
        "averages": 1,
        "slope": config["slope"],
    }

    # Perform VNA scan
    file_name = f"res_{fname}_single.h5"
    if type(hw) is not dict:
        import scresonators.measurement.vna_scan as vna_scan

        data = vna_scan.do_vna_scan(
            hw, file_name, expt_path, scan_config, config["spar"], att=att, plot=False
        )
    else:
        from scresonators.measurement.rfsoc_scan import do_rfsoc_scan

        config = copy.deepcopy(config)
        config["phase_const"] = False
        scan_config["kappa"] = np.nan

        data = do_rfsoc_scan(
            hw, file_name, expt_path, scan_config, config=config, att=att, plot=False
        )

    # Fit resonator to find center frequency and kappa
    min_freq = result.current_frequencies[freq_idx]  # Initial guess
    freq_center, q_total_amp, kappa, fit_params = fit_resonator(data, power, plot=True)

    # Calculate quality factors
    q_internal_amp = fit_params[1] * 1e4
    q_coupling_amp = fit_params[2] * 1e4

    # Calculate photon number
    pin = (
        power
        - result.config["att"]
        - result.config["db_slope"] * (freq_center / 1e9 - result.config["freq_0"])
    )
    photon_number = n(pin, freq_center, q_total_amp, q_coupling_amp)

    # Create and return measurement object
    return ResonatorMeasurement(
        frequency_amp=freq_center,
        power=power,
        power_at_device=power - att,
        q_total_amp=q_total_amp,
        q_internal_amp=q_internal_amp,
        q_coupling_amp=q_coupling_amp,
        kappa=kappa,
        photon_number=photon_number,
        averages=1,
        fit_parameters=fit_params,
        raw_data=data,
    )


def _perform_scan(hw, file_name, expt_path, scan_config, config):
    """
    Perform a scan based on the scan type specified in the config.

    Parameters:
    -----------
    hw : ZNB object or RFSoC object
        The VNA instrument object
    file_name : str
        Name for saving the data
    expt_path : str
        Path for saving data
    scan_config : dict
        scan configuration dictionary
    config : dict
        Main configuration dictionary
    att : float
        Attenuation value

    Returns:
    --------
    dict
        Dictionary containing the measurement data
    """

    if not config["type"] == "rfsoc":
        from scresonators.measurement.vna_scan import do_vna_scan_consolidated

        if config["type"] == "lin":
            scan_type = "standard"
        elif config["type"] == "single":
            scan_type = "single_point"
        else:
            scan_type = "segmented"
        return do_vna_scan_consolidated(
            hw,
            file_name,
            expt_path,
            scan_config,
            scan_type=scan_type,
            spar=config["spar"],
            plot=False,
        )
    # For RFSoC scans
    else:
        return do_rfsoc_scan(
            hw,
            file_name,
            expt_path,
            scan_config,
            config=config,
            att=config["att"],
            plot=False,
        )


def _determine_scan_parameters(config, result, freq_idx, power_idx):
    """
    Determine the scan parameters based on the power index.

    Parameters:
    -----------
    config : dict
        Configuration dictionary
    result : PowerSweepResult
        Current result object
    freq_idx : int
        Index of the frequency being measured
    power_idx : int
        Index of the power being measured

    Returns:
    --------
    tuple
        (npoints, span) - Number of points and span for the scan
    """
    if power_idx < 5:
        return int(np.ceil(1.6 * config["npoints"])), result.spans[freq_idx] * 1.25
    elif power_idx > 0 and "next_time" in globals():
        return config["npoints"], result.spans[freq_idx]
    else:
        return config["npoints"], result.spans[freq_idx]


def _should_stop_measuring(result, freq_idx, next_time):
    """
    Determine if we should stop measuring a frequency.

    Parameters:
    -----------
    result : PowerSweepResult
        Current result object
    freq_idx : int
        Index of the frequency being measured
    next_time : float
        Estimated time for the next measurement in seconds

    Returns:
    --------
    bool
        True if we should stop measuring, False otherwise
    """
    # print(result.q_adjustment_factors[freq_idx])
    # print(next_time)
    low_qother = True
    if low_qother:
        thresh = [0.02, 0.005, -0.05, -0.15]
    else:
        thresh = [0.05, 0.015, -0.0, -0.02]

    return (
        (result.q_adjustment_factors[freq_idx] > 1 - thresh[3] and next_time > 900)
        or (result.q_adjustment_factors[freq_idx] > 1 - thresh[2] and next_time > 1800)
        or (result.q_adjustment_factors[freq_idx] > 1 - thresh[1] and next_time > 2700)
        or (result.q_adjustment_factors[freq_idx] > 1 - thresh[0] and next_time > 3800)
        or next_time > 7200
    )


def _calculate_next_measurement_time(config, result, freq_idx):
    """
    Calculate the estimated time for the next measurement.

    Parameters:
    -----------
    config : dict
        Configuration dictionary
    result : PowerSweepResult
        Current result object
    freq_idx : int
        Index of the frequency being measured

    Returns:
    --------
    float
        Estimated time for the next measurement in seconds
    """
    return (
        1 / config["bandwidth"] * config["npoints"] * result.averaging_factors[freq_idx]
    )


def power_sweep_v2(config, hw):
    """
    Perform a power sweep scan using the do_vna_scan function.
    This is an improved version that uses structured data storage.

    Parameters:
    -----------
    config : dict
        Configuration dictionary with measurement parameters:
        - base_path: Base directory path
        - folder: Folder name for saving data
        - freqs: List of center frequencies to scan
        - nvals: Number of power values to sweep
        - pow_inc: Power increment
        - pow_start: Starting power
        - span_inc: Span increment factor [number of linewidths for scan]
        - kappa_start: Initial kappa value
        - npoints: Number of frequency points
        - bandwidth: Measurement bandwidth in Hz
        - averages: Number of averages
        - att: Attenuation value (optional, default is 0)
        - comment: Optional comment to save as a text file
    VNA : ZNB object
        The VNA instrument object

    Returns:
    --------
    PowerSweepResult
        Object containing all measurement results and parameters
    """
    # Create experiment path
    expt_path = os.path.join(config["base_path"], config["folder"])
    try:
        os.makedirs(expt_path)
    except FileExistsError:
        pass
    except Exception as e:
        print(f"Error creating directory: {str(e)}")
        raise

    # Save comment to a text file if provided
    if "comment" in config:
        comment_file_path = os.path.join(expt_path, "comment.txt")
        try:
            with open(comment_file_path, "w") as f:
                f.write(config["comment"])
            print(f"Comment saved to {comment_file_path}")
        except Exception as e:
            print(f"Error saving comment to file: {str(e)}")

    # Generate power values to sweep
    powers = np.arange(0, config["nvals"]) * config["pow_inc"] + config["pow_start"]

    # Create result object to track state
    result = PowerSweepResult(
        measurements={},
        frequencies=copy.deepcopy(config["freqs"]),
        current_frequencies=copy.deepcopy(config["freqs"]),
        powers=powers,
        config=config,
        spans=config["span_inc"]
        * config["kappa_start"]
        * np.ones(len(config["freqs"])),
        averaging_factors=np.ones(len(config["freqs"])),
        q_adjustment_factors=0.9 * np.ones(len(config["freqs"])),
        keep_measuring=[True] * len(config["freqs"]),
    )

    # Perform power sweep for each frequency
    for freq_idx, freq in enumerate(result.current_frequencies):
        result.measurements[freq_idx] = {}

        # Process each power point for this frequency
        for power_idx, power in enumerate(powers):
            if not result.keep_measuring[freq_idx]:
                continue

            # Set initial averaging for first power point
            if power_idx == 0:
                result.averaging_factors[freq_idx] = 1

            # Create filenames
            fname = f"{result.frequencies[freq_idx]:1.0f}"

            # For first power point, do an initial scan with wider span
            if power_idx == 0:
                # Perform initial scan to find resonance frequency and linewidth
                config["pin"] = power - config["att"]
                measurement = _perform_initial_scan(
                    hw,
                    expt_path,
                    result,
                    freq_idx,
                    power,
                    config.get("att", 0),
                    fname,
                    config,
                )

                # Update frequency and span based on measurement
                result.current_frequencies[freq_idx] = measurement.frequency_amp
                result.spans[freq_idx] = measurement.kappa * config["span_inc"]

                # Store measurement
                result.measurements[freq_idx][power_idx] = measurement

                # Store parameters for next iteration
                prev_q = measurement.q_total_amp
                prev_fit_params = measurement.fit_parameters

            # Use parameters from previous power point
            else:
                prev_measurement = result.measurements[freq_idx][power_idx - 1]
                prev_q = prev_measurement.q_total_amp
                prev_fit_params = prev_measurement.fit_parameters

            # Determine scan parameters based on power index
            npoints, span = _determine_scan_parameters(
                config, result, freq_idx, power_idx
            )

            # Configure VNA scan for this power point
            scan_config = {
                "freq_center": float(result.current_frequencies[freq_idx]),
                "span": span,
                "npoints": npoints,
                "power": power,
                "bandwidth": config["bandwidth"],
                "averages": int(max(result.averaging_factors[freq_idx], 100)),
                "kappa": measurement.kappa / 1e6,
                "slope": config["slope"],
            }

            config["pin"] = (
                power
                - config["att"]
                - config["db_slope"]
                * (measurement.frequency_amp / 1e9 - config["freq_0"])
            )

            # Perform the VNA scan
            tstart = datetime.datetime.now()
            time_expected = (1 / scan_config["bandwidth"] * npoints * scan_config["averages"])

            file_name = f"res_{fname}_{power:.0f}dbm.h5"
            data = _perform_scan(hw, file_name, expt_path, scan_config, config)

            elapsed_time = (datetime.datetime.now() - tstart).total_seconds()
            print( f"Time elapsed: {elapsed_time / 60:.2f} min, expected time: {time_expected / 60:.2f} min")

            result.current_frequencies[freq_idx] = data["freqs"][np.argmin(data["amps"])]

            measurement=_perform_fits(power_idx, freq_idx, result, data, power, prev_fit_params, expt_path, fname, config)

            # Store the measurement
            result.measurements[freq_idx][power_idx] = measurement

            # Save fit parameters to CSV
            _save_fit_to_csv(measurement, freq_idx, power_idx, expt_path)

            # Plot Qi vs power
            _plot_qi_vs_photon(result.measurements, freq_idx, expt_path)

            # Calculate new averaging based on photon number
            if power_idx > 0:
                result.q_adjustment_factors[freq_idx] = measurement.q_total_amp / prev_q

            if "avg_corr" in config:
                tau_prop = (10 ** (-measurement.power_at_device / 10)* (measurement.q_coupling_amp / measurement.q_total_amp) ** 2* 1e-11)
                print(f"Averaging factor: {tau_prop:.3f}")

                result.averaging_factors[freq_idx] = np.round(
                    config["avg_corr"]
                    * tau_prop
                    / result.q_adjustment_factors[freq_idx] ** 2
                )
                print(
                    f"Pin {power - config['att']:.1f}, N photons: {measurement.photon_number:.3g}, navg: {int(result.averaging_factors[freq_idx])}"
                )

            # Determine if we should continue measuring this frequency
            next_time = _calculate_next_measurement_time(config, result, freq_idx)
            print(f"Next time: {next_time / 60:.2f} min, q_adj: {result.q_adjustment_factors[freq_idx]:.3f}")

            if _should_stop_measuring(result, freq_idx, next_time):
                result.keep_measuring[freq_idx] = False
                print(f"Stopping frequency {result.current_frequencies[freq_idx] / 1e9:.5f} GHz")

            # Update span for next measurement
            result.spans[freq_idx] = measurement.kappa * config["span_inc"]

    return result


def _plot_qi_vs_photon(measurements, freq_idx, expt_path):
    """
    Plot internal quality factor vs photon number.

    Parameters:
    -----------
    measurements : dict
        Dictionary of measurements
    freq_idx : int
        Frequency index
    expt_path : str
        Path to save the plot
    """

    config_figs()
    if freq_idx not in measurements:
        return

    # Get data for plotting
    power_indices = sorted(measurements[freq_idx].keys())
    photon_numbers = [measurements[freq_idx][i].photon_number for i in power_indices]
    qi_values = [measurements[freq_idx][i].q_internal_amp for i in power_indices]
    qc_values = [measurements[freq_idx][i].q_coupling_amp for i in power_indices]

    # Get alternative fit data if available
    qi_alt_values = []
    qc_alt_values = []
    for i in power_indices:
        if measurements[freq_idx][i].q_internal is not None:
            qi_alt_values.append(measurements[freq_idx][i].q_internal)
            qc_alt_values.append(measurements[freq_idx][i].q_coupling)

    # Create plot
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))

    # Plot Qi vs photon number
    ax[0].semilogx(photon_numbers, qi_values, "o-", label="Primary fit")
    if qi_alt_values:
        ax[0].semilogx(
            photon_numbers[: len(qi_alt_values)], qi_alt_values, "s--", label="Alt fit"
        )

    # Fit Qi vs power to an exponential if we have enough points
    if len(qi_values) > 6:
        try:
            min_freq = measurements[freq_idx][power_indices[0]].frequency_amp
            q_fitn = lambda n, Qtls0, Qoth, nc, beta: ana_tls.Qtotn(
                n, 0.04, min_freq, Qtls0, Qoth, nc, beta
            )
            p = [np.min(qi_values), np.max(qi_alt_values), 3, 0.4]
            p, err = curve_fit(q_fitn, photon_numbers, qi_alt_values, p0=p)
            ax[0].plot(
                photon_numbers,
                q_fitn(np.array(photon_numbers), *p),
                "r--",
                label="Exp fit",
            )
            # Add fit parameters as text to the plot
            fit_text = (
                f"$Q_{{tls}}$: {p[0]:.2e}\n"
                f"$Q_{{oth}}$: {p[1]:.2e}\n"
                f"$n_c$: {p[2]:.2e}\n"
                f"$\\beta$: {p[3]:.2f}"
            )
            ax[0].text(
                0.95,
                0.05,
                fit_text,
                transform=ax[0].transAxes,
                fontsize=10,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="black"),
            )
            print(f"Fit parameters: {p}")
        except Exception as e:
            print(f"Fit failed: {str(e)}")

    ax[0].set_xlabel("Number of Photons")
    ax[0].set_ylabel("Internal Quality Factor ($Q_i$)")
    ax[0].set_title(
        f"Frequency: {measurements[freq_idx][power_indices[0]].frequency_amp/1e9:.5f} GHz"
    )
    # if len(qi_alt_values) > 0:
    #     ax[0].legend()

    # Plot Qc vs photon number
    ax[1].semilogx(photon_numbers, qc_values, "o-", label="Primary fit")
    if qc_alt_values:
        ax[1].semilogx(
            photon_numbers[: len(qc_alt_values)], qc_alt_values, "s--", label="Alt fit"
        )

    ax[1].set_xlabel("Number of Photons")
    ax[1].set_ylabel("Coupling Quality Factor ($Q_c$)")
    if len(qc_alt_values) > 0:
        ax[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(expt_path, f"Qi_vs_power_{freq_idx}.png"))
    plt.close(fig)

    npl = len(measurements[freq_idx])
    sns.set_palette("crest", npl)
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for i, d in enumerate(np.arange(npl)):
        d = measurements[freq_idx][i].raw_data
        f = (d["freqs"] - np.mean(d["freqs"])) * 1e3
        y = d["amps"]  # -np.max(d['amps'])
        ax[0].plot(f, y, label=f"{i}")
        ax[1].plot(f, d["phases"] - np.mean(d["phases"]), label=f"{i}")
    # ax[0].legend()
    ax[0].set_ylabel("Amplitude (dB)")
    ax[1].set_ylabel("Phase (rad)")
    ax[1].set_xlabel("Frequency Offset (kHz)")
    ax[0].set_title(
        f"Frequency: {measurements[freq_idx][power_indices[0]].frequency_amp:.5f} GHz"
    )
    fig.savefig(os.path.join(expt_path, f"Data_{freq_idx}.png"))
    plt.close(fig)


def _save_fit_to_csv(measurement, freq_idx, power_idx, expt_path):
    """
    Save fit parameters to a CSV file after each measurement.

    Parameters:
    -----------
    measurement : ResonatorMeasurement
        Measurement object containing fit parameters
    freq_idx : int
        Frequency index
    power_idx : int
        Power index
    expt_path : str
        Path to save the CSV file
    """
    import csv

    # Create a filename based on the frequency only (not the scan)
    # Round to nearest kHz to ensure consistent filenames across scans
    freq_mhz = round(measurement.frequency_amp / 1e3) / 1e3  # Round to nearest kHz
    csv_filename = os.path.join(expt_path, f"fit_results_freq_{freq_mhz:.0f}MHz.csv")

    # Define the header and data row
    header = [
        "power_idx",
        "power_dBm",
        "power_at_device_dBm",
        "frequency_amp_Hz",
        "q_total_amp",
        "q_internal_amp",
        "q_coupling_amp",
        "kappa_Hz",
        "photon_number",
        "averages",
        "timestamp",
    ]

    # Add alternative fit parameters to header if available
    if measurement.q_total is not None:
        header.extend(["q_total", "q_internal", "q_coupling", "frequency_Hz"])

        # Add error parameters to header if available
        if measurement.q_internal_err is not None:
            header.extend(
                ["q_internal_err", "q_total_err", "q_coupling_err", "frequency_err", "phase_err"]
            )

    # Create data row
    data_row = [
        power_idx,
        measurement.power,
        measurement.power_at_device,
        measurement.frequency_amp,
        measurement.q_total_amp,
        measurement.q_internal_amp,
        measurement.q_coupling_amp,
        measurement.kappa,
        measurement.photon_number,
        measurement.averages,
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ]

    # Add circle fit parameters to data row if available
    if measurement.q_total is not None:
        data_row.extend(
            [
                measurement.q_total,
                measurement.q_internal,
                measurement.q_coupling,
                measurement.frequency,
            ]
        )

        # Add error parameters to data row if available
        if measurement.q_internal_err is not None:
            data_row.extend(
                [
                    measurement.q_internal_err,
                    measurement.q_total_err,
                    measurement.q_coupling_err,
                    measurement.frequency_err,
                    measurement.phase_err,
                ]
            )

    # Check if file exists to determine if we need to write the header
    file_exists = os.path.isfile(csv_filename)

    # Write to CSV file
    with open(csv_filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow(header)

        # Write data row
        writer.writerow(data_row)

    print(
        f"Saved fit results for frequency {freq_mhz:.6f} MHz, power {measurement.power} dBm to CSV"
    )


def get_default_power_sweep_config(custom_config=None):
    """
    Get default configuration for power_sweep function.

    Parameters:
    -----------
    custom_config : dict, optional
        Dictionary with custom configuration values to override defaults

    Returns:
    --------
    dict
        Configuration dictionary with default values for power_sweep
    """
    # Define default configuration
    default_config = {
        # File paths
        "base_path": "./data",
        "folder": f"power_sweep_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        # Frequency settings
        "freqs": np.array([6]) * 1e9,  # Default center frequency in Hz
        "span_inc": 10,  # Span as multiple of linewidth
        "kappa_start": 30000,  # Initial linewidth estimate in Hz
        # Power sweep settings
        "nvals": 18,  # Number of power points
        "pow_start": -5,  # Starting power in dBm
        "pow_inc": -5,  # Power increment in dB
        # Measurement settings
        "npoints": 201,  # Number of frequency points
        "npoints1": 10,
        "npoints2": 27,
        "bandwidth": 100,  # Measurement bandwidth in Hz
        "averages": 1,  # Number of averages
        "att": 60,  # Attenuation in dB
        "type": "lin",
        "freq_0": 6,
        "db_slope": 4,
        "spar": "S21",
        "slope": 0,
        # Analysis settings
        "avg_corr": 1e6,  # Correction factor for averaging
    }
    if (
        custom_config is not None
        and "type" in custom_config
        and custom_config["type"] == "single"
    ):
        default_config["npoints"] = 31

    # Override defaults with custom values if provided
    if custom_config is not None:
        for key, value in custom_config.items():
            default_config[key] = value

    return default_config


def write_hp_csv(results, config): 
    qhp_list = []
    for i in range(len(results.measurements)):
        q_int = [results.measurements[i][j].q_internal for j in range(len(results.measurements[i]))]
        qhp = np.max(q_int)
        qhp_list.append(np.round(qhp))

    fname = os.path.join(config['base_path'], 'qhp.csv')
    with open(fname, mode='a', newline='') as file:
        writer = csv.writer(file)
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([now] + qhp_list)

def new_hp_csv(path):
    fname = os.path.join(path, 'qhp.csv')
    with open(fname, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', '10','12','14','2','16','4','6','8']) 

def _perform_fits(
    power_idx, freq_idx, result, data, power, prev_fit_params, expt_path, fname, config
):
    if power_idx < 8:
        fitparams = [
            result.current_frequencies[freq_idx],
            prev_fit_params[1],
            prev_fit_params[2],
            prev_fit_params[3],
            np.max(10 ** (data["amps"] / 20)),
        ]
        freq_center, q_total_amp, kappa, fit_params = fit_resonator(
            data, power, fitparams, plot=False
        )
        q_coupling_amp = fit_params[2] * 1e4
    else:
        # For higher power indices, use mean of previous coupling Q values
        qc_values = [
            result.measurements[freq_idx][i].q_coupling_amp
            for i in range(4, 8)
            if i < power_idx
        ]
        qc_best = np.mean(qc_values) if qc_values else prev_fit_params[2] * 1e4

        fitparams = [
            result.current_frequencies[freq_idx],
            prev_fit_params[1],
            prev_fit_params[3],
            np.max(10 ** (data["amps"] / 20)),
        ]
        freq_center, q_total_amp, kappa, fit_params = fit_resonator(
            data, power, fitparams, qc_best
        )
        q_coupling_amp = qc_best

    # Perform alternative fitting
    try:
        data = ResonatorData.fit_phase(data)
        if power_idx < 8:
            output = ResonatorFitter.fit_resonator(
                data, fname, expt_path, plot=True, fix_freq=False
            )
        else:
            qc_values = [
                result.measurements[freq_idx][i].q_coupling
                for i in range(4, 7)
                if i < power_idx
            ]
            qc_best = np.mean(np.array(qc_values))
            output = ResonatorFitter.fit_resonator(
                data,
                fname,
                expt_path,
                plot=True,
                fix_freq=False,
                fit_Qc=False,
                Qc_fix=qc_best,
            )

        q_total, q_coupling, freq, phase = output[0][:4]
        q_err, phase_err, qi_err, qc_err, f_err = (
            output[1][0],
            output[1][4],
            output[1][1],
            output[1][3],
            output[1][5],
        )

        Qc_comp = q_coupling / np.exp(1j * phase)
        q_internal = (q_total**-1 - np.real(Qc_comp**-1)) ** -1
    except Exception as e:
        print(f"Alternative fit failed: {str(e)}")
        q_total = q_coupling = freq = q_internal = None
        phase_err = qi_err = qc_err = f_err = q_err = None

        # Calculate photon number
    pin = (
        power
        - config["att"]
        - config["db_slope"] * (freq_center / 1e9 - config["freq_0"])
    )

    if config['type']=='linear':
        min_avg = 10
    else:
        min_avg = 100

    # Create measurement object
    measurement = ResonatorMeasurement(
        frequency_amp=freq_center,
        power=power,
        power_at_device=pin,
        q_total_amp=q_total_amp,
        q_internal_amp=fit_params[1] * 1e4,
        q_coupling_amp=q_coupling_amp,
        q_total=q_total,
        q_internal=q_internal,
        q_coupling=q_coupling,
        frequency=freq,
        q_internal_err=qi_err,
        q_total_err=q_err,
        q_coupling_err=qc_err,
        frequency_err=f_err,
        phase_err=phase_err,
        kappa=kappa,
        photon_number=n(pin, freq_center, q_total_amp, q_coupling_amp),
        averages=int(max(result.averaging_factors[freq_idx], min_avg)),
        fit_parameters=fit_params,
        raw_data=data,
    )
    return measurement
