import numpy as np
import slab_qick_calib.experiments as meas
import os
import copy


def do_rfsoc_scan(hw, file_name, expt_path, scan_config, config, att=0, plot=False):

    # for now, change gain
    if "attn_id" in config:
        from scresonators.measurement import vaunix_da

        gain = 1
        vaunix_da.set_atten(
            config["attn_id"], config["attn_channel"], -scan_config["power"]
        )
        import time

        time.sleep(0.1)

    else:
        gain = 10 ** (scan_config["power"] / 20)

    if "loop" not in config:
        config["loop"] = False
        config["phase_const"] = False
    params = {
        "span": scan_config["span"] / 1e6,
        "reps": scan_config["averages"],
        "gain": gain,
        "length": int(1e6 / scan_config["bandwidth"]),
        "center": scan_config["freq_center"] / 1e6,
        "expts": scan_config["npoints"],
        "loop": config["loop"],
        "phase_const": config["phase_const"],
        "final_delay": 50,
    }
    oth_par = {
        "span": 0.2,
        "reps": 10,
        "gain": gain,
        "length": 1000,
        "center": scan_config["freq_center"] / 1e6 - 3,
        "expts": 10,
    }
    s = meas.ResSpec(
        hw,
        qi=0,
        params=oth_par,
        save=False,
        display=False,
        analyze=False,
        progress=False,
    )
    # Otherwise the ADCs will saturate
    if scan_config["power"] >= -20:
        params["length"] = np.min((params["length"], 1000))
    if config["loop"]:
        exp = meas.ResSpec2D
        params["expts_count"] = params["reps"]
        params["reps"] = 1
        params["kappa"] = (scan_config["kappa"],)
        if config["pin"] > -88:
            params["length"] = np.min((params["length"], 10000))
    else:
        exp = meas.ResSpec

    rspec = exp(hw, qi=0, params=params, save=False, display=False, analyze=False)

    rspec.data["freqs"] = rspec.data["xpts"] * 1e6

    fix_phase = rspec.data["phases"]
    data = copy.deepcopy(rspec.data)
    data["phases"] = np.unwrap(fix_phase) - np.unwrap(fix_phase)[0]
    data["amps"] = 20 * np.log10(data["amps"])
    rspec.data = data
    rspec.cfg["power"] = scan_config["power"]
    rspec.fname = os.path.join(expt_path, file_name)
    rspec.save_data()

    return data
