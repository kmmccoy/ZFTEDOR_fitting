import subprocess
import numpy as np
import lmfit as lm
import scipy.special as sp
import scipy.constants as cs
from astropy.stats import jackknife_resampling
import nmrglue as ng
import ast
import pandas as pd
from more_itertools import pairwise
import statistics as st


def tedor_ideal(t_mix, a, dist, t2, j_cc, obs='C13', pulsed='N15', vr=14000, return_t=False):
    """
    Makes a SpinEvolution input file from template file "tedor_ideal_template", calls SpinEvolution, parses the output,
    and applies phenomenological scaling and exponential relaxation.

    The tedor_ideal is a calculation for interpreting and ultimately fitting ZF-TEDOR build-up curves

    Parameters
    ----------

    a: float, scaling factor
    dist: float, distance between 13C-15N
    t2: float, $T_2$ relaxations time
    vr: float, MAS speed in HZ
    j_cc: float, carbon carbon J coupling in Hz
    return_t: bool, should the function return t=np.arange(0, n)*tr
    t_mix: array of mixing experimental mixing times in ms
    obs: string, the observed nucleus for the TEDOR experiment
    pulsed: string, the nucleus with the REDOR pulses on it

    Returns
    -------
    signal: array, len(t_mix)

    or

    time; signal: array, len(n); array, len(t_mix)
    """
    # Build the simulation program from the template
    sim_params = {'dist': dist, 'vr': vr / 1000, 'tr': 1 / vr, 'obs': obs, 'pulsed': pulsed}

    with open('templates/tedor_ideal_template', 'r') as fid:
        template = fid.read()

    with open('templates/tedor_ideal_step', 'w') as fid:
        fid.write(template.format(**sim_params))

    cmd = ['/opt/spinev/spinev', 'templates/tedor_ideal_step']

    # Run the simulation
    subprocess.call(cmd)

    # Parse the results
    output_file = 'templates/tedor_ideal_step_re.dat'
    results = np.loadtxt(output_file)
    time = results[:, 0]
    signal = results[:, 1]

    # Apply phenomenological corrections
    signal = a * signal * (np.cos(np.pi * (j_cc * 1000 / 2))**2) * np.exp(-time / t2)

    time_points = []
    signal_points = []

    for i in t_mix:
        ind = (np.where((np.trunc(time * 100) / 100) == i)[0][0])
        time_points.append(time[ind])
        signal_points.append(signal[ind])

    if return_t:
        return time_points, signal_points
    else:
        return signal_points


def tedor_ideal_2n(t_mix, a, dist, t2, x, y, z, j_cc, obs='C13', pulsed='N15', vr=14000, return_t=False):
    """
    Makes a SpinEvolution input file from template file "tedor_ideal_template_2N" using the CNN.cor coordinates file,
    calls SpinEvolution, parses the output, and applies phenomenological scaling and exponential relaxation.

    Parameters
    ----------

    a: float, scaling factor
    dist: float, distance between 13C-15N
    t2: float, $T_2$ relaxations time
    vr: float, MAS speed in HZ
    j_cc: float, carbon carbon J coupling in Hz
    return_t: bool, should the function return t=np.arange(0, n)*tr
    t_mix: array of mixing experimental mixing times in ms
    x: float, distance of second N from C
    y: float, distance of second N from C
    z: float, distance of second N from C
    obs: string, the observed nucleus for the TEDOR experiment
    pulsed: string, the nucleus with the REDOR pulses on it

    Returns
    -------
    signal: array, len(t_mix)

    or

    time; signal: array, len(n); array, len(t_mix)
    """
    # Build the simulation program from the template
    sim_params = {'dist': dist, 'x': x, 'y': y, 'z': z, 'vr': vr / 1000, 'tr': 1 / vr, 'j_cc': j_cc, 'obs': obs,
                  'pulsed': pulsed}

    with open('templates/CNN.cor', 'r') as fid:
        template = fid.read()

    with open('templates/CNN_step.cor', 'w') as fid:
        fid.write(template.format(**sim_params))

    with open('templates/tedor_ideal_template_2N', 'r') as fid:
        template = fid.read()

    with open('templates/tedor_ideal_step_2N', 'w') as fid:
        fid.write(template.format(**sim_params))

    cmd = ['/opt/spinev/spinev', 'templates/tedor_ideal_step_2N']

    # Run the simulation
    subprocess.call(cmd)

    # Parse the results
    output_file = 'templates/tedor_ideal_step_2N_re.dat'
    results = np.loadtxt(output_file)
    time = results[:, 0]
    signal = results[:, 1]

    # Apply phenomenological corrections
    signal = a * signal * (np.cos(np.pi * (j_cc * 1000 / 2))**2) * np.exp(-time / t2)

    time_points = []
    signal_points = []

    for i in t_mix:
        ind = np.where((np.trunc(time * 100)/100) == i)[0][0]
        time_points.append(time[ind])
        signal_points.append(signal[ind])

    if return_t:
        return time_points, signal_points
    else:
        return signal_points


def tedor_fitting_spinev(data, err, t_mix, p0, p1, obs, pulsed, vr=14000, spins=2, method='nelder'):
    """

    :param data: array, transfer efficiency values for fitting
    :param err: array, error for each data point
    :param t_mix: array, mixing times in ms
    :param p0: array, initial guesses for [dist, j_cc, t2, and a]
    :param p1: bool array len(3) -- allows you to turn on/off varying j_cc, t2, and a
    :param obs: string, observed nucleus
    :param pulsed: string, other nucleus
    :param vr: MAS frequency
    :param spins: float, total number of spins in system, either 2 or 3
    :param method: fitting method -- for lmfit

    :return: result - fitting result structure
    """
    if spins == 2:
        spin_model = tedor_ideal
    else:
        spin_model = tedor_ideal_2n

    kws = {"obs": obs, "pulsed": pulsed}

    # Build a model to fit the data - SPINEV function
    tedor_model = lm.Model(spin_model, **kws)
    params = tedor_model.make_params()
    params['dist'].set(value=p0[0], min=2, max=8)
    params['j_cc'].set(value=p0[1], min=0, max=75, vary=p1[0])
    params['t2'].set(value=p0[2], min=2, max=30, vary=p1[1])
    params['a'].set(value=p0[3], min=0, max=1, vary=p1[2])
    params['vr'].set(value=vr, min=10000, max=20000, vary=False)

    if spins == 3:
        params['x'].set(value=2.0, min=1.5, max=7)
        params['y'].set(value=2.0, min=1.5, max=7)
        params['z'].set(value=2.0, min=1.5, max=7)

    # Fit the data
    result = tedor_model.fit(data, t_mix=t_mix, **params, weights=err, method=method)

    return result


def tedor_analytical(t_mix, a, d_active, t2, j_cc, d_p1):

    """

    Analytical equations for TEDOR fitting from Helmus et al 2008 and Jaroniec et al 2002
    Uses Bessel function of first kind order 0 to simulate TEDOR behavior

    Parameters
    ----------

    a: float, scaling factor
    d_active: float, dipolar coupling between 13C and 15N in Hz
    d_p1: float, passive dipolar coupling between 13C and additional 15N in Hz
    t2: float, $T_2$ relaxations time in ms
    j_cc: float, carbon carbon J coupling in Hz
    t_mix: array of mixing experimental mixing times in ms

    Returns
    -------
    signal: array, len(t_mix)

    KMM 11 May 2021
    """

    t2_s = t2 / 1000  # puts t2 in terms of s, must be entered in ms
    time = t_mix / 1000

    signal = a * 0.5 * (1 - (sp.j0(np.sqrt(2) * d_active * time)) ** 2) * (np.cos(np.pi * (j_cc / 2)) ** 2) * \
        (1 + (sp.j0(np.sqrt(2) * d_p1 * time)) ** 2) * np.exp(-time / t2_s)

    return signal


def tedor_fitting_bessel(data, err, t_mix, p0, p1, method='nelder'):
    """

    :param data: array, transfer efficiency values for fitting
    :param err: array, error for each data point
    :param t_mix: array, mixing times in s
    :param p0: array, initial guesses for [d_active, t2, a, d_p1, j_cc]
    :param p1: bool array len(3) -- allows you to turn on/off varying t2, a, d_p1, and j_cc
    :param method: fitting method -- for lmfit

    :return: result - fitting result structure
    """
    tedor_model_analytical = lm.Model(tedor_analytical)
    params_a = tedor_model_analytical.make_params()
    params_a['d_active'].set(value=p0[0], min=0, max=200)
    params_a['t2'].set(value=p0[1], min=2, max=50, vary=p1[0])
    params_a['a'].set(value=p0[2], min=0.0001, max=1, vary=p1[1])
    params_a['d_p1'].set(value=p0[3], min=0, max=150, vary=p1[2])
    params_a['j_cc'].set(value=p0[4], min=0, max=50, vary=p1[3])

    # Fit the data
    result_a = tedor_model_analytical.fit(data, t_mix=t_mix, **params_a, weights=err, method=method)

    return result_a


def jackknife_err(data, error, time, p0, p1, spins=2, obs='C13', pulsed='N15', method='bessel'):

    """
    FUNCTION FOR THE JACKKNIFE ERROR FOR TEDOR FITS USING EITHER THE BESSEL APPROXIMATION OR SPINEVOLUTION FITTING
    AS IMPLEMENTED IN THIS CODE BASE

    Parameters
    ----------
    data: np.array, TEDOR data for fitting error estimate
    error: np.array, error for individual TEDOR data points
    time: np.array, mixing time for TEDOR experiments
    p0: initial guesses for fitting
    p1: boolian, which parameters to vary in fitting
    obs: str, observed nucleus
    pulsed: str, pulsed nucleus
    method: str, fitting method to use - bessel or spinev

    Returns
    -------
    a list of the computed errors for the distance and the t2

    KMM
    15 July 2021

    """
    resamples_data = jackknife_resampling(data)
    resamples_err = jackknife_resampling(error)
    resamples_time = jackknife_resampling(np.array(time))

    dist_jk = []
    t2_jk = []

    if method == 'bessel':
        for jk_data, jk_err, jk_time in zip(resamples_data, resamples_err, resamples_time):
            result_jk = tedor_fitting_bessel(jk_data, jk_err, t_mix=jk_time, p0=p0, p1=p1, method='nelder')
            d_jk = result_jk.params["d_active"].value
            t2_a_jk = result_jk.params["t2"].value

            dist_jk.append(np.round(radius(d_jk, obs, pulsed), 3))
            t2_jk.append(t2_a_jk)
    elif method == 'spinev':
        for jk_data, jk_err, jk_time in zip(resamples_data, resamples_err, resamples_time):
            result_jk = tedor_fitting_spinev(jk_data, jk_err, t_mix=jk_time, p0=p0, p1=p1, spins=spins, obs=obs, pulsed=pulsed,
                                             method='nelder')
            dist = result_jk.params["dist"].value
            t2_a_jk = result_jk.params["t2"].value

            dist_jk.append(dist)
            t2_jk.append(t2_a_jk)

    return dist_jk, t2_jk


def radius(d_active, obs='C13', pulsed='N15'):
    """
    Function that calculates the internuclear distance from the dipolar coupling in Hz
    Currently supports 13C, 15N, 19F, and 31P

    :param
        d_active: float, dipolar coupling in Hz
        obs: string, the observed nucleus, can be C, N, P or F
        pulsed: string, the pulsed nulceus, can be C, N, P or F, but not the same as obs

    :return:
        dist: float, distance between 13C and 15N in Angstrom


    KMM 1 June 2021
    """

    mu = cs.mu_0  # Mu_0 in T^2*m^3/J
    h = cs.hbar  # hbar in J/s
    pi = cs.pi
    y_c = 67.2828 * 1E6  # gamma 13C in Hz/T
    y_n = -27.116 * 1E6  # gamma 15N in Hz/T
    y_p = 108.291 * 1E6  # gamma 31P in Hz/T
    y_f = 251.815 * 1E6  # gamma 19F in Hz/T

    if (obs == 'C13' and pulsed == 'N15') or (obs == 'N15' and pulsed == 'C13'):
        dist = ((-mu * y_c * y_n * h) / (8 * pi * pi * d_active)) ** (1 / 3)
    elif (obs == 'C13' and pulsed == 'P31') or (obs == 'P31' and pulsed == 'C13'):
        dist = ((-mu * y_c * y_p * h) / (8 * pi * pi * d_active)) ** (1 / 3)
    elif (obs == 'C13' and pulsed == 'F19') or (obs == 'F19' and pulsed == 'C13'):
        dist = ((-mu * y_c * y_f * h) / (8 * pi * pi * d_active)) ** (1 / 3)
    elif (obs == 'N15' and pulsed == 'P31') or (obs == 'P31' and pulsed == 'N15'):
        dist = ((-mu * y_n * y_p * h) / (8 * pi * pi * d_active)) ** (1 / 3)
    elif (obs == 'N15' and pulsed == 'F19') or (obs == 'F19' and pulsed == 'N15'):
        dist = ((-mu * y_n * y_f * h) / (8 * pi * pi * d_active)) ** (1 / 3)
    elif (obs == 'F19' and pulsed == 'P31') or (obs == 'P31' and pulsed == 'F19'):
        dist = ((-mu * y_f * y_p * h) / (8 * pi * pi * d_active)) ** (1 / 3)
    elif obs == pulsed:
        raise KeyError("This Doesn't Work For Homonulcear Experiments, Sorry")
    else:
        raise KeyError("Please Enter Valid Nuclei")

    return dist * 1E10


def coupling_strength(dist, obs='C13', pulsed='N15'):
    """
    Function that calculates the dipolar coupling in Hz from the internuclear radius
    Currently supports 13C, 15N, 19F, and 31P

    :param
        dist: float, inter-nuclear distance in A
        obs: string, the observed nucleus, can be C, N, P or F
        pulsed: string, the pulsed nulceus, can be C, N, P or F, but not the same as obs
    :return:
        d_active: float, dipolar coupling strength in Hz


    KMM 19 May 2021
    """

    mu = cs.mu_0  # Mu_0 in T^2*m^3/J
    h = cs.hbar  # hbar in J/s
    pi = cs.pi
    y_c = (67.2828 * 1E6)  # gamma 13C in MHz/T
    y_n = (-27.116 * 1E6)  # gamma 15N in MHz/T
    y_p = 108.291 * 1E6  # gamma 31P in Hz/T
    y_f = 251.815 * 1E6  # gamma 19F in Hz/T

    r = dist / 1E10

    d_cn = (-mu * y_c * y_n * h) / (8 * pi * pi * (r ** 3))
    d_cp = (-mu * y_c * y_p * h) / (8 * pi * pi * (r ** 3))
    d_cf = (-mu * y_c * y_f * h) / (8 * pi * pi * (r ** 3))
    d_np = (-mu * y_n * y_p * h) / (8 * pi * pi * (r ** 3))
    d_nf = (-mu * y_n * y_f * h) / (8 * pi * pi * (r ** 3))

    if (obs == 'C13' and pulsed == 'N15') or (obs == 'N15' and pulsed == 'C13'):
        return d_cn
    elif (obs == 'C13' and pulsed == 'P31') or (obs == 'P31' and pulsed == 'C13'):
        return d_cp
    elif (obs == 'C13' and pulsed == 'F19') or (obs == 'F19' and pulsed == 'C13'):
        return d_cf
    elif (obs == 'N15' and pulsed == 'P31') or (obs == 'P31' and pulsed == 'N15'):
        return d_np
    elif (obs == 'N15' and pulsed == 'F19') or (obs == 'F19' and pulsed == 'N15'):
        return d_nf
    elif obs == pulsed:
        raise KeyError("This Doesn't Work For Homonulcear Experiments, Sorry")
    else:
        raise KeyError("Please Enter Valid Nuclei")


def dist_cn2(x, y, z, dist):
    """
    Function that calculates the internuclear distance between a 13C and a second 15N from cartesian coordinates
    :param
        dist: float, distance between 15N and observed 13C
        x, y, z: floats, cartesian coordiates for second 15N coupled to observed 13C
    :return:
        dist_carbon: float, distance between 13C and 15N in Angstrom
        dist_nitrogen: float, distance between 15N and second 15N


    KMM 11 May 2021
    """

    dist_carbon = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    dist_nitrogen = np.sqrt(np.square(x) + np.square(y) + np.square(z - dist))

    return dist_carbon, dist_nitrogen

def perfectEval(anonstring):
    try:
        ev = ast.literal_eval(anonstring)
        return ev
    except ValueError:
        corrected = "\'" + anonstring + "\'"
        ev = ast.literal_eval(corrected)
        return ev


def get_experiment(filename, label):
    exp_df = pd.read_csv(filename, sep='\t')
    details = exp_df[exp_df.label == label]
    ind = details.index[0]
    experiment = details.file_name[ind]
    buffers = perfectEval(details.tedor_buffers[ind])
    exp = len(buffers)
    cp_buffer = details.cp_buffer[ind]
    scans = float(perfectEval(details.tedor_scans[ind]))

    return experiment, buffers, scans, cp_buffer, exp


def import_tedor_data(file_base, experiment, buffers):

    exp = len(buffers)
    mixing_t = []
    scans = []

    for i in range(0, exp):  # imports data from ZF-TEDOR data sets and constructs a dataframe with all spectra

        spectrum_dir = file_base + experiment + '/' + str(buffers[i]) + '/pdata/1'

        dic, tedor_data = ng.bruker.read_pdata(spectrum_dir)

        udic = ng.bruker.guess_udic(dic, tedor_data)
        uc = ng.convert.fileiobase.uc_from_udic(udic)
        ppm = uc.ppm_scale()

        si, vr, obs, pulsed, mix, ns = get_params(dic)  # extracts relevant experimental details from the bruker file

        mixing_t.append(mix)
        scans.append(ns)

        # find noise level and normalize
        noise_slice = tedor_data[uc(220, 'ppm'): uc(200, 'ppm')]
        noise_level_tedor = np.mean(noise_slice)

        if noise_level_tedor < 0:
            tedor_data -= noise_level_tedor
        else:
            tedor_data += noise_level_tedor

        if i == 0:
            data = pd.DataFrame(np.zeros((exp, si)))

        data.iloc[i, :] = tedor_data

        if i == (exp - 1):
            data.columns = ppm
            data.index = mixing_t

    return data, uc, mixing_t, scans, obs, pulsed


def get_params(dic):

    aq = dic['acqus']
    proc = dic['procs']

    si = proc['SI']
    vr = aq['CNST'][31]
    nucleus1 = aq['NUC1']
    nucleus2 = aq['NUC3']

    obs = nucleus1[-1] + nucleus1[:-1]
    pulsed = nucleus2[-1] + nucleus2[:-1]

    mix = np.floor(4 * (aq['L'][1] / vr) * 1000000) / 1000
    mix = np.trunc(mix * 100) / 100
    scans = aq['NS']

    return si, vr, obs, pulsed, mix, scans

def multispectrum_peakpicking(data, uc, tol, nthresh, save_csv=False, long_ind=8):

    """

    :param data: a pandas dataframe, all the spectrum to pick peaks from
    :param uc: unit converter object to convert Hz to ppm
    :param tol: float, tolerance for identifying peaks as the same peak (ppm)
    :param nthresh: list, noise thresholds for each spectrum
    :param save_csv: bool, if True returns a .csv of the peak list
    :param long_ind: float, lower bound for long mixing times

    :return: list of identified and averaged peaks
    """

    exp = len(data)
    mixing_t = data.index
    peak_locations_ppm = []
    peak_labels = []
    good_peaks = []
    avg_peaks = []
    groups = []
    chunk = []

    # automatic peak picking for all input experiments
    for i in range(0, exp):
        peak_data = data.iloc[i, :]
        peak_table = ng.peakpick.pick(peak_data.values, pthres=nthresh[i], algorithm='downward')
        peak_locations_ppm.append([uc.ppm(j) for j in peak_table['X_AXIS']])
        peak_labels.append(np.round(peak_locations_ppm[i], 2))

    mixing_long_ind = [n for n, i in enumerate(mixing_t) if i > long_ind]  # determines indices for experiments

    # compares each peak in the long mixing time spectra to every other peak and determines which ones are within the
    # input tolerance and then constructs a list of all of these duplicate peaks
    for ind in mixing_long_ind:
        if ind != mixing_long_ind[-1]:
            peaks = peak_labels[ind]

            for i in range(ind, len(mixing_long_ind) - 1):
                other_peaks = peak_labels[i + 1]

                for peak in peaks:
                    sims = other_peaks[np.isclose(other_peaks, peak, atol=tol)]
                    if sims.size:
                        good_peaks.append(peak)
                        for num in sims:
                            good_peaks.append(num)

    good_peaks = sorted(good_peaks)
    good_peaks = sorted(list(set(good_peaks)))
    good_peaks[:] = [number for number in good_peaks if number < 185]
    good_peaks[:] = [number for number in good_peaks if number > 10]

    # averages the close peaks together and outputs a list of all of these peaks
    for peak1, peak2 in pairwise(good_peaks):
        if peak2 - peak1 <= tol:
            chunk.append(peak1)
        elif chunk:
            chunk.append(peak1)
            groups.append(chunk)
            chunk = []

    for group in groups:
        avg = st.mean(group)
        avg_peaks.append(np.round(avg, 2))

    if save_csv is True:
        return np.savetxt("peak_list.csv", avg_peaks, delimiter=",")

    return avg_peaks


def divisors(n):
    large_divisors = []
    for i in range(1, int(np.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor


def peak_integration_fixed(data, peak_list, dx, tol, ppm):

    points = 2 * (tol / dx)
    num_peaks = len(peak_list)
    if type(data) is np.ndarray:
        data = pd.DataFrame(data, index=ppm).transpose()

    exp = len(data.index)
    integrals = pd.DataFrame(np.zeros((exp, num_peaks)), columns=peak_list, index=data.index)
    int_error = pd.DataFrame(np.zeros((exp, num_peaks)), columns=peak_list, index=data.index)

    for peak in peak_list:
        if type(data) is np.ndarray:
            spectrum = data
            region = spectrum[peak+tol:peak-tol]
            integrals.loc[peak] = np.trapz(region, x=None, dx=dx)
            rmsd = np.std(spectrum[220:200])
            int_error.loc[peak] = rmsd * np.sqrt(2 * points) * dx * 0.5
        else:
            for ind, spectrum in data.iterrows():
                region = spectrum[peak+tol:peak-tol]
                integrals.loc[ind, peak] = np.trapz(region, x=None, dx=dx)
                rmsd = np.std(spectrum[220:200])
                int_error.loc[ind, peak] = rmsd * np.sqrt(2 * points) * dx * 0.5

    return integrals, int_error
