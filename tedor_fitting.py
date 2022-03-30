"""
WARNING:
SpinEvolution is closed source.  This python code also has subprocess calls, be careful with the "cmd" variable.
The security consciences should be alert.  Use at your own risk.

This script calls tedor_fitting_functions.py and tedor_bessel_approx.py to fit ZF-TEDOR build-up curves. It requires
SpinEvolution (Veshtort et al. 2008), which requires a paid license. The Bessel approximation (Mueller 1995, Jaroniec
et al. 2002, and Helmus et al. 2008) is an approximate analytical solution that doesn't require SpinEv.

This script requires the input of the location of all relevant spectra, a list of TEDOR mixing times, a peak list, and
various information about the NMR files (number of experiments, length of spectra, number of scans, etc.). You also need
to manually input initial guesses for the fitting and turn on/off fitting parameters (p0 and p1). The min/max for
fitting parameters must be adjusted in the fitting functions in tedor_fitting_functions.py and tedor_bessel_approx.py.
All relevant SpinEv pulse sequence/spin system files are located in /templates/.

All relevant experimental information is extracted from the bruker pdata file. However the number of scans must be input
manually in the input .tsv files because of coadding spectra.

Updated to include nuclei as inputs (1 June 2021)
Updated to auto pick peaks (4 June 2021)
Updated to add looping over peak list and multi-peak plotting (7 June 2021)
Updated to implement jackknife error estimation for the fitting (15 July 2021)
Updated to take .tsv file with list of experiments and buffers as input (21 October 2021)
Updated with bug fixes and some function consolodation (27 October 2021)

KMM 27 October 2021
"""
import matplotlib.pyplot as plt
from tedor_fitting_functions import *

file_base = '/Users/kelseymccoy/Desktop/Spectra/FtsZ/DNP/nmrsu/'
label = 'AG'
experiments = 'tedor_experiments.tsv'  # must be .tsv
peaks_file = 'peak_list_all_samples.tsv'  # must be .tsv
save_label = '_bestfit.csv'

spinev = False  # bool, true if running SpinEvolution
bessel = True  # turns on analytical fitting
spins = 2

plot = False  # turns on plotting the best fit curves
multiplot = False
peak_to_plot = 14.80

loop_all = True  # bool, if true, loops over all fitted peaks
loop_end = 18.86  # int, peak after which to end looping

auto_pick = False  # bool, if true, runs automatic peak picking, if false, must read in a peak list
jackknife = True  # bool, if true, implements jackknife error estimation to determine fitting error.
                  # If false, no fitting error is output.
table = True  # bool, if true outputs results as table in console
save_csv = False  # bool, if true outputs results as a .csv file

# initial guesses for fitting
p0 = [5, 10, 8, 0.6]  # array of initial guesses for fitting [dist, j_cc, t2, a]
p1 = [False, True, True]  # array of whether to vary [j_cc, t2, and a]

p0_a = [15, 7.5, 0.3, 0, 10]  # array of initial guesses for analytical fitting [d_active, t2, a, d_p1, j_cc]
p1_a = [True, True, False, False]  # array of whether to vary [t2, a, d_p1, j_cc]

# integration parameters
tol = 0.5  # window for integral = 2*tol
deex = 0.1  # dx for integral
points = 2 * (tol / deex)

"""CODE THAT DOES STUFF"""

# display formatting for the output table
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# initializes all the empty dataframes and lists
rmsd = []
rmsd_for_pp = []

experiment, buffers, scans, cp_buffer, exp = get_experiment(experiments, label)
data, uc, mixing_t, scans_x, obs, pulsed = import_tedor_data(file_base, experiment, buffers)
peak_list_file = pd.read_csv(peaks_file, sep='\t')
cp_file = file_base + experiment + '/' + str(cp_buffer) + '/pdata/1'  # reads in cp spectrum
ppm = uc.ppm_scale()

dic_cp, cp_data = ng.bruker.read_pdata(cp_file)
udic = ng.bruker.guess_udic(dic_cp, cp_data)
uc_cp = ng.convert.fileiobase.uc_from_udic(udic)
cp_ppm = uc_cp.ppm_scale()
noise_cp = cp_data[uc_cp(220, 'ppm'): uc_cp(200, 'ppm')]
noise_level_cp = np.mean(cp_data[uc_cp(220, 'ppm'): uc_cp(200, 'ppm')])

cp_scans = dic_cp['acqus']['NS']

if noise_level_cp < 0:
    cp_data -= noise_level_cp
else:
    cp_data += noise_level_cp

for i in range(0, exp):
    spectrum = data.iloc[i, :]

    noise_level = spectrum[220:200]
    noise = pd.DataFrame(np.zeros((exp, len(noise_level))))
    noise.iloc[i, :] = noise_level.to_numpy()

    rmsd.append(np.std(noise.iloc[i, :]))  # finds the new rmsd of the scaled noise
    rmsd_for_pp.append(2 * np.std(noise.iloc[i, :]))  # 2 * the rmsd

# auto generates a peak list if auto_pick == True
if auto_pick is True:
    peak_list = multispectrum_peakpicking(data, uc, tol=0.25, nthresh=rmsd_for_pp)
    col_names = peak_list
    num_peaks = len(peak_list)
else:
    peak_list_sample = peak_list_file[peak_list_file.samples == label]
    peak_list = list(peak_list_sample.peak)
    col_names = peak_list
    num_peaks = len(peak_list)

integrals = pd.DataFrame(np.zeros((exp, num_peaks)), columns=col_names, index=mixing_t)
error = pd.DataFrame(np.zeros((exp, num_peaks)), index=mixing_t, columns=col_names)
cp_integrals = pd.DataFrame(np.zeros((1, num_peaks)), columns=col_names)
scaled_integrals = pd.DataFrame(np.zeros((exp, num_peaks)), columns=col_names)
error_scaled = pd.DataFrame(np.zeros((exp, num_peaks)), columns=col_names)

# integrates region around each peak for a total of 1 ppm
if type(scans) == float:
    scale = scans / cp_scans  # scaling factor for transfer efficiency calculation
    cp_scaled = cp_data * scale  # scaled CP spectrum
    noise_scaled = noise_cp * np.sqrt(scale)
    rmsd_cp = np.std(noise_scaled)
else:
    scans = np.array(scans)
    scale = scans / cp_scans
    cp_scaled = cp_data * scale
    noise_scaled = noise_cp * np.sqrt(scale)
    rmsd_cp = np.std(noise_scaled)

integrals, int_error = peak_integration_fixed(data, peak_list, deex, tol, ppm)
cp_integrals, int_error_cp = peak_integration_fixed(cp_scaled, peak_list, deex, tol, cp_ppm)

for i in range(0, num_peaks):  # divides the integrals by the scaled cp_integrals element-wise and calculates error
    for j in range(0, exp):
        value = integrals.iloc[j, i]
        scaled_integrals.iloc[j, i] = value / cp_integrals.iloc[0, i]
        err_value = int_error.iloc[j, i]
        int_err_cp = int_error_cp.iloc[0, i]

        error_scaled.iloc[j, i] = np.sqrt((err_value / value) ** 2 + (int_err_cp / cp_integrals.iloc[0, i]) ** 2)


if loop_all is True:
    loop_end = peak_list[-1]

fits_all = pd.DataFrame(np.zeros((len(peak_list[0:peak_list.index(loop_end)+1]), 8)),
                        index=peak_list[0:peak_list.index(loop_end)+1],
                        columns=["se_dist", "se_err", "se_t2", "se_rmsd", "be_dist", "be_err", "be_t2", "be_rmsd"])
results_se_fits = []
results_a_fits = []

for peak_to_fit in peak_list[0: peak_list.index(loop_end)+1]:

    int_peak = scaled_integrals.loc[:, peak_to_fit]
    int_peak = int_peak.to_numpy()
    err = np.array(error_scaled.loc[:, peak_to_fit] * int_peak)

    # Experimental data to fit - data in form of transfer efficiency
    data_tofit = int_peak  # raw, adjusted data
    ind = peak_list.index(peak_to_fit)

    if spinev is True:

        # Fit the data - SpinEv
        result_se_CN = tedor_fitting_spinev(data_tofit, err, t_mix=mixing_t, p0=p0, p1=p1, obs=obs, pulsed=pulsed,
                                            spins=spins, method='nelder')
        rmsd_fit_se = np.sqrt(np.sum((result_se_CN.best_fit - data_tofit) ** 2) / exp) / np.std(data_tofit)
        dist = result_se_CN.params["dist"].value
        err_se = result_se_CN.params["dist"].stderr
        t2 = result_se_CN.params["t2"].value

        if jackknife is True:
            dist_jk, t2_jk = jackknife_err(data_tofit, err, np.array(mixing_t), p0=p0, p1=p1, spins=spins, method='spinev')
        else:
            dist_jk = np.nan
            t2_jk = np.nan

        fits_all.iloc[ind, 0] = dist
        fits_all.iloc[ind, 1] = np.std(dist_jk)
        fits_all.iloc[ind, 2] = t2
        fits_all.iloc[ind, 3] = rmsd_fit_se
        results_se_fits.append(result_se_CN)

    if bessel is True:

        # Fit the data - analytical
        result_a_CN = tedor_fitting_bessel(data_tofit, err, t_mix=mixing_t, p0=p0_a, p1=p1_a, method='nelder')

        # RMSD of the fit divided by the standard deviation of the data
        rmsd_fit_a = np.sqrt(np.sum((result_a_CN.best_fit - data_tofit) ** 2) / exp) / np.std(data_tofit)

        # convert dipolar coupling to a distance
        D = result_a_CN.params["d_active"].value
        t2_a = result_a_CN.params["t2"].value
        dist_a = np.round(radius(D, obs, pulsed), 3)

        if jackknife is True:
            dist_jk, t2_jk = jackknife_err(data_tofit, err, np.array(mixing_t), p0=p0_a, p1=p1_a, method='bessel')
        else:
            dist_jk = np.nan
            t2_jk = np.nan

        fits_all.iloc[ind, 4] = dist_a
        fits_all.iloc[ind, 5] = np.std(dist_jk)
        fits_all.iloc[ind, 6] = t2_a
        fits_all.iloc[ind, 7] = rmsd_fit_a
        results_a_fits.append(result_a_CN)

if plot is True:
    # Plot the results
    fig, axs = plt.subplots(1, 1, figsize=[4, 4])

    ind = peak_list.index(peak_to_plot)

    int_peak = scaled_integrals.loc[:, peak_to_plot]
    err = np.array(error_scaled.loc[:, peak_to_plot] * int_peak)

    plt.errorbar(mixing_t, int_peak, err, fmt='o', color='black')

    if bessel is True:
        result_a = results_a_fits[ind]
        plt.plot(mixing_t, result_a.best_fit, color='black', linestyle=':', label='Bessel Function Fit')

    if spinev is True:
        result_se = results_se_fits[ind]
        plt.plot(mixing_t, result_se.best_fit, color='black', linestyle="--", label='SpinEv Fit')

    plt.xlabel('Mixing Time (ms)')
    plt.ylabel('Transfer Efficiency')
    plt.title(str(peak_to_plot) + ' ppm')
    plt.legend()

    plt.show()


if multiplot is True:
    fit_no = len(peak_list[0: peak_list.index(loop_end)+1])
    div_list = list(divisors(fit_no))

    if len(div_list) > 2:
        plots_no1 = int(div_list[1])
        plots_no2 = int(div_list[-2])
    elif len(div_list) == 2:
        plots_no1 = int(div_list[0])
        plots_no2 = int(div_list[-1])

    fig, axs = plt.subplots(plots_no1, plots_no2, figsize=[8, 8])
    ind = 0

    for i in range(0, plots_no1):
        for j in range(0, plots_no2):
            peak_id = peak_list[ind]
            int_peak = scaled_integrals.loc[:, peak_id]
            int_peak = int_peak.to_numpy()
            err = error_scaled.loc[:, peak_id] * int_peak

            axs[i, j].errorbar(mixing_t, int_peak, err, fmt='o', color='black')
            axs[i, j].set_title(str(peak_id) + " ppm")

            if spinev is True:
                axs[i, j].plot(mixing_t, results_se_fits[peak_list.index(peak_id)].best_fit, color='black',
                               linestyle='--', label='SpinEv Fit')
            if bessel is True:
                axs[i, j].plot(mixing_t, results_a_fits[peak_list.index(peak_id)].best_fit, color='black', linestyle=':'
                               , label='Bessel Fit')
            ind += 1

    fig.text(0.5, 0.06, 'TEDOR mixing time (ms)', ha='center', va='center', fontsize='16')
    fig.text(0.025, 0.5, 'Normalized Transfer Efficiency', ha='center', va='center', rotation='vertical', fontsize='16')
    # plt.legend(loc='upper right', bbox_to_anchor=(2.4, 3), fontsize='18')
    plt.show()

if save_csv is True:
    fits_all.to_csv(experiment + save_label)

if table is True:
    print(fits_all)
