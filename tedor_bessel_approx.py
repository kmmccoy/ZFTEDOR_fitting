import numpy as np
import scipy.special as sp
import scipy.constants as cs
import lmfit as lm


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
