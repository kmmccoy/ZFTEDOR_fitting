"""
This is a script calls the tedor_ideal function from tedor_ideal_spinev to run a defined SpinEvolution
simulation and outputs a plot. tedor_ideal also applies phenomenological corrections for t2 relaxation
and scaling (a), along with 13C-13C J coupling if desired.

tedor_ideal uses templates/tedor_ideal_template for the SpinEv input file. If you want to alter the simulation
parameters, go there.

SpinEv is a black box and tedor_ideal calls it as a subprocess, so be careful.

This script has some command line flags --> -p will prompt the output of the graph. -n lets you name the saved graph
without either it will save the graph as plot.png but not print it
-o is the identity of the observed nucleus, and -r is the identity of the other nucleus. -o and -r cannot be the same.
Options for nuclei are C13, N15, F19, and P31. If nothing is entered, o = C13 and r = N15
-d is the distance between the nuclei in Angstroms
-t is the t2 value in ms
-j is the carbon carbon J coupling value in Hz (optional)
-m is the passive dipolar coupling for the Bessel simulation (optional)

KMM 1 June 2021
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
from tedor_fitting_functions import tedor_ideal
from tedor_bessel_approx import tedor_analytical, coupling_strength

# adds command line flags for inputs
parser = argparse.ArgumentParser()
parser.add_argument("--plot", "-p", help="show output plot", action='store_true')
parser.add_argument("--filename", "-f", type=str, help="input name for plot file. If there are spaces, must be in '")
parser.add_argument('--nucleus1', "-o", type=str, help="identity of the observed nucleus, must be a string. "
                                                       "Options: C13, N15, P31, and F19")
parser.add_argument('--nucleus2', "-r", type=str, help="identity of the nucleus with redor pulses, must be a string. "
                                                       "Options: C13, N15, P31, and F19. Can,t be the same as Nucleus1")
parser.add_argument("--distance", "-d", type=float, help="Internuclear distance in Angstroms")
parser.add_argument("--t2time", "-t", type=float, help="T2 relaxation time in ms")
parser.add_argument("--jcoupling", "-j", type=float, help="J coupling in Hz")
parser.add_argument("--passive", "-m", type=float, help="Passive dipolar coupling in Hz")
args = parser.parse_args()

a_se = 1.0  # scaling factor for SPINEV
a_bes = 0.5  # scaling factor for bessel function approximation
d_p1 = 0  # passive dipolar coupling for Bessel function approximation

t_mix = np.array([2.0, 4.28, 6.0, 8.28, 10.0, 12.28, 14.0, 16.28, 18.0, 24.28])  # experimental mixing time_points

if args.nucleus1 is None:
    obs = 'C13'
else:
    obs = args.nucleus1

if args.nucleus2 is None:
    pulsed = 'N15'
else:
    pulsed = args.nucleus2

if args.distance is None:
    raise Exception("You must enter a distance (-d) to simulate")
else:
    dist = args.distance

if args.t2time is None:
    raise Exception("You must enter a T2 value (-t) to simulate")
else:
    t2 = args.t2time

if args.jcoupling is None:
    j_cc = 0
else:
    j_cc = args.jcoupling

if args.passive is None:
    d_p1 = 0
else:
    d_p1 = args.passive

d_active = coupling_strength(dist, obs, pulsed)  # calculates the dipolar coupling strength for an internuclear distance

# runs the simulations - comment out signal_se if you don't want to run SpinEv
signal_se = tedor_ideal(t_mix, a_se, dist, t2, j_cc, obs, pulsed)
signal_a = tedor_analytical(t_mix, a_bes, d_active, t2, j_cc, d_p1)

plt.figure()
plt.plot(t_mix, signal_se, color='navy', label='SpinEv Sim')
plt.plot(t_mix, signal_a, color='red', label='Bessel Sim')
plt.title('Simulated TEDOR Build-up, ' + obs + '--' + pulsed + ' ' + str(dist) + ' A')
plt.xlabel('Mixing Time (ms)')
plt.ylabel('Transfer Efficiency')
plt.legend()

if (args.plot is True) and ((args.filename is None) is False):
    plt.show()
    plt.savefig(args.filename + '.png')
elif (args.plot is True) and ((args.filename is None) is True):
    plt.show()
    plt.savefig('plot.png')
elif (args.plot is False) and ((args.filename is None) is False):
    plt.savefig(args.filename + '.png')
else:
    plt.savefig('plot.png')
