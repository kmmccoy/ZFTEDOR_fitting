import matplotlib.pyplot as plt
import numpy as np
import tedor_bessel_approx as td

t_mix = np.linspace(0, 25, 1000)
d_hz = td.coupling_strength(5)
a = 1
t2 = 8
j_cc = 10
d1 = 0

signal = td.tedor_analytical(t_mix, a, d_hz, t2, j_cc, d1)

plt.plot(t_mix, signal)
plt.show()