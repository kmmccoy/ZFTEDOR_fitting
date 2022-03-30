# ZFTEDOR_fitting

Code base for analyzing and fitting TEDOR MAS NMR datasets using SpinEvolution (Veshtort 2011) and Bessel function approximations (Mueller 1995, Jaroniec 2002, Helmus 2008)

### Requirements:
* python 3 (written in Python 3.9)
* **nmrglue** for reading in and analyzing spectra. It is set up for Bruker files. If you're using a different file type, you need to 
modify the code.
* **matplotlib.pyplot** for plotting
* **numpy** for numerical stuff
* **lmfit** for curve fitting
* **scipy** for constants, and some other random stuff
* **pandas** for DataFrame functionality
* **astropy** for jackknife error estimation
* **ast** for safe string evaluation
* **statistics** for statistics
* **more-itertools** for pairwise interation 
* **SpinEvolution** for the actual simulations and fitting.

SpinEvolution is proprietary, so you also need a license for that if you are running the SpinEv code. If you don't have
access to SpinEv, you can just run the Bessel function approximation fitting. 

SpinEvolution pulse sequence and dependent files are located in the /templates folder

Dependent functions are located in tedor_fitting_functions.py . spinev_tedor_sim.py
is a command line function that runs a basic ZF-TEDOR simulation using both SpinEv and Bessel Functions with an
input distance and t2. It can be run for 13C, 15N, 19F, or 31P nuclei for the Bessel functions and any nucleus with
SpinEv. 

The actual fitting function is tedor_fitting.py, which is a script that takes all the relevant information about
the experimental data, including the file structure where the spectra are located, a peak list formatted as a .tsv, and a .tsv
that contains information about the location of the experimental data. You input which 
peak from the peak list that you want to fit, and it outputs that peak's build-up curve and the best fit curves 
for both SpinEv and Bessel fitting, along with the best fit distances. You can run it either for an individual peak,
a set of peaks, or the entire peak list. You can also choose to save the output as a .csv and whether to
plot the best fit curve(s) or whether to save them. 

There is also a function to pick peaks from multiple 1D spectra and select peaks that appear in multiple spectra.
This uses the nmrglue auto peak picking algorithm and should be used with caution. Jackknife error estimation is used to
determine the fitting error for both SpinEv and Bessel function fitting. This adds considerably to the run time, particularly
if using SpinEv. This can be turned off such that no fitting error estimation is provided.

*Last Updated 30 March 2022 by KMM*

