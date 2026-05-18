#   Transfer Function Coefficient Regression

The idea of this package is to be able to regress for filter coefficents where the stages, types of filter etc. can be specified in the optimisation
bounds. Below describes the characteristics of different common filters and how to enforce them in the optimisation

##  How to Use
To run the regression simply run `./run_regression.py` with the required arguments. Run `./run_regression.py --help` to see the arguments.

##  General Structure and Things to Note
The most important thing here is the bounds that are imposed to ensure we get the correct filter type. If any of the ones I've put in are wrong, the 
function `get_bounds` in `bfgs.py` creates a namespace that gets filled with all the optimisation bounds explicitly, so they can be altered there.

As the forward model uses `scipy.signal.freqz`, the poles are zeros are transformed form Polar to Cartesian in `objective_lbfgs` in `bfgs.py`. If the 
filter type is linear then the reciprocals of the current model vector are calculated and concatenated. Finally, the conjugates are calculated and appended in
`g_scipy` in `shared.py`.


##  Filter Characteristics
It is assumed throughout that all poles and zeros are symmetric over the real axis where applicable

### FIR vs IIR
FIR filters have all poles at the origin. This can be enforced by setting `set_poles = np.zeros(nz)`. IIR filters can have poles anywhere but are
no longer stable or causal (NOTE: The regression doesn't currently support IIR filters.).

### Minimum Phase
These are stable, causal FIR filters with ALL zeros inside the unit circle. They are usually FIR but can be approximated with IIR if needed.

### Linear Phase
For exactly linear phase response the filter must be FIR and the zeros come in sets of 4 instead of 2. For a complex zero, z, we need its conjugate, z',
as well as its reciprocal and reciprocal conjugate, 1/z and 1/z'. Again these can be approximated with IIR filters but will not have perfectly linear phase.
The expanded filter coefficients are also either symmetric or anti-symmetric and the locations of the zeros are mirrored around the unit circle.

### Texas Instruments Data
These are some of the values of things from the nanometric filter that the regressed filter should conform to (ish)

    Stop Band:  0.5*sample_rate (250Hz)
    Corner Freq (-3dB): 0.413*sampling_rate (206.5Hz)
    Group Delay:    0.01s for min-phase and 0.126s for linear phase


