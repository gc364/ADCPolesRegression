#   Transfer Function Coefficient Regression

The idea of this package is to be able to regress for filter coefficents where the stages, types of filter etc. can be specified in the optimisation
bounds. Below describes the characteristics of different common filters and how to enforce them in the optimisation

##  Filter Characteristics
It is assumed throughout that all poles and zeros are symmetric over the real axis where applicable

### FIR vs IIR
FIR filters have all poles at the origin. This can be enforced by setting `set_poles = np.zeros(nz)`. IIR filters can have poles anywhere but are
no longer stable or causal.

### Minimum Phase
These are stable, causal FIR filters with ALL zeros inside the unit circle. They are usually FIR but can be approximated with IIR if needed.

### Linear Phase
For exactly linear phase response the filter must be FIR and the zeros come in sets of 4 instead of 2. For a complex zero, z, we need its conjugate, z',
as well as its reciprocal and reciprocal conjugate, 1/z and 1/z'. Again these can be approximated with IIR filters but will not have perfectly linear phase.
