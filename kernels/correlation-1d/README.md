# Project Name
Correlation 1D

## Overview
The developed GPU kernels implement 1 dimensional correlation.

##Usage


## Performance Notes
- Key optimisations
	- The filter is defined as constant memory, aiming at broadcasting filter elements to threads as each thread is using the same filter
	- Used template to determine the filter length
	- Loop unrolling

## Profiling


## Results
