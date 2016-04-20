#!/bin/bash

f2py --verbose --f90flags='-fopenmp -O3 -ftree-vectorize' -lgomp -c -m f90routines f90routines.f90

