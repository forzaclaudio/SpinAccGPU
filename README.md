Copyright (c) 2014-2016, David Claudio Gonzalez
===============================================================================
SpinAccGPU
===============================================================================

This code implements a solution of the Zhang and Li equation using a Runge-
Kutta 4th order integrator.

In order to build a binary it is necesary to modify the Makefile in the main 
directory with the path to the corresponding CUDA installation available.
Once the proper path is indicated you can build the binary with the following
command:

  $ make

Make sure to update the contents of the parameters.h file to setup your
simulation.

Some of the available options are:

  $ make clean

which clears all folders and files from the latest build

  $ make testATW

which builds an executable that solves an ATW for testing purposes.

  $ make testVW

does the same as the previous expect that uses a VW magnetizations as input.
Make sure to check out the README file in the validation directory to perform 
the validation of your codes against the reference *-spin.dat and *-eff.dat
files provided both upVW and ATWpm types of Domain Wall configurations
