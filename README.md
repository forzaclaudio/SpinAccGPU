Copyright (c) 2014, David Claudio Gonzalez
===========
SpinAccGPU
===========

This code implements a solution of the Zhang and Li equation using a Runge Kutta 4th order integrator.

In order to build a binary it is necesary to modify the Makefile in the main 
directory with the path to the corresponding CUDA installation available.
Once the proper path is indicated you can build the binary with the following
command:
$make 

Some of the available options are:
$make clean
clears all folders and files from the latest build

$make testATW
builds an executable and decompresses into the tests dir. In order to run the
test just enter the name of the binary. It's important to use the right 
name file and parameters in the parameters.h file under the include directory
$./SpinAccGPU

$make testVW
same as above except that the data file used corresponds to magnetization 
configuration of a VW

before running validation tests it is necessary to gunzip all the contents of 
validation directory this is accomplished with the following command:
$gunzip -r validation
make sure to check out the README file there to perform the validation of your
improved codes against the reference *-spin.dat and *-eff.dat files there.
We provide reference files for both upVW and ATWpm types of Domain Wall 
configurations
