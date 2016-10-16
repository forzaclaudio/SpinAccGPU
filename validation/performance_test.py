#!/usr/bin/python
# Copyright (C) 2014-2016 David Claudio Gonzalez
# This script is used to compare the *-spin.dat and *-eff.dat
# in case there is a significant disagreement the new code is
# not producing the right result so far testing is taking place
# according to the following formula:
# sum [sum abs(testfileline - reffileline)_i ] / #lines / 3
# where i is the column of the data
import os
import shutil
import optparse
import subprocess
import random

#reference values are first argument, test are second
def test_time(line1,line2):
    ref_values = line1.split()
    test_values = line2.split()
    time_diff = abs(float(test_values[3])-float(ref_values[3]))
    return time_diff

#reference values are first argument, test are second
def test_beta(line1,line2):
    ref_values = line1.split()
    test_values = line2.split()
    beta_diff = abs(float(test_values[2])-float(ref_values[2]))
    return beta_diff

#reference values are first argument, test are second
def get_acceleration(line1,line2):
    ref_values = line1.split()
    test_values = line2.split()
    acceleration = float(ref_values[3])/float(test_values[3])
    return acceleration

def test_files(options):
    test_file = options.testfile
    reference_file = options.referencefile
    if options.verbose:
        print 'Performance testing of file {0}'.format(test_file)
#Compute the size of the files to be compared
    if options.verbose:  
        print 'Extracting diffusive beta from test file...'
    if os.path.isfile(test_file):
        #use grep command to locate line where Diffusive beta value is
        p = subprocess.Popen(['grep', '-i', 'Diffusive beta', test_file], 
                             shell=False, stdout=subprocess.PIPE)
        Diff_beta_test = p.stdout.readline()
        if options.verbose: 
            print 'Diffusive beta test read: %s'% Diff_beta_test
        if options.verbose:  
            print 'Extracting running time from test file...'
        #use grep command to locate line where Diffusive beta value is
        p = subprocess.Popen(['grep', '-i', 'Calculation completed in', 
                              test_file], shell=False, stdout=subprocess.PIPE)
        Calculation_time_test = p.stdout.readline()
        if options.verbose: 
            print 'Calculation time test read: %s'% Calculation_time_test
    else:
        print "File %s test file not found. Aborting!", test_file
        exit(1)
    if os.path.isfile(reference_file):
        #use grep command to locate line where Diffusive beta value is
        p = subprocess.Popen(['grep', '-i', 'Diffusive beta', reference_file],
                             shell=False, stdout=subprocess.PIPE)
        Diff_beta_ref = p.stdout.readline()
        if options.verbose: 
            print 'Diffusive beta test read: %s'% Diff_beta_ref
        #Diff_beta_test = Diff_beta_test.split()
        if options.verbose:  
            print 'Extracting running time from test file...'
        #use grep command to locate line where Diffusive beta value is
        p = subprocess.Popen(['grep', '-i', 'Calculation completed in',
                              reference_file],shell=False, 
                              stdout=subprocess.PIPE)
        Calculation_time_ref = p.stdout.readline()
        if options.verbose: 
            print 'Calculation time test read: %s'% Calculation_time_ref
    else:
        print "File %s test file not found. Aborting!", reference_file
        exit(1)
    beta_value = test_beta(Diff_beta_ref, Diff_beta_test)
    if options.verbose:   
        print "Diffusive beta value is: ", beta_value
    if beta_value <= 1e-16:
        print "Diffusive beta variation is: {0:f} passed!".format(beta_value) 
    else: 
        print "Diffusive beta variation is: {0:f} failed!".format(beat_value)
    time_value = test_time(Calculation_time_ref, Calculation_time_test)
    if options.verbose: 
        print "Execution time difference is: {0:f} ms".format(time_value)
#Check if execution time varies for more than a second
    acceleration_value = get_acceleration(Calculation_time_ref,
                                              Calculation_time_test) 
   
    if time_value < 1000.0: 
        print "Execution time is comparable!"
    elif acceleration_value > 1.0:
        print "Execution time is: {0:.3f}x faster!".format(acceleration_value)
    elif acceleration_value < 1.0:
        print "Execution time is: {0:.3f}x slower".format(1/acceleration_value) 

def main():
    p = optparse.OptionParser()
    p.add_option('--verbose', '-v', action='store_true', default=False, 
                 dest='verbose')
    p.add_option('--testfile', '-t', help = "Name of file with data to test")
    p.add_option('--referencefile', '-r', help="Name of file with reference " +
                 "data")
    options, argument = p.parse_args()
    if not options.testfile: 
        print "A test file is needed!"
    elif not options.referencefile: 
        print "A reference file is needed!"
    else: 
        test_files(options)

if __name__ == '__main__':
    main()