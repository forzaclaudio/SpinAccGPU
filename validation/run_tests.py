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

def unit_test(test_string, options):
#use unit_test.py c to obtain error from random coverage
    average_error = 0.0
    for i in range(int(options.numofruns)):
        if options.verbose: 
            print "Executing command: ./unit_test.py" + test_string
        p = subprocess.Popen('./unit_test.py'+ test_string, shell=True, 
                             stdout=subprocess.PIPE)
        error = p.stdout.readline()
        average_error += float(error)
        if options.verbose: 
             print i, error
    if options.verbose:
        print "Average error is:", average_error/float(options.numofruns)
    else:
        print average_error/float(options.numofruns)
def main():
    p = optparse.OptionParser()
    p.add_option('--verbose', '-v', action="store_true", default=False,
                 dest='verbose')
    p.add_option('--numofruns', '-n', help = "Indicates the number of runs " +
                 "required to obtain average error")
    options, argument = p.parse_args()
    if not options.numofruns: 
        print 'It is necessary to indicate the number of runs'    
    else: 
        unit_test(' -c 1.0 -p ../include/parameters.h -t ' +
                  '../tests/upVW-magn-2.5nm-eff.dat -r upVW-magn-2.5nm-eff.dat',
                  options)

if __name__ == '__main__':
	main()