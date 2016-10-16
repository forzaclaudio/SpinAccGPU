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

def test_lines(line1,line2):
    ref_values = line1.split()
    test_values = line2.split()
    err1 = abs(float(ref_values[0])-float(test_values[0]))
    err2 = abs(float(ref_values[1])-float(test_values[1]))
    err3 = abs(float(ref_values[2])-float(test_values[2]))
    errorinline= err1+err2+err3
    return errorinline

def test_files(options):
    parameters_file = options.paramsfile
    reference_file = options.referencefile
    test_file = options.testfile

    if options.verbose:
        print 'Performing Unit testing of file %s'% test_file
#Compute the size of the files to be compared
    if options.verbose:
        print 'Computing size of files to compare...'
    if os.path.isfile(parameters_file):
        #use grep command to locate line where NX size is especified
        p = subprocess.Popen(['grep','-i','NX', parameters_file], shell=False,
                              stdout=subprocess.PIPE)
        sizeX = p.stdout.readline()
        sizeX = sizeX.split()
        #use grep command to locate line where NY size is especified
        p = subprocess.Popen(['grep', '-i', 'NY', parameters_file], shell=False,
                             stdout=subprocess.PIPE)
        sizeY = p.stdout.readline()
        sizeY = sizeY.split()

        numLines = int(sizeX[2])*int(sizeY[2])
    if options.verbose:
        print "Number of lines in file is: %i" % numLines
# Coverage of testing in % of total lines
        linesCoverage = float(options.coverage)
        if options.verbose:
            print "Coverage of random lines will be: %3.1f %% or %i lines" \
                  % (float(linesCoverage)*100, linesCoverage*numLines)
        linesSampled = random.sample(range(numLines),
                                     int(linesCoverage*numLines))
        linesSampled.sort();
        try:
            f1 = open(reference_file,'r')
            f2 = open(test_file,'r')
        except:
            print 'Error when opening file, Aborting!'
            exit(1)
        f1.seek(0)
        f2.seek(0)
        counter = 0
        index = 0
        linesCovered = int(linesCoverage*numLines)
        Totalerror = 0
        f1.readline()
        f2.readline()
        for lines in f1:
# iterate on file 1 and read a line from file 2
            lines2=f2.readline()
            if index < linesCovered:
                if counter == linesSampled[index]:
                    test_lines(lines,lines2)
                    Totalerror += test_lines(lines,lines2)
                    if options.verbose:
                        print "Error in line is %e" % test_lines(lines,lines2)
                    index=index+1
            counter= counter+1
        f1.close()
        f2.close()
        if options.verbose:
            print 'Total error in file per line is: %e' \
                   % (Totalerror/linesCovered/3)
        else:
            print (Totalerror/linesCovered/3)
    else:
        print 'Parameters file doesn\'t exist aborting'
        exit(1)

def main():
    p = optparse.OptionParser()
    p.add_option('--verbose', '-v', action='store_true', default=False,
                 dest='verbose')
    p.add_option('--coverage', '-c', help = "Number of lines covered by " + 
                 "anaylsis. Has value between 0.0 and 1.0")
    p.add_option('--paramsfile','-p',help = "Name of file of parameters used " +
                 "in compilation")
    p.add_option('--testfile','-t',help = "Name of file with data to test")
    p.add_option('--referencefile','-r',help="Name of file with reference data")
    options, argument = p.parse_args()
    if not options.coverage:
        print 'It is necessary to indicate the coverage'
    elif not options.paramsfile: 
        print 'A file with parameters used for compilation is required'
    elif not options.testfile: 
        print 'A file with data to test is required'
    elif not options.referencefile: 
        print 'A file with reference data is required'
    else:
        if float(options.coverage) > 0 and float(options.coverage) <=1:
            if options.verbose: 
                print "running with %s %s %s %s" %(options.coverage, 
                                                   options.paramsfile, 
                                                   options.testfile, 
                                                   options.referencefile)
            test_files(options)
        else:
            print 'Coverage should be between 0 (>) and 1.0 (<=)! Aborting'
            exit(1)

if __name__ == '__main__':
    main()
