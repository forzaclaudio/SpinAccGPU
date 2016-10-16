#!/bin/bash
for i in {1..20..1}
do 
   ./run_tests.py -n $i
done
