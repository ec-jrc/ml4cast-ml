#!/bin/bash


Cluster=${1}
uset_file=${3}

echo "python3 /eos/jeodpp/data/projects/ML4CAST/condor/PY/condor_launcher_nrt.py $uset_file "

python3 /eos/jeodpp/data/projects/ML4CAST/condor/PY/_condor_launcher_nrt.py $uset_file

#sleep infinity
echo "done"
