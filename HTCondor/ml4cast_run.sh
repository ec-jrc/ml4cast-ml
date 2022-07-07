#!/bin/bash


Cluster=${1}
uset_file=${3}

export PYTHONPATH=/eos/jeodpp/data/projects/ML4CAST/condor/ml4cast-be-ml

echo "python3 /eos/jeodpp/data/projects/ML4CAST/condor/ml4cast-be-ml/condor_launcher.py $uset_file "

python3 /eos/jeodpp/data/projects/ML4CAST/condor/ml4cast-be-ml/HTCondor/condor_launcher.py $uset_file

#sleep infinity
echo "done"
