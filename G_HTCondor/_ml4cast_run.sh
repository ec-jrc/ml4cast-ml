#!/bin/bash


#Cluster=${1}
uset_file=${3}
config_fn=${4}
run_name= ${5}

export PYTHONPATH=/eos/jeodpp/data/projects/ML4CAST/ml4cast-ml

echo "Launching condor on $uset_file "

python3 /eos/jeodpp/data/projects/ML4CAST/ml4cast-ml/G_HTCondor//condor_launcher.py $uset_file $config_fn $run_name

#sleep infinity
echo "done"
