#!/bin/bash
# Remember to change mode of this file amd make executable
# chmod 755 run.sh

#echo "Launching condor on test"
#echo ${3} ${4} ${5}
export MPLCONFIGDIR=/tmp
# conda activate /mnt/jeoproc/envs/ML4CAST/YF3_dock_p311
/mnt/jeoproc/envs/ML4CAST/YF3_dock_p311/bin/python3.11 /eos/jeodpp/data/projects/ML4CAST/ml4cast-ml/condor_launcher.py ${3} ${4} ${5} ${6}
#/mnt/jeoproc/envs/BDA/test_env/condabin/conda run -p /mnt/jeoproc/envs/ML4CAST/YF3_dock_p311/ python3.11 /eos/jeodpp/data/projects/ML4CAST/ml4cast-ml/condor_launcher.py ${3} ${4} ${5}
#echo "done"
