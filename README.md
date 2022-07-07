# Machine Learning methods for regional yield forecasting
Training various Machine Learning algorithms on satellite and meteorological data  time series for regional yield forecasting.
Methods described in the paper [Yield forecasting with machine learning and small data: what gains for grains?](https://doi.org/10.1016/j.agrformet.2021.108555)

### Table of contents
* [Prerequisites](#Prerequisites)
* [Script overview](#Script overview)
* [Contributors](#Contributors)





## Prerequisites
The script assumes that input data are extracted from the JRC ASAP Early Warning System (https://mars.jrc.ec.europa.eu/asap/download.php)
at the appropriate administrative level that corresponds to one which yield statistics refer to.
The code runs on CPU machines. Model tuning can run on single machine but it is very time consuming. We used mostly HT Condor
running on the JRC Big Data Analytics Platform (https://jeodpp.jrc.ec.europa.eu/bdap/).

Use environment.yml to set up the environment.


## Script overview
Main folders described in logical order:
* ```preprocess```: Collection of scripts to load, reformat, analyze and map input data and yield statistics. It includes
the analysis of yield trend and of correlation between explanatory variables
* ```src```: various variables are set constants.py, including: data paths, type of features normalization, hyperparameters
search space, etc.
* ```project root```: main scripts for the tuning of ML models
* ```ml```: classes for the yield forecasting scripts
* ```postprocess_analysis```: main scripts for the evaluation and comparison of tuned ML algorithms
* ```viz```: scrpts for visualizing ML algorithms' performance

Folders containing supporting code or work-in-progress code:
* ```dynamic-masking```: dynamic crop group masking tests (work-in-progress)
* ```HTCondor```: script and bash commands to run the code in parallel on the JRC Big Data Analytics Platform
* ```ope```: scripts to apply best ML models to yield forecasting in operational applications
* ```run_managers```: supporting scripts to chain all necessary steps for a given Area Of Interest

## Contributors
 - [Dr. Michele Meroni](https://scholar.google.com/citations?user=iQk-wj8AAAAJ&hl=en&oi=ao)
 - [Dr. Franz Waldner](https://scholar.google.com/citations?user=4z2zcXwAAAAJ&hl=en&oi=ao)
 - [Dr. Lorenzo Seguini](https://www.researchgate.net/profile/Lorenzo-Seguini)
