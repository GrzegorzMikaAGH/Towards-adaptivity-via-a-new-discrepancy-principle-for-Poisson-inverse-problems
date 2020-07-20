# Towards adaptivity via a new discrepancy principle for Poisson inverse problems

Simulation code accompanying the article G.Mika, Z.Szkutnik, "Towards adaptivity via a new discrepancy principle for Poisson inverse problems".

The replication of simulation results is possible via a provided set of codes. For each of considered methods (*Landweber*, *1-* and *2-times iterated Tikhonov* and *TSVD*) and for each presented function (*Beta*, *Bimodal*, *SMLA*, *SMLB*) there is a separate code file with the following naming convention: `FunctionName_MethodName.py`. The list of requirements is provided in file `requirements.txt`. To execute the code a minimal required version of Python is 3.7.

Each code file produces a set of output files in a format of *csv* files with following naming convention: `MethodName_fun_FunctionName_size_SampleSize_tau_ParamterTau.csv`, where `SampleSize` and `ParamterTau` are customizable parameters and can be edited inside a respective code files. By default, they are set to the values used in the article. 

For Linux-based systems, a shortcut is provided in a format of `shell` file (`run_simulations.sh`). The code expects to be executed inside a repository folder and in a proper python environment with the required packages installed.
