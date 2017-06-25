# Analysis of taxi tips patterns in NYC

This repository contains scirpt for analyzing the NYX taxi data with focus on determining if there are areas with .

## Installation of required libraries

Start by installing the basic packages:

```
$ sudo apt-get install build-essential

```

Install Anaconda from a terminal (this is for 32 bit version):
```
$ wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda-2.3.0-Linux-x86.sh
$ bash Anaconda-2.3.0-Linux-x86.sh
```


Install Anaconda from a terminal (this is for 64 bit version):
```
$ wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda-2.3.0-Linux-x86_64.sh
$ bash Anaconda-2.3.0-Linux-x86_64.sh
```

After Anaconda is installed install python version 2.7 and the basic packages:

```
$ conda create -n py27 python=2.7 anaconda
$ activate py27
$ conda update conda
$ conda install scikit-learn
$ conda install shapely
```


## Running the analysis

The scirpts provided below will try to automatically download the data if it is not located in the disk.
In addition, analysis can be time consuming, with the more detailed level (neigborhoods) taking several hours to runs

To generate the histograms that compare the top level areas (boroughs) of NYC run:

```
$ activate py27
$ python compare_boroughs.py
```

To generate the detailed t-tests that compare the lower level areas (neigborhoods) of NYC run:
```
$ activate py27
$ python compare_neighborhoods.py
```

Finally, to get a feeling of the individual factors that contribute to a higher tip run:
```
$ activate py27
$ python rank_features.py
```

## Notes and TODOs

- [ ] We have re-used code from several existing repositories, for example https://github.com/kthouz/NYC_Green_Taxi
- [ ] In the current analysis population size is not considered explicitly (due to time limitations). We could resample and average a fixed boostrap size for each area to acount for that
- [ ] The feature ranking from tree-based ensembles is biased against categorical predictors (tends to overemphasize continuous predictors). We could do more sophisticated feature ranking  
- [ ] Since we only run less than 50 neigborhood t-tests it is not critical but if more are run (like the full 266) Bonferroni and similar corrections should be applied to counteract the problem of multiple comparisons 

