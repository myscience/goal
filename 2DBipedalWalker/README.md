# GOAL: The Code Repository 

This repository hosts the official implementation of `GOAL` as described in the paper: *Error Based or Target Based? A unifying framework for learning in recurrent spiking networks*, Plos Computation Biology (2022).

## Bipedal Walker 2D
This section of the repository is dedicated to the `Bipedal Walker 2D` task. The core result of the analysis is reported in *Figure 5* of the paper (see below), which highlights how higher different values of \tau_{star} (tolerance to spike timing) bring to different performances.


#### 1. Specification of dependencies


The code is written in `Python 3` and requires the following external dependences to run:

```
matplotlib
numpy
os
tqdm
gym
box2d-py
swig
```

Each package can be installed through `pip3` (or anaconda) by running:

> pip3 install <name_of_package>

With the exception of

```
swig
```

that should be installed with `conda` by running:

> conda install <name_of_package>

#### 2. Training-Evaluation code

The training and the evaluation of the model are performed at the same time. A period of training is alternated to a period of test.

> python3 cloner.py

This script runs the algorithms for different values of the rank [4,500] and of tau_targ [0.1,3.,10.]ms._

#### 3. Visualization code

To plot the results run:

> python3 plotRewards_tau.py

This can be done also without running the previous step, the results are pre-saved.

