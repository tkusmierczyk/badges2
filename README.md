Code used in the following paper:

**T. Kusmierczyk, K. Nørvåg: [On Validation and Predictability of Digital Badges' Influence on Individual Users.](https://arxiv.org/abs/1710.01716) (AAAI-18)**

-----------------------------------------------------------------------------------------------------------------

### Introduction

Badges are a common, and sometimes the only, method of incentivizing users to perform certain actions on online sites. However, due to many competing factors influencing user temporal dynamics, it is difficult to determine whether the badge had (or will have) the intended effect or not. 

We introduce two complementary approaches for determining badge influence on individual users. In the first one, we cluster users' temporal traces (represented with point processes) and apply covariates (user features) to regularize results. In the second approach, we first classify users' temporal traces with a novel statistical framework, and then we refine the classification results with a semi-supervised clustering of covariates. 

Both approaches we evalute on synthetic datasets and experiments on two badges from StackOverflow.

### Requirements

The code was tested on Ubuntu 16.04 (4.4.0-101-generic #124-Ubuntu SMP, x86_64 GNU/Linux) with **Python 3.5.2** (from Anaconda custom (64-bit), default, Jul  2 2016). 
The following libraries are required:
* numpy (1.13.3)
* pandas (0.20.3)
* matplotlib (2.1.0)
* scipy (0.19.1)
* scikit-learn (0.19.0)
* seaborn (0.7.1)
* tensorflow (1.3.0)

Furthermore, to run jupyter nootebooks we used **runipy (0.1.5)**.

### Data preparation

Data can be found in `data/` and `data/badges/` directories.
Files need to be unzipped before using them.

### Code execution    

To reproduce our experiments use the following:
* for synthetic experiments: `RUN.sh` in `src/synthetic/`
* for real data experiments: `RUN.sh` in `src/analytics/`

