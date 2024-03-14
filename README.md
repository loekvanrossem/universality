# Universal Representation Learning Dynamics

This repository contains code for experiments investigating a universal framework for learning dynamics in deep neural networks, as discussed in the paper ["When Representations Align: Universality in Representation Learning Dynamics"](https://arxiv.org/abs/2402.09142).

## Overview

The experiments presented here compare results from an effective theory derived in the paper to the learning dynamics of various architectures and learned representational structures across different toy datasets. Implementations of these experiments can be found in the "experiments" folder.


## Contents

- **source:** This folder contains source code and utilities used for conducting experiments.
- **experiments:** This folder contains implementations of experiments comparing the learning dynamics across different architectures and datasets, as well as a folder containing the hyperparameter settings used in the paper.
- **environment.yaml:** This file lists the required packages and dependencies needed to run the experiments.

## Usage

1. Ensure you have Python 3.10 or higher installed.
2. Set up the environment using the provided `environment.yaml` file.

```bash
    conda env create -f environment.yaml
    conda activate universality
```

3. Navigate to the "experiments" folder and run the desired experiment script.
4. Results will be saved in the "experiments/plots" directory for further analysis.

## Note

- The theoretical prediction for the XOR dataset (`XOR.ipynb`) requires the effective learning rates computed in the rich structure comparison (`rich_structure.ipynb`). Make sure to run the latter first before running experiments related to the XOR dataset.
