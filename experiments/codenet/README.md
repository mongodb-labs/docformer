# CodeNet Java Experiments

In this experiment, we convert Java code snippets from the [CodeNet](https://developer.ibm.com/exchanges/data/all/project-codenet/) dataset into Abstract Syntax Trees and store them as JSON objects.
We then train an ORiGAMi model on these ASTs for a classification task, where the programming problem ID is the target label. More details on the dataset and classification task can be found
in the paper [CodeNet: A Large-Scale AI for Code Dataset for Learning a Diversity of Coding Tasks](https://arxiv.org/abs/2105.12655) by Ruchir Puri et al.

First, make sure you have restored the datasets from the mongo dump file as described in [../EXPERIMENTS.md](../EXPERIMENTS.md). All commands (see below) must be run from the `codenet` directory.

### Training and evaluating the model

Due to resource constraints, we did not perform a hyperparameter optimization. We use a model with 4 transformer layers, 4 heads and 192 embedding dimensionality. All parameters are
configured as defaults in the `guild.yml` file.

To run the training and evaluation on the test set, use:

```bash
guild run train
```

Note: Training with the default parameters requires est. 50 GB of GPU RAM.
