<p align="center">
  <img src="assets/origami_logo.jpg" style="width: 100%; height: auto;">
</p>

# ORiGAMi - Object Representation through Generative Autoregressive Modelling

<p align="center">
| <a href="https://arxiv.org/abs/2412.17348"><b>ORiGAMi Paper on Arxiv</b></a> |
</p>

## Disclaimer

Please note: This tool is not officially supported or endorsed by MongoDB, Inc. The code is released for use "AS IS" without any warranties of any kind, including, but not limited to its installation, use, or performance. Do not run this tool against critical production systems.

## Overview

ORiGAMi is a transformer-based Machine Learning model to learn directly from semi-structured data such as JSON
or Python dictionaries.

Typically, when working with semi-structured data in a Machine Learning context, the data needs to be flattened
into a tabular form first. This flattening can be lossy, especially in the presence of arrays and nested objects, and often requires domain expertise to extract meaningful higher-order features from the raw data. This feature extraction step is manual, slow and expensive and doesn't scale well.

ORiGAMi is a transformer model and follows the trend of many other deep learning models by operating directly on the raw data and discovering meaningful features itself. Preprocessing is fully automated (apart from some hyper-parameters that can improve the model performance).

### Use Cases

Once an ORiGAMi model is trained on a collection of JSON objects, it can be used in several ways:

1. **Prediction**: ORiGAMi models can predict the value for any key of the dataset. This is different to typical discriminative models such as Logistic Regression or Random Forests, which have to be trained with a particular target key in mind. ORiGAMi is a generative model trained in order-agnostic fashion, and a single trained model can predict any target, given any subset of key/value pairs as input.
2. **Autocompletion**: ORiGAMi can auto-complete partial objects based on the probabilities it has learned from the training data, by iteratively sampling next tokens. This also allows it to predict complex multi-token values such as nested objects or arrays.
3. **Generation**: ORiGAMi can generate synthetic mock data by sampling from the distribution it has learned from the training data.
<!-- 4. **Embeddings**: As a deep neural network, ORiGAMi creates contextualized embeddings which can be extracted at the last hidden layer. These embeddings represent the objects in latent space and can be used as inputs to other ML algorithms, for data visualization or similarity search. -->

Check out the Juypter notebooks under [`./notebooks/`](./notebooks/) for examples for each of these use cases.

## Installation

ORiGAMi requires Python version 3.10 or higher. We recommend using a virtual environment, such as
Python's native [`venv`](https://docs.python.org/3/library/venv.html).

To install ORiGAMi with `pip`, use

```shell
pip install origami-ml
```

## Usage

ORiGAMi comes with a command line interface (CLI) and a Python SDK.

### Usage from the Command Line

The CLI allows to train a model and make predictions from a trained model. After installation, run `origami` from your shell to see an overview of available commands.

Help for specific commands is available with `origami <command> --help`, where `<command>` is currently one of `train` or `predict`.

Detailed documentation for the CLI and available options can be found in [`CLI.md`](CLI.md).

### Usage with Python

To see an example on how to use ORiGAMi from Python, take a look at the provided [./notebooks](./notebooks/) folder, e.g. the [`example_origami_dungeons.ipynb`](./notebooks/example_origami_dungeons.ipynb) notebook.

## Experiment Reproduction

This code is released alongside our paper, which can be found on Arxiv: [ORIGAMI: A generative transformer architecture for predictions from semi-structured data](https://arxiv.org/abs/2412.17348). To reproduce the experiments in the paper, see the instructions in the [`./experiments/`](./experiments/) directory.
