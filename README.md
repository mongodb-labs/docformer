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

ORiGAMi is a transformer-based Machine Learning model for supervised classification from semi-structured data such as MongoDB documents or JSON files.

Typically, when working with semi-structured data in a Machine Learning context, the data needs to be flattened into a tabular format first. This flattening can be lossy, especially in the presence of arrays and nested objects, and often requires domain expertise to extract meaningful higher-order features from the raw data. This feature extraction step is manual, slow and expensive and doesn't scale well.

ORiGAMi circumvents this by directly operating on JSON data. Once a model is trained, it can be used to make predictions on any field in the dataset.

## Installation

ORiGAMi requires Python version 3.10 or 3.11. We recommend using a virtual environment, such as
Python's native [`venv`](https://docs.python.org/3/library/venv.html).

To install ORiGAMi with `pip`, use

```shell
pip install origami-ml
```

You can also clone the repository to your local machine and install the dependencies manually:

```shell
git clone https://github.com/mongodb-labs/origami.git
cd origami
pip install -r requirements.txt
pip install -e .
```

## Usage

ORiGAMi comes with a command line interface (CLI) and a Python SDK.

### Usage from the Command Line

The CLI allows to train a model and make predictions from a trained model. After installation, run `origami` from your shell to see an overview of available commands.

Help for specific commands is available with `origami <command> --help`, where `<command>` is currently one of `train` or `predict`. Note that the first time you run the `origami` CLI tool can take longer.

Detailed documentation for the CLI and available options can be found in [`CLI.md`](CLI.md).

### Usage with Python

To see an example on how to use ORiGAMi from Python, take a look at the provided [./notebooks](./notebooks/) folder, e.g. the [`example_origami_dungeons.ipynb`](./notebooks/example_origami_dungeons.ipynb) notebook.

## Experiment Reproduction

This code is released alongside our paper, which can be found on Arxiv: [ORIGAMI: A generative transformer architecture for predictions from semi-structured data](https://arxiv.org/abs/2412.17348). To reproduce the experiments in the paper, see the instructions in the [`./experiments/`](./experiments/) directory.
