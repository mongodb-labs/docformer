# json2vec Experiments

We compare our model against baselines (Logistic Regression, Random Forests, XGBoost, LightGBM) on the same benchmark datasets proposed in [A Framework for End-to-End Learning on Semantic Tree-Structured Data](https://arxiv.org/abs/2002.05707) by William Woof and Ke Chen. These datasets were originally taken from the UCI repository and have been converted from tabular form to JSON structure.

First, make sure you have restored the datasets from the mongo dump file as described in [../README.md](../README.md). All commands (see below) must be run from the `json2vec` directory.

### Hyper-parameter tuning

To conduct a hyper-parameter search for a model, use the following command:

```
NUMPY_EXPERIMENTAL_DTYPE_API=1 guild run <model>:hyperopt dataset=<dataset> --optimizer random --max-trials <num>
```

This will evaluate `<num>` random combinations for model `<model>` on a 5-fold cross-validation for the dataset `<dataset>`:

- `<model>` is the model name, choose from `origami`, `logreg`, `rf`, `xgboost`, `lightgbm`.
- `<dataset>` is the dataset config filename under [`./datasets`](./datasets/). For example `json2vec-car` refers to the file `json2vec-car.yml` file.

Each parameter combination is executed as a separate guild run. To see the best parameters, you can use

```
guild compare -Fo <model>:hyperopt -F"dataset=<dataset>" -u
```

Alternatively you can provide a `--label <label>` as part of the run command and filter the comparison like so:

```
guild compare -Fl <label> -u
```

Search for the column `test_acc_mean` and sort in descending order (press `S`). Take note of the run ID (an 8-digit hash) of the best run (first column).

To retrieve the flags of this particular run, use:

```
guild runs info <run-id>
```

### Running a hyperparameter configuration

To run a particular parameter configuration on a dataset, use the following command:

```
guild run <model>:all dataset=<dataset> <param1>=<value1> <param2=value> ...
```

- `<model>` is the model name, choose from `origami`, `logreg`, `rf`, `xgboost`, `lightgbm`.
- `<dataset>` is the dataset config name under `./datasets`. For example `json2vec-car` refers to the file `json2vec-car.yml` file.
- parameters are provided as `<param>=<value>`. For example, to change the number of layers in the model to 6, use `model.n_layer=6`. All available parameters can be found in the [`./flags.yaml`](./flags.yml) file.

### Best ORiGAMi parameters for each dataset

For convenience, we list the invocations with the best hyperparameters we provided in the paper.

#### automobile dataset

```
guild run origami:all dataset=json2vec-automobile model.n_embd=160 model.n_head=8 model.n_layer=5 pipeline.sequence_order=SHUFFLED pipeline.n_bins=10 pipeline.upscale=400 train.batch_size=10 train.n_batches=10000 train.learning_rate=4e-5 cross_val=5-fold
```

#### bank dataset

```
guild run origami:all dataset=json2vec-bank model.n_embd=160 model.n_head=4 model.n_layer=5 pipeline.sequence_order=SHUFFLED pipeline.upscale=4 train.batch_size=50 train.n_batches=10000 cross_val=5-fold
```

#### car dataset

```
guild run origami:all dataset=json2vec-car model.n_embd=64 model.n_head=4 model.n_layer=4 pipeline.sequence_order=SHUFFLED pipeline.upscale=4 train.batch_size=100 train.n_batches=10000 cross_val=5-fold
```

#### contraceptive dataset

```
guild run origami:all dataset=json2vec-contraceptive model.n_embd=24 model.n_head=4 model.n_layer=3 pipeline.sequence_order=SHUFFLED pipeline.upscale=1000 train.batch_size=100 train.n_batches=30000 cross_val=5-fold
```

#### mushroom dataset

```
guild run origami:all dataset=json2vec-mushroom model.n_embd=64 model.n_head=4 model.n_layer=4 pipeline.sequence_order=SHUFFLED pipeline.upscale=100 train.batch_size=100 train.n_batches=10000 cross_val=5-fold
```

#### nursery dataset

```
guild run origami:all dataset=json2vec-nursery model.n_embd=64 model.n_head=4 model.n_layer=4 pipeline.sequence_order=SHUFFLED pipeline.upscale=40 train.batch_size=100 train.n_batches=10000 cross_val=5-fold
```

#### seismic dataset

```
guild run origami:all dataset=json2vec-seismic model.n_embd=16 model.n_head=4 model.n_layer=4 pipeline.sequence_order=SHUFFLED pipeline.upscale=1000 train.batch_size=100 train.n_batches=10000 cross_val=5-fold
```

#### student dataset

```
guild run origami:all dataset=json2vec-student model.n_embd=52 model.n_head=4 model.n_layer=4 pipeline.sequence_order=SHUFFLED pipeline.upscale=1000 train.batch_size=10 train.n_batches=10000 cross_val=5-fold
```
