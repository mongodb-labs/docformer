<!-- ## General Notes

### Cross-validation and train/test splits

The behaviour is controlled by the `cross_val` flag.

- `cross_val=none` disables cross-validation and uses a simple train/test split
- `cross_val=5-fold` creates 5 folds for cross-validation
- `cross_val=catalog` uses the pre-defined split indices in the `openml.catalog` collection (only for OpenML datasets)

Additional parameters:

- `train.test_split` is the fraction of the test dataset when cross-validation is disabled
- `train.shuffle_split` whether or not to shuffle rows (both for cross-validation splits and train/test splits)

Some examples below:

#### Single run with default train/test split

Default test split is 0.2 and shuffled.

```
guild run <model>:all dataset=<dataset> cross_val=none
```

#### Single run with custom train/test split

We choose a split of 60/40 and no shuffling.

```
guild run <model>:all dataset=<dataset> cross_val=none train.test_split=0.4 train.shuffle_split=no
```

#### 5-fold cross validation, unshuffled

`train.test_split` is ignored.

```
guild run <model>:all dataset=<dataset> cross_val=5-fold train.shuffle_split=no
```

#### k-fold cross-validation from catalog

This loads the split indices in the `openml.catalog` collection, which are stored
under the field path `task.cross_validation`.

`k` is usually 10, but may potentially differ, based on the splits defined in the `catalog` collection.

`train.test_split` and `train.shuffle_split` are ignored.

```
guild run <model>:all dataset=tictactoe cross_val=catalog
``` -->

# Reproducing the results from our paper

We use the open source library [guild.ai](https://guild.ai) for experiment management and result tracking.

### Datasets

We bundled all datasets used in the paper in a convenient [MongoDB dump file](). To reproduce the results, first
you need MongoDB installed on your system (or a remote server). Then, download the dump file, unzip it, and restore it into your MongoDB instance:

```
mongorestore dump/
```

This assumes your `mongod` server is running on `localhost` on default port 27017 and without authentication. If your setup varies, consult the [documentation](https://www.mongodb.com/docs/database-tools/mongorestore/) for `mongorestore` on how to restore the data.

If your database setup (URI, port, authentication) differs, also make sure to update the [`.env.local`](.env.local) file in this directory accordingly.

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
guild run <model>:all dataset=<dataset> <param1>=<value1> <param2=value>
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
