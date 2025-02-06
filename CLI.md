# Command Line Interface

An ORiGAMi model can be trained through a command line interface (CLI) from a MongoDB collection or JSON or CSV files.

## `train` Command

The general invocation of the `train` command is:

```shell
origami train SOURCE [OPTIONS]
```

### General Options

#### Help and Verbosity

To list an overview of the available options, use the `--help` option.

To print more verbose information about pipeline execution, model details and configuration options, use the `--verbose` (or `-v`) option.

#### Model Path

The trained model will be saved to disk, with a default path and filename of `./model.origami`. To change the path or name, use the `--model-path` (or `-m`) option.

For example, to change the model file name to `./snapshots/orders-dev.origami`, use the followig command:

```shell
origami train <SOURCE> --model-path ./snapshots/orders-dev.origami
```

#### Changing the Random Seed

To choose a particular random seed (e.g. for reproducibility), use the `--seed` option (default is 1234).

For example, to change the random seed to 42, use the command:

```shell
origami train <SOURCE> --seed 42
```

### Source Selection

The CLI allows training from a number of different sources:

- A running MongoDB instance: use `mongodb://<host>:<port>` connection URI as SOURCE and `--source-db`, `--source-coll` options to specify the database and collection name
- a `.jsonl` file, where each line is a JSON object
- a `.json` file, which needs to contain an single array of JSON objects
- a `.csv` file with header row
- a directory containing any of the supported file types: specify the directory path as SOURCE

### Additional Source Options

#### MongoDB Source

When the source is a MongoDB instance, the parameters `--source-db` (or `-d`) and `--source-coll` (or `-c`) are required.

For example, when the source is the `ecommerce.orders` namespace in a MongoDB instance running on localhost and standard port 27017, the source can be specified as:

```shell
origami train mongodb://localhost:27017 --source-db ecommerce --source-coll orders
```

When training from a file or a directory containing supported files, these two options are ignored.

#### Including and Excluding Fields

To train a model on a subset of fields in the object, use either the `--include-fields` (or `-i`) or the `--exclude-fields` (or `-e`) options, followed by a comma-separated list of field names. Note that if neither option is specified, the `_id` field is excluded by default (if it exists).

Nested fields can be included or excluded by using dot notation, e.g. `items.color` for a `color` field inside an `items` array of sub-documents.

For example, to train a model only on the fields `price`, `items`, `category`, use the following command:

```shell
origami train <SOURCE> --include-fields price,items,category
```

To exclude the field `extra.comment` in a dataset (in addition to `_id`), use the following command:

```shell
origami train <SOURCE> --exclude-fields _id,extra.comment
```

#### Limit and Skip

To limit the number of objects to train on, use the option `--limit` (or `-l`).

For example, to only train on the first 1000 objects, use the following command:

```shell
origami train <SOURCE> --limit 1000
```

To skip a number of objects to train on, use the option `--skip` (or `-s`).

For example, to ignore the first 1000 objects from the source, use the following command:

```shell
origami train <SOURCE> --skip 1000
```

### Model Configuration Options

Note: All model configuration options use upper-case letters in their short version.

#### Number of transformer layers

To change the number of transformer layers (default is 4), use the `--num-layers` (or `-T`) option.

For example, to train a model with 8 layers, use the command:

```shell
origami train <SOURCE> --num-layers 8
```

#### Number of attention heads

To change the number of attenion heads per transformer layer (default is 4), use the `--num-attn-heads` (or `-A`) option.

For example, to train a model with 8 attention heads, use the command:

```shell
origami train <SOURCE> --num-attn-heads 8
```

Note that the hidden dimension size must be a multiple of the attention heads.

#### Hidden Dimension Size

To change the number of dimensions in the hidden layers (default is 128), use the `--hidden-dim` (or `-H`) option.

For example, to train a model with a hidden dimension size of 64, use the command:

```shell
origami train <SOURCE> --hidden-dim 64
```

Note that the hidden dimension size must be a multiple of the attention heads.

#### Setting the Learning Rate

ORiGAMi uses a combined warmup and decay schedule, slowly increasing the learning rate over 1000 batches to the maximum learning rate, then decaying the learning rate over the course of training to 1% of the max. learning rate.

To change the maximum learning rate (default is 1e-3), use the `--learning-rate` (or `-L`) option.

For example, to change the learning rate to 5e-4, use the following command:

```shell
origami train <SOURCE> --learning-rate 5e-4
```

#### Number of Training Batches

ORiGAMi uses _number of batches_ rather than epochs to specify how long a model should be trained. The default is 10,000. To change the number of training batches, use the `--num-batches` (or `-N`) option.

For example, to train for only 2000 batches, use the following command:

```shell
origami train <SOURCE> --num-batches 2000
```

#### Batch Size

To change the batch size (default 100), use the `--batch-size` (or `-B`) option. This can be useful if the JSON objects are large and full batches of 100 objects would not fit into GPU RAM.

For example, to change the batch size to 10, use the following command:

```shell
origami train <SOURCE> --batch-size 10
```

#### Shuffling and Upscaling

One of ORiGAMi's features is that it can be trained with shuffled key/value pairs, which allows the model to predict any field with a single trained model, and mitigates overfitting. Additionally, when shuffling the key/value pairs, the dataset can be upscaled by including different permutations of the pair orders.

The default mode is `--shuffled`, which can be turned off with the `--ordered` option. When `--shuffled` is active, the upscaling factor (default 5) can be changed with the `--upscaling` (or `-U`) option.

For example, to train with shuffled objective and an upscaling factor of 100, use the following command:

```shell
origami train <SOURCE> --upscaling 100
```

To disable shuffling and use the original key/value pair order, use the option `--ordered`. In this case, the `--upscaling` value is ignored.

Hint: Higher upscaling values are especially useful for small datasets. As a rule of thumb, use an upscaling factor of 1,000,000 divided by the size of the dataset. With a batch size of 100 and 10,000 training steps, this will train the model for 1 epoch, presenting each permutation exactly once.

#### Guardrails

As described in our paper, we use guardrails during training and inference, suppressing logits that would lead to invalid JSON objects. During training, this leads to faster convergence, as the model does not need to learn the structure of the JSON objects and can focus on the values.

ORiGAMi offers 3 different guardrail modes: `STRUCTURE_AND_VALUES`, `STRUCTURE_ONLY`, and `NONE`, with the default being `STRUCTURE_AND_VALUES`.

When the guardrails mode is `NONE`, guardrails is disabled. While a model can still be trained in this way, during inference it's possible that the model produces grammatically invalid sequences which cannot be parsed back into a JSON object (for example by closing an array bracket `]` before opening one).

When the guardrails mode is `STRUCTURE_ONLY`, ORiGAMi will ensure valid JSON structure, but will allow any value token at any position. For example, for a field `age` that only contains numbers, it is theoretically possible for the model to sample a token `Sydney` during inference, which only occured as value under the field `city`.

When the guardrails mode is `STRUCTURE_AND_VALUES`, in addition to enforcing valid JSON structure, ORiGAMi will only allow previously seen value tokens for a given field. This is done by parsing the schema before training and tracking the set of values for each of the leaf fields in the objects.

To change the guardrails mode to e.g. `STRUCTURE_ONLY`, use the following command:

```shell
origami train <SOURCE> --guardrails STRUCTURE_ONLY
```

We recommend using the default `STRUCTURE_AND_VALUES` mode unless used for comparisons, or in combination with the `--ignore-field-path` option (see below).

#### Ignoring Field Paths

By default, ORiGAMi's preprocessing pipeline creates field tokens containing the full nested path, e.g. `address.city`. In cases with deeply nested objects and repeating field names at various nesting levels, for example when dealing with Abstract Syntax Trees, this approach can lead to an extremely large vocabulary, as almost all leaf fields have unique field paths. In such a case, it can make sense to use the `--ignore-field-path` option.

When enabled, it will create field tokens only based on the inner most field name. In the example above, this would create a field token for the field name `city` only. In such cases, this can reduce the vocabulary size significantly.

Note that this option is not compatible with the guardrails mode `STRUCTURE_AND_VALUES`, as the schema parser is no longer able to uniquely identify fields and their values. Using this option implicitly sets `--guardrails STRUCTURE_ONLY` as well.

#### Limiting Vocabulary Size

For datasets with a large number of unique values, the vocabulary size can become a bottleneck. To limit the size of the vocabulary to a fixed value, the `--max-vocab-size` (or `-V`) option can be used. This will replace the least frequently seen tokens with an `[UNKNOWN]` token during preprocessing.

For example, to limit the vocabulary to 1000 tokens, use the command:

```shell
origami train <SOURCE> --max-vocab-size 1000
```

### Evaluation Options

During training, it can be helpful to monitor classification accuracy on a portion of the training (and optionally validation) set. For this evaluation, a target field to predict needs to be specified with the `--target-field` (or `-t`) option. Currently, only top-level fields are supported as target fields.

If specificed, ORiGAMi will take random samples of 5 \* batch size instances from the training set and print out classification accuracy (`train_acc`) over that sample after every 100 batches.

Optionally, the training dataset can be split into training and validation portions, with the `--val-split-ratio` (or `-r`) option specifying the ratio. If this option is specified, the model is trained only on the training portion, and additionally evaluated on samples of 5 \* batch size instances from the validation portion of the data after every 100 batches (`test_acc`). This also prints out the validation loss (`test_loss`) at the same time.

For example, to use `income` as the target field and use 20% of the dataset for validation, use the following command:

```shell
origami train <SOURCE> --target-field income --val-split-ratio 0.2
```

Note that if only the `--val-split-ratio` is specified without a target field, only the `test_loss` is printed, not the accuracy.

### Advanced Configuration Options

ORiGAMi has a few more configuration parameters that are not accessible via dedicated CLI options. These can still be changed with the `--set-parameter` (or `-p`) option, using the format `<parameter>=<value>`. Parameter names are grouped into `data`, `pipeline`, `model` and `train` using dot notation, e.g. `model.tie_weights` or `pipeline.n_bins`.

The full list of options is defined in `~/origami/utils/config.py` and can also be seen printed out when the `--verbose` option is used.

For example, to change after how many batches the evaluation on the validation set should happen, use the following command:

```shell
origami train <SOURCE> --target-field <TARGET> --val-split-ratio <RATIO> --set-parameter train.eval_every=500
```

Several parameters can be changed by providing the `--set-parameter` or `-p` option multiple times.

## `predict` command

The general invocation of the `predict` command is:

```shell
origami predict SOURCE [OPTIONS]
```

The `predict` command allows to make a prediction from a trained model for a given target field in the dataset. If the target field is present in the dataset, the values will be used as ground through to calculate prediction accuracy. This is useful to validate a model's accuracy on a separate test dataset.

The predictions are printed to stdout, while all other output (e.g. the accuracy result or output from the `--verbose` option) is printed to stderr. This allows to store only the predictions by piping them into a file, e.g.

```shell
origami predict <SOURCE> [OPTIONS] > predictions.txt
```

### General Options

#### Help and Verbosity

To list an overview of the available options, use the `--help` option.

To print more verbose information about pipeline execution, model details and configuration options, use the `--verbose` (or `-v`) option.

#### Model Path

Specify the trained model path with the `--model-path` (or `-m`) option. If no path is specified, the default path and filename of `./model.origami` will be used.

For example, to make predictions from a model located at `./snapshots/orders-dev.origami`, use the followig command:

```shell
origami predict <SOURCE> --model-path ./snapshots/orders-dev.origami
```

#### Target Field

To make predictions, a target field needs to be specified. This is a required parameter for the `predict` command. Use the `--target-field` (or `-t`) option to provide the name of the target field. Currently, only top-level fields are supported as target.

For example, to make predictions for the `income` field in a collection, use the following command:

```shell
origami predict <SOURCE> --target-field income
```

### Source Selection

The CLI allows making predictions from a number of different sources:

- A running MongoDB instance: use `mongodb://<host>:<port>` connection URI as SOURCE and `--source-db`, `--source-coll` options to specify the database and collection name
- a `.jsonl` file, where each line is a JSON object
- a `.json` file, which needs to contain an single array of JSON objects
- a `.csv` file with header row
- a directory containing any of the supported file types: specify the directory path as SOURCE

### Additional Source Options

#### MongoDB Source

When the source is a MongoDB instance, the parameters `--source-db` (or `-d`) and `--source-coll` (or `-c`) are required.

For example, when the source is the `ecommerce.orders` namespace in a MongoDB instance running on localhost and standard port 27017, the source can be specified as:

```shell
origami predict mongodb://localhost:27017 --source-db ecommerce --source-coll orders
```

When training from a file or a directory containing supported files, these two options are ignored.

#### Including and Excluding Fields

To in- or exclude fields in the objects when making predictions, use either the `--include-fields` (or `-i`) or the `--exclude-fields` (or `-e`) options, followed by a comma-separated list of field names. Note that if neither option is specified, the `_id` field is excluded by default (if it exists).

Fields can be excluded during prediction even if the model was trained on the fields. This can be useful for feature selection to determine which of the fields are most helpful in making predictions.

Nested fields can be included or excluded by using dot notation, e.g. `items.color` for a `color` field inside an `items` array of sub-documents.

For example, to make predictions only from the fields `price`, `items`, `category`, use the following command:

```shell
origami predict <SOURCE> --include-fields price,items,category
```

To exclude the field `extra.comment` in a dataset (in addition to `_id`), use the following command:

```shell
origami predict <SOURCE> --exclude-fields _id,extra.comment
```

#### Limit and Skip

To limit the number of objects to make predictions for, use the option `--limit` (or `-l`).

For example, to only make predictions on the first 1000 objects, use the following command:

```shell
origami predict <SOURCE> --limit 1000
```

To skip a number of objects to make predictions for, use the option `--skip` (or `-s`).

For example, to ignore the first 1000 objects from the source, use the following command:

```shell
origami predict <SOURCE> --skip 1000
```

### Output Format Options

#### JSON Output

By default, the `predict` command only prints the values for the target field. Use the `--json` (or `-j`) option to print the entire
JSON object including the target field and predicted value.

This option is useful if you want to fill in missing values in a dataset but keep the format as JSON. You can also pipe the output to `mongoimport` to write the resulting documents back to a MongoDB collection.

For example, if you have a MongoDB collection `product.catalog` where the `category` field is missing, and you want to use the models' predictions to fill in the values, you can use the following command:

```shell
origami predict <URI> -d product -c catalog -t category --json | mongoimport -d product -c catalog_predicted
```
