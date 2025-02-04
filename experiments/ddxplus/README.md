# DDXPlus Experiments

In this experiment we train the model on the [DDXPlus dataset](https://arxiv.org/abs/2205.09148), a dataset for automated medical diagnosis. We devise a task to predict the most likely differential diagnoses for each instance, a multi-label prediction task.

For ORiGAMi, we reformat the dataset into JSON format with two different representations:

- A flat representation, in which we store the evidences and their values as strings.
- An object representation, where the evidences are stored as object containing array values.

We compare our model against baselines: Logistic Regression, Random Forests, XGBoost, LightGBM. The baselines are trained on a
flat representation by converting the evidence-value strings into a multi-label binary matrix. We wrap each model in a scikit-learn
[MultiOutputClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html).

First, make sure you have restored the datasets from the mongo dump file as described in [../README.md](../README.md). All commands (see below) must be run from the `ddxplus` directory.

## ORiGAMi

We train a model with the `medium` size preset by default: 6 layers, 6 heads, 192 embedding dimensionality. To train with other model sizes, append `model_size=<size>` to the command, using one of the following options: `xs`, `small`, `medium`, `large`, `xl`.

To train and evaluate ORiGAMi on the flat evidences structure, run the following:

```bash
guild run origami:train evidences=flat eval_data=test seed="[1, 2, 3, 4, 5]"
```

For the object representation of evidences, run instead:

```bash
guild run origami:train evidences=object eval_data=test seed="[1, 2, 3, 4, 5]"
```

This will repeat the training and evaluation 5 times with different random seeds and evaluate on the test set.

## Baselines

### Hyperparameter optimization

First perform HPO, supplying the `<model>` as one of `lr` (Logistic Regression), `rf` (Random Forest), `xgb` (XGBoost), `lgb` (LightGBM) and the appropriate number of trial runs with `--max-trials <num>`, and give the run a name with `<label>`, e.g.

```bash
 NUMPY_EXPERIMENTAL_DTYPE_API=1 guild run lr:hyperopt --optimizer random --max-trials 20 --label <label>
```

To find the best parameters on the validation dataset, use:

```bash
guild compare -Fl <label> -u
```

Sort the `f1_val_mean` column in descending order (press `S` key) and pick the run ID (first column) of the best configuration.

Get the hyperparameters (= flags) with `guild runs info <run-id>`.

### Evaluate best hyperparameters on test dataset

Once the optimal hyperparameters are found, run the model with the optimal hyperparameters, e.g.:

```bash
guild run lr:train <param1>=<value1> <param2=value2> ...
```

Replace the `<param>` and `<value>` placeholders with the optimal hyperparameters. You can ignore `model_name` and `n_random_seeds` here.
By default, the evaluation is done 5 times with different random seeds.

The `<metric>_test_mean` and `<metric_test_val>` scores show the evaluation on the test dataset, where `<metric>` is one of `f1`, `precision`, `recall`.
