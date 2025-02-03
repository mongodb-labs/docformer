# DDXPlus Experiments

In this experiment we train the model on the [DDXPlus dataset](https://arxiv.org/abs/2205.09148), a dataset for automated medical diagnosis. We devise a task to predict the most likely differential diagnoses for each instance, a multi-label prediction task.

For ORIGAMI, we reformat the dataset into JSON format with two different representations:

- A flat representation, in which we store the evidences and their values as strings.
- An object representation, where the evidences are stored as object containing array values.

We compare our model against baselines: Logistic Regression, Random Forests, XGBoost, LightGBM. The baselines are trained on a
flat representation by converting the evidence-value strings into a multi-label binary matrix. We wrap each model in a scikit-learn
[MultiOutputClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html).

First, make sure you have restored the datasets from the mongo dump file as described in [../EXPERIMENTS.md](../EXPERIMENTS.md). All commands (see below) must be run from the `ddxplus` directory.

## ORiGAMi

## Baselines

### Single Run

Running a single model, using `guild.yml` settings, as:

```bash
guild run lr:hyperopt lr_C=10.0
```

**Note**: parameters passed through the CLI overwrite values in the notebook and in `guild.yml`.

It is also possible to run the notebook directly, though this will create a number of additional quantities tracked
by guild (e.g. `TARGET_FIELD`) which we may not be interested in, and as such, this method is for quick local checks only.

```bash
guild run baseline.ipynb model_name=LogisticRegression lr_C=0.1
```

### Experimental Runs

First perform HPO, supplying `limit=0` and the appropriate number of `--max-trials`:

```bash
guild run lr:hyperopt limit=1000 --optimizer random --max-trials 3
```

Once the optimal hyperparameters are found:

- update the `prod` sections in the `guild.yml` file in this folder with the optimal hyperparameters values
- run the model with the optimal hyperparameters, 5 repetitions with 5 different seeds:

```bash
guild run lr:prod
```

## Retrieving Results

To retrieve the values for the individual runs, we have 2 alternatives:

```python
from axon.utils.guild import get_runs
from guild import ipy, tfevent

# 1st alternative
runs = get_runs()
run = runs[0]

for _path, _digest, scalars in tfevent.scalar_readers(run.dir):
    for tag, value, step in scalars:
        print(tag, value, step)

# 2nd alternative (as pandas dataframe)
runs = ipy.runs()
sd_df = runs.scalars_detail()
sd_df['run_id'] = sd_df['run'].apply(lambda x: x.id)
print(sd_df[sd_df['run_id'] == run.id])
```
