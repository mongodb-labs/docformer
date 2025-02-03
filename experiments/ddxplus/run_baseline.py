import warnings
from collections import defaultdict

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from utils import get_scores
from xgboost import XGBClassifier

from origami.preprocessing import load_df_from_mongodb
from origami.utils.guild import load_secrets, print_guild_scalars

# experiment flags
model_name = "LogisticRegression"  # "XGBoost" # "RandomForest"
limit = 1000
n_random_seeds = 5

print(f"Running {model_name=}, {limit=}, {n_random_seeds=}")

# defaul model hyper parameters

# logistic regression
lr_C = 1.0
lr_penalty = "none"
lr_max_iter = 50
lr_fit_intercept = True

# xgboost
xgb_learning_rate = 0.1
xgb_max_depth = 5
xgb_subsample = 1.0
xgb_colsample_bytree = 1.0
xgb_colsample_bylevel = 1.0
xgb_min_child_weight = 1.0
xgb_reg_alpha = 0.0
xgb_reg_lambda = 1.0
xgb_gamma = 0
xgb_n_estimators = 100

# random forest
rf_n_estimators = 100
rf_max_features = "none"
rf_max_depth = "none"
rf_min_samples_split = 5

# lightgbm
lgb_num_leaves = 10
lgb_max_depth = 5
lgb_learning_rate = 0.1
lgb_n_estimators = 100
lgb_min_child_weight = 1.0
lgb_subsample = 0.8
lgb_colsample_bytree = 0.8
lgb_reg_alpha = 0.0
lgb_reg_lambda = 1.0

secrets = load_secrets()

PROJECTION = {"_id": 0, "DIFFERENTIAL_DIAGNOSIS": 0}
TARGET_FIELD = "DIFFERENTIAL_DIAGNOSIS_NOPROB"


def load_docs(collection_name):
    return load_df_from_mongodb(
        uri=secrets["MONGO_URI"],
        db=secrets["DATABASE"],
        coll=collection_name,
        projection=PROJECTION,
        sort=[("_id", 1)],
        limit=limit,
    )


def preprocess_dataset(df):
    # pull up relevant fields at the top of the df
    df["EVIDENCES"] = df["docs"].apply(lambda x: x["EVIDENCES"])
    df["DIFFERENTIAL_DIAGNOSIS_NOPROB"] = df["docs"].apply(lambda x: x["DIFFERENTIAL_DIAGNOSIS_NOPROB"])
    df["PATHOLOGY"] = df["docs"].apply(lambda x: x["PATHOLOGY"])
    return df


train_docs_df = load_docs(collection_name="train-noprob").pipe(preprocess_dataset)
test_docs_df = load_docs(collection_name="test-noprob").pipe(preprocess_dataset)
val_docs_df = load_docs(collection_name="validate-noprob").pipe(preprocess_dataset)


def get_classifier(model_name, seed):
    match model_name:
        case "LogisticRegression":
            clf = LogisticRegression(
                random_state=seed,
                C=lr_C if lr_penalty != "none" else 1.0,
                penalty=lr_penalty if lr_penalty != "none" else None,
                max_iter=lr_max_iter,
                fit_intercept=True if lr_fit_intercept == 1 else False,
                solver="saga",
            )
        case "XGBoost":
            clf = XGBClassifier(
                random_state=seed,
                max_depth=xgb_max_depth,
                learning_rate=xgb_learning_rate,
                n_estimators=xgb_n_estimators,
                subsample=xgb_subsample,
                colsample_bytree=xgb_colsample_bytree,
                colsample_bylevel=xgb_colsample_bylevel,
                min_child_weight=xgb_min_child_weight,
                reg_alpha=xgb_reg_alpha,
                reg_lambda=xgb_reg_lambda,
                gamma=xgb_gamma,
            )
        case "RandomForest":
            clf = RandomForestClassifier(
                random_state=seed,
                n_estimators=rf_n_estimators,
                max_features=rf_max_features if rf_max_features != "none" else None,
                max_depth=rf_max_depth if rf_max_depth != "none" else None,
                min_samples_split=rf_min_samples_split,
            )
        case "LightGBM":
            clf = LGBMClassifier(
                random_state=seed,
                verbose=-1,
                num_leaves=lgb_num_leaves,
                max_depth=lgb_max_depth,
                learning_rate=lgb_learning_rate,
                n_estimators=lgb_n_estimators,
                min_child_weight=lgb_min_child_weight,
                subsample=lgb_subsample,
                colsample_bytree=lgb_colsample_bytree,
                reg_alpha=lgb_reg_alpha,
                reg_lambda=lgb_reg_lambda,
            )

        case _:
            raise ValueError(f"Unknown model {model_name}")
    return clf


# encode data
mlb_ddx = MultiLabelBinarizer()
mlb_evd = MultiLabelBinarizer()

# train
X_train = mlb_evd.fit_transform(train_docs_df["EVIDENCES"])
y_train = mlb_ddx.fit_transform(train_docs_df["DIFFERENTIAL_DIAGNOSIS_NOPROB"])

# val
X_val = mlb_evd.transform(val_docs_df["EVIDENCES"])
y_val = mlb_ddx.transform(val_docs_df["DIFFERENTIAL_DIAGNOSIS_NOPROB"])
y_pathology_val = mlb_ddx.transform(
    val_docs_df["PATHOLOGY"].apply(
        lambda x: [
            x,
        ]
    )
)
y_pathology_val = np.where(y_pathology_val > 0.5)[1]

# test
X_test = mlb_evd.transform(test_docs_df["EVIDENCES"])
y_test = mlb_ddx.transform(test_docs_df["DIFFERENTIAL_DIAGNOSIS_NOPROB"])
y_pathology_test = mlb_ddx.transform(
    test_docs_df["PATHOLOGY"].apply(
        lambda x: [
            x,
        ]
    )
)
y_pathology_test = np.where(y_pathology_test > 0.5)[1]

results = defaultdict(list)

for clf_seed in range(n_random_seeds):
    clf = get_classifier(model_name=model_name, seed=clf_seed)
    multi_output_clf = MultiOutputClassifier(clf, n_jobs=4)
    print(f"Training {clf}")

    # train
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=ConvergenceWarning)
        multi_output_clf.fit(X_train, y_train)

    # evaluate dev
    y_pred_val = multi_output_clf.predict_proba(X_val)
    y_pred_val = np.hstack([y_pred_val_i[:, 1].reshape(-1, 1) for y_pred_val_i in y_pred_val])

    scores_val = get_scores(y_target=y_val, y_pred=y_pred_val, y_pathology=y_pathology_val, postfix="_val")
    for score_name, score in scores_val.items():
        results[score_name].append(score)

    # evaluate test
    y_pred_test = multi_output_clf.predict_proba(X_test)
    y_pred_test = np.hstack([y_pred_test_i[:, 1].reshape(-1, 1) for y_pred_test_i in y_pred_test])

    scores_test = get_scores(y_target=y_test, y_pred=y_pred_test, y_pathology=y_pathology_test, postfix="_test")
    for score_name, score in scores_test.items():
        results[score_name].append(score)

    guild_output = {"step": clf_seed} | scores_val | scores_test
    print_guild_scalars(**guild_output)

print("\n\nAggregated metrics:")
keys = list(results.keys())
scalars = {}
for key in keys:
    scalars[f"{key}_mean"] = np.mean(results[key])
    scalars[f"{key}_std"] = np.std(results[key])
    scalars[f"{key}_min"] = np.min(results[key])
    scalars[f"{key}_max"] = np.max(results[key])

# print rounded scalars
print_guild_scalars(**{k: f"{v:.4f}" for k, v in scalars.items()})
print()
