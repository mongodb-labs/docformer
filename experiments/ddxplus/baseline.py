from types import SimpleNamespace

from pymongo import MongoClient
from sklearn.model_selection import StratifiedKFold

from axon.gpt.data import load_df_from_mongodb
from axon.gpt.utils import set_seed
from axon.utils.guild import load_secrets

flags = SimpleNamespace()
secrets = load_secrets()

set_seed(flags.seed)

# load PATHOLOGY fields for stratified cv splits
client = MongoClient(secrets["MONGO_URI"])
collection = client.ddxplus["train-noprob"]

pathologies = [
    d["PATHOLOGY"] for d in collection.find({}, projection={"PATHOLOGY": 1}, limit=flags.limit, sort=[("_id", 1)])
]

# now load data properly for training, same sort order
PROJECTION = {"_id": 0, "PATHOLOGY": 0, "DIFFERENTIAL_DIAGNOSIS": 0}
TARGET_FIELD = "DIFFERENTIAL_DIAGNOSIS_NOPROB"

docs_df = load_df_from_mongodb(
    secrets["MONGO_URI"], "ddxplus", "train-noprob", projection=PROJECTION, limit=flags.limit, sort=[("_id", 1)]
)

cv_scores = []

# create cross-validation splits
kfold = StratifiedKFold(n_splits=flags.n_cv_splits, shuffle=True, random_state=flags.seed)
splits = list(kfold.split(docs_df, pathologies))
splits = [(train.tolist(), test.tolist()) for train, test in splits]

for k, (train_ixs, test_ixs) in enumerate(splits):
    pass
    # TODO train models

    # print results for this fold
    # print_guild_scalars(fold=k, ddr=ddr, ddp=ddp, f1=f1, gtpa_at_1=gtpa_at_1, gtpa=gtpa)
    # cv_scores.append({"ddr": ddr, "ddp": ddp, "f1": f1, "gtpa_at_1": gtpa_at_1, "gtpa": gtpa})


# print("cross-validation results:")
# keys = list(cv_scores[0].keys())
# scalars = {}
# for key in keys:
#     scalars[f"{key}_mean"] = np.mean([e[key] for e in cv_scores])
#     scalars[f"{key}_std"] = np.std([e[key] for e in cv_scores])
#     scalars[f"{key}_min"] = np.min([e[key] for e in cv_scores])
#     scalars[f"{key}_max"] = np.max([e[key] for e in cv_scores])

# print rounded scalars
# print_guild_scalars(**{k: f"{v:.4f}" for k, v in scalars.items()})
