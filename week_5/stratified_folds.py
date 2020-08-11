import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedKFold
from pathlib import Path


train_df = pd.read_csv('data.csv')

# print(df.columns)


def stratified_k_fold(description,n_folds):
    folds = []

    X = description
    y = description.target

    stratifier = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    stratifier.get_n_splits(X, y)

    for _, test_indexes in stratifier.split(X, y):
        folds.append(X.iloc[test_indexes])

    folds = [pd.DataFrame(fold, columns=description.columns) for fold in folds]

    return folds


# ax = sns.countplot(x="target", data=df)

folds = stratified_k_fold(train_df, 5)

# for idx, fold in enumerate(folds):
#     ax = sns.countplot(x="target", data=fold)
#     plt.show()


CURRENT_PATH = os.getcwd()
folds_folder = Path(CURRENT_PATH + "/folds")

for idx, fold in enumerate(folds):
    fold_filename =  folds_folder / f"fold_{idx}.csv"
    fold.to_csv(fold_filename)


def load_splits(folds_folder, val_folds=[0], train_folds=None):
    folds = [int(fn.stem.split('_')[-1]) for fn in folds_folder.glob("fold_?.csv")]

    if train_folds is None:
        train_folds = [f for f in folds if f not in val_folds]

    if val_folds is None:
        train_folds = [f for f in folds if f not in train_folds]

    val = pd.concat([pd.read_csv(folds_folder / f"fold_{fi}.csv") for fi in val_folds])
    train = pd.concat([pd.read_csv(folds_folder / f"fold_{fi}.csv") for fi in train_folds])
    val = val.reset_index()
    train = train.reset_index()

    folds = {}
    folds["train"] = train
    folds["val"] = val

    return folds


for i in range(5):
    splits = load_splits(folds_folder, val_folds=[i])

    val_size = splits["val"].shape[0]
    train_size = splits["train"].shape[0]
    # print("val size: ", val_size)
    # print("train size: ", train_size)
    # print("overall: ", val_size + train_size)


splits = load_splits(folds_folder)
# print(splits["train"])