import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# change the path!
# from week_5.stratified_folds import load_splits
from week_5.move_files import move_all_imgs

# move and rename images names
move_all_imgs()

train_df = pd.read_csv('data.csv')

y = train_df['target']
ImageId = train_df['ImageId']

train_df = train_df.drop('target', 1)
train_df = train_df.drop('ImageId', 1)

X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=0.3, random_state=42)

# rf = RandomForestClassifier(bootstrap=True, max_depth=90, max_features=2, min_samples_leaf=5, min_samples_split=12,
#                             n_estimators=1000)
rf = RandomForestClassifier()
svm = SVC(kernel='linear')
# lgb = LGBMClassifier(learning_rate=0.03,num_iterations=1000)
lgb = LGBMClassifier()

clf = VotingClassifier(estimators=[('rf', rf), ('lgb', lgb), ('svm', svm)], voting='hard')
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# rf.fit(X_train, y_train)
# y_pred = rf.predict(X_test)

# lgb_param_grid = {
#     'learning_rate': [0.01, 0.03, 0.1],
#     'num_iterations': [300, 500, 1000,2000],
#     'max_depth': [10, 15],
#     'num_leaves': [12, 24, 32, 50]
# }

# rf_param_grid = {
#     'bootstrap': [True],
#     'max_depth': [80, 90, 100, 110],
#     'max_features': [2, 3],
#     'min_samples_leaf': [3, 4, 5],
#     'min_samples_split': [8, 10, 12],
#     'n_estimators': [100, 200, 300, 1000]
# }

# grid_search = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=3, n_jobs=-1, verbose=2)

# grid_search.fit(X_train, y_train)
# print(grid_search.best_params_)


# folds_folder = Path("/folds")

# def cross_val():
#     result = {}
#     result["weighted_acc"] = []
#
#     for val_fold in range(5):
#         splits = load_splits(folds_folder, [val_fold])
#
#         train_x = splits["train"]
#         val_x = splits["val"]
#
#         train_y = splits["train"].target
#         val_y = splits["val"].target
#
#         model = LGBMClassifier(random_state=42)
#         model.fit(train_x, train_y)
#
#         val_pred = model.predict(val_x)
#         result["weighted_acc"].append(weighted_accuracy(val_y, val_pred))
#         print("val fold:", val_fold, "weighted acc:", result["weighted_acc"][-1])
#     return result


def weighted_accuracy(y_test, y_pred):
    w = y_test//2 + 1
    return accuracy_score(y_test, y_pred, sample_weight=w)

# result = cross_val()
# print(np.mean(result["weighted_acc"]))

conf_matrix = confusion_matrix(y_test, y_pred)
print('Weighted Accuracy: ', weighted_accuracy(y_test, y_pred))
print('Accuracy: ', accuracy_score(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred, average='weighted'))
print("Precision: ", precision_score(y_test, y_pred, average='weighted'))
print("Recall: ", recall_score(y_test, y_pred, average='weighted'))

print('confusion matrix: ', conf_matrix)