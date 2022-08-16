import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold


def get_classification_report(y_true, y_pred):
    labels = sorted(np.unique(y_true))

    precisions, recalls, f1_scores, supports = precision_recall_fscore_support(y_true, y_pred)
    report_df = pd.DataFrame((precisions, recalls, f1_scores, supports)).T
    report_df.columns = ['precision', 'recall', 'f1', 'support']
    report_df.index = labels
    weighted_f1 = round(f1_score(y_true, y_pred, average='weighted'), 4)

    return report_df, weighted_f1


def get_confusion_matrix(y_true, y_pred):
    labels = sorted(np.unique(y_true))

    confusion_df = pd.DataFrame(confusion_matrix(y_true, y_pred))
    confusion_df.columns = [f'Pred {label}' for label in labels]
    confusion_df.index = [f'True {label}' for label in labels]
    confusion_df.replace(0, '')

    return confusion_df


def get_cross_validation_report(X, y, *, n_splits=5, model_factory=None, seed=None):
    if model_factory is None:
        model_factory = lambda: RandomForestClassifier(random_state=seed)

    total_y_true, total_y_pred = np.array([]), np.array([])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_index, test_index in tqdm(kf.split(y), total=n_splits):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        model = model_factory().fit(X_train, y_train)
        y_pred = model.predict(X_test)

        total_y_true = np.concatenate((total_y_true, y_test), axis=0)
        total_y_pred = np.concatenate((total_y_pred, y_pred), axis=0)

    report_df, weighted_f1 = get_classification_report(total_y_true, total_y_pred)
    confusion_df = get_confusion_matrix(total_y_true, total_y_pred)
    return weighted_f1, report_df, confusion_df


def save_reports(model_name, classification_report_df, confusion_report_df, f1_weighted, *, default_path='data/reports'):
    folder_path = os.path.join(default_path, model_name)
    os.makedirs(folder_path, exist_ok=True)

    classification_report_df.to_excel(f'{folder_path}/classification_report.xlsx')
    confusion_report_df.to_excel(f'{folder_path}/confusion_report.xlsx')
    with open(f'{folder_path}/f1_weighted.txt', 'w+') as file:
        file.write(str(f1_weighted))
