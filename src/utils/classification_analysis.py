from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from .reporting import *
from tqdm import tqdm


def get_debug_classification_report(dataframe, entry_column, label_column, *, n_splits=5, model_factory=None, seed=None):
    if model_factory is None:
        model_factory = lambda: RandomForestClassifier(random_state=seed)

    dataframe = dataframe.copy()
    total_y_true, total_y_pred = np.array([]), np.array([])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_index, test_index in tqdm(kf.split(dataframe[entry_column]), total=n_splits):
        X_train, y_train = dataframe.iloc[train_index][entry_column], dataframe.iloc[train_index][label_column]
        X_test, y_test = dataframe.iloc[test_index][entry_column], dataframe.iloc[test_index][label_column]

        model = model_factory().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        dataframe.loc[dataframe.iloc[test_index].index, 'predicted'] = y_pred

        total_y_true = np.concatenate((total_y_true, y_test), axis=0)
        total_y_pred = np.concatenate((total_y_pred, y_pred), axis=0)

    report_df, weighted_f1 = get_classification_report(total_y_true, total_y_pred)
    confusion_df = get_confusion_matrix(total_y_true, total_y_pred)
    return weighted_f1, report_df, confusion_df, dataframe


def get_y_true_y_pred(confusion_df):
    y_true, y_pred = [], []
    for index in confusion_df.index:
        true_label = index.replace('True ', '')
        for column in confusion_df.columns:
            pred_label = column.replace('Pred ', '')

            value = confusion_df.loc[index, column]
            y_true += [true_label] * value
            y_pred += [pred_label] * value

    return np.array((y_true, y_pred)).T


def get_possible_f1_improvement(calculate_f1, default_f1, predicted_label, true_label, truth_array):
    fixed_truth_array = truth_array.copy()
    fixed_truth_array[
        (fixed_truth_array[:, 0] == true_label) &
        (fixed_truth_array[:, 1] == predicted_label),
        1
    ] = true_label
    fixed_f1 = calculate_f1(fixed_truth_array)
    improvement = round(fixed_f1 - default_f1, 4)
    return improvement


def _get_misclassifications_base(confusion_df, priority_func, priority_column):
    misclassifications = []
    misclassifications_set = set()
    for index in tqdm(confusion_df.index):
        true_label = index.replace('True ', '')

        for column in confusion_df.columns:
            predicted_label = column.replace('Pred ', '')

            if true_label == predicted_label:
                continue

            key = tuple(sorted((true_label, predicted_label)))
            if key in misclassifications_set:
                continue
            misclassifications_set.add(key)

            value = confusion_df.loc[index, column]
            if value == 0:
                continue

            priority_value = priority_func(confusion_df, true_label, predicted_label)
            misclassifications.append({
                'true_label': true_label,
                'predicted_label': predicted_label,
                priority_column: priority_value
            })

    misclassification_df = pd.DataFrame(misclassifications)
    misclassification_df['priority'] = MinMaxScaler().fit_transform(misclassification_df[priority_column].values.reshape(-1, 1))

    misclassification_df = misclassification_df.sort_values('priority', ascending=False)
    return misclassification_df.reset_index(drop=True)


def get_misclassifications_report(confusion_df, *, calculate_improvements=False):
    if calculate_improvements:
        truth_array = get_y_true_y_pred(confusion_df)
        calculate_f1 = lambda arr: f1_score(arr[:, 0], arr[:, 1], average='weighted')
        default_f1 = calculate_f1(truth_array)
        priority_func = lambda df, true_label, predicted_label: get_possible_f1_improvement(
            calculate_f1, default_f1, predicted_label, true_label, truth_array
        )
        return _get_misclassifications_base(confusion_df, priority_func, 'f1_possible_improvement')
    else:
        priority_func = lambda df, true_label, predicted_label: confusion_df.loc[f'True {true_label}', f'Pred {predicted_label}']
        return _get_misclassifications_base(confusion_df, priority_func, 'count')


def get_class_reliability(classification_df):
    scaled_support = MinMaxScaler().fit_transform(classification_df['support'].values.reshape(-1, 1)).reshape(-1)
    classification_df['reliability'] = scaled_support * classification_df['f1']
    classification_df['reliability'] = MinMaxScaler().fit_transform(classification_df['reliability'].values.reshape(-1, 1))
    return classification_df.sort_values('reliability', ascending=False)

