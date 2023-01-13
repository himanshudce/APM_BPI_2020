import os
import re
import pandas as pd
import numpy as np
import pickle
from tabulate import tabulate

from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_validate
np.random.seed(42)

# Event Log & Data Loading

# function to load the data from pickle
def load_data(load_path):
    with open(load_path, 'rb') as handle:
        data = pickle.load(handle)
    return data


def scale_features(data_dict):
    ## Scaling
    # If needed, would take place here. Not required for binary values.
    scaler_x = MinMaxScaler()
    data_scaled = scaler_x.fit_transform(data_dict['X'])

    scaler_y = FunctionTransformer() # for binary values scaling does not make sense at all but we keep it for symetry and apply the "NoOp" scaler
    target_scaled = scaler_y.fit_transform(data_dict['y'].reshape(-1, 1))

def build_evaluation_dataframe():
    ## Setting up Model Evaluation DataFrame
    # All the results will be saved to the `results_df` dataframe for comparison.

    results = pd.DataFrame(columns=['Model', 'Encoding', 'Trace Length', 'F-score', 'Precision', 'Recall', 'Accuracy'])
    return results


def train_test_decision_tree(X_train, y_train, X_test, y_test, X_val, y_val):

    dt = DecisionTreeClassifier(random_state=42)  # Create Decision Tree classifier object
    dt_fit = dt.fit(X_train, y_train)  # Train Decision Tree Classifier

    dt_predict = dt_fit.predict(X_test)  # Predict the response for test dataset
    test_results = pd.Series({'Model': 'Decision Tree (Test)', 'F-score': f1_score(y_test, dt_predict, average='macro'),
               'Precision': precision_score(y_test, dt_predict), 'Recall': recall_score(
            y_test, dt_predict), 'Accuracy': accuracy_score(y_test, dt_predict)})

    dt_val = dt_fit.predict(X_val)  # validation
    test_val = pd.Series({'Model': 'Decision Tree (Val)', 'F-score': f1_score(y_val, dt_val, average='macro'),
         'Precision': precision_score(y_val, dt_val), 'Recall': recall_score(
            y_val, dt_val), 'Accuracy': accuracy_score(y_val, dt_val)})

    return dt_fit, test_results, test_val


def train_test_logistic_regression(X_train, y_train, X_test, y_test, X_val, y_val):
    lr = LogisticRegression(random_state=42)
    lr_fit = lr.fit(X_train, y_train)
    
    lr_predict = lr_fit.predict(X_test)
    test_results = pd.Series({'Model': 'Logistic Regression (Default)', 'F-score': f1_score(y_test, lr_predict, average='macro'), 'Precision': precision_score(y_test, lr_predict), 'Recall': recall_score(
        y_test, lr_predict), 'Accuracy': accuracy_score(y_test, lr_predict)})

    lr_val = lr_fit.predict(X_val)  # validation
    test_val = pd.Series({'Model': 'Logistic Regression (Val)', 'F-score': f1_score(y_val, lr_val, average='macro'),
         'Precision': precision_score(y_val, lr_val), 'Recall': recall_score(
            y_val, lr_val), 'Accuracy': accuracy_score(y_val, lr_val)})

    return lr_fit, test_results, test_val


def train_test_logstic_regression_grid_search(X_train, y_train, X_test, y_test):
    # TODO: Return is not giving only the value, needs to be fixed. What I am seeing when I call
    #   cross_val_results['test_f1_macro'].mean() is different from what is being written to the dataframe
    # define grid search
    # solvers = ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga']
    penalty = ['l1', 'l2', 'elasticnet', 'None']
    # c_values = [100, 10, 1.0, 0.1, 0.01]
    # max_iteration = [20, 50, 100, 200, 500, 1000, 2000, 5000]
    max_iteration = [2000]
    c_values = [0.1]
    solvers = ['liblinear']
    grid = dict(solver=solvers, penalty=penalty, C=c_values, max_iter=max_iteration)

    lr = LogisticRegression(random_state=42)
    lr_grid = GridSearchCV(estimator=lr, param_grid=grid, n_jobs=-1, scoring='accuracy', error_score=0)
    lr_grid_fit = lr_grid.fit(X_train, y_train)

    cross_val_results = pd.DataFrame(cross_validate(lr_grid_fit.best_estimator_, X_train, y_train,
                                                    scoring=['f1_macro', 'precision_macro', 'recall_macro',
                                                             'accuracy']))

    return pd.Series({'Model': 'Logistic Regression (Hyper Parameter Tuning)', 'F-score': cross_val_results['test_f1_macro'].mean(), 'Precision': cross_val_results['test_precision_macro'].mean(), 'Recall': cross_val_results['test_recall_macro'].mean(), 'Accuracy': cross_val_results['test_accuracy'].mean()})


def train_test_random_forest(X_train, y_train, X_test, y_test, X_val, y_val):
    rf = RandomForestClassifier(n_estimators=20, random_state=42, max_depth=4)
    rf_fit = rf.fit(X_train, y_train)
    # Predicting the Test set results
    
    rf_predict = rf_fit.predict(X_test)
    test_results = pd.Series({'Model': 'Random Forest (Default)', 'F-score': f1_score(y_test, rf_predict, average='macro'), 'Precision': precision_score(y_test, rf_predict), 'Recall': recall_score(
        y_test, rf_predict), 'Accuracy': accuracy_score(y_test, rf_predict)})

    rf_val = rf_fit.predict(X_val)  # validation
    test_val = pd.Series({'Model': 'Logistic Regression (Val)', 'F-score': f1_score(y_val, rf_val, average='macro'),
         'Precision': precision_score(y_val, rf_val), 'Recall': recall_score(
            y_val, rf_val), 'Accuracy': accuracy_score(y_val, rf_val)})

    return rf_fit, test_results, test_val


def train_test_neural_network(X_train, y_train, X_test, y_test, X_val, y_val):
    nnet = MLPClassifier(hidden_layer_sizes=[100,],
                         alpha=0.0001,
                         activation='relu',
                         max_iter=500,
                         solver='adam', random_state=42)
    nnet_fit = nnet.fit(X_train, y_train)
    # Predicting the Test set results

    nnet_predict = nnet_fit.predict(X_test)
    test_results =  pd.Series({'Model': 'Neural Net', 'F-score': f1_score(y_test, nnet_predict, average='macro'),
                    'Precision': precision_score(y_test, nnet_predict), 'Recall': recall_score(
                    y_test, nnet_predict), 'Accuracy': accuracy_score(y_test, nnet_predict)})

    nnet_val = nnet_fit.predict(X_val)  # validation
    test_val = pd.Series({'Model': 'Neural Net (Val)', 'F-score': f1_score(y_val, nnet_val, average='macro'),
         'Precision': precision_score(y_val, nnet_val), 'Recall': recall_score(
            y_val, nnet_val), 'Accuracy': accuracy_score(y_val, nnet_val)})

    return nnet_fit, test_results, test_val


def iterate_data_and_create_x_y(dataset_directory, dataset_file, dataset_filename_split):
    filename_test = dataset_file
    filename_train = dataset_filename_split
    filename_train[2] = 'train'
    filename_train = '_'.join(filename_train)  # recombines filename, replacing test with train
    filename_val = dataset_filename_split
    filename_val[2] = 'val'
    filename_val = '_'.join(filename_val)  # replace test with val
    data_test = load_data(os.path.join(dataset_directory, filename_test))
    data_train = load_data(os.path.join(dataset_directory, filename_train))
    data_val = load_data(os.path.join(dataset_directory, filename_val))
    X_train_data = np.array(data_train['X'])
    X_test_data = np.array(data_test['X'])
    X_val_data = np.array(data_val['X'])
    y_train_data = np.array(data_train['y'])
    y_test_data = np.array(data_test['y'])
    y_val_data = np.array(data_val['y'])

    return X_train_data, X_test_data, X_val_data, y_train_data, y_test_data, y_val_data


def train_and_test_models(dataset_filename_split, X_train_data, X_test_data, X_val_data, y_train_data, y_test_data, y_val_data):
    encoding_results = pd.DataFrame(columns=['Encoding', 'Trace Length'])
    encoding = dataset_filename_split[0]
    trace_length = dataset_filename_split[5][:-7]  # slice to remove ".pickle" from this metadata

    model, model_results, val_results = train_test_decision_tree(X_train_data, y_train_data, X_test_data, y_test_data, X_val_data, y_val_data)
    encoding_results = pd.concat([encoding_results, model_results.to_frame().T], ignore_index=True)
    encoding_results = pd.concat([encoding_results, val_results.to_frame().T], ignore_index=True)
    filename = encoding + '_' + trace_length + '_decision_tree_model.sav'
    pickle.dump(model, open(os.path.join('../models', filename), 'wb'))  # save the model to disk

    model, model_results, val_results = train_test_logistic_regression(X_train_data, y_train_data, X_test_data, y_test_data, X_val_data, y_val_data)
    encoding_results = pd.concat([encoding_results, model_results.to_frame().T], ignore_index=True)
    encoding_results = pd.concat([encoding_results, val_results.to_frame().T], ignore_index=True)
    filename = encoding + '_' + trace_length + '_logistic_regression_model.sav'
    pickle.dump(model, open(os.path.join('../models', filename), 'wb'))  # save the model to disk

    model, model_results, val_results = train_test_random_forest(X_train_data, y_train_data, X_test_data, y_test_data, X_val_data, y_val_data)
    encoding_results = pd.concat([encoding_results, model_results.to_frame().T], ignore_index=True)
    encoding_results = pd.concat([encoding_results, val_results.to_frame().T], ignore_index=True)
    filename = encoding + '_' + trace_length + '_random_forest_model.sav'
    pickle.dump(model, open(os.path.join('../models', filename), 'wb'))  # save the model to disk

    model, model_results, val_results = train_test_neural_network(X_train_data, y_train_data, X_test_data, y_test_data, X_val_data, y_val_data)
    encoding_results = pd.concat([encoding_results, model_results.to_frame().T], ignore_index=True)
    encoding_results = pd.concat([encoding_results, val_results.to_frame().T], ignore_index=True)
    filename = encoding + '_' + trace_length + '_neural_network_model.sav'
    pickle.dump(model, open(os.path.join('../models', filename), 'wb'))  # save the model to disk

    encoding_results['Encoding'] = encoding
    encoding_results['Trace Length'] = trace_length

    return encoding_results


def run_model_training_and_evaluation():
    evaluation_results = build_evaluation_dataframe()
    training_data_dir = '../data/training_data'
    for filename in os.listdir(training_data_dir):  # iterate over all files in directory DIR
        if not filename.startswith('.'):  # do not process hidden files
            if 'test' in filename:
                filename_split = re.split('[_]', filename)  # splits the filename on '-' and '.' -> creates a list
                X_train_data, X_test_data, X_val_data, y_train_data, y_test_data, y_val_data = iterate_data_and_create_x_y(training_data_dir, filename, filename_split)
                new_results = train_and_test_models(filename_split, X_train_data, X_test_data, X_val_data, y_train_data, y_test_data, y_val_data)
                evaluation_results = pd.concat([evaluation_results, new_results], ignore_index=True)
    evaluation_results = evaluation_results.sort_values(by='F-score', ascending=False)

    evaluation_results.to_csv('../data/processed/model_evaluation_results.csv', index=False)
    print(tabulate(evaluation_results, headers="keys", tablefmt="github", showindex=False))


def load_model(model_filename):
    model_filename = os.path.join('../models', model_filename)
    return pickle.load(open(model_filename, 'rb'))

# TODO: Build function to validate boolean encoding w/ trace length 6 model
#   1) load saved model
#   2) fit model to validation data
#   3) return results

def run_selected_model_validation(model_file):
    validation_model = load_model(model_file)
    validation_results = build_evaluation_dataframe()
    training_data_dir = '../data/training_data'
    model_filename_split = re.split('[_]', model_file)
    for filename in os.listdir(training_data_dir):  # iterate over all files in directory DIR
        if model_filename_split[0] in filename and model_filename_split[1] in filename and 'val' in filename:
            data_val = load_data(os.path.join(training_data_dir, filename))
            X_val = np.array(data_val['X'])
            y_val = np.array(data_val['y'])
            model_name = model_filename_split[2] + ' ' + model_filename_split[3] + ' with ' + model_filename_split[0] + ' encoding and trace length ' + model_filename_split[1]
            val_predict = validation_model.predict(X_val)
            model_results = pd.Series({'Model': model_name, 'F-score': f1_score(y_val, val_predict, average='macro'),
                       'Precision': precision_score(y_val, val_predict), 'Recall': recall_score(
                    y_val, val_predict), 'Accuracy': accuracy_score(y_val, val_predict)})
            validation_results = pd.concat([validation_results, model_results.to_frame().T], ignore_index=True)

    validation_results.to_csv('../data/processed/validation_results.csv', index=False)
    print(tabulate(validation_results, headers="keys", tablefmt="github", showindex=False))


run_model_training_and_evaluation()

# run_selected_model_validation('boolean_6_decision_tree_model.sav')