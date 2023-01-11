import os
import re

import pandas as pd
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
import numpy as np
import sklearn
from tqdm.auto import tqdm
import pickle
import numpy

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import KFold,train_test_split, RandomizedSearchCV
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

    results = pd.DataFrame(columns=['Encoding', 'Model', 'F-score', 'Precision', 'Recall', 'Accuracy'], index=[0])
    return results



def train_test_decision_tree(X_train, y_train, X_test, y_test, results, encoding):

    dt = DecisionTreeClassifier(random_state=42)  # Create Decision Tree classifier object
    dt_fit = dt.fit(X_train, y_train)  # Train Decision Tree Classifier
    dt_predict = dt_fit.predict(X_test)  #Predict the response for test dataset

    new_results = pd.Series({'Encoding': encoding, 'Model': 'Decision Tree (Default)', 'F-score': f1_score(y_test, dt_predict, average='macro'), 'Precision': precision_score(y_test, dt_predict), 'Recall': recall_score(
        y_test, dt_predict), 'Accuracy': accuracy_score(y_test, dt_predict)})
    results = pd.concat([results, new_results.to_frame().T], ignore_index=True)

    # results.loc['Decision Tree (Default)', :] = [f1_score(y_test, dt_predict, average='macro'), precision_score(y_test, dt_predict), recall_score(
    #     y_test, dt_predict), accuracy_score(y_test, dt_predict)]
    # results.sort_values(by='F-score', ascending=False)

    tree.plot_tree(dt_fit)

def train_test_logistic_regression(X_train, y_train, X_test, y_test, results, encoding):
    lr = LogisticRegression(random_state=42)
    lr_fit = lr.fit(X_train, y_train)
    lr_predict = lr_fit.predict(X_test)

    new_results = pd.Series({'Encoding': encoding, 'Model': 'Logistic Regression (Default)', 'F-score': f1_score(y_test, lr_predict, average='macro'), 'Precision': precision_score(y_test, lr_predict), 'Recall': recall_score(
        y_test, lr_predict), 'Accuracy': accuracy_score(y_test, lr_predict)})
    results = pd.concat([results, new_results.to_frame().T], ignore_index=True)

    # results.loc['Logistic Regression (Default)', :] = [f1_score(y_test, lr_predict, average='macro'),
    #                                                       precision_score(y_test, lr_predict),
    #                                                       recall_score(y_test, lr_predict),
    #                                                       accuracy_score(y_test, lr_predict)]
    results.sort_values(by='F-score', ascending=False)


def train_test_logstic_regression_grid_search(X_train, y_train, X_test, y_test, results):
    # define grid search
    solvers = ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga']
    penalty = ['l1', 'l2', 'elasticnet', 'None']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    max_iteration = [20, 50, 100, 200, 500, 1000, 2000, 5000]
    grid = dict(solver=solvers, penalty=penalty, C=c_values, max_iter=max_iteration)

    lr = LogisticRegression(random_state=42)
    lr_grid = GridSearchCV(estimator=lr, param_grid=grid, n_jobs=-1, scoring='accuracy', error_score=0)
    lr_grid_fit = lr_grid.fit(X_train, y_train)

    cross_val_results = pd.DataFrame(cross_validate(lr_grid_fit.best_estimator_, X_train, y_train,
                                                    scoring=['f1_macro', 'precision_macro', 'recall_macro',
                                                             'accuracy']))

    results.loc['Logistic Regression (Hyper Parameter Tuning)', :] = cross_val_results[['test_f1_macro',
                                                                                           'test_precision_macro',
                                                                                           'test_recall_macro',
                                                                                           'test_accuracy']].mean().values

    results.sort_values(by='F-score', ascending=False)

results_df = build_evaluation_dataframe()
training_data = '../data/training_data'
for filename in tqdm(os.listdir(training_data)):  # iterate over all files in directory DIR
    if not filename.startswith('.'):  # do not process hidden files
        file_metadata = re.split('[_]', filename)  # splits the filename on '-' and '.' -> creates a list
        if file_metadata[2] == 'test':
            filename_test = filename
            filename_train = file_metadata
            filename_train[2] = 'train'
            filename_train = '_'.join(filename_train)
            filename_val = file_metadata
            filename_val[2] = 'val'
            filename_val = '_'.join(filename_val)
            encoding = file_metadata[0]
            data_test = load_data(os.path.join(training_data, filename_test))
            data_train = load_data(os.path.join(training_data, filename_train))
            data_val = load_data(os.path.join(training_data, filename_val))
            X_train_data = np.array(data_train['X'])
            X_test_data = np.array(data_test['X'])
            y_train_data = np.array(data_train['y'])
            y_test_data = np.array(data_test['y'])
            train_test_decision_tree(X_train=X_train_data, y_train=y_train_data, X_test=X_test_data, y_test=y_test_data,
                                     results=results_df, encoding=encoding)
            train_test_logistic_regression(X_train=X_train_data, y_train=y_train_data, X_test=X_test_data,
                                           y_test=y_test_data, results=results_df, encoding=encoding)
            print(results_df)
            print('hello')



# train_test_logstic_regression_grid_search(X_train=X_train_data, y_train=y_train_data, X_test=X_test_data, y_test=y_test_data,
#                              results=results_df)
print(results_df)



def other_models():
    # Logistic Regression


    # Training the Logistic Regression model on the Training set
    lr = LogisticRegression(random_state=42)
    lr_fit = lr.fit(X_train_data, y_train_data)
    # Predicting the Test set results
    lr_predict = lr_fit.predict(X_test_data)

    results_df.loc['Logistic Regression (Default)',:] = [f1_score(y_test_data, lr_predict, average='macro'), precision_score(y_test_data, lr_predict), recall_score(y_test_data, lr_predict), accuracy_score(y_test_data, lr_predict)]
    results_df.sort_values(by='F-score', ascending=False)

    solvers = ['lbfgs','newton-cg','liblinear','sag','saga']
    penalty = ['l1', 'l2', 'elasticnet', 'None']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    max_iteration= [20, 50, 100, 200, 500, 1000, 2000, 5000]
    # define grid search
    grid = dict(solver=solvers,penalty=penalty,C=c_values,max_iter=max_iteration)

    lr_grid = GridSearchCV(estimator=lr, param_grid=grid, n_jobs=-1, scoring='accuracy',error_score=0)
    lr_grid_fit = lr_grid.fit(X_train_data, y_train_data)

    cross_val_results = pd.DataFrame(cross_validate(lr_grid.best_estimator_, X_train_data, y_train_data, scoring = ['f1_macro', 'precision_macro', 'recall_macro', 'accuracy']))

    results_df.loc['Logistic Regression (Hyper Parameter Tuning)',:] = cross_val_results[['test_f1_macro',
           'test_precision_macro', 'test_recall_macro','test_accuracy']].mean().values

    results_df.sort_values(by='F-score', ascending=False)

    # Random Forest Classifier

    rf = RandomForestClassifier()
    n_estimators = [1,5,10,40,100,200,500,1000,2000,5000]
    max_features = ['sqrt', 'log2']
    max_depth = [1,3,5,7,9,11,13,15,20]
    criterion= ['gini', 'entropy']

    grid = dict(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, criterion=criterion)

    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = grid, n_iter = 100, verbose=2, random_state=42, n_jobs = -1)
    rf_random_fit = rf_random.fit(X_train_data, y_train_data)

    cross_val_results = pd.DataFrame(cross_validate(rf_random.best_estimator_, X_train_data, y_train_data, scoring = ['f1_macro', 'precision_macro', 'recall_macro', 'accuracy']))

    results_df.loc['Random Forest (Randomized Parameters)',:] = cross_val_results[['test_f1_macro',
           'test_precision_macro', 'test_recall_macro','test_accuracy']].mean().values

    results_df.sort_values(by='F-score', ascending=False)

    # MLP (Neural Network) Classifier

    sizes = [2*i for i in range(1,5)]
    sizes = sizes + [[1*i,1*i] for i in range(1,5)]
    sizes = sizes + [[1*i,1*i, 1*i] for i in range(1,5)]

    decays = [3,4,5,6,7,8]

    nnet = MLPClassifier(alpha=0,
                         activation='logistic',
                               max_iter=500,
                               solver='lbfgs',
                               random_state=42)

    nnet_grid = GridSearchCV(estimator=nnet,
                       scoring=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy'],
                       param_grid={'hidden_layer_sizes': sizes,
                                  'alpha': decays,},
                       return_train_score=True,
                       refit='f1_macro')

    nnet_grid_fit = nnet_grid.fit(X_train_data, y_train_data)

    cross_val_results = pd.DataFrame(cross_validate(nnet_grid.best_estimator_, X_train_data, y_train_data, scoring = ['f1_macro', 'precision_macro', 'recall_macro', 'accuracy']))

    results_df.loc['MLP (Hyper Parameter Tuning)',:] = cross_val_results[['test_f1_macro',
           'test_precision_macro', 'test_recall_macro','test_accuracy']].mean().values

    results_df.sort_values(by='F-score', ascending=False)

    print(nnet_grid.best_params_)
    print(nnet_grid.best_estimator_.n_iter_)
    print(nnet_grid.best_estimator_.n_layers_)

