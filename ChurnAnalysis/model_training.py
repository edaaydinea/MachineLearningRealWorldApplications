# -*- coding: utf-8 -*-
"""
Model Training Module

This module contains the ModelTraining class which handles the training,
evaluation, and tuning of the machine learning model.
"""

#### Importing Libraries ####

import random
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from config import LOGISTIC_REGRESSION_PARAMS, GRID_SEARCH_PARAMS, FEATURE_IMPORTANCE_DROP_COLS
from data_preprocessing import DataPreprocessing
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelTraining:
    """
    A class used to train and evaluate a machine learning model.

    Attributes
    ----------
    data_path : str
        Path to the dataset file.
    dataset : pd.DataFrame
        The dataset loaded from the data_path.
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    y_train : pd.Series
        Training labels.
    y_test : pd.Series
        Test labels.
    classifier : LogisticRegression
        The logistic regression model.

    Methods
    -------
    preprocess_data():
        Preprocesses the dataset for model training.
    build_model():
        Builds and trains the logistic regression model.
    evaluate_model():
        Evaluates the model on the test set.
    cross_validate_model():
        Performs k-fold cross-validation on the model.
    tune_model():
        Tunes the model using grid search.
    save_results(y_pred, test_identity):
        Saves the final results to a DataFrame.
    """

    def __init__(self, data_path, logistic_regression_params, grid_search_params):
        """
        Constructs all the necessary attributes for the ModelTraining object.

        Parameters
        ----------
        data_path : str
            Path to the dataset file.
        logistic_regression_params : dict
            Parameters for the logistic regression model.
        grid_search_params : dict
            Parameters for grid search.
        """
        self.data_path = data_path
        self.dataset = pd.read_csv(data_path)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.classifier = None
        self.logistic_regression_params = logistic_regression_params
        self.grid_search_params = grid_search_params
        self.user_identifier = None


    def preprocess_data(self):
        """Preprocesses the dataset for model training."""
        self._split_data()
        self._balance_data()
        self._scale_features()

    def _data_preparation(self):
        self.user_identifier = self.dataset["user"]
        response = self.dataset["churn"]
        self.dataset = self.dataset.drop(columns="churn")
        return response
    
    def _feature_engineering(self):
        """Performs feature engineering on the dataset."""
        logging.info("Performing feature engineering on the dataset")
        self.dataset = pd.get_dummies(self.dataset)
        self.dataset = self.dataset.drop(columns=FEATURE_IMPORTANCE_DROP_COLS)

    def _split_data(self):
        """Splits the dataset into training and test sets."""
        response = self._data_preparation()
        self._feature_engineering()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataset, response, test_size=0.2, random_state=0)

        
    def _balance_data(self):
        pos_index = self.y_train[self.y_train.values == 1].index
        neg_index = self.y_train[self.y_train.values == 0].index
        
        if len(pos_index) > len(neg_index):
            higher = pos_index
            lower = neg_index
        else:
            higher = neg_index
            lower = pos_index
            
        random.seed(0)
        higher = np.random.choice(higher, size=len(lower))
        lower = np.asarray(lower)
        new_indexes = np.concatenate((lower, higher))
        
        self.X_train = self.X_train.loc[new_indexes,]
        self.y_train = self.y_train[new_indexes]


    def _scale_features(self):
        """Scales the features using StandardScaler."""
        sc_X = StandardScaler()
        X_train2 = pd.DataFrame(sc_X.fit_transform(self.X_train))
        X_test2 = pd.DataFrame(sc_X.transform(self.X_test))
        X_train2.columns = self.X_train.columns.values
        X_test2.columns = self.X_test.columns.values
        X_train2.index = self.X_train.index.values
        X_test2.index = self.X_test.index.values
        self.X_train = X_train2
        self.X_test = X_test2

    def build_model(self):
        """Builds and trains the logistic regression model."""
        self.classifier = LogisticRegression(**self.logistic_regression_params)
        self.classifier.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """Evaluates the model on the test set."""
        self.y_pred = self.classifier.predict(self.X_test)
        cm = self._compute_confusion_matrix()
        accuracy, precision, recall, f1 = self._compute_metrics()
        self._plot_confusion_matrix(cm)
        print("Test Data Accuracy: %0.4f" % accuracy)
        return accuracy, precision, recall, f1

    def _compute_confusion_matrix(self):
        """Computes the confusion matrix."""
        return confusion_matrix(self.y_test, self.y_pred)

    def _compute_metrics(self):
        """Computes accuracy, precision, recall, and f1 score."""
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        return accuracy, precision, recall, f1

    def _plot_confusion_matrix(self, cm):
        """Plots the confusion matrix."""
        df_cm = pd.DataFrame(cm, index=(0, 1), columns=(0, 1))
        plt.figure(figsize=(10, 7))
        sn.set_theme(font_scale=1.4)
        sn.heatmap(df_cm, annot=True, fmt='g')

    def cross_validate_model(self):
        """Performs k-fold cross-validation on the model."""
        accuracies = cross_val_score(estimator=self.classifier, X=self.X_train, y=self.y_train, cv=10)
        print("SVM Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))
        return accuracies.mean(), accuracies.std() * 2

    def tune_model(self):
        """Tunes the model using grid search."""
        grid_search = GridSearchCV(estimator=self.classifier, param_grid=self.grid_search_params, scoring="accuracy", cv=10, n_jobs=-1)
        t0 = time.time()
        grid_search = grid_search.fit(self.X_train, self.y_train)
        t1 = time.time()
        print("Took %0.2f seconds" % (t1 - t0))
        best_accuracy = grid_search.best_score_
        best_parameters = grid_search.best_params_
        best_score = grid_search.best_score_
        return best_accuracy, best_parameters, best_score

    def save_results(self):
        """Saves the final results to a DataFrame."""
        final_results = pd.concat([self.y_test, self.user_identifier], axis=1).dropna()
        final_results['predicted_churn'] = self.y_pred
        final_results = final_results[['user', 'churn', 'predicted_churn']].reset_index(drop=True)
        return final_results
